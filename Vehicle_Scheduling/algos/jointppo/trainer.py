from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from .buffer import PPOBuffer


@dataclass
class TrainStats:
    pi_loss: float
    v_loss: float
    entropy_loss: float
    total_loss: float
    entropy_coef: float


class JointPPOTrainer:
    def __init__(self, policy, cfg: Dict, device: torch.device):
        self.policy = policy
        self.cfg = cfg
        self.device = device

        tcfg = cfg["train"]
        self.clip_ratio = float(tcfg["clip_ratio"])
        self.train_epochs = int(tcfg["train_epochs"])
        self.minibatch_size = int(tcfg["minibatch_size"])
        self.vf_coef = float(tcfg["vf_coef"])
        self.ent_coef = float(tcfg.get("ent_coef", 0.0))
        self.ent_decay = tcfg.get("ent_decay", None)
        self.max_grad_norm = float(tcfg.get("max_grad_norm", 0.5))

        actor_params = []
        actor_params += list(self.policy.gat.parameters())
        actor_params += list(self.policy.actor_input_proj.parameters())
        actor_params += list(self.policy.agent_pos_emb.parameters())
        actor_params += list(self.policy.encoder_blocks.parameters())
        actor_params += list(self.policy.action_emb.parameters())
        actor_params += list(self.policy.attr_proj.parameters())
        actor_params += list(self.policy.dec_pos_emb.parameters())
        actor_params += list(self.policy.memory_proj.parameters())
        actor_params += list(self.policy.decoder_blocks.parameters())
        actor_params += list(self.policy.policy_head.parameters())

        self.optimizer_actor = torch.optim.Adam(actor_params, lr=float(tcfg["lr_actor"]))
        self.optimizer_critic = torch.optim.Adam(self.policy.critic.parameters(), lr=float(tcfg["lr_critic"]))

    def current_entropy_coef(self, iteration: int) -> float:
        if self.ent_decay is None:
            return self.ent_coef
        return float(self.ent_coef * np.exp(-float(self.ent_decay) * iteration))

    def update(self, batch_np: Dict, iteration: int) -> TrainStats:
        batch = PPOBuffer.to_torch(batch_np, self.device)
        n = batch["actions"].shape[0]
        ent_coef = self.current_entropy_coef(iteration)

        pi_losses = []
        v_losses = []
        ent_losses = []
        total_losses = []

        idx_all = np.arange(n)
        for _ in range(self.train_epochs):
            np.random.shuffle(idx_all)
            for start in range(0, n, self.minibatch_size):
                idx = idx_all[start : start + self.minibatch_size]
                obs_mb = {k: v[idx] for k, v in batch["obs"].items()}
                act_mb = batch["actions"][idx]
                old_logp_mb = batch["logprobs"][idx]
                adv_mb = batch["advantages"][idx]
                ret_mb = batch["returns"][idx]

                new_logp, entropy, value = self.policy.evaluate_actions(obs_mb, act_mb)
                ratio = torch.exp(new_logp - old_logp_mb)
                surr1 = ratio * adv_mb
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv_mb
                pi_loss = -torch.min(surr1, surr2).mean()
                v_loss = F.mse_loss(value, ret_mb)
                entropy_loss = -entropy.mean()
                total_loss = pi_loss + self.vf_coef * v_loss + ent_coef * entropy_loss

                self.optimizer_actor.zero_grad()
                self.optimizer_critic.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer_actor.step()
                self.optimizer_critic.step()

                pi_losses.append(float(pi_loss.item()))
                v_losses.append(float(v_loss.item()))
                ent_losses.append(float((ent_coef * entropy_loss).item()))
                total_losses.append(float(total_loss.item()))

        return TrainStats(
            pi_loss=float(np.mean(pi_losses)) if pi_losses else 0.0,
            v_loss=float(np.mean(v_losses)) if v_losses else 0.0,
            entropy_loss=float(np.mean(ent_losses)) if ent_losses else 0.0,
            total_loss=float(np.mean(total_losses)) if total_losses else 0.0,
            entropy_coef=float(ent_coef),
        )