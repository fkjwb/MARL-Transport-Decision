from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from algos.jointppo.buffer import RolloutBuffer
from algos.jointppo.order_module import acyclicity_constraint, depth_constraint
from algos.jointppo.policy import JointActorCritic


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    train_epochs: int = 4
    minibatch_size: int = 256
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    actor_betas: tuple[float, float] = (0.9, 0.999)
    actor_eps: float = 1e-8
    actor_weight_decay: float = 0.0
    critic_betas: tuple[float, float] = (0.9, 0.999)
    critic_eps: float = 1e-8
    critic_weight_decay: float = 0.0


@dataclass
class OrderConfig:
    """Decision-order module training config (REINFORCE + AugLag)."""

    lr: float = 3e-4
    train_epochs: int = 1
    minibatch_size: int = 256

    mu_init: float = 1e-3
    rho: float = 1.3
    mu_max: float = 1e9
    tol: float = 1e-7
    depth_k: int = 3

    # Stabilize order constraints to avoid collapsing to near-empty graphs.
    normalize_constraints: bool = True
    h_coef: float = 1.0
    c_coef: float = 1.0

    max_grad_norm: float = 1.0
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0


class EntropyScheduler:
    def __init__(self, base: float, decay: Optional[float] = None):
        self.base = float(base)
        self.decay = decay

    def coef(self, iteration: int) -> float:
        if self.decay is None:
            return self.base
        return float(self.base * math.exp(-self.decay * iteration))


class JointPPOTrainer:
    """Joint PPO trainer for truck/belt actor and order module."""

    def __init__(
        self,
        model: JointActorCritic,
        buffer: RolloutBuffer,
        ppo_cfg: PPOConfig,
        order_cfg: Optional[OrderConfig] = None,
        device: str = "cpu",
        entropy_scheduler: Optional[EntropyScheduler] = None,
    ):
        self.model = model
        self.buffer = buffer
        self.ppo_cfg = ppo_cfg
        self.order_cfg = order_cfg or OrderConfig(depth_k=3)
        self.device = device

        self.entropy_scheduler = entropy_scheduler or EntropyScheduler(ppo_cfg.ent_coef, None)

        # PPO update: actor(truck+belt) and critic use independent optimizer groups.
        self._truck_actor_params = list(self.model.truck_actor.parameters())
        self._belt_actor_params = list(self.model.belt_actor.parameters())
        self._critic_params = list(self.model.critic.parameters())
        actor_params = self._truck_actor_params + self._belt_actor_params
        critic_params = self._critic_params
        self._ppo_params = actor_params + critic_params

        self.opt_ppo = torch.optim.Adam(
            [
                {
                    "params": actor_params,
                    "lr": float(self.ppo_cfg.lr_actor),
                    "betas": tuple(self.ppo_cfg.actor_betas),
                    "eps": float(self.ppo_cfg.actor_eps),
                    "weight_decay": float(self.ppo_cfg.actor_weight_decay),
                },
                {
                    "params": critic_params,
                    "lr": float(self.ppo_cfg.lr_critic),
                    "betas": tuple(self.ppo_cfg.critic_betas),
                    "eps": float(self.ppo_cfg.critic_eps),
                    "weight_decay": float(self.ppo_cfg.critic_weight_decay),
                },
            ]
        )

        self.opt_order = torch.optim.Adam(
            self.model.order_module.parameters(),
            lr=float(self.order_cfg.lr),
            betas=tuple(self.order_cfg.betas),
            eps=float(self.order_cfg.eps),
            weight_decay=float(self.order_cfg.weight_decay),
        )

        # AugLag multipliers for h(W), c(W^k).
        self.lambda_h = torch.zeros((), device=self.device)
        self.lambda_c = torch.zeros((), device=self.device)
        self.mu_h = torch.tensor(float(self.order_cfg.mu_init), device=self.device)
        self.mu_c = torch.tensor(float(self.order_cfg.mu_init), device=self.device)

    def _aug_lag_terms(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.lambda_h * h + 0.5 * self.mu_h * (h * h) + self.lambda_c * c + 0.5 * self.mu_c * (c * c)

    @torch.no_grad()
    def _update_multipliers(self, h_val: float, c_val: float) -> None:
        h = float(h_val)
        c = float(c_val)

        self.lambda_h += self.mu_h * torch.tensor(h, device=self.device)
        self.lambda_c += self.mu_c * torch.tensor(c, device=self.device)

        if abs(h) > float(self.order_cfg.tol):
            self.mu_h = torch.minimum(
                self.mu_h * float(self.order_cfg.rho),
                torch.tensor(float(self.order_cfg.mu_max), device=self.device),
            )
        if abs(c) > float(self.order_cfg.tol):
            self.mu_c = torch.minimum(
                self.mu_c * float(self.order_cfg.rho),
                torch.tensor(float(self.order_cfg.mu_max), device=self.device),
            )

    def _scaled_constraints(self, probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return raw and scaled constraints."""
        h_raw = acyclicity_constraint(probs)
        c_raw = depth_constraint(probs, k=int(self.order_cfg.depth_k))

        h = h_raw
        c = c_raw
        if bool(self.order_cfg.normalize_constraints):
            B = int(probs.shape[0])
            depth_k = max(int(self.order_cfg.depth_k), 1)
            h = h / float(max(B * (B - 1), 1))
            c = c / float(max(B**depth_k, 1))

        h = h * float(self.order_cfg.h_coef)
        c = c * float(self.order_cfg.c_coef)
        return h_raw, c_raw, h, c

    @staticmethod
    def _grad_norm(params) -> float:
        total_sq = 0.0
        for p in params:
            if p.grad is None:
                continue
            g = p.grad.detach()
            total_sq += float((g * g).sum().item())
        return math.sqrt(total_sq) if total_sq > 0.0 else 0.0

    def update(self, iteration: int) -> Dict[str, float]:
        batch = self.buffer.get()
        ent_coef = self.entropy_scheduler.coef(iteration)

        # ---------------- PPO ----------------
        T = batch.obs_critic.shape[0]
        idx = torch.randperm(T, device=self.device)

        pi_losses, v_losses, ent_means, total_losses = [], [], [], []
        truck_actor_grad_norms, belt_actor_grad_norms, critic_grad_norms = [], [], []

        for _ in range(self.ppo_cfg.train_epochs):
            for start in range(0, T, self.ppo_cfg.minibatch_size):
                mb = idx[start : start + self.ppo_cfg.minibatch_size]

                new_logprob, entropy, value = self.model.evaluate_actions(
                    batch.obs_truck[mb],
                    batch.obs_belt[mb],
                    batch.obs_critic[mb],
                    batch.act_truck[mb],
                    batch.act_belt[mb],
                    batch.belt_order[mb],
                )

                old_logprob = batch.logprob[mb]
                ratio = torch.exp(new_logprob - old_logprob)

                adv = batch.advantages[mb]
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - self.ppo_cfg.clip_ratio, 1.0 + self.ppo_cfg.clip_ratio) * adv
                pi_loss = -torch.min(surr1, surr2).mean()

                ret = batch.returns[mb]
                v_loss = ((value - ret) ** 2).mean()
                ent_loss = -entropy.mean()

                loss = pi_loss + self.ppo_cfg.vf_coef * v_loss + ent_coef * ent_loss

                self.opt_ppo.zero_grad(set_to_none=True)
                loss.backward()
                truck_actor_grad_norms.append(self._grad_norm(self._truck_actor_params))
                belt_actor_grad_norms.append(self._grad_norm(self._belt_actor_params))
                critic_grad_norms.append(self._grad_norm(self._critic_params))
                nn.utils.clip_grad_norm_(self._ppo_params, self.ppo_cfg.max_grad_norm)
                self.opt_ppo.step()

                pi_losses.append(float(pi_loss.item()))
                v_losses.append(float(v_loss.item()))
                ent_means.append(float(entropy.mean().item()))
                total_losses.append(float(loss.item()))

        # -------------- ORDER --------------
        order_losses = []
        h_raw_vals, c_raw_vals = [], []
        h_scaled_vals, c_scaled_vals = [], []

        idx2 = torch.randperm(T, device=self.device)

        for _ in range(int(self.order_cfg.train_epochs)):
            for start in range(0, T, int(self.order_cfg.minibatch_size)):
                mb = idx2[start : start + int(self.order_cfg.minibatch_size)]

                loss_mb = 0.0
                h_raw_mb = 0.0
                c_raw_mb = 0.0
                h_scaled_mb = 0.0
                c_scaled_mb = 0.0

                for t in mb.tolist():
                    obs_belt = batch.obs_belt[t]  # [B,D]
                    order_in = obs_belt

                    adj_sample = batch.order_adj[t]  # [B,B]
                    adv_t = batch.adv_raw[t].detach()

                    logp, _, probs = self.model.order_module.logprob_of_adj(order_in, adj_sample)
                    h_raw, c_raw, h_scaled, c_scaled = self._scaled_constraints(probs)

                    reinforce = -(adv_t * logp)
                    aug = self._aug_lag_terms(h_scaled, c_scaled)
                    loss_t = reinforce + aug

                    loss_mb = loss_mb + loss_t
                    h_raw_mb = h_raw_mb + h_raw.detach()
                    c_raw_mb = c_raw_mb + c_raw.detach()
                    h_scaled_mb = h_scaled_mb + h_scaled.detach()
                    c_scaled_mb = c_scaled_mb + c_scaled.detach()

                loss_mb = loss_mb / max(len(mb), 1)
                h_raw_mb = h_raw_mb / max(len(mb), 1)
                c_raw_mb = c_raw_mb / max(len(mb), 1)
                h_scaled_mb = h_scaled_mb / max(len(mb), 1)
                c_scaled_mb = c_scaled_mb / max(len(mb), 1)

                self.opt_order.zero_grad(set_to_none=True)
                loss_mb.backward()
                nn.utils.clip_grad_norm_(self.model.order_module.parameters(), float(self.order_cfg.max_grad_norm))
                self.opt_order.step()

                order_losses.append(float(loss_mb.item()))
                h_raw_vals.append(float(h_raw_mb.item()))
                c_raw_vals.append(float(c_raw_mb.item()))
                h_scaled_vals.append(float(h_scaled_mb.item()))
                c_scaled_vals.append(float(c_scaled_mb.item()))

        h_raw_mean = float(sum(h_raw_vals) / max(len(h_raw_vals), 1)) if h_raw_vals else 0.0
        c_raw_mean = float(sum(c_raw_vals) / max(len(c_raw_vals), 1)) if c_raw_vals else 0.0
        h_scaled_mean = float(sum(h_scaled_vals) / max(len(h_scaled_vals), 1)) if h_scaled_vals else 0.0
        c_scaled_mean = float(sum(c_scaled_vals) / max(len(c_scaled_vals), 1)) if c_scaled_vals else 0.0
        self._update_multipliers(h_scaled_mean, c_scaled_mean)

        return {
            "pi_loss": float(sum(pi_losses) / max(len(pi_losses), 1)),
            "v_loss": float(sum(v_losses) / max(len(v_losses), 1)),
            "entropy": float(sum(ent_means) / max(len(ent_means), 1)),
            "total_loss": float(sum(total_losses) / max(len(total_losses), 1)),
            "ent_coef": float(ent_coef),
            "truck_actor_grad_norm": float(sum(truck_actor_grad_norms) / max(len(truck_actor_grad_norms), 1)),
            "belt_actor_grad_norm": float(sum(belt_actor_grad_norms) / max(len(belt_actor_grad_norms), 1)),
            "critic_grad_norm": float(sum(critic_grad_norms) / max(len(critic_grad_norms), 1)),
            "order_loss": float(sum(order_losses) / max(len(order_losses), 1)) if order_losses else 0.0,
            "h(W)": float(h_raw_mean),
            "c(W^k)": float(c_raw_mean),
            "h_scaled": float(h_scaled_mean),
            "c_scaled": float(c_scaled_mean),
            "mu_h": float(self.mu_h.item()),
            "mu_c": float(self.mu_c.item()),
            "lambda_h": float(self.lambda_h.item()),
            "lambda_c": float(self.lambda_c.item()),
        }
