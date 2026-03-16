from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from algos.jointppo.order_module import GraphOrderModule  # type: ignore


def activation_factory(name: str) -> nn.Module:
    n = str(name).lower()
    if n == "tanh":
        return nn.Tanh()
    if n == "relu":
        return nn.ReLU()
    if n == "gelu":
        return nn.GELU()
    if n == "silu":
        return nn.SiLU()
    if n in ("leakyrelu", "leaky_relu"):
        return nn.LeakyReLU(negative_slope=0.01)
    raise ValueError(f"Unknown activation: {name}")


class BetaDiagDist:
    """独立 Beta 分布（每个维度一个 Beta）。"""

    def __init__(self, alpha: torch.Tensor, beta: torch.Tensor, eps: float = 1e-6):
        self.alpha = torch.clamp(alpha, min=eps)
        self.beta = torch.clamp(beta, min=eps)
        self.dist = torch.distributions.Beta(self.alpha, self.beta)

    def sample(self) -> torch.Tensor:
        return self.dist.rsample()

    def mean(self) -> torch.Tensor:
        return self.dist.mean

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return self.dist.log_prob(x)

    def entropy(self) -> torch.Tensor:
        return self.dist.entropy()


def mlp(in_dim: int, hidden: List[int], out_dim: int, activation: str = "tanh") -> nn.Sequential:
    act = activation_factory(activation)
    layers: List[nn.Module] = []
    last = int(in_dim)
    for h in hidden:
        layers.append(nn.Linear(last, int(h)))
        layers.append(act)
        last = int(h)
    layers.append(nn.Linear(last, int(out_dim)))
    return nn.Sequential(*layers)


class TruckActorMLP(nn.Module):
    """卡车 actor：MLP 输出 Beta(alpha,beta) 参数。"""

    def __init__(self, obs_dim: int, K: int, hidden: List[int], activation: str = "tanh"):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.K = int(K)
        self.net = mlp(obs_dim, hidden, out_dim=2 * K, activation=activation)

    def sample_actions(self, obs: torch.Tensor, deterministic: bool = False):
        """obs: [Nbatch, D]"""
        ab = self.net(obs)
        alpha, beta = torch.chunk(ab, 2, dim=-1)
        alpha = F.softplus(alpha) + 1e-3
        beta = F.softplus(beta) + 1e-3
        dist = BetaDiagDist(alpha, beta)
        a = dist.mean() if deterministic else dist.sample()
        logp = dist.log_prob(a).sum(dim=-1)
        ent = dist.entropy().sum(dim=-1)
        return a, logp, ent

    def evaluate_actions(self, obs: torch.Tensor, act: torch.Tensor):
        ab = self.net(obs)
        alpha, beta = torch.chunk(ab, 2, dim=-1)
        alpha = F.softplus(alpha) + 1e-3
        beta = F.softplus(beta) + 1e-3
        dist = BetaDiagDist(alpha, beta)
        logp = dist.log_prob(act).sum(dim=-1)
        ent = dist.entropy().sum(dim=-1)
        return logp, ent


class BeltActorTransformer(nn.Module):
    """皮带 actor：Transformer Encoder-Decoder，自回归输出离散 one-hot 动作(K+1)。"""

    def __init__(
        self,
        obs_dim: int,
        num_combos: int,
        K: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.num_combos = int(num_combos)
        self.K = int(K)
        self.d_model = int(d_model)

        self.obs_proj = nn.Linear(self.obs_dim, self.d_model)
        self.combo_embed = nn.Embedding(self.num_combos, self.d_model)
        self.pos_embed = nn.Embedding(self.num_combos, self.d_model)  # 决策步位置
        self.bos = nn.Parameter(torch.zeros(self.d_model))
        self.prev_act_proj = nn.Linear(self.K + 1, self.d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(nhead),
            dim_feedforward=4 * self.d_model,
            dropout=float(dropout),
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(num_layers))

        dec_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=int(nhead),
            dim_feedforward=4 * self.d_model,
            dropout=float(dropout),
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=int(num_layers))

        # logits over (K materials + 1 no-transport)
        self.head = nn.Linear(self.d_model, self.K + 1)

    def _encode(self, obs_belt: torch.Tensor) -> torch.Tensor:
        x = self.obs_proj(obs_belt)  # [Nbatch,B,d]
        combo_ids = torch.arange(self.num_combos, device=obs_belt.device).unsqueeze(0)
        x = x + self.combo_embed(combo_ids)
        return self.encoder(x)

    def _causal_mask(self, L: int, device: torch.device) -> torch.Tensor:
        m = torch.full((L, L), float("-inf"), device=device)
        return torch.triu(m, diagonal=1)

    def sample_actions(self, obs_belt: torch.Tensor, order: torch.Tensor, deterministic: bool = False):
        """obs_belt: [Nbatch,B,D], order: [Nbatch,B]
        返回:
          act_combo: [Nbatch,B,K+1] one-hot（最后一维=不运输）
          logp_sum: [Nbatch]
          ent_sum:  [Nbatch]
        """
        Nbatch, B, _ = obs_belt.shape
        memory = self._encode(obs_belt)

        act_seq = torch.zeros((Nbatch, B, self.K + 1), device=obs_belt.device, dtype=obs_belt.dtype)
        logp_sum = torch.zeros((Nbatch,), device=obs_belt.device)
        ent_sum = torch.zeros((Nbatch,), device=obs_belt.device)

        for s in range(B):
            combo_ids = order[:, : s + 1]  # [Nbatch,s+1]
            pos_ids = torch.arange(s + 1, device=obs_belt.device).unsqueeze(0).expand(Nbatch, s + 1)

            prev_seq = torch.zeros((Nbatch, s + 1, self.K + 1), device=obs_belt.device, dtype=obs_belt.dtype)
            if s >= 1:
                prev_seq[:, 1:, :] = act_seq[:, :s, :]

            # token = BOS + id + pos + prev_action_token
            tgt = self.combo_embed(combo_ids) + self.pos_embed(pos_ids) + self.prev_act_proj(prev_seq)
            tgt = tgt + self.bos.view(1, 1, -1)

            out = self.decoder(tgt=tgt, memory=memory, tgt_mask=self._causal_mask(s + 1, obs_belt.device))
            h = out[:, -1, :]
            logits = self.head(h)  # [Nbatch,K+1]
            dist = torch.distributions.Categorical(logits=logits)

            idx = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
            lp = dist.log_prob(idx)
            en = dist.entropy()

            a = F.one_hot(idx, num_classes=self.K + 1).to(dtype=obs_belt.dtype)

            act_seq[:, s, :] = a
            logp_sum += lp
            ent_sum += en

        # scatter 回 combo 维度（order 是“决策顺序 -> combo_id”的排列）
        act_combo = torch.zeros((Nbatch, B, self.K + 1), device=obs_belt.device, dtype=obs_belt.dtype)
        act_combo.scatter_(dim=1, index=order.unsqueeze(-1).expand(Nbatch, B, self.K + 1), src=act_seq)
        return act_combo, logp_sum, ent_sum

    def evaluate_actions(self, obs_belt: torch.Tensor, order: torch.Tensor, act_combo: torch.Tensor):
        """teacher forcing 计算 logprob/entropy。
        act_combo: [Nbatch,B,K+1] one-hot（combo 维度）
        """
        Nbatch, B, _ = obs_belt.shape
        memory = self._encode(obs_belt)

        idx = order.unsqueeze(-1).expand(Nbatch, B, self.K + 1)
        act_seq = torch.gather(act_combo, dim=1, index=idx)  # [Nbatch,B,K+1]

        prev_seq = torch.zeros_like(act_seq)
        prev_seq[:, 1:, :] = act_seq[:, :-1, :]

        combo_tok = self.combo_embed(order)
        pos_ids = torch.arange(B, device=obs_belt.device).unsqueeze(0).expand(Nbatch, B)
        pos_tok = self.pos_embed(pos_ids)
        prev_tok = self.prev_act_proj(prev_seq)

        tgt = combo_tok + pos_tok + prev_tok
        tgt = tgt + self.bos.view(1, 1, -1)

        out = self.decoder(tgt=tgt, memory=memory, tgt_mask=self._causal_mask(B, obs_belt.device))
        logits = self.head(out)  # [Nbatch,B,K+1]
        dist = torch.distributions.Categorical(logits=logits)

        act_idx = torch.argmax(act_seq, dim=-1)  # [Nbatch,B]
        logp = dist.log_prob(act_idx).sum(dim=-1)
        ent = dist.entropy().sum(dim=-1)
        return logp, ent


class CriticMLP(nn.Module):
    def __init__(self, obs_dim: int, hidden: List[int], activation: str = "tanh"):
        super().__init__()
        self.net = mlp(obs_dim, hidden, out_dim=1, activation=activation)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


@dataclass
class PolicyOutput:
    action_env: Dict[str, np.ndarray]
    action_aux: Dict[str, np.ndarray]
    logprob_ppo: float
    value: float


class JointActorCritic(nn.Module):
    """拆分网络的联合策略：truck MLP、belt Transformer、critic MLP、order GAT+MLP。"""

    def __init__(
        self,
        Et: int,
        B: int,
        K: int,
        truck_obs_dim: int,
        belt_obs_dim: int,
        critic_obs_dim: int,
        truck_actor_mlp_hidden: List[int],
        critic_mlp_hidden: List[int],
        activation: str,
        transformer_cfg: Dict[str, int],
        order_cfg: Dict[str, int],
    ):
        super().__init__()
        self.Et = int(Et)
        self.B = int(B)
        self.K = int(K)

        self.truck_actor = TruckActorMLP(truck_obs_dim, K, truck_actor_mlp_hidden, activation=activation)
        self.belt_actor = BeltActorTransformer(
            obs_dim=belt_obs_dim,
            num_combos=B,
            K=K,
            d_model=int(transformer_cfg.get("d_model", 128)),
            nhead=int(transformer_cfg.get("nhead", 4)),
            num_layers=int(transformer_cfg.get("num_layers", 2)),
            dropout=float(transformer_cfg.get("dropout", 0.0)),
        )
        self.critic = CriticMLP(critic_obs_dim, critic_mlp_hidden, activation=activation)
        order_obs_dim = int(belt_obs_dim)

        self.order_module = GraphOrderModule(
            obs_dim=order_obs_dim,
            num_agents=B,
            gat_hidden=int(order_cfg.get("gat_hidden", 64)),
            gat_heads=int(order_cfg.get("gat_heads", 4)),
            gat_layers=int(order_cfg.get("gat_layers", 2)),
            gat_dropout=float(order_cfg.get("gat_dropout", 0.0)),
            decoder_hidden=int(order_cfg.get("decoder_hidden", 256)),
            activation=str(order_cfg.get("activation", activation)),
            depth_k=int(order_cfg.get("depth_k", 3)),
        )

    def act(self, obs: Dict[str, np.ndarray], deterministic: bool = False) -> PolicyOutput:
        device = next(self.parameters()).device
        with torch.no_grad():
            obs_truck = torch.tensor(obs["truck_actor"], dtype=torch.float32, device=device)
            obs_belt = torch.tensor(obs["belt_actor"], dtype=torch.float32, device=device)
            obs_critic = torch.tensor(obs["critic"], dtype=torch.float32, device=device)

            order_in = obs_belt
            order_out = self.order_module.sample_order(order_in, deterministic=deterministic)
            order = torch.tensor(order_out.order_indices, device=device, dtype=torch.int64).unsqueeze(0)

            a_truck, lp_truck, _ = self.truck_actor.sample_actions(obs_truck, deterministic=deterministic)
            a_belt, lp_belt, _ = self.belt_actor.sample_actions(
                obs_belt.unsqueeze(0), order=order, deterministic=deterministic
            )

            logprob = float((lp_truck.sum() + lp_belt.squeeze(0)).item())
            value = float(self.critic(obs_critic.unsqueeze(0)).item())

            action_env = {
                "truck": a_truck.cpu().numpy().reshape(-1).astype(np.float32),
                "belt": a_belt.squeeze(0).cpu().numpy().reshape(-1).astype(np.float32),
                "belt_order": np.asarray(order_out.order_indices, dtype=np.int64),
            }
            action_aux = {
                "order_adj": order_out.adj_sample.cpu().numpy().astype(np.float32),
            }

        return PolicyOutput(
            action_env=action_env,
            action_aux=action_aux,
            logprob_ppo=logprob,
            value=value,
        )

    def evaluate_actions(
        self,
        obs_truck: torch.Tensor,    # [T,Et,D]
        obs_belt: torch.Tensor,     # [T,B,D]
        obs_critic: torch.Tensor,   # [T,Dc]
        act_truck: torch.Tensor,    # [T,Et,K]
        act_belt: torch.Tensor,     # [T,B,K+1]
        belt_order: torch.Tensor,   # [T,B]
    ):
        T, Et, _ = obs_truck.shape

        logp_truck, ent_truck = self.truck_actor.evaluate_actions(
            obs_truck.reshape(T * Et, -1),
            act_truck.reshape(T * Et, -1),
        )
        logp_truck = logp_truck.view(T, Et).sum(dim=-1)
        ent_truck = ent_truck.view(T, Et).sum(dim=-1)

        logp_belt, ent_belt = self.belt_actor.evaluate_actions(obs_belt, order=belt_order, act_combo=act_belt)

        logp = logp_truck + logp_belt
        ent = ent_truck + ent_belt
        value = self.critic(obs_critic)
        return logp, ent, value
