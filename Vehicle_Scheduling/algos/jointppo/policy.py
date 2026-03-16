from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def activation_factory(name: str) -> nn.Module:
    name = str(name).lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"unknown activation: {name}")


def mlp(in_dim: int, hidden: List[int], out_dim: int, activation: str) -> nn.Sequential:
    layers: List[nn.Module] = []
    last = int(in_dim)
    for h in hidden:
        layers.append(nn.Linear(last, int(h)))
        layers.append(activation_factory(activation))
        last = int(h)
    layers.append(nn.Linear(last, int(out_dim)))
    return nn.Sequential(*layers)


class DenseGATLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, heads: int, dropout: float, activation: str):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.heads = int(heads)
        self.dropout = nn.Dropout(float(dropout))
        self.act = activation_factory(activation)

        self.W = nn.Parameter(torch.empty(self.heads, self.in_dim, self.out_dim))
        self.a_src = nn.Parameter(torch.empty(self.heads, self.out_dim))
        self.a_dst = nn.Parameter(torch.empty(self.heads, self.out_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a_src.unsqueeze(-1))
        nn.init.xavier_uniform_(self.a_dst.unsqueeze(-1))

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        Wh = torch.einsum("bnf,hfo->bhno", x, self.W)
        alpha_src = torch.einsum("bhno,ho->bhn", Wh, self.a_src)
        alpha_dst = torch.einsum("bhno,ho->bhn", Wh, self.a_dst)
        e = alpha_src.unsqueeze(-1) + alpha_dst.unsqueeze(-2)
        e = F.leaky_relu(e, negative_slope=0.2)

        eye = torch.eye(N, device=adj.device, dtype=adj.dtype).unsqueeze(0)
        mask = ((adj + eye) > 0).unsqueeze(1)
        e = e.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(e, dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum("bhij,bhjo->bhio", attn, Wh)
        out = out.permute(0, 2, 1, 3).reshape(B, N, self.heads * self.out_dim)
        return self.act(out)


class GATEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, heads: int, layers: int, dropout: float, activation: str):
        super().__init__()
        self.layers = nn.ModuleList()
        cur_in = int(in_dim)
        head_out = max(1, int(hidden_dim) // int(heads))
        for _ in range(int(layers)):
            self.layers.append(DenseGATLayer(cur_in, head_out, heads, dropout, activation))
            cur_in = int(heads) * head_out
        self.out_dim = cur_in

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, adj)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float):
        super().__init__()
        self.d_model = int(d_model)
        self.heads = int(heads)
        assert self.d_model % self.heads == 0
        self.d_head = self.d_model // self.heads
        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.k_proj = nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model)
        self.o_proj = nn.Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(float(dropout))

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        return x.view(B, L, self.heads, self.d_head).transpose(1, 2)

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        B, H, L, Dh = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, H * Dh)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False) -> torch.Tensor:
        qh = self._split(self.q_proj(q))
        kh = self._split(self.k_proj(k))
        vh = self._split(self.v_proj(v))
        scores = torch.matmul(qh, kh.transpose(-2, -1)) / (self.d_head ** 0.5)
        if causal:
            Lq = q.shape[1]
            Lk = k.shape[1]
            mask = torch.triu(torch.ones((Lq, Lk), device=q.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, vh)
        out = self._merge(out)
        return self.o_proj(out)


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float, activation: str):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            activation_factory(activation),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.attn(x, x, x, causal=False)))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float, activation: str):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            activation_factory(activation),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        tgt = self.norm1(tgt + self.dropout(self.self_attn(tgt, tgt, tgt, causal=True)))
        tgt = self.norm2(tgt + self.dropout(self.cross_attn(tgt, memory, memory, causal=False)))
        tgt = self.norm3(tgt + self.dropout(self.ff(tgt)))
        return tgt


@dataclass
class PolicyActResult:
    actions: np.ndarray
    logprob: float
    entropy: float
    value: float


class JointPolicy(nn.Module):
    def __init__(self, cfg: Dict, env_info: Dict):
        super().__init__()
        self.cfg = cfg
        self.env_info = env_info

        self.N = int(env_info["num_nodes"])
        self.K = int(env_info["num_materials"])
        self.A = int(env_info["num_agents"])
        self.n_obs = int(env_info["n_obs"])
        self.action_dim = int(env_info["action_dim"])
        self.max_vehicle_id = int(env_info["max_vehicle_id"])
        self.max_capacity = float(env_info["max_capacity"])
        self.max_degree = float(env_info["max_degree"])

        mcfg = cfg["model"]
        act_name = str(mcfg.get("activation", "tanh"))

        self.d_hidden = int(mcfg["actor"]["d_hidden"])
        self.actor_heads = int(mcfg["actor"]["heads"])
        self.actor_layers = int(mcfg["actor"]["layers"])
        self.actor_dropout = float(mcfg["actor"]["dropout"])
        self.d_model = int(mcfg["actor"].get("d_model", self.d_hidden))

        gat_hidden = int(mcfg["gat"]["d_hidden_gat"])
        gat_heads = int(mcfg["gat"]["gat_heads"])
        gat_layers = int(mcfg["gat"]["gat_layers"])
        gat_dropout = float(mcfg["gat"]["gat_dropout"])

        node_raw_dim = self.n_obs * (self.N - 1) * self.K + 3
        self.gat = GATEncoder(node_raw_dim, gat_hidden, gat_heads, gat_layers, gat_dropout, act_name)
        self.d_gat = self.gat.out_dim

        actor_raw_dim = (self.N - 1) * (self.K * self.n_obs + self.d_gat) + 7
        self.actor_input_proj = mlp(actor_raw_dim, [self.d_hidden], self.d_hidden, act_name)
        self.agent_pos_emb = nn.Embedding(self.A, self.d_hidden)
        self.encoder_blocks = nn.ModuleList(
            [EncoderBlock(self.d_hidden, self.actor_heads, self.actor_dropout, act_name) for _ in range(self.actor_layers)]
        )

        self.action_emb = nn.Embedding(self.action_dim + 1, self.d_model)
        self.attr_proj = nn.Linear(3, self.d_model)
        self.dec_pos_emb = nn.Embedding(self.A + 2, self.d_model)
        self.bos_idx = self.action_dim
        self.memory_proj = nn.Linear(self.d_hidden, self.d_model)
        self.decoder_blocks = nn.ModuleList(
            [DecoderBlock(self.d_model, self.actor_heads, self.actor_dropout, act_name) for _ in range(self.actor_layers)]
        )
        self.policy_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            activation_factory(act_name),
            nn.Linear(self.d_model, self.action_dim),
        )

        critic_hidden = list(mcfg["critic"]["mlp_hidden"])
        critic_in_dim = self.N * ((self.N - 1) * self.K * self.n_obs + self.d_gat) + 2 * self.A + 2
        self.critic = mlp(critic_in_dim, critic_hidden, 1, act_name)

    def act(self, obs: Dict[str, np.ndarray], deterministic: bool = False) -> PolicyActResult:
        obs_t = {k: self._to_tensor(v) for k, v in obs.items()}
        with torch.no_grad():
            actions_t, logp_t, ent_t, value_t = self._forward_actor_critic(obs_t, actions=None, deterministic=deterministic)
        return PolicyActResult(
            actions=actions_t.squeeze(0).cpu().numpy().astype(np.int64),
            logprob=float(logp_t.item()),
            entropy=float(ent_t.item()),
            value=float(value_t.item()),
        )

    def evaluate_actions(self, obs: Dict[str, torch.Tensor], actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, logp, entropy, value = self._forward_actor_critic(obs, actions=actions, deterministic=False)
        return logp, entropy, value

    def _forward_actor_critic(
        self,
        obs: Dict[str, torch.Tensor],
        actions: torch.Tensor | None,
        deterministic: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _, node_emb = self._build_node_features(obs)
        actor_raw, decode_order, inv_order = self._build_actor_raw(obs, node_emb)
        actor_ctx = self._encode_agents(actor_raw)
        critic_in = self._build_critic_input(obs, node_emb)
        value = self.critic(critic_in).squeeze(-1)
        actions_ordered, logp, entropy = self._decode_actions(obs, actor_ctx, decode_order, actions, deterministic)
        actions_env = torch.gather(actions_ordered, 1, inv_order)
        return actions_env, logp, entropy, value

    def _build_node_features(self, obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        task = obs["node_task_window"].float()
        counts = obs["node_vehicle_counts"].float()
        degree = obs["node_degrees"].float() / max(self.max_degree, 1.0)
        adj = obs["adjacency"].float()
        B = task.shape[0]

        feat_list = []
        for src in range(self.N):
            dsts = [j for j in range(self.N) if j != src]
            local = task[:, src, dsts, :, :].reshape(B, -1)
            node_feat = torch.cat([local, counts[:, src, :], degree[:, src, :]], dim=-1)
            feat_list.append(node_feat)
        node_raw = torch.stack(feat_list, dim=1)
        node_emb = self.gat(node_raw, adj)
        return node_raw, node_emb

    def _build_actor_raw(self, obs: Dict[str, torch.Tensor], node_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        task = obs["node_task_window"].float()
        counts = obs["node_vehicle_counts"].float()
        vehicle_node = obs["vehicle_node"].long()
        vehicle_type = obs["vehicle_type"].float()
        vehicle_cap = obs["vehicle_cap"].float()
        vehicle_id = obs["vehicle_id"].long()
        prev_state = obs["vehicle_prev_state"].float()
        fi = obs["fi_satisfy"].float()

        B = vehicle_node.shape[0]
        raw_per_agent = []
        for a in range(self.A):
            src = vehicle_node[:, a]
            local_rows = []
            other_emb_rows = []
            counts_rows = []
            fi_rows = []
            src_norm = []
            for b in range(B):
                s = int(src[b].item())
                dsts = [j for j in range(self.N) if j != s]
                local_rows.append(task[b, s, dsts, :, :].reshape(-1))
                other_emb_rows.append(node_emb[b, dsts, :].reshape(-1))
                counts_rows.append(counts[b, s, :])
                fi_rows.append(fi[b, s].view(1))
                src_norm.append(torch.tensor([(s + 1) / max(self.N, 1)], device=task.device, dtype=torch.float32))
            local_rows_t = torch.stack(local_rows, dim=0)
            other_emb_rows_t = torch.stack(other_emb_rows, dim=0)
            counts_rows_t = torch.stack(counts_rows, dim=0)
            fi_rows_t = torch.stack(fi_rows, dim=0)
            src_norm_t = torch.stack(src_norm, dim=0)
            agent_feat = torch.cat(
                [
                    vehicle_type[:, a:a+1],
                    vehicle_cap[:, a:a+1] / max(self.max_capacity, 1.0),
                    src_norm_t,
                    local_rows_t,
                    counts_rows_t,
                    fi_rows_t,
                    prev_state[:, a:a+1] / 2.0,
                    other_emb_rows_t,
                ],
                dim=-1,
            )
            raw_per_agent.append(agent_feat)
        actor_raw = torch.stack(raw_per_agent, dim=1)

        decode_order = torch.argsort(vehicle_id, dim=1, descending=True)
        inv_order = torch.argsort(decode_order, dim=1)
        actor_raw = torch.gather(actor_raw, 1, decode_order.unsqueeze(-1).expand(-1, -1, actor_raw.shape[-1]))
        return actor_raw, decode_order, inv_order

    def _encode_agents(self, actor_raw: torch.Tensor) -> torch.Tensor:
        _, A, _ = actor_raw.shape
        x = self.actor_input_proj(actor_raw)
        pos = self.agent_pos_emb(torch.arange(A, device=actor_raw.device)).unsqueeze(0)
        x = x + pos
        for blk in self.encoder_blocks:
            x = blk(x)
        return x

    def _build_critic_input(self, obs: Dict[str, torch.Tensor], node_emb: torch.Tensor) -> torch.Tensor:
        vehicle_node = obs["vehicle_node"].float()
        prev_state = obs["vehicle_prev_state"].float()
        unused_prev = obs["n_unused_prev"].float() / max(self.A, 1)
        timestep = obs["timestep"].float()

        B = node_emb.shape[0]
        task = obs["node_task_window"].float()
        per_node = []
        for src in range(self.N):
            dsts = [j for j in range(self.N) if j != src]
            local = task[:, src, dsts, :, :].reshape(B, -1)
            per_node.append(torch.cat([local, node_emb[:, src, :]], dim=-1))
        node_part = torch.stack(per_node, dim=1).reshape(B, -1)
        return torch.cat(
            [
                node_part,
                (vehicle_node + 1.0) / max(self.N, 1),
                prev_state / 2.0,
                unused_prev,
                timestep,
            ],
            dim=-1,
        )

    def _decode_actions(
        self,
        obs: Dict[str, torch.Tensor],
        actor_ctx: torch.Tensor,
        decode_order: torch.Tensor,
        actions_env: torch.Tensor | None,
        deterministic: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = actor_ctx.device
        B = actor_ctx.shape[0]
        vehicle_id = obs["vehicle_id"].long()
        vehicle_cap = obs["vehicle_cap"].float()
        vehicle_type = obs["vehicle_type"].float()

        ordered_vehicle_id = torch.gather(vehicle_id, 1, decode_order)
        ordered_vehicle_cap = torch.gather(vehicle_cap, 1, decode_order)
        ordered_vehicle_type = torch.gather(vehicle_type, 1, decode_order)

        if actions_env is not None:
            actions_env = actions_env.long()
            actions_ordered = torch.gather(actions_env, 1, decode_order)
        else:
            actions_ordered = torch.zeros((B, self.A), device=device, dtype=torch.long)

        logp_sum = torch.zeros((B,), device=device)
        ent_sum = torch.zeros((B,), device=device)
        bos = self.action_emb.weight[self.bos_idx].view(1, 1, -1).expand(B, -1, -1)

        for s in range(self.A):
            mem = self.memory_proj(actor_ctx[:, s:s+1, :])
            attr = torch.stack(
                [
                    ordered_vehicle_id[:, s].float() / max(self.max_vehicle_id, 1),
                    ordered_vehicle_cap[:, s] / max(self.max_capacity, 1.0),
                    ordered_vehicle_type[:, s],
                ],
                dim=-1,
            )
            attr_tok = self.attr_proj(attr).unsqueeze(1)

            if s > 0:
                prev_tok = self.action_emb(actions_ordered[:, :s])
                tgt = torch.cat([bos, prev_tok, attr_tok], dim=1)
            else:
                tgt = torch.cat([bos, attr_tok], dim=1)

            pos = self.dec_pos_emb(torch.arange(tgt.shape[1], device=device)).unsqueeze(0)
            tgt = tgt + pos
            for blk in self.decoder_blocks:
                tgt = blk(tgt, mem)
            logits = self.policy_head(tgt[:, -1, :])
            dist = torch.distributions.Categorical(logits=logits)

            if actions_env is None:
                a = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
                actions_ordered[:, s] = a
            else:
                a = actions_ordered[:, s]

            logp_sum = logp_sum + dist.log_prob(a)
            ent_sum = ent_sum + dist.entropy()

        return actions_ordered, logp_sum, ent_sum

    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        x = torch.as_tensor(arr, device=next(self.parameters()).device)
        return x.unsqueeze(0)