from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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


@dataclass
class OrderOutput:
    order_indices: List[int]     # 长度 B 的排列
    adj_sample: torch.Tensor     # [B,B] 0/1 采样邻接矩阵（对角为 0）
    probs: torch.Tensor          # [B,B] 连续权重 W=σ(logits)（对角为 0）
    logprob: torch.Tensor        # 标量
    entropy: torch.Tensor        # 标量


class GATLayer(nn.Module):
    """简化版 GAT（默认完全图注意力）。"""

    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)

        self.W = nn.Linear(self.in_dim, self.out_dim * self.num_heads, bias=False)
        self.a_src = nn.Parameter(torch.empty(self.num_heads, self.out_dim))
        self.a_dst = nn.Parameter(torch.empty(self.num_heads, self.out_dim))

        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [B,Fin]
        attn_mask: [B,B] bool，True 表示允许 i->j；None 表示完全允许
        return: [B, H*D]
        """
        B = x.shape[0]
        h = self.W(x).view(B, self.num_heads, self.out_dim)  # [B,H,D]

        e_src = (h * self.a_src.unsqueeze(0)).sum(dim=-1)  # [B,H]
        e_dst = (h * self.a_dst.unsqueeze(0)).sum(dim=-1)  # [B,H]
        e = e_src.unsqueeze(1) + e_dst.unsqueeze(0)        # [B,B,H]
        e = F.leaky_relu(e, negative_slope=0.2)

        if attn_mask is not None:
            m = attn_mask.unsqueeze(-1).to(dtype=torch.bool)
            e = e.masked_fill(~m, float("-inf"))

        alpha = torch.softmax(e, dim=1)  # over j
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = torch.einsum("ijh,jhd->ihd", alpha, h)  # [B,H,D]
        out = out.reshape(B, self.num_heads * self.out_dim)
        return F.elu(out)


class GATEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, num_heads: int = 4, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        fin = int(in_dim)
        for _ in range(int(num_layers)):
            self.layers.append(GATLayer(fin, int(hidden_dim), num_heads=int(num_heads), dropout=float(dropout)))
            fin = int(hidden_dim) * int(num_heads)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)
        return x


class PairwiseAdjDecoder(nn.Module):
    """MLP 解码边 logits。"""

    def __init__(self, node_dim: int, hidden: int = 256, activation: str = "tanh"):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(int(node_dim) * 2, int(hidden)),
            activation_factory(activation),
            nn.Linear(int(hidden), int(hidden)),
            activation_factory(activation),
            nn.Linear(int(hidden), 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        B, D = h.shape
        hi = h.unsqueeze(1).expand(B, B, D)
        hj = h.unsqueeze(0).expand(B, B, D)
        x = torch.cat([hi, hj], dim=-1)  # [B,B,2D]
        logits = self.mlp(x).squeeze(-1)
        eye = torch.eye(B, device=logits.device, dtype=logits.dtype)
        logits = logits * (1.0 - eye) + (-20.0) * eye   # sigmoid(-20)≈2e-9，接近 0
        return logits


def acyclicity_constraint(W: torch.Tensor) -> torch.Tensor:
    """h(W)=tr(exp(W∘W)) - B."""
    B = W.shape[0]
    M = W * W
    return torch.trace(torch.matrix_exp(M)) - float(B)


def depth_constraint(W: torch.Tensor, k: int = 3) -> torch.Tensor:
    """c(W^k)=sum(W^k)，W^k 是矩阵乘法幂。"""
    if k <= 1:
        return W.sum()
    M = W
    for _ in range(k - 1):
        M = M @ W
    return M.sum()


class GraphOrderModule(nn.Module):
    """决策顺序模块：GAT encoder + MLP decoder，采样邻接并输出拓扑顺序。"""

    def __init__(
        self,
        obs_dim: int,
        num_agents: int,
        gat_hidden: int = 64,
        gat_heads: int = 4,
        gat_layers: int = 2,
        gat_dropout: float = 0.0,
        decoder_hidden: int = 256,
        activation: str = "tanh",
        depth_k: int = 3,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.num_agents = int(num_agents)
        self.depth_k = int(depth_k)

        self.encoder = GATEncoder(
            in_dim=self.obs_dim,
            hidden_dim=int(gat_hidden),
            num_heads=int(gat_heads),
            num_layers=int(gat_layers),
            dropout=float(gat_dropout),
        )
        node_dim = int(gat_hidden) * int(gat_heads)
        self.decoder = PairwiseAdjDecoder(node_dim=node_dim, hidden=int(decoder_hidden), activation=activation)

    def forward(self, obs_nodes: torch.Tensor) -> torch.Tensor:
        """obs_nodes: [B,D] or [1,B,D] -> logits [B,B]"""
        if obs_nodes.dim() == 3:
            if obs_nodes.shape[0] != 1:
                raise ValueError("GraphOrderModule only supports batch=1 for 3D input.")
            obs_nodes = obs_nodes[0]
        h = self.encoder(obs_nodes)
        return self.decoder(h)

    @torch.no_grad()
    def _break_cycles(self, adj: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        B = adj.shape[0]

        def find_cycle(a: torch.Tensor) -> Optional[List[int]]:
            visited = [0] * B
            parent = [-1] * B

            def dfs(u: int) -> Optional[List[int]]:
                visited[u] = 1
                for v in range(B):
                    if a[u, v].item() == 0:
                        continue
                    if visited[v] == 0:
                        parent[v] = u
                        cyc = dfs(v)
                        if cyc:
                            return cyc
                    elif visited[v] == 1:
                        cycle = [v]
                        cur = u
                        while cur != v and cur != -1:
                            cycle.append(cur)
                            cur = parent[cur]
                        cycle.append(v)
                        cycle.reverse()
                        return cycle
                visited[u] = 2
                return None

            for s in range(B):
                if visited[s] == 0:
                    cyc = dfs(s)
                    if cyc:
                        return cyc
            return None

        a = adj.clone()
        while True:
            cyc = find_cycle(a)
            if not cyc:
                break
            edges = [(cyc[i], cyc[i + 1]) for i in range(len(cyc) - 1)]
            min_edge = min(edges, key=lambda uv: float(weights[uv[0], uv[1]].item()))
            a[min_edge[0], min_edge[1]] = 0.0
        return a

    def _toposort(self, adj_dag: torch.Tensor, deterministic: bool = False) -> List[int]:
        B = adj_dag.shape[0]
        indeg = adj_dag.sum(dim=0).to(dtype=torch.int64)
        order: List[int] = []
        avail = [i for i in range(B) if indeg[i].item() == 0]
        while avail:
            if deterministic or len(avail) == 1:
                u = min(avail)
            else:
                pick_idx = int(torch.randint(low=0, high=len(avail), size=(1,), device=adj_dag.device).item())
                u = avail[pick_idx]
            avail.remove(u)
            order.append(u)
            out = adj_dag[u].to(dtype=torch.int64)
            for v in range(B):
                if out[v].item() == 1:
                    indeg[v] -= 1
                    if indeg[v].item() == 0:
                        avail.append(v)
        if len(order) < B:
            tail = [i for i in range(B) if i not in order]
            if (not deterministic) and len(tail) > 1:
                perm = torch.randperm(len(tail), device=adj_dag.device).tolist()
                tail = [tail[i] for i in perm]
            order.extend(tail)
        return order

    def sample_order(self, obs_nodes: torch.Tensor, deterministic: bool = False) -> OrderOutput:
        """obs_nodes: [B,D] 或 [1,B,D]"""
        if obs_nodes.dim() == 3:
            if obs_nodes.shape[0] != 1:
                raise ValueError("sample_order only supports batch=1 for 3D input.")
            obs_nodes_ = obs_nodes[0]
        else:
            obs_nodes_ = obs_nodes

        logits = self.forward(obs_nodes_)
        probs = torch.sigmoid(logits)
        bern = torch.distributions.Bernoulli(probs=probs)
        if deterministic:
            adj = (probs >= 0.5).to(dtype=probs.dtype)
        else:
            adj = bern.sample()
        adj.fill_diagonal_(0.0)

        logprob = bern.log_prob(adj).sum()
        entropy = bern.entropy().sum()

        adj_dag = self._break_cycles(adj, probs)
        order = self._toposort(adj_dag, deterministic=deterministic)

        return OrderOutput(order_indices=order, adj_sample=adj, probs=probs, logprob=logprob, entropy=entropy)

    def logprob_of_adj(self, obs_nodes: torch.Tensor, adj_sample: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """给定 adj_sample，返回 (logprob, entropy, probs)。"""
        if obs_nodes.dim() == 3:
            if obs_nodes.shape[0] != 1:
                raise ValueError("logprob_of_adj only supports batch=1 for 3D input.")
            obs_nodes_ = obs_nodes[0]
        else:
            obs_nodes_ = obs_nodes

        logits = self.forward(obs_nodes_)
        probs = torch.sigmoid(logits)
        bern = torch.distributions.Bernoulli(probs=probs)

        adj = adj_sample.to(dtype=probs.dtype)
        adj = adj * (1.0 - torch.eye(self.num_agents, device=adj.device, dtype=adj.dtype))
        logprob = bern.log_prob(adj).sum()
        entropy = bern.entropy().sum()
        return logprob, entropy, probs
