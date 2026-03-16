from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch


@dataclass
class RolloutBatch:
    """用于 PPO + 决策顺序模块（GAT+AugLag）更新的一批数据。"""

    # 观测
    obs_truck: torch.Tensor   # [T, Et, D_truck]
    obs_belt: torch.Tensor    # [T, B,  D_belt]
    obs_critic: torch.Tensor  # [T, D_critic]

    # 动作（truck 为连续 u∈(0,1)；belt 为离散 one-hot(K+1)，最后一维=不运输）
    act_truck: torch.Tensor   # [T, Et, K]
    act_belt: torch.Tensor    # [T, B,  K+1]
    belt_order: torch.Tensor  # [T, B]  int64，排列（每个位置是 combo_idx）
    order_adj: torch.Tensor   # [T, B, B] float32，0/1 采样邻接矩阵（对角为 0）

    # PPO 相关
    logprob: torch.Tensor     # [T] 旧策略 logπ_old(a)
    returns: torch.Tensor     # [T]
    advantages: torch.Tensor  # [T] 归一化优势（用于 PPO）
    adv_raw: torch.Tensor     # [T] 未归一化优势（用于 order 模块）
    values: torch.Tensor      # [T] critic 旧值


class RolloutBuffer:
    """多模块采样缓冲区：同时服务 PPO（truck/belt actor）和 order 模块（REINFORCE+AugLag）。"""

    def __init__(
        self,
        Et: int,
        B: int,
        K: int,
        truck_obs_dim: int,
        belt_obs_dim: int,
        critic_obs_dim: int,
        capacity: int,
        device: str = "cpu",
    ):
        self.Et, self.B, self.K = int(Et), int(B), int(K)
        self.truck_obs_dim = int(truck_obs_dim)
        self.belt_obs_dim = int(belt_obs_dim)
        self.critic_obs_dim = int(critic_obs_dim)
        self.capacity = int(capacity)
        self.device = str(device)
        self.reset()

    def reset(self) -> None:
        self.ptr = 0
        self.full = False

        self.obs_truck = np.zeros((self.capacity, self.Et, self.truck_obs_dim), dtype=np.float32)
        self.obs_belt = np.zeros((self.capacity, self.B, self.belt_obs_dim), dtype=np.float32)
        self.obs_critic = np.zeros((self.capacity, self.critic_obs_dim), dtype=np.float32)

        self.act_truck = np.zeros((self.capacity, self.Et, self.K), dtype=np.float32)
        self.act_belt = np.zeros((self.capacity, self.B, self.K + 1), dtype=np.float32)

        self.belt_order = np.zeros((self.capacity, self.B), dtype=np.int64)
        self.order_adj = np.zeros((self.capacity, self.B, self.B), dtype=np.float32)

        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.float32)
        self.logprob = np.zeros((self.capacity,), dtype=np.float32)
        self.values = np.zeros((self.capacity,), dtype=np.float32)

        self.returns = np.zeros((self.capacity,), dtype=np.float32)
        self.advantages = np.zeros((self.capacity,), dtype=np.float32)
        self.adv_raw = np.zeros((self.capacity,), dtype=np.float32)

    def add(
        self,
        obs: Dict[str, np.ndarray],
        action: Dict[str, np.ndarray],
        logprob: float,
        value: float,
        reward: float,
        done: bool,
    ) -> None:
        """写入 transition。

        obs key: truck_actor, belt_actor, critic
        action key: truck, belt, belt_order, order_adj
        """
        if self.ptr >= self.capacity:
            self.full = True
            return

        self.obs_truck[self.ptr] = obs["truck_actor"]
        self.obs_belt[self.ptr] = obs["belt_actor"]
        self.obs_critic[self.ptr] = obs["critic"]

        self.act_truck[self.ptr] = action["truck"].reshape(self.Et, self.K)
        self.act_belt[self.ptr] = action["belt"].reshape(self.B, self.K + 1)

        self.belt_order[self.ptr] = np.asarray(action["belt_order"], dtype=np.int64)
        self.order_adj[self.ptr] = np.asarray(action["order_adj"], dtype=np.float32)

        self.logprob[self.ptr] = float(logprob)
        self.values[self.ptr] = float(value)
        self.rewards[self.ptr] = float(reward)
        self.dones[self.ptr] = 1.0 if bool(done) else 0.0

        self.ptr += 1
        if self.ptr >= self.capacity:
            self.full = True

    def compute_returns_advantages(self, last_value: float, gamma: float, gae_lambda: float) -> Tuple[np.ndarray, np.ndarray]:
        """GAE-Lambda with bootstrap.

        Args:
            last_value: V(s_{T}) used for bootstrapping (0 if terminal).
            gamma: discount factor.
            gae_lambda: GAE lambda.

        Returns:
            returns: discounted returns [T]
            adv_raw: unnormalized advantages [T] (for order module / logging)
        """
        T = self.ptr
        last_gae = 0.0
        next_value = float(last_value)

        for t in reversed(range(T)):
            done = float(self.dones[t])
            reward = float(self.rewards[t])
            value = float(self.values[t])

            next_nonterminal = 1.0 - done
            delta = reward + float(gamma) * next_value * next_nonterminal - value
            last_gae = delta + float(gamma) * float(gae_lambda) * next_nonterminal * last_gae

            self.adv_raw[t] = last_gae
            self.advantages[t] = last_gae
            self.returns[t] = self.advantages[t] + value

            next_value = value

        adv = self.advantages[:T]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        self.advantages[:T] = adv.astype(np.float32)

        return self.returns[:T].copy(), self.adv_raw[:T].copy()


    def get(self) -> RolloutBatch:
        """取出当前缓冲区（0..ptr-1）。"""
        T = self.ptr

        batch = RolloutBatch(
            obs_truck=torch.tensor(self.obs_truck[:T], dtype=torch.float32, device=self.device),
            obs_belt=torch.tensor(self.obs_belt[:T], dtype=torch.float32, device=self.device),
            obs_critic=torch.tensor(self.obs_critic[:T], dtype=torch.float32, device=self.device),
            act_truck=torch.tensor(self.act_truck[:T], dtype=torch.float32, device=self.device),
            act_belt=torch.tensor(self.act_belt[:T], dtype=torch.float32, device=self.device),
            belt_order=torch.tensor(self.belt_order[:T], dtype=torch.int64, device=self.device),
            order_adj=torch.tensor(self.order_adj[:T], dtype=torch.float32, device=self.device),
            logprob=torch.tensor(self.logprob[:T], dtype=torch.float32, device=self.device),
            returns=torch.tensor(self.returns[:T], dtype=torch.float32, device=self.device),
            advantages=torch.tensor(self.advantages[:T], dtype=torch.float32, device=self.device),
            adv_raw=torch.tensor(self.adv_raw[:T], dtype=torch.float32, device=self.device),
            values=torch.tensor(self.values[:T], dtype=torch.float32, device=self.device),
        )
        return batch