from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch


class PPOBuffer:
    def __init__(self, gamma: float, gae_lambda: float):
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.reset()

    def reset(self) -> None:
        self.obs_list: List[Dict[str, np.ndarray]] = []
        self.actions: List[np.ndarray] = []
        self.logprobs: List[float] = []
        self.values: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.advantages: np.ndarray | None = None
        self.returns: np.ndarray | None = None

    def store(
        self,
        obs: Dict[str, np.ndarray],
        action: np.ndarray,
        logprob: float,
        value: float,
        reward: float,
        done: bool,
    ) -> None:
        self.obs_list.append({k: np.array(v, copy=True) for k, v in obs.items()})
        self.actions.append(np.array(action, copy=True))
        self.logprobs.append(float(logprob))
        self.values.append(float(value))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))

    def finish_path(self, last_value: float = 0.0) -> None:
        rewards = np.asarray(self.rewards, dtype=np.float32)
        values = np.asarray(self.values + [float(last_value)], dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.float32)
        T = len(rewards)
        adv = np.zeros((T,), dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * values[t + 1] * nonterminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * nonterminal * gae
            adv[t] = gae
        ret = adv + values[:-1]
        adv = (adv - adv.mean()) / max(adv.std(), 1e-8)
        self.advantages = adv.astype(np.float32)
        self.returns = ret.astype(np.float32)

    def get(self) -> Dict[str, Any]:
        if self.advantages is None or self.returns is None:
            raise RuntimeError("call finish_path() before get()")
        obs_batch: Dict[str, np.ndarray] = {}
        keys = self.obs_list[0].keys()
        for k in keys:
            obs_batch[k] = np.stack([obs[k] for obs in self.obs_list], axis=0)
        return {
            "obs": obs_batch,
            "actions": np.stack(self.actions, axis=0).astype(np.int64),
            "logprobs": np.asarray(self.logprobs, dtype=np.float32),
            "advantages": self.advantages,
            "returns": self.returns,
        }

    @staticmethod
    def to_torch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
        obs_t = {k: torch.as_tensor(v, device=device) for k, v in batch["obs"].items()}
        out = {
            "obs": obs_t,
            "actions": torch.as_tensor(batch["actions"], device=device, dtype=torch.long),
            "logprobs": torch.as_tensor(batch["logprobs"], device=device, dtype=torch.float32),
            "advantages": torch.as_tensor(batch["advantages"], device=device, dtype=torch.float32),
            "returns": torch.as_tensor(batch["returns"], device=device, dtype=torch.float32),
        }
        return out