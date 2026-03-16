from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from logistics_env import LogisticsEnv, build_env_spec, load_yaml
from algos.jointppo import JointActorCritic


CONFIG_PATH = Path('configs/config.yaml')


def _require(d: Dict[str, Any], key: str, ctx: str) -> Any:
    if key not in d:
        raise KeyError(f"Missing key '{key}' in {ctx}")
    return d[key]


def main() -> None:
    root = load_yaml(CONFIG_PATH)
    model_cfg = _require(root, 'model', 'root')
    eval_cfg = _require(root, 'eval', 'root')

    device = str(eval_cfg.get('device', 'cpu'))
    seed = int(eval_cfg.get('seed', 0))
    episodes = int(eval_cfg.get('episodes', 20))

    ckpt_path = str(_require(eval_cfg, 'ckpt_path', 'eval'))
    ckpt = torch.load(ckpt_path, map_location=device)

    spec = build_env_spec(root)
    env = LogisticsEnv(spec)

    actor_obs_dim = env.observation_space['actor'].shape[0]
    critic_obs_dim = env.observation_space['critic'].shape[0]

    actor_arch = str(model_cfg.get('actor_arch', 'mlpTransformer'))
    activation = str(model_cfg.get('activation', 'tanh'))
    actor_mlp_hidden = list(model_cfg.get('trunk_actor_mlp_hidden', model_cfg.get('actor_mlp_hidden', [256, 256])))
    critic_mlp_hidden = list(model_cfg.get('critic_mlp_hidden', [256, 256]))
    transformer_cfg = dict(model_cfg.get('transformer', {'d_model': 128, 'nhead': 4, 'num_layers': 2}))
    order_hidden = int(model_cfg.get('order_hidden', 256))
    order_activation = str(model_cfg.get('order_activation', activation))

    model = JointActorCritic(
        actor_obs_dim=actor_obs_dim,
        critic_obs_dim=critic_obs_dim,
        Et=env.Et,
        B=env.B,
        K=env.K,
        actor_arch=actor_arch,
        activation=activation,
        truck_actor_mlp_hidden=actor_mlp_hidden,
        critic_mlp_hidden=critic_mlp_hidden,
        transformer=transformer_cfg,
        order_mode=spec.belt_order_mode,
        order_hidden=order_hidden,
        order_activation=order_activation,
    ).to(device)

    model.load_state_dict(ckpt['model'])
    model.eval()

    returns = []
    term_reasons = Counter()

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_ret = 0.0
        last_reason = ''

        while not done:
            obs_actor = torch.tensor(obs['actor'], device=device)
            obs_critic = torch.tensor(obs['critic'], device=device)
            with torch.no_grad():
                out = model.act(obs_actor, obs_critic, deterministic=True)

            action_np = {
                'truck': out.action['truck'].detach().cpu().numpy().astype(np.float32),
                'belt': out.action['belt'].detach().cpu().numpy().astype(np.float32),
                'belt_order': out.action['belt_order'].detach().cpu().numpy().astype(np.float32),
            }

            obs, r, terminated, truncated, info = env.step(action_np)
            ep_ret += float(r)
            done = bool(terminated or truncated)
            last_reason = info.get('terminate_reason', '')

        returns.append(ep_ret)
        term_reasons[last_reason] += 1

    print(f'episodes={episodes}')
    print(f'avg_return={np.mean(returns):.3f} std_return={np.std(returns):.3f}')
    print('terminate_reason_counts:')
    for k, v in term_reasons.most_common():
        print(f'  {k}: {v}')


if __name__ == '__main__':
    main()
