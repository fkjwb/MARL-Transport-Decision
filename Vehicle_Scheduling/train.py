from __future__ import annotations

import argparse
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from algos.jointppo import JointPPOTrainer, JointPolicy, PPOBuffer
from envs import VehicleSchedulingEnv, build_env_spec, dump_step_jsonl, load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/default_3node.yaml")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def timestamp_str() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())


def make_run_dirs(runs_dir: str) -> Dict[str, Path]:
    root = Path(runs_dir) / timestamp_str()
    ckpt_dir = root / "ckpt"
    tb_dir = root / "tb"
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)
    return {
        "root": root,
        "ckpt": ckpt_dir,
        "tb": tb_dir,
        "steps": root / "steps.jsonl",
        "config": root / "config.yaml",
    }


def build_env_info(env: VehicleSchedulingEnv) -> Dict[str, Any]:
    return {
        "num_nodes": env.N,
        "num_materials": env.K,
        "num_agents": env.A,
        "n_obs": env.n_obs,
        "action_dim": env.action_dim,
        "max_vehicle_id": int(env.vehicle_ids.max()),
        "max_capacity": float(env.vehicle_caps.max()),
        "max_degree": float(env.degrees.max()),
    }


def _get_action_dim(env: VehicleSchedulingEnv, names: List[str], default: int = 0) -> int:
    for name in names:
        if hasattr(env, name):
            value = getattr(env, name)
            try:
                return int(value)
            except Exception:
                pass
    return int(default)


def _extract_first_shape(x: Any):
    if isinstance(x, torch.Tensor):
        return tuple(int(v) for v in x.shape)
    if isinstance(x, np.ndarray):
        return tuple(int(v) for v in x.shape)
    if isinstance(x, (list, tuple)):
        for item in x:
            s = _extract_first_shape(item)
            if s is not None:
                return s
    if isinstance(x, dict):
        for item in x.values():
            s = _extract_first_shape(item)
            if s is not None:
                return s
    return None


def _capture_policy_io_shapes(policy: JointPolicy, obs: Dict[str, Any]) -> Dict[str, Any]:
    info = {
        "actor_in": None,
        "critic_in": None,
        "actor_out": None,
    }
    handles = []

    def _pre_hook(key):
        def hook(module, inputs):
            if info[key] is None:
                info[key] = _extract_first_shape(inputs)
        return hook

    def _fwd_hook(key):
        def hook(module, inputs, output):
            if info[key] is None:
                info[key] = _extract_first_shape(output)
        return hook

    actor_mod = getattr(policy, "actor", None)
    critic_mod = getattr(policy, "critic", None)

    if isinstance(actor_mod, torch.nn.Module):
        handles.append(actor_mod.register_forward_pre_hook(_pre_hook("actor_in")))
        handles.append(actor_mod.register_forward_hook(_fwd_hook("actor_out")))

    if isinstance(critic_mod, torch.nn.Module):
        handles.append(critic_mod.register_forward_pre_hook(_pre_hook("critic_in")))

    was_training = policy.training
    policy.eval()
    with torch.no_grad():
        policy.act(obs, deterministic=True)
    if was_training:
        policy.train()

    for h in handles:
        h.remove()

    return info


def print_env_actor_critic_spaces(
    env: VehicleSchedulingEnv,
    policy: JointPolicy,
    obs: Dict[str, Any],
    cfg: Dict[str, Any],
) -> None:
    io_shapes = _capture_policy_io_shapes(policy, obs)

    actor_shape = io_shapes["actor_in"]
    critic_shape = io_shapes["critic_in"]
    actor_out_shape = io_shapes["actor_out"]

    # hook 没抓到时，用配置兜底
    if actor_shape is None or critic_shape is None:
        N = int(getattr(policy, "N", env.N))
        K = int(getattr(policy, "K", env.K))
        A = int(getattr(policy, "A", env.A))
        n_obs = int(getattr(policy, "n_obs", env.n_obs))
        d_gat = int(cfg["model"]["gat"]["d_hidden_gat"])

        actor_dim = (N - 1) * (K * n_obs + d_gat) + 7
        critic_dim = N * ((N - 1) * K * n_obs + d_gat) + 2 * A + 2

        actor_shape = (A, actor_dim)
        critic_shape = (critic_dim,)

    actor_dim = int(actor_shape[-1])
    critic_dim = int(critic_shape[-1]) if len(critic_shape) > 0 else 1

    if actor_out_shape is not None and len(actor_out_shape) > 0:
        act_dim = int(actor_out_shape[-1])
    else:
        act_dim = _get_action_dim(env, ["action_dim", "act_dim"], default=0)

    print(f"[Env Obs Dim] actor={actor_dim} | critic={critic_dim}")
    print(f"[Env Obs Shape] actor={actor_shape} | critic={critic_shape}")
    print(f"[Env Action Dim] act_dim={act_dim}")


def summarize_first_episode(it: int, trace: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for item in trace:
        rec = {"iter": int(it)}
        rec.update(item)
        out.append(rec)
    return out


def save_checkpoint(path: Path, policy: JointPolicy, trainer: JointPPOTrainer, iteration: int) -> None:
    ckpt = {
        "iteration": int(iteration),
        "policy": policy.state_dict(),
        "optimizer_actor": trainer.optimizer_actor.state_dict(),
        "optimizer_critic": trainer.optimizer_critic.state_dict(),
    }
    torch.save(ckpt, path)


def maybe_load_resume(path: str | None, policy: JointPolicy, trainer: JointPPOTrainer, device: torch.device) -> int:
    if not path:
        return 0
    ckpt = torch.load(path, map_location=device)
    policy.load_state_dict(ckpt["policy"])
    trainer.optimizer_actor.load_state_dict(ckpt["optimizer_actor"])
    trainer.optimizer_critic.load_state_dict(ckpt["optimizer_critic"])
    return int(ckpt.get("iteration", 0))


def _format_duration(seconds: float) -> str:
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def main() -> None:
    train_start_ts = time.time()
    train_start_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(train_start_ts))
    print(f"train_start_time: {train_start_str}")

    args = parse_args()
    cfg = load_yaml(args.config)
    tcfg = cfg["train"]

    seed = int(tcfg.get("seed", 0))
    set_seed(seed)

    device = torch.device(args.device if args.device is not None else str(tcfg.get("device", "cpu")))

    env = VehicleSchedulingEnv(build_env_spec(cfg))
    env_info = build_env_info(env)

    obs = env.reset()

    # 先建 policy，再打印
    policy = JointPolicy(cfg, env_info).to(device)
    print_env_actor_critic_spaces(env, policy, obs, cfg)

    trainer = JointPPOTrainer(policy, cfg, device)

    start_iter = maybe_load_resume(tcfg.get("resume_ckpt", None), policy, trainer, device)

    run_dirs = make_run_dirs(str(tcfg.get("runs_dir", "./runs")))
    shutil.copyfile(args.config, run_dirs["config"])
    writer = SummaryWriter(log_dir=str(run_dirs["tb"]))

    iterations = int(tcfg["iterations"])
    steps_per_iter = int(tcfg["steps_per_iter"])
    ckpt_every = int(tcfg.get("ckpt_every", 50))
    debug_print = bool(tcfg.get("debug_first_full_episode_each_iter", False))

    for it in range(start_iter + 1, iterations + 1):
        buffer = PPOBuffer(gamma=float(tcfg["gamma"]), gae_lambda=float(tcfg["gae_lambda"]))
        ep_returns: List[float] = []
        ep_lens: List[int] = []
        reward_task_list: List[float] = []
        reward_unused_list: List[float] = []
        reward_total_list: List[float] = []
        reward_terminal_list: List[float] = []

        first_episode_trace: List[Dict[str, Any]] = []
        first_episode_done = False
        cur_ep_trace: List[Dict[str, Any]] = []
        cur_ep_ret = 0.0
        cur_ep_len = 0

        for _ in range(steps_per_iter):
            act_res = policy.act(obs, deterministic=False)
            next_obs, reward, done, info = env.step(act_res.actions)

            buffer.store(
                obs=obs,
                action=act_res.actions,
                logprob=act_res.logprob,
                value=act_res.value,
                reward=reward,
                done=done,
            )

            cur_ep_trace.append(info)
            cur_ep_ret += float(reward)
            cur_ep_len += 1

            reward_task_list.append(float(info["R_task"]))
            reward_unused_list.append(float(info["R_unused_penalty"]))
            reward_total_list.append(float(info["R_step"]))
            if done and ("Re" in info) and (info["Re"] is not None):
                reward_terminal_list.append(float(sum(info["Re"].values())))

            obs = next_obs

            if done:
                ep_returns.append(cur_ep_ret)
                ep_lens.append(cur_ep_len)
                if not first_episode_done:
                    first_episode_trace = list(cur_ep_trace)
                    first_episode_done = True
                obs = env.reset()
                cur_ep_trace = []
                cur_ep_ret = 0.0
                cur_ep_len = 0

        with torch.no_grad():
            if buffer.dones and not buffer.dones[-1]:
                last_value = policy.act(obs, deterministic=True).value
            else:
                last_value = 0.0
        buffer.finish_path(last_value=last_value)
        batch_np = buffer.get()
        stats = trainer.update(batch_np, iteration=it)

        if first_episode_done and first_episode_trace:
            jsonl_records = summarize_first_episode(it, first_episode_trace)
            dump_step_jsonl(str(run_dirs["steps"]), jsonl_records)
            if debug_print:
                for rec in jsonl_records:
                    print(rec)

        if it % ckpt_every == 0 or it == iterations:
            save_checkpoint(run_dirs["ckpt"] / f"ckpt_{it:06d}.pt", policy, trainer, it)

        ep_ret_mean = float(np.mean(ep_returns)) if ep_returns else 0.0
        ep_len_mean = float(np.mean(ep_lens)) if ep_lens else 0.0
        r_task_mean = float(np.mean(reward_task_list)) if reward_task_list else 0.0
        r_unused_mean = float(np.mean(reward_unused_list)) if reward_unused_list else 0.0
        r_step_mean = float(np.mean(reward_total_list)) if reward_total_list else 0.0
        r_terminal_mean = float(np.mean(reward_terminal_list)) if reward_terminal_list else 0.0

        writer.add_scalar("loss/pi_loss", stats.pi_loss, it)
        writer.add_scalar("loss/v_loss", stats.v_loss, it)
        writer.add_scalar("loss/entropy_loss", stats.entropy_loss, it)
        writer.add_scalar("loss/total_loss", stats.total_loss, it)
        writer.add_scalar("reward/R_task", r_task_mean, it)
        writer.add_scalar("reward/R_unused_penalty", r_unused_mean, it)
        writer.add_scalar("reward/R_step", r_step_mean, it)
        writer.add_scalar("reward/Re", r_terminal_mean, it)
        writer.add_scalar("episode/return_mean", ep_ret_mean, it)
        writer.add_scalar("episode/len_mean", ep_len_mean, it)
        writer.add_scalar("misc/entropy_coef", stats.entropy_coef, it)

        print(
            f"[iter {it}] "
            f"ep_ret={ep_ret_mean:.3f} "
            f"R_task={r_task_mean:.3f} R_unused={r_unused_mean:.3f} R_step={r_step_mean:.3f} Re={r_terminal_mean:.3f} "
            f"pi_loss={stats.pi_loss:.4f} v_loss={stats.v_loss:.4f} ent_loss={stats.entropy_loss:.4f} total_loss={stats.total_loss:.4f} "
            f"ent_coef={stats.entropy_coef:.6f}"
        )

    writer.close()
    train_end_ts = time.time()
    train_end_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(train_end_ts))
    train_duration_str = _format_duration(train_end_ts - train_start_ts)
    print(f"train_end_time: {train_end_str}")
    print(f"train_duration: {train_duration_str}")
    print(f"run_dir: {run_dirs['root']}")


if __name__ == "__main__":
    main()


'''
PC：
conda activate dl__py_3.9
tensorboard --logdir=runs\20260314-161821\tb
-i https://pypi.tuna.tsinghua.edu.cn/simple
conda env export > C:/Users/19419/Desktop/研三/物流调度/模型设计/多物料/model_1_七节点多物料汽运/pj_env.yaml
服务器：
salloc -p tyhcnormal -n 8 -N 1 --mem=24G        -n 8：请求8个CPU核心   -N 1：在1个节点上运行
salloc -p tyhcnormal -n 64 -N 1 --mem=200G
conda activate yxh__py_3.9     python train.py
'''