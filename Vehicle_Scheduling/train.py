from __future__ import annotations

import argparse
from copy import deepcopy
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from algos.jointppo import JointPPOTrainer, JointPolicy, PPOBuffer
from envs import VehicleSchedulingEnv, build_env_spec, dump_step_jsonl, load_yaml


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "--configs", dest="config", type=str, default="configs/default_3node.yaml")
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


def _path_candidates(raw_path: str | Path, *, runs_dir: Optional[Path] = None) -> List[Path]:
    path = Path(str(raw_path)).expanduser()
    if path.is_absolute():
        return [path]

    candidates: List[Path] = []
    for candidate in (Path.cwd() / path, PROJECT_ROOT / path):
        candidate = candidate.resolve()
        if candidate not in candidates:
            candidates.append(candidate)

    normalized_parts = [part for part in path.parts if part not in ("", ".")]
    if runs_dir is not None and (not normalized_parts or normalized_parts[0].lower() != runs_dir.name.lower()):
        runs_candidate = (runs_dir / path).resolve()
        if runs_candidate not in candidates:
            candidates.append(runs_candidate)

    return candidates


def _resolve_existing_path(raw_path: str | Path, *, label: str) -> Path:
    candidates = _path_candidates(raw_path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    tried = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"{label} not found: {raw_path}. Tried: {tried}")


def _resolve_runs_dir(raw_path: str | Path) -> Path:
    path = Path(str(raw_path)).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _latest_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    if not ckpt_dir.is_dir():
        return None
    ckpts = sorted(ckpt_dir.glob("ckpt_*.pt"))
    if not ckpts:
        return None
    return ckpts[-1]


def _run_config_path(run_dir: Path) -> Optional[Path]:
    cfg_path = run_dir / "config.yaml"
    return cfg_path if cfg_path.is_file() else None


def _resolve_resume_checkpoint(
    raw_path: str | None,
    *,
    runs_dir: Path,
    emit_logs: bool = True,
) -> Optional[Path]:
    if raw_path is None:
        return None

    raw_text = str(raw_path).strip()
    if not raw_text:
        return None

    candidates = _path_candidates(raw_text, runs_dir=runs_dir)
    details: List[str] = []

    for candidate in candidates:
        if candidate.is_file():
            return candidate
        if candidate.is_dir():
            latest = _latest_checkpoint(candidate)
            if latest is not None:
                return latest
            latest = _latest_checkpoint(candidate / "ckpt")
            if latest is not None:
                return latest
            details.append(f"{candidate} exists but contains no 'ckpt_*.pt' checkpoint")
        else:
            details.append(f"{candidate} does not exist")

    tried = ", ".join(str(p) for p in candidates)
    reason = "; ".join(details) if details else "no candidate matched"
    message = (
        f"resume checkpoint not found from '{raw_text}'. "
        f"Expected a checkpoint file, a timestamp run directory, or a ckpt directory. "
        f"Tried: {tried}. Details: {reason}"
    )
    if emit_logs:
        print(f"[resume] {message}")
    raise FileNotFoundError(
        message
    )


def _resume_config_path(resume_ckpt: Path) -> Optional[Path]:
    return _run_config_path(_resume_run_dir(resume_ckpt))


def _resume_run_dir(resume_ckpt: Path) -> Path:
    return resume_ckpt.parent.parent if resume_ckpt.parent.name.lower() == "ckpt" else resume_ckpt.parent


def _merge_resume_config(saved_cfg: Dict[str, Any], requested_cfg: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(saved_cfg)
    merged_train = deepcopy(saved_cfg.get("train", {}) or {})
    merged_train.update(deepcopy(requested_cfg.get("train", {}) or {}))
    merged["train"] = merged_train

    for key, value in requested_cfg.items():
        if key == "train":
            continue
        if key not in merged:
            merged[key] = deepcopy(value)

    return merged


def _config_path_repr(path: Path) -> str:
    try:
        rel = path.resolve().relative_to(PROJECT_ROOT)
        return f"./{rel.as_posix()}"
    except ValueError:
        return str(path)


def _resolve_training_config(config_path: Path) -> tuple[Dict[str, Any], Path, Optional[Path], Path, Optional[Path]]:
    requested_cfg = load_yaml(str(config_path))
    requested_tcfg = requested_cfg.get("train", {}) or {}

    runs_dir = _resolve_runs_dir(str(requested_tcfg.get("runs_dir", "./runs")))
    resume_ckpt = _resolve_resume_checkpoint(requested_tcfg.get("resume_ckpt", None), runs_dir=runs_dir)

    effective_cfg = requested_cfg
    effective_config_path = config_path
    resume_config_path = None

    if resume_ckpt is not None and bool(requested_tcfg.get("resume_use_checkpoint_config", True)):
        resume_config_path = _resume_config_path(resume_ckpt)
        if resume_config_path is not None:
            saved_cfg = load_yaml(str(resume_config_path))
            effective_cfg = _merge_resume_config(saved_cfg, requested_cfg)
            effective_config_path = resume_config_path

            effective_tcfg = effective_cfg.get("train", {}) or {}
            runs_dir = _resolve_runs_dir(str(effective_tcfg.get("runs_dir", "./runs")))
            resume_ckpt = _resolve_resume_checkpoint(
                effective_tcfg.get("resume_ckpt", None),
                runs_dir=runs_dir,
                emit_logs=False,
            )

    if resume_ckpt is not None:
        effective_cfg.setdefault("train", {})["resume_ckpt"] = _config_path_repr(resume_ckpt)

    return effective_cfg, effective_config_path, resume_config_path, runs_dir, resume_ckpt


def _write_config_snapshot(cfg: Dict[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)


def _torch_load_checkpoint(path: str | Path, map_location: Any) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def make_run_dirs(runs_dir: Path, run_dir: Optional[Path] = None) -> Dict[str, Path]:
    root = run_dir if run_dir is not None else (runs_dir / timestamp_str())
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


def save_checkpoint(path: Path, policy: JointPolicy, trainer: JointPPOTrainer, iteration: int, cfg: Dict[str, Any]) -> None:
    ckpt = {
        "iteration": int(iteration),
        "config": deepcopy(cfg),
        "policy": policy.state_dict(),
        "optimizer_actor": trainer.optimizer_actor.state_dict(),
        "optimizer_critic": trainer.optimizer_critic.state_dict(),
    }
    torch.save(ckpt, path)


def maybe_load_resume(path: str | None, policy: JointPolicy, trainer: JointPPOTrainer, device: torch.device) -> int:
    if not path:
        return 0
    ckpt = _torch_load_checkpoint(path, map_location=device)
    try:
        policy.load_state_dict(ckpt["policy"])
    except RuntimeError as exc:
        raise RuntimeError(
            f"Failed to load checkpoint '{path}' into the current Vehicle model. "
            f"This usually means the checkpoint was trained with a different env/model config. "
            f"Keep train.resume_use_checkpoint_config=true to reuse the checkpoint's config snapshot.\n"
            f"Original error:\n{exc}"
        ) from exc
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
    requested_config_path = _resolve_existing_path(args.config, label="config")
    cfg, effective_config_path, resume_config_path, runs_dir, resume_ckpt = _resolve_training_config(
        requested_config_path
    )
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

    start_iter = maybe_load_resume(str(resume_ckpt) if resume_ckpt is not None else None, policy, trainer, device)
    resume_in_place = resume_ckpt is not None and bool(tcfg.get("resume_continue_in_place", True))

    run_dirs = make_run_dirs(runs_dir, _resume_run_dir(resume_ckpt) if resume_in_place else None)
    print(f"[path] project_root={PROJECT_ROOT}")
    print(f"[path] requested_config={requested_config_path}")
    print(f"[path] effective_config={effective_config_path}")
    if resume_config_path is not None:
        print(f"[path] resume_config={resume_config_path}")
    print(f"[path] runs_dir={runs_dir}")
    print(f"[path] run_dir={run_dirs['root']}")
    if resume_ckpt is not None:
        print(f"[path] resume_ckpt={resume_ckpt}")
    _write_config_snapshot(cfg, run_dirs["config"])
    writer_kwargs = {"log_dir": str(run_dirs["tb"])}
    if resume_in_place:
        writer_kwargs["purge_step"] = start_iter + 1
    writer = SummaryWriter(**writer_kwargs)

    iterations = int(tcfg["iterations"])
    steps_per_iter = int(tcfg["steps_per_iter"])
    ckpt_every = int(tcfg.get("ckpt_every", 50))
    debug_print = bool(tcfg.get("debug_first_full_episode_each_iter", False))
    if start_iter >= iterations:
        print(f"[resume] start iteration {start_iter} already reaches target iterations {iterations}; no new update will run.")

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
            save_checkpoint(run_dirs["ckpt"] / f"ckpt_{it:06d}.pt", policy, trainer, it, cfg)

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
