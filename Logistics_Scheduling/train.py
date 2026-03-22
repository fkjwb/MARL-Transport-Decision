from __future__ import annotations

import argparse
from copy import deepcopy
import inspect
import json
import os
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, List

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from envs.env import LogisticsEnv, build_env_spec, load_yaml
from algos.jointppo.buffer import RolloutBuffer
from algos.jointppo.policy import JointActorCritic
from algos.jointppo.trainer import EntropyScheduler, JointPPOTrainer, OrderConfig, PPOConfig


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser()
    # 仅保留配置文件路径；其余配置项从 yaml 读取
    p.add_argument("--configs", "--config", dest="configs", type=str, default="configs/default.yaml")
    return p.parse_args()


def now_timestamp() -> str:
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
    raw_path: Any,
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


def _format_wallclock(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_jsonable(x: Any) -> Any:
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if torch.is_tensor(x):
        return x.detach().cpu().numpy().tolist()
    if hasattr(x, "__dict__"):
        return _to_jsonable(vars(x))
    return str(x)


def _parse_betas(raw: Any, default: tuple[float, float] = (0.9, 0.999)) -> tuple[float, float]:
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        try:
            return float(raw[0]), float(raw[1])
        except Exception:
            return default
    return default


def _get_last_trace(env: Any) -> Optional[Dict[str, Any]]:
    # Preferred: env.last_trace() -> dict
    last_trace = getattr(env, "last_trace", None)
    if callable(last_trace):
        try:
            tr = last_trace()
            if tr is not None:
                return tr if isinstance(tr, dict) else {"trace": tr}
        except Exception:
            pass

    # Fallback: env.trace[-1]
    tr_list = getattr(env, "trace", None)
    if isinstance(tr_list, list) and len(tr_list) > 0:
        tr = tr_list[-1]
        if isinstance(tr, dict):
            return tr
        try:
            return vars(tr)
        except Exception:
            return {"trace": str(tr)}

    return None


def _attach_node_ids(x: Any, node_ids: List[Any]) -> Any:
    """
    将 [N, K] 形式的 Wi / Yi / Pi 转成带节点 id 的 dict：
    {
        "1": [...],
        "2": [...],
        ...
    }
    若不是按节点展开的二维结构，则保持原值不变。
    """
    try:
        rows = _to_jsonable(x)
        if not isinstance(rows, list):
            return rows
        if len(rows) != len(node_ids):
            return rows
        if any(not isinstance(row, list) for row in rows):
            return rows
        return {str(nid): row for nid, row in zip(node_ids, rows)}
    except Exception:
        return x


def _round_reward_obj(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _round_reward_obj(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_round_reward_obj(v) for v in x]
    if isinstance(x, (int, float)):
        return round(float(x), 3)
    return x



def _compact_json_line(x: Any) -> str:
    return json.dumps(_to_jsonable(x), ensure_ascii=False, separators=(",", ":"))


def _format_debug_dict_lines(data: Dict[str, Any], indent: str = "  ") -> List[str]:
    lines: List[str] = []
    for k, v in data.items():
        lines.append(f"{indent}{k}: {_compact_json_line(v)}")
    return lines


def _format_debug_actor_block(title: str, rows: Any, item_label: str) -> List[str]:
    lines: List[str] = [title]
    if not isinstance(rows, list) or len(rows) == 0:
        lines.append("  (empty)")
        return lines

    for idx, row in enumerate(rows):
        lines.append(f"  {item_label}[{idx}]:")
        if isinstance(row, dict):
            lines.extend(_format_debug_dict_lines(row, indent="    "))
        else:
            lines.append(f"    value: {_compact_json_line(row)}")
    return lines


def _format_debug_timestep_block(row: Dict[str, Any]) -> str:
    t = int(row.get("t", -1))
    lines: List[str] = [f"[t={t}]"]

    meta = {k: v for k, v in row.items() if k not in ("t", "truck_actor", "belt_actor", "critic")}
    if meta:
        lines.extend(_format_debug_dict_lines(meta, indent="  "))
        lines.append("")

    lines.extend(_format_debug_actor_block("actor.truck:", row.get("truck_actor", []), "edge"))
    lines.append("")
    lines.extend(_format_debug_actor_block("actor.belt:", row.get("belt_actor", []), "combo"))
    lines.append("")

    lines.append("critic:")
    critic = row.get("critic", {})
    if isinstance(critic, dict) and critic:
        lines.extend(_format_debug_dict_lines(critic, indent="  "))
    elif isinstance(critic, dict):
        lines.append("  (empty)")
    else:
        lines.append(f"  value: {_compact_json_line(critic)}")

    return "\n".join(lines)


def _print_first_episode_debug(it: int, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    print(f"[debug][iter={it}] first_complete_episode begin")
    print()
    for idx, row in enumerate(rows):
        row = row if isinstance(row, dict) else {"value": row}
        print(_format_debug_timestep_block(row))
        if idx != len(rows) - 1:
            print()
            print()
    print()
    print(f"[debug][iter={it}] first_complete_episode end")


def _format_steptrace(tr: Dict[str, Any], it: int, node_ids: List[Any]) -> Dict[str, Any]:
    """Format StepTrace dict for steps.jsonl with fixed key order."""
    tr = dict(tr) if tr is not None else {}
    tr.pop("global_step", None)

    # steps 日志中的 R_t_star 只在命中 T* 的当步写出，避免其他行重复记录 0。
    for k in ("R_demand", "R_level23_penalty", "R_t_star", "R_step", "Re", "episode_return"):
        if k in tr and tr[k] is not None:
            tr[k] = _round_reward_obj(tr[k])

    for k in ("Wi", "Yi", "Yi_before", "Y_after", "Pi"):
        if k in tr:
            tr[k] = _attach_node_ids(tr[k], node_ids)

    ordered = [
        "iter",
        "t",
        "Wi",
        "Yi_before",
        "Y_after",
        "Pi",
        "X",
        "B",
        "f",
        "Decision_seq",
        "R_demand",
        "R_level23_penalty",
        "R_t_star",
        "R_step",
        "Re",
        "episode_return",
        "terminate_reason",
        "terminate_detail",
    ]
    out: Dict[str, Any] = {"iter": int(it)}
    for k in ordered:
        if k == "iter":
            continue
        if k in tr and tr[k] is not None:
            out[k] = tr[k]
    return out


def main():
    args = parse_args()
    train_start_ts = time.time()

    requested_config_path = _resolve_existing_path(args.configs, label="config")
    cfg, effective_config_path, resume_config_path, runs_dir, resume_ckpt_path = _resolve_training_config(
        requested_config_path
    )
    tcfg = cfg.get("train", {})
    seed = int(tcfg.get("seed", 0))
    set_seed(seed)

    env_spec = build_env_spec(cfg)
    env = LogisticsEnv(env_spec)
    env.check_before_training()

    # 一级节点的行索引（用于 logging）
    level1_rows = [i for i, nid in enumerate(env.node_ids) if int(env.spec.nodes[nid].level) == 1]

    device = torch.device(str(tcfg.get("device", "cpu")))

    # dims
    Et = env.Et
    B = env.B

    # K：物料种类数（materials），皮带动作维度为 K_belt = K + 1（最后一维表示“不运输”）
    K_mat = int(env.K)
    K_belt = int(getattr(env, "K_belt", K_mat + 1))
    K = K_mat  # 兼容旧变量名：K 仍表示物料数

    truck_obs_dim = env.truck_obs_dim
    belt_obs_dim = env.belt_obs_dim
    critic_obs_dim = env.critic_obs_dim
    # 训练开始前打印观测维度/shape
    print(
        f"[Env Obs Dim] truck_actor={truck_obs_dim} | "
        f"belt_actor={belt_obs_dim} | critic={critic_obs_dim}"
    )
    # 同时打印 observation_space 的 shape，避免维度含义歧义
    print(
        f"[Env Obs Shape] truck_actor={env.observation_space['truck_actor'].shape} | "
        f"belt_actor={env.observation_space['belt_actor'].shape} | "
        f"critic={env.observation_space['critic'].shape}"
    )

    # 动作维度提示：truck 动作为 K（每种物料一个维度）；belt 动作为 K+1（最后一维表示“不运输”）
    print(f"[Env Action Dim] truck_act_dim={K_mat} | belt_act_dim={K_belt}")

    # model cfg
    mcfg = cfg.get("model", {})
    truck_actor_hidden = list(mcfg.get("trunk_actor_mlp_hidden", mcfg.get("actor_mlp_hidden", [256, 256])))
    critic_hidden = list(mcfg.get("critic_mlp_hidden", [256, 256]))
    activation = str(mcfg.get("activation", "tanh"))

    transformer_cfg = dict(mcfg.get("belt_transformer", {"d_model": 128, "nhead": 4, "num_layers": 2, "dropout": 0.0}))
    order_cfg = dict(
        mcfg.get(
            "order_module",
            {
                "gat_hidden": 64,
                "gat_heads": 4,
                "gat_layers": 2,
                "gat_dropout": 0.0,
                "decoder_hidden": 256,
                "activation": "tanh",
                "depth_k": 3,
            },
        )
    )


    # ---- model ----
    # Build model kwargs.

    model_kwargs = dict(
        Et=Et,
        B=B,
        K=K_mat,
        truck_obs_dim=truck_obs_dim,
        belt_obs_dim=belt_obs_dim,
        critic_obs_dim=critic_obs_dim,
        truck_actor_mlp_hidden=truck_actor_hidden,
        critic_mlp_hidden=critic_hidden,
        activation=activation,
        transformer_cfg=transformer_cfg,
        order_cfg=order_cfg,
    )

    # Compatibility with variants that expose K_belt under different argument names.
    _sig = inspect.signature(JointActorCritic.__init__)
    if "K_belt" in _sig.parameters:
        model_kwargs["K_belt"] = K_belt
    elif "belt_act_dim" in _sig.parameters:
        model_kwargs["belt_act_dim"] = K_belt
    elif "belt_action_dim" in _sig.parameters:
        model_kwargs["belt_action_dim"] = K_belt

    model = JointActorCritic(**model_kwargs).to(device)

    # train cfg
    optim_tcfg = dict(tcfg.get("optimizer", {}) or {})
    ppo_actor_optim = dict(optim_tcfg.get("ppo_actor", {}) or {})
    ppo_critic_optim = dict(optim_tcfg.get("ppo_critic", {}) or {})
    order_optim = dict(optim_tcfg.get("order", {}) or {})

    ppo_cfg = PPOConfig(
        gamma=float(tcfg.get("gamma", 0.99)),
        gae_lambda=float(tcfg.get("gae_lambda", 0.95)),
        clip_ratio=float(tcfg.get("clip_ratio", 0.2)),
        lr_actor=float(tcfg.get("lr_actor", 3e-4)),
        lr_critic=float(tcfg.get("lr_critic", 3e-4)),
        train_epochs=int(tcfg.get("train_epochs", 4)),
        minibatch_size=int(tcfg.get("minibatch_size", 256)),
        vf_coef=float(tcfg.get("vf_coef", 0.5)),
        ent_coef=float(tcfg.get("ent_coef", 0.01)),
        max_grad_norm=float(tcfg.get("max_grad_norm", 0.5)),
        actor_betas=_parse_betas(ppo_actor_optim.get("betas", [0.9, 0.999])),
        actor_eps=float(ppo_actor_optim.get("eps", 1e-8)),
        actor_weight_decay=float(ppo_actor_optim.get("weight_decay", 0.0)),
        critic_betas=_parse_betas(ppo_critic_optim.get("betas", [0.9, 0.999])),
        critic_eps=float(ppo_critic_optim.get("eps", 1e-8)),
        critic_weight_decay=float(ppo_critic_optim.get("weight_decay", 0.0)),
    )

    order_tcfg = tcfg.get("order", {})
    order_cfg_train = OrderConfig(
        lr=float(order_tcfg.get("lr", 3e-4)),
        train_epochs=int(order_tcfg.get("train_epochs", 1)),
        minibatch_size=int(order_tcfg.get("minibatch_size", 256)),
        mu_init=float(order_tcfg.get("mu_init", 1e-3)),
        rho=float(order_tcfg.get("rho", 1.3)),
        mu_max=float(order_tcfg.get("mu_max", 1e9)),
        tol=float(order_tcfg.get("tol", 1e-7)),
        depth_k=int(order_tcfg.get("depth_k", 3)),
        normalize_constraints=bool(order_tcfg.get("normalize_constraints", True)),
        h_coef=float(order_tcfg.get("h_coef", 1.0)),
        c_coef=float(order_tcfg.get("c_coef", 1.0)),
        max_grad_norm=float(order_tcfg.get("max_grad_norm", 1.0)),
        betas=_parse_betas(order_optim.get("betas", [0.9, 0.999])),
        eps=float(order_optim.get("eps", 1e-8)),
        weight_decay=float(order_optim.get("weight_decay", 0.0)),
    )

    ent_decay = tcfg.get("ent_decay", None)
    ent_scheduler = EntropyScheduler(ppo_cfg.ent_coef, decay=float(ent_decay) if ent_decay is not None else None)

    # rollout
    steps_per_iter = int(tcfg.get("steps_per_iter", 2048))
    debug_first_full_episode_each_iter = bool(tcfg.get("debug_first_full_episode_each_iter", True))
    iterations = int(tcfg.get("iterations", 200))
    ckpt_every = int(tcfg.get("ckpt_every", 10))

    resume_in_place = resume_ckpt_path is not None and bool(tcfg.get("resume_continue_in_place", True))

    # ---- runs dir layout (tb/ckpt/steps.jsonl/config.yaml) ----
    run_dir = _resume_run_dir(resume_ckpt_path) if resume_in_place else (runs_dir / now_timestamp())
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[path] project_root={PROJECT_ROOT}")
    print(f"[path] requested_config={requested_config_path}")
    print(f"[path] effective_config={effective_config_path}")
    if resume_config_path is not None:
        print(f"[path] resume_config={resume_config_path}")
    print(f"[path] runs_dir={runs_dir}")
    print(f"[path] run_dir={run_dir}")
    if resume_ckpt_path is not None:
        print(f"[path] resume_ckpt={resume_ckpt_path}")

    # buffer
    buffer_kwargs = dict(
        Et=Et,
        B=B,
        K=K_mat,
        truck_obs_dim=truck_obs_dim,
        belt_obs_dim=belt_obs_dim,
        critic_obs_dim=critic_obs_dim,
        capacity=steps_per_iter,
        device=str(device),
    )
    _bsig = inspect.signature(RolloutBuffer.__init__)
    if "K_belt" in _bsig.parameters:
        buffer_kwargs["K_belt"] = K_belt
    elif "belt_act_dim" in _bsig.parameters:
        buffer_kwargs["belt_act_dim"] = K_belt
    elif "belt_action_dim" in _bsig.parameters:
        buffer_kwargs["belt_action_dim"] = K_belt

    buffer = RolloutBuffer(**buffer_kwargs)

    trainer = JointPPOTrainer(
        model=model,
        buffer=buffer,
        ppo_cfg=ppo_cfg,
        order_cfg=order_cfg_train,
        device=str(device),
        entropy_scheduler=ent_scheduler,
    )

    # ---- checkpointing ----
    def save_ckpt(it: int):
        ckpt = {
            "it": int(it),
            "model": model.state_dict(),
            "config": deepcopy(cfg),
            "ppo_cfg": asdict(ppo_cfg),
            "order_cfg": asdict(order_cfg_train),
            "env_spec": asdict(env_spec),
            "rng": {
                "np": np.random.get_state(),
                "torch": torch.random.get_rng_state(),
            },
            "optimizers": {
                "opt_ppo": trainer.opt_ppo.state_dict(),
                "opt_order": trainer.opt_order.state_dict(),
            },
            "trainer_state": {
                "lambda_h": trainer.lambda_h.detach().cpu(),
                "lambda_c": trainer.lambda_c.detach().cpu(),
                "mu_h": trainer.mu_h.detach().cpu(),
                "mu_c": trainer.mu_c.detach().cpu(),
            },
        }

        path = ckpt_dir / f"ckpt_{it:06d}.pt"
        torch.save(ckpt, path)
        print(f"[ckpt] saved: {path}")

    def load_ckpt(path: str, load_opt: bool = False, load_trainer_state: bool = False) -> int:
        data = _torch_load_checkpoint(path, map_location=device)
        try:
            model.load_state_dict(data["model"], strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Failed to load checkpoint '{path}' into the current Logistics model. "
                f"This usually means the checkpoint was trained with a different env/model config. "
                f"Keep train.resume_use_checkpoint_config=true to reuse the checkpoint's config snapshot.\n"
                f"Original error:\n{exc}"
            ) from exc

        optimizers = data.get("optimizers", {})
        if load_opt:
            opt_ppo_state = optimizers.get("opt_ppo")
            opt_order_state = optimizers.get("opt_order")
            if opt_ppo_state is not None:
                trainer.opt_ppo.load_state_dict(opt_ppo_state)
            else:
                print("[ckpt] resume_load_optimizer=True but opt_ppo state not stored in this checkpoint.")
            if opt_order_state is not None:
                trainer.opt_order.load_state_dict(opt_order_state)
            else:
                print("[ckpt] resume_load_optimizer=True but opt_order state not stored in this checkpoint.")

        trainer_state = data.get("trainer_state", {})
        if load_trainer_state:
            restored = []
            for name in ("lambda_h", "lambda_c", "mu_h", "mu_c"):
                if name not in trainer_state or not hasattr(trainer, name):
                    continue
                current = getattr(trainer, name)
                saved = trainer_state[name]
                if not torch.is_tensor(saved):
                    saved = torch.tensor(saved, dtype=current.dtype)
                setattr(trainer, name, saved.to(device=current.device, dtype=current.dtype))
                restored.append(name)
            if not restored:
                print("[ckpt] resume_load_trainer_state=True but trainer state not stored in this checkpoint.")

        it0 = int(data.get("it", 0))
        print(f"[ckpt] loaded: {path} (it={it0})")
        return it0

    start_it = 0
    if resume_ckpt_path is not None:
        start_it = load_ckpt(
            str(resume_ckpt_path),
            load_opt=bool(tcfg.get("resume_load_optimizer", False)),
            load_trainer_state=bool(tcfg.get("resume_load_trainer_state", False)),
        )
    if start_it >= iterations:
        print(f"[resume] start iteration {start_it} already reaches target iterations {iterations}; no new update will run.")

    _write_config_snapshot(cfg, run_dir / "config.yaml")

    tb_dir = run_dir / "tb"
    tb_dir.mkdir(exist_ok=True)
    writer_kwargs = {"log_dir": str(tb_dir)}
    if resume_in_place:
        writer_kwargs["purge_step"] = start_it + 1
    writer = SummaryWriter(**writer_kwargs)

    ckpt_dir = run_dir / "ckpt"
    ckpt_dir.mkdir(exist_ok=True)

    steps_path = run_dir / "steps.jsonl"
    steps_mode = "a" if (resume_in_place and steps_path.exists()) else "w"

    # ---- train loop ----
    global_step = 0
    obs, _ = env.reset(seed=seed)
    done = False

    # logs per-iter
    ep_returns: List[float] = []
    ep_lens: List[int] = []

    # per-iter summary
    term_counts: Dict[str, int] = {}
    term_f_list: List[float] = []
    term_level1_inv_after_list: List[float] = []
    success_tstar_list: List[float] = []

    # reward parts
    reward_sums = {
        "R_step": 0.0,
        "R_base": 0.0,
        "R_demand": 0.0,
        "R_t_star": 0.0,
        "R_level23_penalty": 0.0,
        "R_T_left": 0.0,
        "R_f_penalty": 0.0,
        "R_level1": 0.0,
        "R_demand_not_fulfilled": 0.0,
        "R_over_Ui": 0.0,
        "R_over_Uk": 0.0,
        "R_below_0": 0.0,
        "R_ultra_limit": 0.0,
        "Re": 0.0,
    }
    sum_r_total_raw = 0.0
    sum_r_total_scaled = 0.0
    n_steps = 0

    with open(steps_path, steps_mode, encoding="utf-8") as f_steps:
        for it in range(start_it + 1, iterations + 1):
            buffer.reset()

            # steps.jsonl: each iteration records its first complete episode only.
            # Note: the iteration may start in the middle of an unfinished episode from the
            # previous iteration, so we arm logging only after seeing a fresh step with t == 0.
            log_started = False
            log_finished = False
            ep_stepbuf: List[Dict[str, Any]] = []

            debug_started = False
            debug_finished = False
            debug_rows: List[Dict[str, Any]] = []

            ep_returns = []
            ep_lens = []
            term_counts = {}
            term_f_list = []
            term_level1_inv_after_list = []
            success_tstar_list = []

            for _k in reward_sums:
                reward_sums[_k] = 0.0
            sum_r_total_raw = 0.0
            sum_r_total_scaled = 0.0
            n_steps = 0

            # per-episode stats (within this iter)
            ep_ret = 0.0
            ep_len = 0

            for _ in range(steps_per_iter):
                global_step += 1

                if debug_first_full_episode_each_iter and (not debug_finished) and (not debug_started):
                    if int(getattr(env, "_t", -1)) == 0:
                        debug_started = True
                        debug_rows = []

                if debug_first_full_episode_each_iter and debug_started and (not debug_finished):
                    try:
                        dbg = env.get_debug_snapshot()
                    except Exception as e:
                        dbg = {"t": int(getattr(env, "_t", -1)), "debug_error": str(e)}
                    dbg = dict(dbg)
                    dbg["iter"] = int(it)
                    debug_rows.append(dbg)

                # policy act
                out = model.act(obs, deterministic=False)

                action_env = out.action_env
                action_aux = out.action_aux

                # env step
                next_obs, reward, terminated, truncated, info = env.step(
                    {
                        "truck": action_env["truck"],
                        "belt": action_env["belt"],
                        "belt_order": action_env["belt_order"],
                    }
                )
                done = bool(terminated or truncated)

                # buffer add
                buffer.add(
                    obs=obs,
                    action={
                        "truck": action_env["truck"],
                        "belt": action_env["belt"],
                        "belt_order": action_env["belt_order"],
                        "order_adj": action_aux.get("order_adj", np.zeros((B, B), dtype=np.float32)),
                    },
                    logprob=float(out.logprob_ppo),
                    value=float(out.value),
                    reward=float(reward),
                    done=bool(done),
                )

                # stats
                ep_ret += float(reward)
                ep_len += 1

                for _k in reward_sums:
                    reward_sums[_k] += float(info.get(_k, 0.0))
                sum_r_total_raw += float(info.get("reward_raw", 0.0))
                sum_r_total_scaled += float(reward)
                n_steps += 1

                # steps.jsonl: start when we see t==0 at a step boundary
                if (not log_finished) and (not log_started):
                    tr0 = _get_last_trace(env)
                    if tr0 is not None:
                        t0 = int(tr0.get("t", -1))
                        if t0 == 0:
                            log_started = True
                            ep_stepbuf = []

                if done:
                    reason = str(info.get("terminate_reason", "unknown"))
                    # 1) 仅记录因 success 终止回合的终止步 f(T)
                    if reason == "success":
                        term_f_list.append(float(getattr(env, "_f_switch", 0)))
                    # 2) 仅记录因 success 终止回合的一级节点总库存量（状态转移后）
                    if reason == "success":
                        Wi_after = getattr(env, "_Wi", None)
                        if Wi_after is not None and len(level1_rows) > 0:
                            term_level1_inv_after_list.append(float(np.asarray(Wi_after)[level1_rows, :].sum()))
                        else:
                            term_level1_inv_after_list.append(0.0)
                    # 3) 因 success 终止的回合记录 T*（由 env 在回合内维护）
                    if reason == "success":
                        t_star = info.get("T_star", getattr(env, "_t_star", None))
                        if t_star is not None:
                            success_tstar_list.append(float(t_star))
                    # 4) 写入 StepTrace（env.trace 的最后一步），用于 steps.jsonl 终止步带上 episode_return
                    if getattr(env, "trace", None) and len(env.trace) > 0:
                        env.trace[-1].episode_return = float(ep_ret)

                # steps.jsonl: buffer the first full episode only
                if log_started and (not log_finished):
                    tr = _get_last_trace(env)
                    if tr is None:
                        # Fallback (should be rare): infer t as env._t-1 after stepping
                        t_last = int(getattr(env, "_t", 0)) - 1
                        tr = {"t": t_last, "reward": float(reward), "done": bool(done)}
                    tr = _format_steptrace(tr, it, env.node_ids)
                    ep_stepbuf.append(tr)

                if done:
                    if log_started and (not log_finished):
                        # flush buffered episode (guaranteed to start at t=0 within this iter)
                        for row in ep_stepbuf:
                            f_steps.write(json.dumps(_to_jsonable(row), ensure_ascii=False) + "\n")
                        f_steps.flush()
                        os.fsync(f_steps.fileno())
                        log_finished = True
                        log_started = False
                        ep_stepbuf = []

                    if debug_first_full_episode_each_iter and debug_started and (not debug_finished):
                        _print_first_episode_debug(it, debug_rows)
                        debug_finished = True
                        debug_started = False
                        debug_rows = []

                    ep_returns.append(ep_ret)
                    ep_lens.append(ep_len)
                    term_counts[reason] = term_counts.get(reason, 0) + 1

                    obs, _ = env.reset(seed=seed + it + global_step)
                    done = False
                    ep_ret = 0.0
                    ep_len = 0
                else:
                    obs = next_obs

            # bootstrap value
            with torch.no_grad():
                obs_critic = torch.tensor(obs["critic"], device=device).unsqueeze(0)
                last_v = float(model.critic(obs_critic).item())

            returns, adv_raw = buffer.compute_returns_advantages(last_value=last_v, gamma=ppo_cfg.gamma, gae_lambda=ppo_cfg.gae_lambda)
            metrics = trainer.update(iteration=it)

            # tensorboard
            writer.add_scalar("loss/pi_loss", metrics["pi_loss"], it)
            writer.add_scalar("loss/v_loss", metrics["v_loss"], it)
            writer.add_scalar("loss/entropy", metrics["entropy"], it)
            writer.add_scalar("loss/total_loss", metrics["total_loss"], it)
            writer.add_scalar("train/ent_coef", metrics["ent_coef"], it)
            writer.add_scalar("grad_norm/truck_actor", metrics["truck_actor_grad_norm"], it)
            writer.add_scalar("grad_norm/belt_actor", metrics["belt_actor_grad_norm"], it)
            writer.add_scalar("grad_norm/critic", metrics["critic_grad_norm"], it)

            writer.add_scalar("order/loss", metrics["order_loss"], it)
            writer.add_scalar("order/h_W", metrics["h(W)"], it)
            writer.add_scalar("order/c_Wk", metrics["c(W^k)"], it)
            writer.add_scalar("order/h_scaled", metrics["h_scaled"], it)
            writer.add_scalar("order/c_scaled", metrics["c_scaled"], it)
            writer.add_scalar("order/mu_h", metrics["mu_h"], it)
            writer.add_scalar("order/mu_c", metrics["mu_c"], it)
            writer.add_scalar("order/lambda_h", metrics["lambda_h"], it)
            writer.add_scalar("order/lambda_c", metrics["lambda_c"], it)

            if len(term_f_list) > 0:
                writer.add_scalar("train/optimal object/f(T)", float(np.mean(term_f_list)), it)
            if len(term_level1_inv_after_list) > 0:
                writer.add_scalar("train/optimal object/level1_inventory(T)", float(np.mean(term_level1_inv_after_list)), it)
            if len(success_tstar_list) > 0:
                writer.add_scalar("train/optimal object/T* in success ep", float(np.mean(success_tstar_list)), it)

            if n_steps > 0:
                for _k, _v in reward_sums.items():
                    writer.add_scalar(f"train/reward/{_k}_mean", _v / n_steps, it)
                writer.add_scalar("train/reward/reward_raw_mean", sum_r_total_raw / n_steps, it)
                writer.add_scalar("train/reward/reward_scaled_mean", sum_r_total_scaled / n_steps, it)

            n_eps = len(ep_returns)
            if n_eps > 0:
                writer.add_scalar("train/episodes/return_mean", float(np.mean(ep_returns)), it)
                writer.add_scalar("train/episodes/return_std", float(np.std(ep_returns)), it)
                writer.add_scalar("train/episodes/len_mean", float(np.mean(ep_lens)), it)
                for k, v in sorted(term_counts.items()):
                    writer.add_scalar(f"train/termination/count/{k}", v, it)
                    writer.add_scalar(f"train/termination/rate/{k}", float(v) / float(n_eps), it)

            # terminal print per-iteration (no per-episode print)
            print(
                f"iter={it} pi_loss={metrics['pi_loss']:.3f} v_loss={metrics['v_loss']:.3f} "
                f"entropy={metrics['entropy']:.3f} order_loss={metrics['order_loss']:.3f} "
                f"h={metrics['h(W)']:.3e} c={metrics['c(W^k)']:.3e}"
            )

            if ckpt_every > 0 and (it % ckpt_every == 0):
                save_ckpt(it)

    save_ckpt(iterations)

    train_end_ts = time.time()
    print(f"[train] end_time={_format_wallclock(train_end_ts)}")
    print(f"[train] total_duration={_format_duration(train_end_ts - train_start_ts)}")

    writer.close()


if __name__ == "__main__":
    main()

'''
PC：
conda activate dl__py_3.9
tensorboard --logdir=runs\20260320-094551\tb
-i https://pypi.tuna.tsinghua.edu.cn/simple
conda env export > C:/Users/19419/Desktop/研三/物流调度/模型设计/多物料/model_1_七节点多物料汽运/pj_env.yaml
服务器：
salloc -p tyhcnormal -n 8 -N 1 --mem=24G        -n 8：请求8个CPU核心   -N 1：在1个节点上运行
salloc -p tyhcnormal -n 64 -N 1 --mem=200G
conda activate yxh__py_3.9     python train.py
'''
