from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import importlib.util
import random
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd


# =============================
# Config
# =============================
CONFIG_PATH = "configs/default.yaml"
OUTPUT_CSV_PATH = Path(__file__).resolve().parent / "ga.csv"
VERBOSE = False
RANDOM_SEED = 42

# -----------------------------
# Multi-objective weight config
# -----------------------------
OBJ_WEIGHT_FINAL_UNMET = 100
OBJ_WEIGHT_EARLY_SERVICE = -1
OBJ_WEIGHT_SWITCH_COST = 100.0
OBJ_WEIGHT_LEVEL1_INVENTORY = 0.5

# -----------------------------
# GA config
# -----------------------------
GA_POP_SIZE = 160
GA_GENERATIONS = 180
GA_TOURNAMENT_SIZE = 4
GA_ELITE_SIZE = 8
GA_CROSSOVER_RATE = 0.90
GA_MUTATION_RATE_TRUCK = 0.12
GA_MUTATION_RATE_BELT = 0.08
GA_MUTATION_RATE_ORDER = 0.10
GA_TRUCK_MUTATION_SIGMA = 0.20
GA_SPARSITY_PROB_TRUCK_ZERO = 0.35
GA_NO_TRANSPORT_PROB_BELT = 0.45
GA_PHASE1_GENERATIONS = 120
GA_PHASE2_GENERATIONS = 120
GA_WEIGHTED_GENERATIONS = 180
EARLY_STOP_PATIENCE = 30

# -----------------------------
# Penalty config
# -----------------------------
PENALTY_ABNORMAL = 1.0e6
PENALTY_FINAL_UNMET = 1.0e5
PENALTY_NOT_SUCCESS = 5.0e4
PENALTY_INVALID_FIXED_FINAL_UNMET = 2.0e5


# =============================
# Env import helpers
# =============================
def _load_env_module():
    """Try importing the env implementation in a few compatible ways."""
    try:
        from envs.env import load_yaml, build_env_spec, LogisticsEnv  # type: ignore
        return load_yaml, build_env_spec, LogisticsEnv
    except Exception:
        pass

    try:
        from env import load_yaml, build_env_spec, LogisticsEnv  # type: ignore
        return load_yaml, build_env_spec, LogisticsEnv
    except Exception:
        pass

    here = Path(__file__).resolve().parent
    candidates = [
        here / "env.py",
        here / "envs" / "env.py",
        Path.cwd() / "env.py",
        Path.cwd() / "envs" / "env.py",
    ]
    for p in candidates:
        if p.exists():
            spec = importlib.util.spec_from_file_location("ga_env_module", p)
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
            return mod.load_yaml, mod.build_env_spec, mod.LogisticsEnv
    raise ImportError("Cannot find env implementation (env.py or envs/env.py).")


load_yaml, build_env_spec, LogisticsEnv = _load_env_module()


# =============================
# Common helpers
# =============================
def resolve_config_path(config_path: str | Path = CONFIG_PATH) -> Path:
    p = Path(config_path)
    project_root = Path(__file__).resolve().parents[2]

    candidates = [
        p if p.is_absolute() else None,
        Path.cwd() / p,
        Path(__file__).resolve().parent / p,
        project_root / p,                     
        project_root / "configs/default.yaml" # 兜底
    ]

    for cand in candidates:
        if cand is not None and cand.exists():
            return cand.resolve()

    return p


@dataclass
class SolveResult:
    solved: bool
    status: str
    objective_value: Optional[float]
    action_table: pd.DataFrame
    summary: Dict[str, Any]


@dataclass
class Chromosome:
    truck: np.ndarray          # [horizon, Et, K], in [0, 1]
    belt_mat: np.ndarray       # [horizon, B], int in [0, K] where K means no transport
    order_keys: np.ndarray     # [horizon, B], real-valued random keys for permutation

    def copy(self) -> "Chromosome":
        return Chromosome(
            truck=self.truck.copy(),
            belt_mat=self.belt_mat.copy(),
            order_keys=self.order_keys.copy(),
        )


class FullHorizonGAScheduler:
    def __init__(self, yaml_path: Path | str):
        self.yaml_path = Path(yaml_path)
        self.cfg = load_yaml(str(self.yaml_path))
        self.spec = build_env_spec(self.cfg)
        self.env = LogisticsEnv(self.spec)

        self.node_ids = list(sorted(self.spec.nodes.keys()))
        self.node_index = {nid: r for r, nid in enumerate(self.node_ids)}
        self.K = self.spec.K()
        self.T = self.spec.T
        self.material_names = [m.name or f"C{k + 1}" for k, m in enumerate(self.spec.materials)]

        self.truck_edges = [te.edge_id for te in self.spec.truck_edges]
        self.belt_combos = list(sorted(self.spec.belt_combos.keys()))
        self.level1_nodes = [nid for nid in self.node_ids if int(self.spec.nodes[nid].level) == 1]
        self.rng = np.random.default_rng(RANDOM_SEED)
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

    # -----------------------------
    # Horizon helpers (same logic as MILP)
    # -----------------------------
    def _exo_demand(self, nid: int, k: int, t: int) -> int:
        node = self.spec.nodes[nid]
        if node.y is None:
            return 0
        if k >= len(node.y) or t >= len(node.y[k]):
            return 0
        return int(node.y[k][t])

    def _future_demand_zero_after(self, t: int) -> bool:
        for nid in self.node_ids:
            for k in range(self.K):
                for tt in range(t + 1, self.T):
                    if self._exo_demand(nid, k, tt) != 0:
                        return False
        return True

    def _candidate_success_times(self) -> List[int]:
        return [t for t in range(self.T) if self._future_demand_zero_after(t)]

    # -----------------------------
    # Encoding / decoding
    # -----------------------------
    def _make_random_chromosome(self, horizon: int) -> Chromosome:
        truck = self.rng.random((horizon, len(self.truck_edges), self.K), dtype=np.float32)
        zero_mask = self.rng.random((horizon, len(self.truck_edges), self.K)) < GA_SPARSITY_PROB_TRUCK_ZERO
        truck[zero_mask] = 0.0

        belt_mat = self.rng.integers(0, self.K + 1, size=(horizon, len(self.belt_combos)), endpoint=False)
        no_transport_mask = self.rng.random((horizon, len(self.belt_combos))) < GA_NO_TRANSPORT_PROB_BELT
        belt_mat[no_transport_mask] = self.K

        order_keys = self.rng.normal(0.0, 1.0, size=(horizon, len(self.belt_combos))).astype(np.float32)
        return Chromosome(truck=truck.astype(np.float32), belt_mat=belt_mat.astype(np.int32), order_keys=order_keys)

    def _make_zero_chromosome(self, horizon: int) -> Chromosome:
        return Chromosome(
            truck=np.zeros((horizon, len(self.truck_edges), self.K), dtype=np.float32),
            belt_mat=np.full((horizon, len(self.belt_combos)), self.K, dtype=np.int32),
            order_keys=np.zeros((horizon, len(self.belt_combos)), dtype=np.float32),
        )

    def _decode_action(self, chrom: Chromosome, t: int) -> Dict[str, np.ndarray]:
        truck = np.clip(chrom.truck[t], 0.0, 1.0).astype(np.float32)
        belt = np.zeros((len(self.belt_combos), self.K + 1), dtype=np.float32)
        belt[:, -1] = 1.0
        for cidx in range(len(self.belt_combos)):
            mk = int(chrom.belt_mat[t, cidx])
            belt[cidx, :] = 0.0
            if 0 <= mk < self.K:
                belt[cidx, mk] = 1.0
            else:
                belt[cidx, -1] = 1.0
        belt_order = np.argsort(-chrom.order_keys[t]).astype(np.int64)
        return {"truck": truck, "belt": belt, "belt_order": belt_order}

    # -----------------------------
    # Evaluation
    # -----------------------------
    def _extract_exec_action_table(self, trace: List[Any], trace_len: int) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for step_idx in range(trace_len):
            tr = trace[step_idx]
            x_dict = tr.X if hasattr(tr, "X") else {}
            b_dict = tr.B if hasattr(tr, "B") else {}

            for raw_eid, amt_list in x_dict.items():
                eid = self.spec.edge_raw_ids.index(raw_eid)
                edge = self.spec.edges[eid]
                for k, qty in enumerate(list(amt_list)):
                    qty = int(qty)
                    if qty <= 0:
                        continue
                    rows.append({
                        "t": step_idx,
                        "mode": "truck",
                        "action_id": raw_eid,
                        "src": edge.src,
                        "dst": edge.dst,
                        "material": self.material_names[k],
                        "qty": qty,
                    })

            for raw_cid, amt_list in b_dict.items():
                cid = self.spec.belt_combo_raw_ids.index(raw_cid)
                combo = self.spec.belt_combos[cid]
                edge = self.spec.edges[combo.edge_id]
                for k, qty in enumerate(list(amt_list)):
                    qty = int(qty)
                    if qty <= 0:
                        continue
                    rows.append({
                        "t": step_idx,
                        "mode": "belt",
                        "action_id": raw_cid,
                        "src": edge.src,
                        "dst": edge.dst,
                        "material": self.material_names[k],
                        "qty": qty,
                    })
        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(columns=["t", "mode", "action_id", "src", "dst", "material", "qty"])
        return df.sort_values(["t", "mode", "action_id", "material"]).reset_index(drop=True)

    def _evaluate_chromosome(
        self,
        chrom: Chromosome,
        horizon: int,
        objective_mode: str,
        fixed_final_unmet: Optional[float] = None,
    ) -> Dict[str, Any]:
        env = LogisticsEnv(self.spec)
        env.reset(seed=RANDOM_SEED)

        early_service = 0.0
        level1_inventory = 0.0
        last_info: Dict[str, Any] = {}
        last_done = False
        stop_step = -1

        for t in range(horizon):
            action = self._decode_action(chrom, t)
            _, _, done, _, info = env.step(action)
            last_info = info
            last_done = bool(done)
            stop_step = t

            tr = env.trace[-1]
            yi_before = np.asarray(tr.Yi_before, dtype=np.int32)
            y_after = np.asarray(tr.Y_after, dtype=np.int32)
            sat_t = int((yi_before - y_after).sum())
            early_service += float((horizon - t) * sat_t)

            wi_after = np.asarray(env._Wi, dtype=np.int32)
            for nid in self.level1_nodes:
                r = self.node_index[nid]
                level1_inventory += float(wi_after[r, :].sum())

            if done:
                break

        trace_len = len(env.trace)
        if trace_len == 0:
            final_unmet = float("inf")
            switch_cost = float("inf")
            terminate_reason = "no_step"
        else:
            final_unmet = float(np.asarray(env._Yi, dtype=np.int32).sum())
            switch_cost = float(int(getattr(env, "_f_switch", 0)))
            terminate_reason = str(last_info.get("terminate_reason", "running"))

        abnormal = terminate_reason == "abnormal"
        success = terminate_reason == "success"

        weighted_obj = (
            OBJ_WEIGHT_FINAL_UNMET * final_unmet
            + OBJ_WEIGHT_EARLY_SERVICE * early_service
            + OBJ_WEIGHT_SWITCH_COST * switch_cost
            + OBJ_WEIGHT_LEVEL1_INVENTORY * level1_inventory
        )
        phase2_obj = (
            OBJ_WEIGHT_FINAL_UNMET * final_unmet
            + OBJ_WEIGHT_SWITCH_COST * switch_cost
            + OBJ_WEIGHT_LEVEL1_INVENTORY * level1_inventory
        )

        penalty = 0.0
        if abnormal:
            penalty += PENALTY_ABNORMAL
        if fixed_final_unmet is not None and abs(final_unmet - float(fixed_final_unmet)) > 1e-9:
            penalty += PENALTY_INVALID_FIXED_FINAL_UNMET + PENALTY_FINAL_UNMET * abs(final_unmet - float(fixed_final_unmet))

        if objective_mode == "min_final_unmet":
            fitness = final_unmet + penalty
            if success:
                # Prefer earlier success under the same final_unmet=0 target.
                fitness += 1e-3 * trace_len
        elif objective_mode == "optimize_after_success":
            if final_unmet > 1e-9:
                penalty += PENALTY_NOT_SUCCESS + PENALTY_FINAL_UNMET * final_unmet
            fitness = phase2_obj + penalty
        elif objective_mode == "weighted":
            fitness = weighted_obj + penalty
        else:
            raise ValueError(f"Unknown objective_mode: {objective_mode}")

        action_table = self._extract_exec_action_table(env.trace, trace_len)
        return {
            "fitness": float(fitness),
            "success": bool(success),
            "abnormal": bool(abnormal),
            "terminate_reason": terminate_reason,
            "trace_len": trace_len,
            "final_unmet": float(final_unmet),
            "early_service": float(early_service),
            "switch_cost": float(switch_cost),
            "level1_inventory": float(level1_inventory),
            "weighted_objective": float(weighted_obj),
            "phase2_objective": float(phase2_obj),
            "last_done": bool(last_done),
            "stop_step": int(stop_step),
            "info": last_info,
            "action_table": action_table,
        }

    # -----------------------------
    # GA operators
    # -----------------------------
    def _tournament_pick(self, population: List[Chromosome], scores: List[float]) -> Chromosome:
        idxs = self.rng.integers(0, len(population), size=GA_TOURNAMENT_SIZE)
        best_idx = min(idxs, key=lambda i: scores[int(i)])
        return population[int(best_idx)]

    def _crossover(self, p1: Chromosome, p2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        c1 = p1.copy()
        c2 = p2.copy()
        if self.rng.random() > GA_CROSSOVER_RATE:
            return c1, c2

        # truck: arithmetic blend
        alpha = self.rng.random(size=p1.truck.shape, dtype=np.float32)
        c1.truck = np.clip(alpha * p1.truck + (1.0 - alpha) * p2.truck, 0.0, 1.0)
        c2.truck = np.clip(alpha * p2.truck + (1.0 - alpha) * p1.truck, 0.0, 1.0)

        # belt material: uniform crossover
        mask_belt = self.rng.random(size=p1.belt_mat.shape) < 0.5
        c1.belt_mat = np.where(mask_belt, p1.belt_mat, p2.belt_mat)
        c2.belt_mat = np.where(mask_belt, p2.belt_mat, p1.belt_mat)

        # order keys: uniform/arithmetic blend
        mask_ord = self.rng.random(size=p1.order_keys.shape) < 0.5
        mix = self.rng.random(size=p1.order_keys.shape, dtype=np.float32)
        ord1 = mix * p1.order_keys + (1.0 - mix) * p2.order_keys
        ord2 = mix * p2.order_keys + (1.0 - mix) * p1.order_keys
        c1.order_keys = np.where(mask_ord, ord1, p1.order_keys).astype(np.float32)
        c2.order_keys = np.where(mask_ord, ord2, p2.order_keys).astype(np.float32)
        return c1, c2

    def _mutate(self, chrom: Chromosome) -> None:
        # truck
        mut_mask_t = self.rng.random(size=chrom.truck.shape) < GA_MUTATION_RATE_TRUCK
        noise = self.rng.normal(0.0, GA_TRUCK_MUTATION_SIGMA, size=chrom.truck.shape).astype(np.float32)
        chrom.truck = np.where(mut_mask_t, np.clip(chrom.truck + noise, 0.0, 1.0), chrom.truck)
        zero_mask = self.rng.random(size=chrom.truck.shape) < (GA_MUTATION_RATE_TRUCK * 0.25)
        chrom.truck[zero_mask] = 0.0

        # belt material
        mut_mask_b = self.rng.random(size=chrom.belt_mat.shape) < GA_MUTATION_RATE_BELT
        reset_vals = self.rng.integers(0, self.K + 1, size=chrom.belt_mat.shape, endpoint=False)
        no_trans_mask = self.rng.random(size=chrom.belt_mat.shape) < GA_NO_TRANSPORT_PROB_BELT
        reset_vals[no_trans_mask] = self.K
        chrom.belt_mat = np.where(mut_mask_b, reset_vals, chrom.belt_mat).astype(np.int32)

        # order keys
        mut_mask_o = self.rng.random(size=chrom.order_keys.shape) < GA_MUTATION_RATE_ORDER
        noise_o = self.rng.normal(0.0, 1.0, size=chrom.order_keys.shape).astype(np.float32)
        chrom.order_keys = np.where(mut_mask_o, chrom.order_keys + noise_o, chrom.order_keys).astype(np.float32)

    def _run_ga(
        self,
        horizon: int,
        objective_mode: str,
        generations: int,
        fixed_final_unmet: Optional[float] = None,
    ) -> Tuple[Chromosome, Dict[str, Any]]:
        population: List[Chromosome] = [self._make_zero_chromosome(horizon)]
        population.extend(self._make_random_chromosome(horizon) for _ in range(GA_POP_SIZE - 1))

        best_eval: Optional[Dict[str, Any]] = None
        best_chrom: Optional[Chromosome] = None
        stagnant = 0

        for gen in range(generations):
            evals = [self._evaluate_chromosome(ind, horizon, objective_mode, fixed_final_unmet) for ind in population]
            scores = [e["fitness"] for e in evals]
            order = np.argsort(scores)

            gen_best_idx = int(order[0])
            gen_best_eval = evals[gen_best_idx]
            if (best_eval is None) or (gen_best_eval["fitness"] < best_eval["fitness"]):
                best_eval = gen_best_eval
                best_chrom = population[gen_best_idx].copy()
                stagnant = 0
            else:
                stagnant += 1

            if VERBOSE and (gen % 10 == 0 or gen == generations - 1):
                print(
                    f"[GA] mode={objective_mode:<18} horizon={horizon:<2d} gen={gen:<4d} "
                    f"best={gen_best_eval['fitness']:.3f} final_unmet={gen_best_eval['final_unmet']:.1f} "
                    f"reason={gen_best_eval['terminate_reason']}"
                )

            if objective_mode == "min_final_unmet" and best_eval is not None and best_eval["final_unmet"] <= 1e-9 and best_eval["success"]:
                break
            if stagnant >= EARLY_STOP_PATIENCE:
                break

            next_population: List[Chromosome] = [population[int(i)].copy() for i in order[:GA_ELITE_SIZE]]
            while len(next_population) < GA_POP_SIZE:
                p1 = self._tournament_pick(population, scores)
                p2 = self._tournament_pick(population, scores)
                c1, c2 = self._crossover(p1, p2)
                self._mutate(c1)
                self._mutate(c2)
                next_population.append(c1)
                if len(next_population) < GA_POP_SIZE:
                    next_population.append(c2)
            population = next_population[:GA_POP_SIZE]

        assert best_chrom is not None and best_eval is not None
        return best_chrom, best_eval

    # -----------------------------
    # Solve / verification
    # -----------------------------
    def verify_with_env(self, action_table: pd.DataFrame) -> Dict[str, Any]:
        """For GA, the action table already comes from env execution. Re-check by replaying approximately."""
        env = LogisticsEnv(self.spec)
        env.reset(seed=RANDOM_SEED)

        trace_rows: List[Dict[str, Any]] = []
        info: Dict[str, Any] = {}
        raw_combo_to_idx = {self.spec.belt_combo_raw_ids[cid]: cid for cid in self.belt_combos}
        raw_edge_to_idx = {self.spec.edge_raw_ids[eid]: eid for eid in self.truck_edges}
        mat_pos = {name: k for k, name in enumerate(self.material_names)}

        for t in range(self.T):
            df_t = action_table[action_table["t"] == t] if not action_table.empty else action_table
            truck = np.zeros((len(self.truck_edges), self.K), dtype=np.float32)
            belt = np.zeros((len(self.belt_combos), self.K + 1), dtype=np.float32)
            belt[:, -1] = 1.0
            belt_order = np.arange(len(self.belt_combos), dtype=np.int64)

            # Use executed quantities to reconstruct a close replay action.
            for row in df_t.itertuples(index=False):
                k = mat_pos[row.material]
                if row.mode == "truck":
                    eid = raw_edge_to_idx[row.action_id]
                    edge = self.spec.edges[eid]
                    cap = float(getattr(self.spec.nodes[edge.dst], "Mj_truck_unload", 0.0) or 0.0)
                    if cap <= 0:
                        cap = float(next(te.capacity for te in self.spec.truck_edges if te.edge_id == eid))
                    te_pos = self.truck_edges.index(eid)
                    truck[te_pos, k] = float(row.qty) / max(cap, 1.0)
                else:
                    cid = raw_combo_to_idx[row.action_id]
                    cidx = self.belt_combos.index(cid)
                    belt[cidx, :] = 0.0
                    belt[cidx, k] = 1.0

            _, reward, done, _, info = env.step({"truck": truck, "belt": belt, "belt_order": belt_order})
            last = env.last_trace() or {}
            trace_rows.append({
                "t": t,
                "reward": reward,
                "done": done,
                "reason": info.get("terminate_reason") if isinstance(info, dict) else None,
                "unmet_sum": int(np.asarray(last.get("Y_after", np.zeros((1, 1))), dtype=np.int32).sum()) if last else None,
            })
            if done:
                break

        final_trace = env.last_trace() or {}
        return {
            "env_steps": trace_rows,
            "terminate_reason": info.get("terminate_reason") if isinstance(info, dict) else None,
            "trace_len": len(env.trace),
            "final_t": int(final_trace.get("t", -1)) if final_trace else -1,
        }

    def solve(self) -> SolveResult:
        chosen_chrom: Optional[Chromosome] = None
        chosen_eval: Optional[Dict[str, Any]] = None
        chosen_status = "UNKNOWN"
        solve_mode = "ga_weighted_full_horizon"
        success_horizon: Optional[int] = None
        phase1_best_final_unmet: Optional[float] = None

        for success_t in self._candidate_success_times():
            horizon = success_t + 1
            chrom1, eval1 = self._run_ga(
                horizon=horizon,
                objective_mode="min_final_unmet",
                generations=GA_PHASE1_GENERATIONS,
            )
            if eval1["final_unmet"] > 1e-9 or not eval1["success"]:
                continue

            success_horizon = horizon
            phase1_best_final_unmet = eval1["final_unmet"]
            chrom2, eval2 = self._run_ga(
                horizon=horizon,
                objective_mode="optimize_after_success",
                generations=GA_PHASE2_GENERATIONS,
                fixed_final_unmet=phase1_best_final_unmet,
            )
            if eval2["final_unmet"] <= 1e-9 and eval2["success"]:
                chosen_chrom = chrom2
                chosen_eval = eval2
                chosen_status = "SUCCESS"
                solve_mode = "ga_earliest_success_then_optimize"
            else:
                chosen_chrom = chrom1
                chosen_eval = eval1
                chosen_status = "SUCCESS_PHASE1_ONLY"
                solve_mode = "ga_earliest_success_phase1_only"
            break

        if chosen_chrom is None or chosen_eval is None:
            chosen_chrom, chosen_eval = self._run_ga(
                horizon=self.T,
                objective_mode="weighted",
                generations=GA_WEIGHTED_GENERATIONS,
            )
            if chosen_eval["abnormal"]:
                chosen_status = "ABNORMAL_BEST_FOUND"
            elif chosen_eval["success"]:
                chosen_status = "SUCCESS"
            else:
                chosen_status = "FEASIBLE_BEST_FOUND"

        action_table = chosen_eval["action_table"]
        verify = self.verify_with_env(action_table)
        objective_breakdown = {
            "final_unmet": float(chosen_eval["final_unmet"]),
            "early_service": float(chosen_eval["early_service"]),
            "switch_cost": float(chosen_eval["switch_cost"]),
            "level1_inventory": float(chosen_eval["level1_inventory"]),
            "weighted_objective": float(chosen_eval["weighted_objective"]),
            "phase2_objective": float(chosen_eval["phase2_objective"]),
        }
        summary = {
            "status": chosen_status,
            "objective_value": float(chosen_eval["fitness"]),
            "objective_breakdown": objective_breakdown,
            "action_count": int(len(action_table)),
            "horizon_used": int(chosen_eval["trace_len"]),
            "solve_mode": solve_mode,
            "success_horizon": success_horizon,
            "phase1_best_final_unmet": phase1_best_final_unmet,
            "terminate_reason": chosen_eval["terminate_reason"],
            "verify": verify,
            "ga_config": {
                "population": GA_POP_SIZE,
                "phase1_generations": GA_PHASE1_GENERATIONS,
                "phase2_generations": GA_PHASE2_GENERATIONS,
                "weighted_generations": GA_WEIGHTED_GENERATIONS,
                "elite": GA_ELITE_SIZE,
                "tournament": GA_TOURNAMENT_SIZE,
                "seed": RANDOM_SEED,
            },
        }
        return SolveResult(True, chosen_status, float(chosen_eval["fitness"]), action_table, summary)


def main() -> None:
    yaml_path = resolve_config_path(CONFIG_PATH)
    solver = FullHorizonGAScheduler(yaml_path)
    result = solver.solve()

    print(f"yaml_path: {yaml_path}")
    print(f"status: {result.status}")
    print(f"objective_value: {result.objective_value}")
    print(f"summary: {result.summary}")

    if result.action_table.empty:
        print("No non-zero executed action was found.")
    else:
        print("\nExecuted action table:")
        print(result.action_table.to_string(index=False))
        result.action_table.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")
        print(f"\nSaved action table to: {OUTPUT_CSV_PATH.resolve()}")


if __name__ == "__main__":
    main()
