from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

from envs.env import load_yaml, build_env_spec, LogisticsEnv, allowed_mask_for_edge_mode


CONFIG_PATH = "configs/default.yaml"
TIME_LIMIT_SECONDS = 60
MIP_GAP = 0.0
VERBOSE = False
OUTPUT_CSV_PATH = Path(__file__).resolve().parent / "milp.csv"

# -----------------------------
# Multi-objective weight config
# -----------------------------
OBJ_WEIGHT_FINAL_UNMET = 100           # 调大：更强地压制期末未满足需求，模型会更优先保供；调小：允许用更多切换/库存等代价换取少量缺口
OBJ_WEIGHT_EARLY_SERVICE = -1        # 该项是“负权重奖励”，绝对值调大：更鼓励提前满足需求；绝对值调小或更接近0：提前满足的驱动力减弱
OBJ_WEIGHT_SWITCH_COST = 100.0            # 调大：更强地惩罚运输/生产切换，方案更平滑稳定但可能牺牲响应性；调小：更允许频繁切换以追求保供或降库存
OBJ_WEIGHT_LEVEL1_INVENTORY = 0.5         # 调大：更强地压低一级节点库存，减少一级节点积压；调小：更容忍一级节点持库存以换取供给灵活性


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


class FullHorizonMILPScheduler:
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
        self.combo_overlap_pairs = self._build_overlap_pairs()

    def _build_overlap_pairs(self) -> List[Tuple[int, int]]:
        pairs: List[Tuple[int, int]] = []
        for i, cid_i in enumerate(self.belt_combos):
            belts_i = set(self.spec.belt_combos[cid_i].sub_belts)
            for j in range(i + 1, len(self.belt_combos)):
                cid_j = self.belt_combos[j]
                belts_j = set(self.spec.belt_combos[cid_j].sub_belts)
                if belts_i.intersection(belts_j):
                    pairs.append((cid_i, cid_j))
        return pairs

    def _pi_mask(self, src_nid: int, t: int, mode: str) -> np.ndarray:
        src = self.spec.nodes[src_nid]
        pi_row = None
        if src.Pi is not None:
            pi_row = [
                float(src.Pi[k][t]) if (k < len(src.Pi) and t < len(src.Pi[k])) else 0.5
                for k in range(self.K)
            ]
        return allowed_mask_for_edge_mode(pi_row, self.K, mode)

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

    def _status_to_str(self, status: int) -> str:
        status_map = {
            GRB.OPTIMAL: "OPTIMAL",
            GRB.TIME_LIMIT: "TIME_LIMIT",
            GRB.SUBOPTIMAL: "SUBOPTIMAL",
            GRB.INFEASIBLE: "INFEASIBLE",
            GRB.INF_OR_UNBD: "INF_OR_UNBD",
            GRB.UNBOUNDED: "UNBOUNDED",
        }
        return status_map.get(status, str(status))

    def _has_usable_solution(self, model: gp.Model) -> bool:
        return model.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL) and model.SolCount > 0

    def build_model(
        self,
        horizon: Optional[int] = None,
        objective_mode: str = "weighted",
        fixed_final_unmet: Optional[float] = None,
    ) -> Tuple[gp.Model, Dict[str, Any]]:
        spec = self.spec
        horizon = int(horizon or self.T)
        if horizon <= 0 or horizon > self.T:
            raise ValueError(f"Invalid horizon: {horizon}")

        model = gp.Model("full_horizon_milp_scheduler")
        model.Params.OutputFlag = 1 if VERBOSE else 0
        model.Params.TimeLimit = TIME_LIMIT_SECONDS
        model.Params.MIPGap = MIP_GAP

        W = model.addVars(self.node_ids, range(self.K), range(horizon), vtype=GRB.INTEGER, lb=0, name="W")
        Y = model.addVars(self.node_ids, range(self.K), range(horizon), vtype=GRB.INTEGER, lb=0, name="Y")
        X = model.addVars(self.truck_edges, range(self.K), range(horizon), vtype=GRB.INTEGER, lb=0, name="X")
        Z = model.addVars(self.belt_combos, range(self.K), range(horizon), vtype=GRB.BINARY, name="Z")
        B = model.addVars(self.belt_combos, range(self.K), range(horizon), vtype=GRB.INTEGER, lb=0, name="B")
        U = model.addVars(self.belt_combos, range(horizon), vtype=GRB.BINARY, name="U")
        S = model.addVars(self.node_ids, range(self.K), range(horizon), vtype=GRB.INTEGER, lb=0, name="S")
        MAT_SW = model.addVars(self.belt_combos, range(1, horizon), vtype=GRB.BINARY, name="MAT_SW")
        OVL_SW = model.addVars(self.belt_combos, range(1, horizon), vtype=GRB.BINARY, name="OVL_SW")

        for cid in self.belt_combos:
            cap = int(self.spec.belt_combos[cid].capacity)
            for t in range(horizon):
                model.addConstr(gp.quicksum(Z[cid, k, t] for k in range(self.K)) == U[cid, t], name=f"belt_one_material_{cid}_{t}")
                for k in range(self.K):
                    model.addConstr(B[cid, k, t] == cap * Z[cid, k, t], name=f"belt_cap_link_{cid}_{k}_{t}")

        for cid_i, cid_j in self.combo_overlap_pairs:
            for t in range(horizon):
                model.addConstr(U[cid_i, t] + U[cid_j, t] <= 1, name=f"overlap_{cid_i}_{cid_j}_{t}")

        for eid in self.truck_edges:
            edge = spec.edges[eid]
            for t in range(horizon):
                mask = self._pi_mask(edge.src, t, mode="truck")
                dst_mj = int(getattr(spec.nodes[edge.dst], "Mj_truck_unload", 0.0) or 0.0)
                if dst_mj <= 0:
                    dst_mj = int(next(te.capacity for te in spec.truck_edges if te.edge_id == eid))
                for k in range(self.K):
                    if mask[k] <= 0.0:
                        model.addConstr(X[eid, k, t] == 0, name=f"truck_pi_block_{eid}_{k}_{t}")
                    else:
                        model.addConstr(X[eid, k, t] <= dst_mj, name=f"truck_edge_cap_{eid}_{k}_{t}")

        for cid in self.belt_combos:
            combo = spec.belt_combos[cid]
            edge = spec.edges[combo.edge_id]
            for t in range(horizon):
                mask = self._pi_mask(edge.src, t, mode="belt")
                for k in range(self.K):
                    if mask[k] <= 0.0:
                        model.addConstr(Z[cid, k, t] == 0, name=f"belt_pi_block_{cid}_{k}_{t}")

        dst_to_truck_edges: Dict[int, List[int]] = {}
        for eid in self.truck_edges:
            dst = spec.edges[eid].dst
            dst_to_truck_edges.setdefault(dst, []).append(eid)
        for dst, eids in dst_to_truck_edges.items():
            mj = int(getattr(spec.nodes[dst], "Mj_truck_unload", 0.0) or 0.0)
            if mj > 0:
                for t in range(horizon):
                    model.addConstr(
                        gp.quicksum(X[eid, k, t] for eid in eids for k in range(self.K)) <= mj,
                        name=f"truck_dest_mj_{dst}_{t}",
                    )

        for nid in self.node_ids:
            node = spec.nodes[nid]
            for k in range(self.K):
                for t in range(horizon):
                    inflow = gp.LinExpr()
                    outflow = gp.LinExpr()

                    for eid in self.truck_edges:
                        edge = spec.edges[eid]
                        if edge.dst == nid:
                            inflow += X[eid, k, t]
                        if edge.src == nid:
                            outflow += X[eid, k, t]

                    for cid in self.belt_combos:
                        combo = spec.belt_combos[cid]
                        edge = spec.edges[combo.edge_id]
                        if edge.dst == nid:
                            inflow += B[cid, k, t]
                        if edge.src == nid:
                            outflow += B[cid, k, t]

                    prev_w = int(node.W0[k]) if t == 0 else W[nid, k, t - 1]
                    model.addConstr(W[nid, k, t] == prev_w + inflow - outflow, name=f"inv_bal_{nid}_{k}_{t}")

                    exo = self._exo_demand(nid, k, t)
                    prev_y = 0 if t == 0 else Y[nid, k, t - 1]
                    if node.y is None:
                        model.addConstr(S[nid, k, t] == 0, name=f"no_demand_sat_{nid}_{k}_{t}")
                        model.addConstr(Y[nid, k, t] == 0, name=f"no_demand_state_{nid}_{k}_{t}")
                    else:
                        model.addConstr(S[nid, k, t] <= inflow, name=f"sat_le_inflow_{nid}_{k}_{t}")
                        model.addConstr(S[nid, k, t] <= prev_y + exo, name=f"sat_le_need_{nid}_{k}_{t}")
                        model.addConstr(Y[nid, k, t] == prev_y + exo - S[nid, k, t], name=f"dem_bal_{nid}_{k}_{t}")

            for t in range(horizon):
                model.addConstr(
                    gp.quicksum(W[nid, k, t] for k in range(self.K)) <= int(node.Ui),
                    name=f"node_ui_{nid}_{t}",
                )

        lvl23_nodes = [nid for nid in self.node_ids if int(spec.nodes[nid].level) in (2, 3)]
        for k in range(self.K):
            uk = int(spec.materials[k].Uk)
            for t in range(horizon):
                model.addConstr(
                    gp.quicksum(W[nid, k, t] for nid in lvl23_nodes) <= uk,
                    name=f"lvl23_uk_{k}_{t}",
                )

        for cid in self.belt_combos:
            overlap_neighbors = [c2 for c1, c2 in self.combo_overlap_pairs if c1 == cid] + [c1 for c1, c2 in self.combo_overlap_pairs if c2 == cid]
            for t in range(1, horizon):
                for k1 in range(self.K):
                    for k2 in range(self.K):
                        if k1 == k2:
                            continue
                        model.addConstr(MAT_SW[cid, t] >= Z[cid, k1, t - 1] + Z[cid, k2, t] - 1, name=f"mat_sw_{cid}_{t}_{k1}_{k2}")

                if overlap_neighbors:
                    for c_prev in overlap_neighbors:
                        model.addConstr(
                            OVL_SW[cid, t] >= U[c_prev, t - 1] + U[cid, t] - 1,
                            name=f"ovl_sw_{cid}_{c_prev}_{t}",
                        )
                else:
                    model.addConstr(OVL_SW[cid, t] == 0, name=f"ovl_sw_zero_{cid}_{t}")

        level1_nodes = [nid for nid in self.node_ids if int(spec.nodes[nid].level) == 1]

        final_unmet = gp.quicksum(Y[nid, k, horizon - 1] for nid in self.node_ids for k in range(self.K))
        switch_cost = gp.quicksum(MAT_SW[cid, t] + OVL_SW[cid, t] for cid in self.belt_combos for t in range(1, horizon))
        level1_inventory = gp.quicksum(W[nid, k, t] for nid in level1_nodes for k in range(self.K) for t in range(horizon))

        if fixed_final_unmet is not None:
            model.addConstr(final_unmet == fixed_final_unmet, name="fixed_final_unmet")

        if objective_mode == "weighted":
            early_service = gp.quicksum((horizon - t) * S[nid, k, t] for nid in self.node_ids for k in range(self.K) for t in range(horizon))
            model.setObjective(
                OBJ_WEIGHT_FINAL_UNMET * final_unmet
                + OBJ_WEIGHT_EARLY_SERVICE * early_service
                + OBJ_WEIGHT_SWITCH_COST * switch_cost
                + OBJ_WEIGHT_LEVEL1_INVENTORY * level1_inventory,
                GRB.MINIMIZE,
            )
        elif objective_mode == "min_final_unmet":
            early_service = gp.LinExpr(0.0)
            model.setObjective(final_unmet, GRB.MINIMIZE)
        elif objective_mode == "optimize_after_success":
            early_service = gp.LinExpr(0.0)
            model.setObjective(
                OBJ_WEIGHT_FINAL_UNMET * final_unmet
                + OBJ_WEIGHT_SWITCH_COST * switch_cost
                + OBJ_WEIGHT_LEVEL1_INVENTORY * level1_inventory,
                GRB.MINIMIZE,
            )
        else:
            raise ValueError(f"Unknown objective_mode: {objective_mode}")

        return model, {
            "W": W,
            "Y": Y,
            "X": X,
            "Z": Z,
            "B": B,
            "U": U,
            "S": S,
            "MAT_SW": MAT_SW,
            "OVL_SW": OVL_SW,
            "horizon": horizon,
            "objective_exprs": {
                "final_unmet": final_unmet,
                "early_service": early_service,
                "switch_cost": switch_cost,
                "level1_inventory": level1_inventory,
            },
        }

    def _extract_solution(self, var_dict: Dict[str, Any]) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        X = var_dict["X"]
        B = var_dict["B"]
        U = var_dict["U"]

        horizon = int(var_dict.get("horizon", self.T))

        for t in range(horizon):
            for eid in self.truck_edges:
                edge = self.spec.edges[eid]
                for k in range(self.K):
                    qty = int(round(X[eid, k, t].X))
                    if qty <= 0:
                        continue
                    rows.append({"t": t, "mode": "truck", "action_id": self.spec.edge_raw_ids[eid], "src": edge.src, "dst": edge.dst, "material": self.material_names[k], "qty": qty})

            for cid in self.belt_combos:
                if int(round(U[cid, t].X)) <= 0:
                    continue
                combo = self.spec.belt_combos[cid]
                edge = self.spec.edges[combo.edge_id]
                for k in range(self.K):
                    qty = int(round(B[cid, k, t].X))
                    if qty <= 0:
                        continue
                    rows.append({"t": t, "mode": "belt", "action_id": self.spec.belt_combo_raw_ids[cid], "src": edge.src, "dst": edge.dst, "material": self.material_names[k], "qty": qty})

        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(columns=["t", "mode", "action_id", "src", "dst", "material", "qty"])
        return df.sort_values(["t", "mode", "action_id", "material"]).reset_index(drop=True)

    def _build_env_action(self, df_t: pd.DataFrame) -> Dict[str, np.ndarray]:
        truck = np.zeros((len(self.truck_edges), self.K), dtype=np.float32)
        belt = np.zeros((len(self.belt_combos), self.K + 1), dtype=np.float32)
        belt[:, -1] = 1.0
        belt_order = np.arange(len(self.belt_combos), dtype=np.int64)

        edge_pos = {eid: i for i, eid in enumerate(self.truck_edges)}
        mat_pos = {name: k for k, name in enumerate(self.material_names)}
        combo_pos = {cid: i for i, cid in enumerate(self.belt_combos)}
        raw_combo_to_cid = {self.spec.belt_combo_raw_ids[cid]: cid for cid in self.belt_combos}
        raw_edge_to_eid = {self.spec.edge_raw_ids[eid]: eid for eid in self.truck_edges}

        for row in df_t.itertuples(index=False):
            k = mat_pos[row.material]
            if row.mode == "truck":
                eid = raw_edge_to_eid[row.action_id]
                edge = self.spec.edges[eid]
                mj = float(getattr(self.spec.nodes[edge.dst], "Mj_truck_unload", 0.0) or 0.0)
                if mj <= 0:
                    mj = float(next(te.capacity for te in self.spec.truck_edges if te.edge_id == eid))
                truck[edge_pos[eid], k] = float(row.qty) / mj
            else:
                cid = raw_combo_to_cid[row.action_id]
                cidx = combo_pos[cid]
                belt[cidx, :] = 0.0
                belt[cidx, k] = 1.0

        return {"truck": truck, "belt": belt, "belt_order": belt_order}

    def verify_with_env(self, action_table: pd.DataFrame) -> Dict[str, Any]:
        env = LogisticsEnv(self.spec)
        env.reset(seed=0)

        trace_rows: List[Dict[str, Any]] = []
        info: Dict[str, Any] = {}
        for t in range(self.T):
            df_t = action_table[action_table["t"] == t] if not action_table.empty else action_table
            action = self._build_env_action(df_t)
            _, reward, done, _, info = env.step(action)
            last = env.last_trace() or {}
            trace_rows.append({
                "t": t,
                "reward": reward,
                "done": done,
                "reason": info.get("terminate_reason") if isinstance(info, dict) else None,
                "unmet_sum": int(np.asarray(last.get("Yi", np.zeros((1, 1))), dtype=np.int32).sum()) if last else None,
            })
            if done:
                break

        final_trace = env.last_trace() or {}
        return {"env_steps": trace_rows, "terminate_reason": info.get("terminate_reason") if isinstance(info, dict) else None, "trace_len": len(env.trace), "final_t": int(final_trace.get("t", -1)) if final_trace else -1}

    def solve(self) -> SolveResult:
        total_runtime = 0.0
        chosen_model: Optional[gp.Model] = None
        chosen_var_dict: Optional[Dict[str, Any]] = None
        chosen_status = "UNKNOWN"
        solve_mode = "full_horizon_weighted"
        success_horizon: Optional[int] = None
        phase1_best_final_unmet: Optional[float] = None

        for success_t in self._candidate_success_times():
            horizon = success_t + 1
            model1, var_dict1 = self.build_model(horizon=horizon, objective_mode="min_final_unmet")
            model1.optimize()
            total_runtime += float(model1.Runtime)

            if not self._has_usable_solution(model1):
                continue

            phase1_final_unmet = float(var_dict1["objective_exprs"]["final_unmet"].getValue())
            if phase1_final_unmet > 1e-6:
                continue

            success_horizon = horizon
            phase1_best_final_unmet = phase1_final_unmet
            model2, var_dict2 = self.build_model(
                horizon=horizon,
                objective_mode="optimize_after_success",
                fixed_final_unmet=phase1_final_unmet,
            )
            model2.optimize()
            total_runtime += float(model2.Runtime)

            if self._has_usable_solution(model2):
                chosen_model = model2
                chosen_var_dict = var_dict2
                chosen_status = self._status_to_str(model2.Status)
                solve_mode = "earliest_success_then_optimize"
            else:
                chosen_model = model1
                chosen_var_dict = var_dict1
                chosen_status = self._status_to_str(model1.Status)
                solve_mode = "earliest_success_phase1_only"
            break

        if chosen_model is None or chosen_var_dict is None:
            fallback_model, fallback_var_dict = self.build_model(horizon=self.T, objective_mode="weighted")
            fallback_model.optimize()
            total_runtime += float(fallback_model.Runtime)
            if not self._has_usable_solution(fallback_model):
                return SolveResult(False, self._status_to_str(fallback_model.Status), None, pd.DataFrame(), {"message": "模型未得到可用解。", "solve_time_seconds": total_runtime})
            chosen_model = fallback_model
            chosen_var_dict = fallback_var_dict
            chosen_status = self._status_to_str(fallback_model.Status)

        action_table = self._extract_solution(chosen_var_dict)
        verify = self.verify_with_env(action_table)
        objective_exprs = chosen_var_dict.get("objective_exprs", {})
        objective_breakdown = {
            name: float(expr.getValue()) for name, expr in objective_exprs.items()
        }
        summary = {
            "status": chosen_status,
            "objective_value": float(chosen_model.ObjVal),
            "objective_breakdown": objective_breakdown,
            "solve_time_seconds": total_runtime,
            "action_count": int(len(action_table)),
            "horizon_used": int(chosen_var_dict.get("horizon", self.T)),
            "solve_mode": solve_mode,
            "success_horizon": success_horizon,
            "phase1_best_final_unmet": phase1_best_final_unmet,
            "env_verify": verify,
        }
        return SolveResult(True, chosen_status, float(chosen_model.ObjVal), action_table, summary)



def run() -> SolveResult:
    scheduler = FullHorizonMILPScheduler(resolve_config_path(CONFIG_PATH))
    result = scheduler.solve()

    if result.action_table.empty:
        print("Action table:")
        print("<empty>")
    else:
        print("Action table:")
        print(result.action_table.to_string(index=False))
        result.action_table.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")

    objective_breakdown = result.summary.get("objective_breakdown", {})
    print(f"final_unmet: {objective_breakdown.get('final_unmet')}")
    print(f"switch_cost: {objective_breakdown.get('switch_cost')}")
    print(f"level1_inventory: {objective_breakdown.get('level1_inventory')}")
    print(f"solve_time_seconds: {result.summary.get('solve_time_seconds')}")
    print(f"objective_value: {result.summary.get('objective_value')}")

    return result


if __name__ == "__main__":
    run()
