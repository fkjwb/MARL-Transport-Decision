from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml


STATE_STAY = 0
STATE_MOVE_EMPTY = 1
STATE_TRANSPORT = 2


@dataclass
class VehicleSpec:
    vehicle_id: int
    vehicle_type: int  # 0 small, 1 large
    capacity: int
    init_node: int


@dataclass
class EnvSpec:
    name: str
    T: int
    n_obs: int
    node_ids: List[int]
    edge_list: List[Tuple[int, int]]
    material_names: List[str]
    material_ids: List[int]
    vehicles: List[VehicleSpec]
    demand: np.ndarray  # [T, N, N, K]

    @property
    def N(self) -> int:
        return len(self.node_ids)

    @property
    def K(self) -> int:
        return len(self.material_names)

    @property
    def A(self) -> int:
        return len(self.vehicles)


class VehicleSchedulingEnv:
    """车辆班次计划环境。"""

    def __init__(self, spec: EnvSpec):
        self.spec = spec
        self.N = spec.N
        self.K = spec.K
        self.A = spec.A
        self.T = spec.T
        self.n_obs = spec.n_obs
        self.node_ids = list(spec.node_ids)
        self.node_id_to_idx = {nid: i for i, nid in enumerate(self.node_ids)}

        self.vehicle_ids = np.array([v.vehicle_id for v in spec.vehicles], dtype=np.int64)
        self.vehicle_types = np.array([v.vehicle_type for v in spec.vehicles], dtype=np.int64)
        self.vehicle_caps = np.array([v.capacity for v in spec.vehicles], dtype=np.int64)
        self.init_vehicle_nodes = np.array([self.node_id_to_idx[v.init_node] for v in spec.vehicles], dtype=np.int64)

        self.adjacency = np.zeros((self.N, self.N), dtype=np.float32)
        for src, dst in spec.edge_list:
            i = self.node_id_to_idx[src]
            j = self.node_id_to_idx[dst]
            self.adjacency[i, j] = 1.0
            self.adjacency[j, i] = 1.0
        self.degrees = self.adjacency.sum(axis=1, keepdims=True).astype(np.float32)

        self.total_demand_sum = float(spec.demand.sum())

        self.t = 0
        self.outstanding = np.zeros((self.N, self.N, self.K), dtype=np.float32)
        self.vehicle_nodes = self.init_vehicle_nodes.copy()
        self.vehicle_prev_states = np.zeros((self.A,), dtype=np.int64)
        self.n_schedule = 0
        self.n_unused_prev = 0
        self.episode_return = 0.0

    @staticmethod
    def from_yaml(path: str) -> "VehicleSchedulingEnv":
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return VehicleSchedulingEnv(build_env_spec(cfg))

    @property
    def action_dim(self) -> int:
        return self.K * (self.N - 1) + 1

    @property
    def small_capacity(self) -> int:
        vals = [v.capacity for v in self.spec.vehicles if v.vehicle_type == 0]
        return int(vals[0]) if vals else 1

    @property
    def large_capacity(self) -> int:
        vals = [v.capacity for v in self.spec.vehicles if v.vehicle_type == 1]
        return int(vals[0]) if vals else 2

    def reset(self) -> Dict[str, np.ndarray]:
        self.t = 0
        self.outstanding = self.spec.demand[0].copy().astype(np.float32)
        self.vehicle_nodes = self.init_vehicle_nodes.copy()
        self.vehicle_prev_states = np.zeros((self.A,), dtype=np.int64)
        self.n_schedule = 0
        self.n_unused_prev = 0
        self.episode_return = 0.0
        return self._build_obs(done=False)

    def step(self, actions: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        actions = np.asarray(actions, dtype=np.int64)
        if actions.shape != (self.A,):
            raise ValueError(f"actions shape must be ({self.A},), got {actions.shape}")

        t = int(self.t)
        tasks_before = self.outstanding.copy()
        vehicle_nodes_before = self.vehicle_nodes.copy()

        tasks_after = self.outstanding.copy()
        new_nodes = self.vehicle_nodes.copy()
        new_states = np.zeros((self.A,), dtype=np.int64)
        log_actions: List[Dict[str, Any]] = []

        order = np.argsort(-self.vehicle_ids)
        for idx in order:
            src_idx = int(vehicle_nodes_before[idx])
            src_node_id = self.node_ids[src_idx]
            action = int(actions[idx])
            stay_idx = self.action_dim - 1

            if action == stay_idx:
                new_nodes[idx] = src_idx
                new_states[idx] = STATE_STAY
                log_actions.append(
                    {
                        "A_id": int(self.vehicle_ids[idx]),
                        "A_i_before": int(src_node_id),
                        "A_i_after": int(src_node_id),
                        "A_state": int(new_states[idx]),
                        "action_index": action,
                        "action": "stay",
                    }
                )
                continue

            dst_idx, mat_idx = self.decode_action(src_idx, action)
            dst_node_id = self.node_ids[dst_idx]
            remaining = float(tasks_after[src_idx, dst_idx, mat_idx])
            load = min(float(self.vehicle_caps[idx]), max(remaining, 0.0))
            tasks_after[src_idx, dst_idx, mat_idx] = max(remaining - float(self.vehicle_caps[idx]), 0.0)
            new_nodes[idx] = dst_idx
            new_states[idx] = STATE_TRANSPORT if load > 0 else STATE_MOVE_EMPTY
            self.n_schedule += 1
            log_actions.append(
                {
                    "A_id": int(self.vehicle_ids[idx]),
                    "A_i_before": int(src_node_id),
                    "A_i_after": int(dst_node_id),
                    "A_state": int(new_states[idx]),
                    "action_index": action,
                    "action": {
                        "dst_node": int(dst_node_id),
                        "material": self.spec.material_names[mat_idx],
                        "transported": round(float(load), 3),
                    },
                }
            )

        log_actions.sort(key=lambda x: x["A_id"])

        self.outstanding = tasks_after
        self.vehicle_nodes = new_nodes
        self.vehicle_prev_states = new_states

        fi_curr = self.compute_fi_satisfy(self.outstanding, self.vehicle_nodes)
        n_unused_curr = self.compute_n_unused(fi_curr, self.vehicle_nodes, self.vehicle_prev_states)

        r_task = 0.1 * float(tasks_before.sum() - tasks_after.sum())
        r_unused_penalty = -0.2 * float(n_unused_curr)
        reward = r_task + r_unused_penalty

        done = (t == self.T - 1)
        re_detail = None
        terminate_reason = None
        if done:
            remaining = float(tasks_after.sum())
            if remaining <= 1e-9:
                re_detail = {
                    "R_base": 8.0,
                    "R_schedule_penalty": round(-0.05 * self.n_schedule, 3),
                }
                terminate_reason = "completed"
            else:
                re_detail = {
                    "R_base": -4.0,
                    "R_schedule_penalty": round(-0.05 * self.n_schedule, 3),
                    "R_task_not_completed_penalty": round(
                        -min(5.0, 20.0 * remaining / max(self.total_demand_sum, 1.0)),
                        3,
                    ),
                }
                terminate_reason = "uncompleted"
            reward += float(sum(re_detail.values()))

        self.episode_return += reward

        info: Dict[str, Any] = {
            "t": t,
            "Jij_k_before": self._tasks_to_nested_dict(tasks_before),
            "Jij_k_after": self._tasks_to_nested_dict(tasks_after),
            "vehicles": log_actions,
            "R_task": round(r_task, 3),
            "R_unused_penalty": round(r_unused_penalty, 3),
            "R_step": round(r_task + r_unused_penalty, 3),
        }
        if done:
            info["Re"] = re_detail
            info["terminate_reason"] = terminate_reason
            info["episode_return"] = round(float(self.episode_return), 3)

        self.n_unused_prev = n_unused_curr
        if not done:
            self.t += 1
            self.outstanding = self.outstanding + self.spec.demand[self.t]
            next_obs = self._build_obs(done=False)
        else:
            next_obs = self._build_obs(done=True)
        return next_obs, float(reward), done, info

    def _build_obs(self, done: bool = False) -> Dict[str, np.ndarray]:
        task_window = np.zeros((self.N, self.N, self.K, self.n_obs), dtype=np.float32)
        for offset in range(self.n_obs):
            future_t = self.t + offset
            if offset == 0:
                task_window[..., 0] = self.outstanding
            elif future_t < self.T:
                task_window[..., offset] = self.spec.demand[future_t]

        obs = {
            "adjacency": self.adjacency.astype(np.float32),
            "node_task_window": task_window,
            "node_vehicle_counts": self.get_vehicle_counts(self.vehicle_nodes).astype(np.float32),
            "node_degrees": self.degrees.astype(np.float32),
            "vehicle_node": self.vehicle_nodes.astype(np.int64),
            "vehicle_type": self.vehicle_types.astype(np.int64),
            "vehicle_cap": self.vehicle_caps.astype(np.float32),
            "vehicle_id": self.vehicle_ids.astype(np.int64),
            "vehicle_prev_state": self.vehicle_prev_states.astype(np.int64),
            "fi_satisfy": self.compute_fi_satisfy(self.outstanding, self.vehicle_nodes).astype(np.float32),
            "n_unused_prev": np.array([self.n_unused_prev], dtype=np.float32),
            "timestep": np.array([self.t / max(self.T, 1)], dtype=np.float32) if not done else np.array([1.0], dtype=np.float32),
        }
        return obs

    def get_vehicle_counts(self, vehicle_nodes: np.ndarray) -> np.ndarray:
        counts = np.zeros((self.N, 2), dtype=np.float32)
        for a, node_idx in enumerate(vehicle_nodes):
            counts[int(node_idx), int(self.vehicle_types[a])] += 1.0
        return counts

    def compute_fi_satisfy(self, outstanding: np.ndarray, vehicle_nodes: np.ndarray) -> np.ndarray:
        counts = self.get_vehicle_counts(vehicle_nodes)
        cap_sum = counts[:, 0] * self.small_capacity + counts[:, 1] * self.large_capacity
        task_sum = outstanding.sum(axis=(1, 2))
        return (task_sum <= cap_sum + 1e-9).astype(np.float32)

    def compute_n_unused(self, fi_satisfy: np.ndarray, vehicle_nodes: np.ndarray, vehicle_states: np.ndarray) -> int:
        total = 0
        for idx in range(self.A):
            node_idx = int(vehicle_nodes[idx])
            state = int(vehicle_states[idx])
            if fi_satisfy[node_idx] <= 0.5 and state in (STATE_STAY, STATE_MOVE_EMPTY):
                total += 1
        return total

    def get_dest_list(self, src_idx: int) -> List[int]:
        return [j for j in range(self.N) if j != src_idx]

    def decode_action(self, src_idx: int, action_idx: int) -> Tuple[int, int]:
        if action_idx < 0 or action_idx >= self.action_dim - 1:
            raise ValueError(f"invalid action index={action_idx}")
        dest_list = self.get_dest_list(src_idx)
        dest_pos = action_idx // self.K
        mat_idx = action_idx % self.K
        return int(dest_list[dest_pos]), int(mat_idx)

    def _tasks_to_nested_dict(self, tasks: np.ndarray) -> Dict[str, Dict[str, List[float]]]:
        out: Dict[str, Dict[str, List[float]]] = {}
        for src_idx, src_node_id in enumerate(self.node_ids):
            item: Dict[str, List[float]] = {}
            for dst_idx, dst_node_id in enumerate(self.node_ids):
                if src_idx == dst_idx:
                    continue
                vals = tasks[src_idx, dst_idx]
                if vals.sum() <= 1e-9:
                    continue
                item[str(dst_node_id)] = [round(float(x), 3) for x in vals.tolist()]
            out[str(src_node_id)] = item
        return out


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_env_spec(cfg: Dict[str, Any]) -> EnvSpec:
    ecfg = cfg["env"]
    T = int(ecfg["T"])
    n_obs = int(ecfg["n_obs"])

    materials = ecfg["materials"]
    if isinstance(materials[0], dict):
        material_names = [m["name"] for m in materials]
        material_ids = [int(m.get("material_id", idx)) for idx, m in enumerate(materials)]
    else:
        material_names = list(materials)
        material_ids = list(range(len(material_names)))

    node_entries = ecfg["nodes"]
    node_ids = [int(x["node_id"]) for x in node_entries]
    node_id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    N = len(node_ids)
    K = len(material_names)

    demand = np.zeros((T, N, N, K), dtype=np.float32)

    vehicles_cfg = ecfg["vehicles"]
    small_ids = [int(x) for x in vehicles_cfg["A_small"]["A_s_id"]]
    large_ids = [int(x) for x in vehicles_cfg["A_large"]["A_l_id"]]
    small_cap = int(vehicles_cfg["A_small"]["capacity"])
    large_cap = int(vehicles_cfg["A_large"]["capacity"])

    init_pos: Dict[int, int] = {}

    for node in node_entries:
        src_id = int(node["node_id"])
        src_idx = node_id_to_idx[src_id]
        for vid in node.get("A_s_id", []):
            init_pos[int(vid)] = src_id
        for vid in node.get("A_l_id", []):
            init_pos[int(vid)] = src_id

        Jij_list = node.get("Jij_list", {})
        for mat_name, entries in Jij_list.items():
            if mat_name not in material_names:
                raise ValueError(f"unknown material {mat_name}")
            k = material_names.index(mat_name)
            for entry in entries:
                if len(entry) != 1:
                    raise ValueError(f"invalid entry: {entry}")
                dst_str, arr = next(iter(entry.items()))
                dst_id = int(dst_str)
                dst_idx = node_id_to_idx[dst_id]
                arr = list(arr)
                if len(arr) != T:
                    raise ValueError(f"task length must equal T={T}, got {len(arr)}")
                demand[:, src_idx, dst_idx, k] = np.asarray(arr, dtype=np.float32)

    vehicles: List[VehicleSpec] = []
    for vid in sorted(small_ids):
        if vid not in init_pos:
            raise ValueError(f"small vehicle {vid} not assigned to any node")
        vehicles.append(VehicleSpec(vehicle_id=vid, vehicle_type=0, capacity=small_cap, init_node=init_pos[vid]))
    for vid in sorted(large_ids):
        if vid not in init_pos:
            raise ValueError(f"large vehicle {vid} not assigned to any node")
        vehicles.append(VehicleSpec(vehicle_id=vid, vehicle_type=1, capacity=large_cap, init_node=init_pos[vid]))
    vehicles.sort(key=lambda x: x.vehicle_id)

    edge_list: List[Tuple[int, int]] = []
    for edge in ecfg["edges"]:
        edge_list.append((int(edge["src"]), int(edge["dst"])))

    return EnvSpec(
        name=str(ecfg.get("name", "Vehicle_scheduling")),
        T=T,
        n_obs=n_obs,
        node_ids=node_ids,
        edge_list=edge_list,
        material_names=material_names,
        material_ids=material_ids,
        vehicles=vehicles,
        demand=demand,
    )


def dump_step_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    def _round(x: Any) -> Any:
        if isinstance(x, float):
            return round(x, 3)
        if isinstance(x, dict):
            return {k: _round(v) for k, v in x.items()}
        if isinstance(x, list):
            return [_round(v) for v in x]
        return x

    with open(path, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(_round(rec), ensure_ascii=False) + "\n")