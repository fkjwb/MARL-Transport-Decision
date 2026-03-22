from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import yaml


# ==================================
# 0) 一些工具
# ==================================
ComboId = int
EdgeId = int
NodeId = int


def _floor_int(x):
    """Floor to int.

    - scalar -> int
    - ndarray -> ndarray[int] (elementwise)
    """
    arr = np.asarray(x)
    if arr.shape == ():
        xv = float(arr)
        if xv <= 0:
            return 0
        return int(math.floor(xv + 1e-9))
    out = np.floor(arr.astype(np.float32) + 1e-9).astype(np.int32)
    out[out < 0] = 0
    return out


def allowed_mask_for_edge_mode(Pi_row: Optional[List[float]], K: int, transport_mode: str) -> np.ndarray:
    """根据 Pi(i,k,t) 允许运输的物料做 mask。

    约定（对每个节点 i、物料 k、时刻 t）：
      - Pi_k(t) = 0   : 只能通过车辆运输（belt 禁止）
      - Pi_k(t) = 1   : 只能通过皮带运输（truck 禁止）
      - Pi_k(t) = 0.5 : 车辆/皮带均可运输（默认）
    """
    if Pi_row is None:
        # 未配置 Pi -> 默认两种方式都允许
        return np.ones((K,), dtype=np.float32)

    mode = str(transport_mode).lower()
    if mode not in ("truck", "belt"):
        raise ValueError(f"Unknown transport_mode={transport_mode!r}, expected 'truck' or 'belt'.")

    eps = 1e-9
    mask = np.zeros((K,), dtype=np.float32)
    for k in range(K):
        v = float(Pi_row[k])
        if mode == "truck":
            # Pi==1 -> 只能皮带 -> truck 禁止；其他(0/0.5/缺省)允许
            mask[k] = 0.0 if abs(v - 1.0) <= eps else 1.0
        else:  # belt
            # Pi==0 -> 只能车辆 -> belt 禁止；其他(0.5/1/缺省)允许
            mask[k] = 0.0 if abs(v - 0.0) <= eps else 1.0
    return mask


# ==================================
# 1) spec
# ==================================
@dataclass
class MaterialSpec:
    Uk: float
    Lk: float
    name: str = ""


@dataclass
class NodeSpec:
    # --- static capacities ---
    Ui: float
    Ui_s: float
    Li_s: float

    # --- metadata (for logging / filtering) ---
    level: int = 0
    name: str = ""
    Li: float = 0.0

    # --- truck unloading capability at this node (yaml: Mj_truck_unload) ---
    Mj_truck_unload: float = 0.0

    # --- initial inventory ---
    W0: List[int] = None  # type: ignore

    # --- exogenous demand (yaml: Y_exo) and transport-mode availability Pi ---
    # y[k][t] : exogenous demand of material k at time t
    y: Optional[List[List[int]]] = None
    # Pi[k][t] : availability of transporting material k at time t (per-mode mask)
    Pi: Optional[List[List[float]]] = None


@dataclass
class EdgeSpec:
    src: NodeId
    dst: NodeId
    mode: str


@dataclass
class TruckEdgeSpec:
    edge_id: EdgeId
    capacity: float


@dataclass
class BeltComboSpec:
    edge_id: EdgeId
    capacity: float
    sub_belts: List[int]


@dataclass
class EnvSpec:
    nodes: Dict[NodeId, NodeSpec]
    edges: Dict[EdgeId, EdgeSpec]

    # truck edges are a subset of edges, with (per-edge) capacity
    truck_edges: List[TruckEdgeSpec]

    # belt combos are a set of "virtual belts" (combination of sub-belts) used by belt actor
    belt_combos: Dict[ComboId, BeltComboSpec]

    # internal combo index -> raw yaml combo_id
    belt_combo_raw_ids: List[str]

    # internal edge index -> raw yaml edge_id
    edge_raw_ids: List[str]

    materials: List[MaterialSpec]
    T: int

    # belt decision order:
    # - manual: follow belt_decision_seq if provided (must cover all combos), else sorted combo ids
    # - model : order can be provided by action 'belt_order'
    belt_order_mode: str = "manual"
    belt_decision_seq: Optional[List[ComboId]] = None

    # observable horizon length n (includes current step)
    n_obs: int = 1

    def K(self) -> int:
        return len(self.materials)

    def N(self) -> int:
        return len(self.nodes)

    def Et(self) -> int:
        return len(self.truck_edges)

    def B(self) -> int:
        return len(self.belt_combos)

    def belt_combo_order_default(self) -> List[ComboId]:
        if self.belt_decision_seq is not None:
            seq = list(self.belt_decision_seq)
            if sorted(seq) == list(sorted(self.belt_combos.keys())):
                return seq
        return list(sorted(self.belt_combos.keys()))



# ==================================
# 2) 动作处理
# ==================================
@dataclass
class TruckDecisionResult:
    edge_amounts: Dict[EdgeId, np.ndarray]


@dataclass
class BeltDecisionResult:
    combo_amounts: Dict[ComboId, np.ndarray]
    ignored_due_to_overlap: List[ComboId]
    selected_material: Dict[ComboId, Optional[int]]


def process_truck_actions(
    spec: EnvSpec,
    truck_edges: List[EdgeId],
    t: int,
    action_truck: np.ndarray,
) -> TruckDecisionResult:
    """卡车动作处理（增加两层 Mj 缩放）。"""
    K = spec.K()
    Et = len(truck_edges)
    a = np.clip(action_truck.reshape(Et, K), 0.0, 1.0).astype(np.float32)

    # 1) 先做 Pi mask
    masked = np.zeros_like(a, dtype=np.float32)
    for eidx, eid in enumerate(truck_edges):
        edge = spec.edges[eid]
        src = spec.nodes[edge.src]

        Pi_row = None
        if src.Pi is not None:
            Pi_row = [
                float(src.Pi[k][t]) if (k < len(src.Pi) and t < len(src.Pi[k])) else 0.5
                for k in range(K)
            ]
        mask = allowed_mask_for_edge_mode(Pi_row, K, transport_mode="truck")
        masked[eidx, :] = a[eidx, :] * mask.astype(np.float32)

    # 2) 每条边先乘该边目标节点的 Mj，得到初步运输量
    prelim = np.zeros_like(masked, dtype=np.float32)
    dst_to_eidxs: Dict[NodeId, List[int]] = {}

    for eidx, eid in enumerate(truck_edges):
        edge = spec.edges[eid]
        dst = edge.dst
        Mj = float(getattr(spec.nodes[dst], "Mj_truck_unload", 0.0) or 0.0)
        if Mj <= 0.0:
            Mj = float(spec.truck_edges[eidx].capacity)

        amt = masked[eidx, :] * Mj  # 初步 Xij_k(t)

        # 2.1) 若该边各物料总和超过 Mj，则按各物料占比缩放
        total_edge = float(amt.sum())
        if total_edge > Mj and total_edge > 1e-8:
            amt = amt * (Mj / total_edge)

        prelim[eidx, :] = amt
        dst_to_eidxs.setdefault(dst, []).append(eidx)

    # 3) 对同一目标节点，再做一次总卸车能力缩放
    for dst, eidxs in dst_to_eidxs.items():
        Mj = float(getattr(spec.nodes[dst], "Mj_truck_unload", 0.0) or 0.0)
        if Mj <= 0.0:
            # 与现有逻辑保持一致：若节点未配，则回退到这些边的 capacity（通常相同）
            Mj = max(float(spec.truck_edges[eidx].capacity) for eidx in eidxs)

        total_dst = float(sum(prelim[eidx, :].sum() for eidx in eidxs))
        if total_dst > Mj and total_dst > 1e-8:
            scale = Mj / total_dst
            for eidx in eidxs:
                prelim[eidx, :] *= scale

    # 4) floor -> int
    edge_amounts: Dict[EdgeId, np.ndarray] = {}
    for eidx, eid in enumerate(truck_edges):
        edge_amounts[eid] = _floor_int(prelim[eidx, :]).astype(np.int32)

    return TruckDecisionResult(edge_amounts=edge_amounts)


def process_belt_actions(
    spec: EnvSpec,
    belt_combos: List[ComboId],
    combo_overlap: Dict[ComboId, List[ComboId]],
    t: int,
    action_belt: np.ndarray,
    belt_order: List[ComboId],
) -> BeltDecisionResult:
    """皮带动作处理（离散 one-hot: K+1，最后一维=不运输）。"""
    K = spec.K()
    B = len(belt_combos)

    a = np.asarray(action_belt, dtype=np.float32).reshape(B, K + 1)

    # 先做 one-hot 规整化（允许策略输出为 soft one-hot，env 内取 argmax）
    chosen_idx = np.argmax(a, axis=-1).astype(np.int32)  # [B], 0..K (K=不运输)

    # 物料可运输掩码（Pi 约束）仅作用于前 K 维；若选了不可运输物料 -> 置为不运输
    for cidx, cid in enumerate(belt_combos):
        if int(chosen_idx[cidx]) >= K:
            continue

        combo = spec.belt_combos[cid]
        edge = spec.edges[combo.edge_id]
        src = spec.nodes[edge.src]

        Pi_row = None
        if src.Pi is not None:
            Pi_row = [float(src.Pi[k][t]) if (k < len(src.Pi) and t < len(src.Pi[k])) else 0.5 for k in range(K)]
        mask = allowed_mask_for_edge_mode(Pi_row, K, transport_mode="belt")
        if float(mask[int(chosen_idx[cidx])]) <= 0.0:
            chosen_idx[cidx] = K  # 不运输

    # 重叠检测：只对“执行运输”(idx<K) 的组合起作用；后序（belt_order）若重叠则置为不运输
    chosen_transport: List[ComboId] = []
    ignored: List[ComboId] = []
    cid_to_cidx = {cid: i for i, cid in enumerate(belt_combos)}

    for cid in belt_order:
        cidx = cid_to_cidx[cid]
        if int(chosen_idx[cidx]) >= K:
            continue  # 不运输不参与重叠

        has_overlap = any((cid2 in combo_overlap.get(cid, [])) for cid2 in chosen_transport)
        if has_overlap:
            ignored.append(cid)
            chosen_idx[cidx] = K  # 置为不运输
        else:
            chosen_transport.append(cid)

    # one-hot（float）与选择结果
    onehot = np.zeros((B, K + 1), dtype=np.float32)
    selected_mat: Dict[ComboId, Optional[int]] = {}
    for cidx, cid in enumerate(belt_combos):
        idx = int(chosen_idx[cidx])
        onehot[cidx, idx] = 1.0
        selected_mat[cid] = None if idx >= K else idx

    # 运输量：选择的物料按运力上限运输（cap），不运输则为 0
    amounts: Dict[ComboId, np.ndarray] = {}
    for cidx, cid in enumerate(belt_combos):
        cap = float(spec.belt_combos[cid].capacity)
        cap_int = _floor_int(cap)

        amt = np.zeros((K,), dtype=np.int32)
        idx = int(chosen_idx[cidx])
        if idx < K:
            amt[idx] = cap_int
        amounts[cid] = amt

    return BeltDecisionResult(combo_amounts=amounts, ignored_due_to_overlap=ignored, selected_material=selected_mat)


# ==================================
# 3) 奖励
# ==================================
@dataclass
class RewardParts:
    R_base: float = 0.0
    R_demand: float = 0.0
    R_t_star: float = 0.0
    R_level23_penalty: float = 0.0
    R_T_left: float = 0.0
    R_f_penalty: float = 0.0
    R_level1: float = 0.0
    R_demand_not_fulfilled: float = 0.0
    R_over_Ui: float = 0.0
    R_over_Uk: float = 0.0
    R_below_0: float = 0.0

    def __add__(self, other: "RewardParts") -> "RewardParts":
        return RewardParts(
            R_base=float(self.R_base + other.R_base),
            R_demand=float(self.R_demand + other.R_demand),
            R_t_star=float(self.R_t_star + other.R_t_star),
            R_level23_penalty=float(self.R_level23_penalty + other.R_level23_penalty),
            R_T_left=float(self.R_T_left + other.R_T_left),
            R_f_penalty=float(self.R_f_penalty + other.R_f_penalty),
            R_level1=float(self.R_level1 + other.R_level1),
            R_demand_not_fulfilled=float(self.R_demand_not_fulfilled + other.R_demand_not_fulfilled),
            R_over_Ui=float(self.R_over_Ui + other.R_over_Ui),
            R_over_Uk=float(self.R_over_Uk + other.R_over_Uk),
            R_below_0=float(self.R_below_0 + other.R_below_0),
        )

    @property
    def R_step(self) -> float:
        return float(self.R_demand + self.R_t_star + self.R_level23_penalty)

    @property
    def R_ultra_limit(self) -> float:
        raw = float(self.R_over_Ui + self.R_over_Uk + self.R_below_0)
        return float(max(-5.0, min(0.0, raw)))

    @property
    def Re(self) -> float:
        return float(
            self.R_base
            + self.R_T_left
            + self.R_f_penalty
            + self.R_level1
            + self.R_demand_not_fulfilled
            + self.R_ultra_limit
        )

    @property
    def total(self) -> float:
        return float(self.R_step + self.Re)


def compute_reward(
    spec: EnvSpec,
    Wi_before: np.ndarray,
    Wi_after: np.ndarray,
    Yi_before: np.ndarray,
    Yi_after: np.ndarray,
    f_before: int,
    f_after: int,
    Y_exo_total: float,
) -> RewardParts:
    """每步奖励（非终止步）。"""
    del Wi_before, Yi_before, f_before, f_after

    # R_demand 直接反映“当前剩余需求 / 总需求”的完成度
    denom = float(Y_exo_total) if float(Y_exo_total) > 0.0 else 1.0
    R_demand = 1.0 - (float(Yi_after.sum()) / denom)

    node_ids = list(sorted(spec.nodes.keys()))
    lvl23_rows = [i for i, nid in enumerate(node_ids) if int(spec.nodes[nid].level) in (2, 3)]

    if len(lvl23_rows) == 0:
        R_level23_penalty = 0.0
    else:
        pen_sum = 0.0
        violate_cnt = 0

        for r in lvl23_rows:
            nid = node_ids[r]
            node = spec.nodes[nid]
            Ui = float(node.Ui) if float(node.Ui) > 0 else 1.0
            total = float(Wi_after[r, :].sum())

            lo = max(float(node.Li_s) - total, 0.0)
            hi = max(total - float(node.Ui_s), 0.0)

            pen_i = (lo + hi) / Ui
            if pen_i > 0.0:
                pen_sum += pen_i
                violate_cnt += 1

        R_level23_penalty = -3.0 * float(pen_sum / violate_cnt) if violate_cnt > 0 else 0.0

    return RewardParts(
        R_demand=float(R_demand),
        R_level23_penalty=float(R_level23_penalty),
    )


def compute_t_star_reward(spec: EnvSpec, t_star: int) -> RewardParts:
    """首次达到 T* 时立即支付的时间奖励。"""
    del spec, t_star
    rp = RewardParts()
    rp.R_t_star = 1.0
    return rp


def compute_terminal_reward(
    spec: EnvSpec,
    node_ids: List[NodeId],
    lvl1_rows: List[int],
    lvl23_rows: List[int],
    W_after: np.ndarray,
    Y_after: np.ndarray,
    t: int,
    f_switch: int,
    reason: str,
    Y_exo_total: float,
    *,
    T_star: Optional[int] = None,
    neg_inventory: bool = False,
    over_ui_rows: Optional[List[int]] = None,
    lvl23_material_totals: Optional[np.ndarray] = None,
    over_uk_mats: Optional[List[int]] = None,
) -> RewardParts:
    """终止奖励（success / demand_not_fulfill / abnormal）。"""
    rp = RewardParts()

    if reason == "success":
        rp.R_base = 6.0
        t_star = int(T_star) if T_star is not None else int(t)
        remain_steps = max(int(spec.T) - 1 - t_star, 0)     # 因为 t_star 取得是t时刻而非第t步
        rp.R_T_left = 3.0 * float(remain_steps)
        rp.R_f_penalty = -0.5 * float(f_switch)

        if len(lvl1_rows) > 0:
            vals = []
            for r in lvl1_rows:
                nid = node_ids[r]
                Ui = float(spec.nodes[nid].Ui) if float(spec.nodes[nid].Ui) > 0 else 1.0
                total = float(W_after[r, :].sum())
                vals.append(2.0 * (1.0 - (total / Ui)))
            rp.R_level1 = float(sum(vals) / float(len(vals)))

    elif reason == "demand_not_fulfill":
        rp.R_base = 0.0
        rp.R_f_penalty = -0.5 * float(f_switch)

        unmet = float(Y_after.sum())
        denom = float(Y_exo_total) if float(Y_exo_total) > 0 else 1.0
        pen = -20.0 * (unmet / denom)
        if pen < -5.0:
            pen = -5.0
        rp.R_demand_not_fulfilled = float(pen)

    elif reason == "abnormal":
        rp.R_base = -5.0
        rp.R_f_penalty = -0.5 * float(f_switch)

        if over_ui_rows:
            vals = []
            for r in over_ui_rows:
                nid = node_ids[r]
                Ui = float(spec.nodes[nid].Ui) if float(spec.nodes[nid].Ui) > 0 else 1.0
                total = float(W_after[r, :].sum())
                exceed = max(total - Ui, 0.0)
                ratio = exceed / (0.01 * Ui) if Ui > 0 else 0.0
                vals.append(min(ratio, 5.0))
            if len(vals) > 0:
                rp.R_over_Ui = -float(sum(vals) / float(len(vals)))

        if (over_uk_mats is not None) and (lvl23_material_totals is not None):
            vals = []
            for k in over_uk_mats:
                Uk = float(spec.materials[int(k)].Uk) if float(spec.materials[int(k)].Uk) > 0 else 1.0
                total_k = float(lvl23_material_totals[int(k)])
                exceed = max(total_k - Uk, 0.0)
                ratio = exceed / (0.01 * Uk) if Uk > 0 else 0.0
                vals.append(min(ratio, 5.0))
            if len(vals) > 0:
                rp.R_over_Uk = -float(sum(vals) / float(len(vals)))

        if bool(neg_inventory):
            rp.R_below_0 = -5.0

    return rp

# ==================================
# 4) 环境
# ==================================
@dataclass
class StepTrace:
    t: int
    Wi: Any
    Yi_before: Any
    Y_after: Any
    Pi: Any
    f: int
    X: Any
    B: Any
    Decision_seq: Any
    R_demand: float
    R_level23_penalty: float
    R_step: float
    R_t_star: Optional[float] = None
    Re: Optional[Any] = None
    terminate_reason: Optional[str] = None
    terminate_detail: Optional[List[str]] = None
    episode_return: Optional[float] = None


class LogisticsEnv:
    def __init__(self, spec: EnvSpec):
        self.spec = spec
        self.K = spec.K()
        self.N = spec.N()
        self.Et = spec.Et()
        self.B = spec.B()

        self.node_ids: List[NodeId] = list(sorted(spec.nodes.keys()))

        # ---------- reward/terminal 预计算 ----------
        # 行索引与 node_id 对齐（env 内部 Wi/Yi 的行顺序与此一致：sorted(spec.nodes.keys()))
        self._node_id_to_row: Dict[NodeId, int] = {nid: i for i, nid in enumerate(self.node_ids)}
        self._lvl1_rows: List[int] = [i for i, nid in enumerate(self.node_ids) if int(spec.nodes[nid].level) == 1]
        self._lvl23_rows: List[int] = [i for i, nid in enumerate(self.node_ids) if int(spec.nodes[nid].level) in (2, 3)]

        # yaml 中所有节点所有物料的外生需求总和（用于终止时“剩余未满足需求惩罚”归一化）
        y_exo_total = 0.0
        for nid in self.node_ids:
            node = spec.nodes[nid]
            if node.y is None:
                continue
            for k in range(len(node.y)):
                y_exo_total += float(sum(node.y[k]))
        self._Y_exo_total: float = float(y_exo_total) if float(y_exo_total) > 0.0 else 1.0

        self.truck_edges: List[EdgeId] = [te.edge_id for te in spec.truck_edges]
        self.belt_combos: List[ComboId] = list(sorted(spec.belt_combos.keys()))

        # overlap 预计算
        self._combo_overlap: Dict[ComboId, List[ComboId]] = {}
        for i in self.belt_combos:
            self._combo_overlap[i] = []
            for j in self.belt_combos:
                if i == j:
                    continue
                si = set(spec.belt_combos[i].sub_belts)
                sj = set(spec.belt_combos[j].sub_belts)
                if len(si.intersection(sj)) > 0:
                    self._combo_overlap[i].append(j)

        # same dst/src 计数（每个节点上有多少 combo 指向/来自该节点）
        self._belt_dst_count: Dict[NodeId, int] = {}
        self._belt_src_count: Dict[NodeId, int] = {}
        for cid in self.belt_combos:
            combo = spec.belt_combos[cid]
            edge = spec.edges[combo.edge_id]
            self._belt_dst_count[edge.dst] = self._belt_dst_count.get(edge.dst, 0) + 1
            self._belt_src_count[edge.src] = self._belt_src_count.get(edge.src, 0) + 1

        # same dst/src 计数（每个节点上有多少 truck edge 指向/来自该节点）
        self._truck_dst_count: Dict[NodeId, int] = {}
        self._truck_src_count: Dict[NodeId, int] = {}
        self._truck_out_edges_by_src: Dict[NodeId, List[EdgeId]] = {}
        self._truck_in_edges_by_dst: Dict[NodeId, List[EdgeId]] = {}
        for eid in self.truck_edges:
            e = spec.edges[eid]
            self._truck_dst_count[e.dst] = self._truck_dst_count.get(e.dst, 0) + 1
            self._truck_src_count[e.src] = self._truck_src_count.get(e.src, 0) + 1
            self._truck_out_edges_by_src.setdefault(e.src, []).append(eid)
            self._truck_in_edges_by_dst.setdefault(e.dst, []).append(eid)

        self._belt_out_combos_by_src: Dict[NodeId, List[ComboId]] = {}
        self._belt_in_combos_by_dst: Dict[NodeId, List[ComboId]] = {}
        for cid in self.belt_combos:
            combo = spec.belt_combos[cid]
            edge = spec.edges[combo.edge_id]
            self._belt_out_combos_by_src.setdefault(edge.src, []).append(cid)
            self._belt_in_combos_by_dst.setdefault(edge.dst, []).append(cid)

        # ---------- 观测维度 ----------
        self.n_obs = int(getattr(spec, "n_obs", 1))
        n = int(self.n_obs)

        # truck actor: 2K*(1+n)+12
        # = caps8(src/dst: Ui,Li,Ui_s,Li_s) + Wi2K + Y nK + Pi nK + id2 + Mj1 + time_norm1
        self.truck_obs_dim = 2 * self.K * (1 + n) + 12

        # belt actor: K*(2n+3)+15
        # = (last_action_onehot(K+1) + flag_overlap1) + caps8 + Wi2K + Y nK + Pi nK + cap1 + id3 + time_norm1
        self.belt_obs_dim = self.K * (2 * n + 3) + 15

        # centralized critic: 4N + (2 + N + n*N + n*N)*K + 2 + 2*B
        # 说明：
        # - 每个节点静态容量拼接 Ui/Li/Ui_s/Li_s；
        # - y / Pi 窗口按所有 node_ids 逐节点拼接；
        # - t-1 时刻的 belt 决策顺序 Decision_seq（B）与 Flag_overlap（B）。
        self.critic_obs_dim = (
            4 * self.N
            + 2 * self.K
            + (self.N + self.n_obs * self.N + self.n_obs * self.N) * self.K
            + 2
            + 2 * self.B
        )

        # ---------- 状态 ----------
        self._t = 0
        self._Wi: np.ndarray = np.zeros((self.N, self.K), dtype=np.int32)
        self._Yi: np.ndarray = np.zeros((self.N, self.K), dtype=np.int32)
        self._f_switch = 0
        self._t_star: Optional[int] = None

        # 上一时刻 belt one-hot 动作 / overlap 标志 / 决策顺序
        self._last_belt_action = np.zeros((self.B, self.K + 1), dtype=np.float32)  # last raw belt action
        self._last_belt_flag_overlap = np.ones((self.B,), dtype=np.float32)  # 1=normal, 0=ignored due to overlap
        self._last_belt_selected = np.full((self.B,), -1, dtype=np.int32)  # material idx (0..K-1), -1=no-transport
        self._combo_id_to_idx: Dict[ComboId, int] = {cid: i for i, cid in enumerate(self.belt_combos)}
        self._last_decision_seq = np.asarray(
            [self._combo_id_to_idx[cid] for cid in self.spec.belt_combo_order_default()],
            dtype=np.float32,
        )

        self.trace: List[StepTrace] = []
        self._pretrain_checked = False

        # ---- train.py expects these attributes for shape printing ----
        self.K_belt = int(self.K) + 1  # belt action dim = K + 1 (last is "no transport")
        self.reward_scale = 0.25

        # train.py only needs `.shape` (keep numpy placeholders)
        self.observation_space = {
            "truck_actor": np.zeros((self.Et, self.truck_obs_dim), dtype=np.float32),
            "belt_actor": np.zeros((self.B, self.belt_obs_dim), dtype=np.float32),
            "critic": np.zeros((self.critic_obs_dim,), dtype=np.float32),
        }

    def _node_row(self, nid: NodeId) -> int:
        return self._node_id_to_row[nid]

    def _demand_nodes(self) -> List[NodeId]:
        out = []
        for nid in self.node_ids:
            if self.spec.nodes[nid].y is not None:
                out.append(nid)
        return out

    def _pi_nodes(self) -> List[NodeId]:
        out = []
        for nid in self.node_ids:
            if self.spec.nodes[nid].Pi is not None:
                out.append(nid)
        return out

    @property
    def demand_nodes(self) -> List[NodeId]:
        return self._demand_nodes()

    @property
    def pi_nodes(self) -> List[NodeId]:
        return self._pi_nodes()

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            np.random.seed(int(seed))

        self._t = 0
        self._f_switch = 0
        self._t_star = None
        self.trace = []

        Wi0 = []
        for nid in self.node_ids:
            Wi0.append(list(self.spec.nodes[nid].W0))
        self._Wi = np.asarray(Wi0, dtype=np.int32)

        self._Yi = np.zeros((self.N, self.K), dtype=np.int32)

        self._last_belt_action = np.zeros((self.B, self.K + 1), dtype=np.float32)
        self._last_belt_flag_overlap = np.ones((self.B,), dtype=np.float32)
        self._last_belt_selected = np.full((self.B,), -1, dtype=np.int32)
        self._last_decision_seq = np.asarray(
            [self._combo_id_to_idx[cid] for cid in self.spec.belt_combo_order_default()],
            dtype=np.float32,
        )

        obs = self._make_obs()
        return obs, {}
    
    def check_before_training(self) -> None:
        """Sanity-check obs shapes before training."""
        obs, _ = self.reset(seed=0)
        assert obs["truck_actor"].shape == (self.Et, self.truck_obs_dim)
        assert obs["belt_actor"].shape == (self.B, self.belt_obs_dim)
        assert obs["critic"].shape == (self.critic_obs_dim,)
        self._pretrain_checked = True

    def last_trace(self) -> Optional[Dict[str, Any]]:
        if len(self.trace) == 0:
            return None
        tr = self.trace[-1]
        try:
            return dict(vars(tr))
        except Exception:
            return {"trace": str(tr)}

    def _current_pi_matrix(self, t: int) -> np.ndarray:
        # 未配置 Pi 的节点默认 0.5（车辆/皮带均可）
        Pi = np.full((self.N, self.K), 0.5, dtype=np.float32)
        for r, nid in enumerate(self.node_ids):
            node = self.spec.nodes[nid]
            if node.Pi is None:
                continue
            for k in range(self.K):
                if k < len(node.Pi) and t < len(node.Pi[k]):
                    Pi[r, k] = float(node.Pi[k][t])
        return Pi

    def _future_exogenous_demand_all_zero(self, t: int) -> bool:
        """检查未来时刻 (t+1..T-1) 的全部 Y_exo 是否均为 0。"""
        for nid in self.node_ids:
            node = self.spec.nodes[nid]
            if node.y is None:
                continue
            for k in range(self.K):
                if k >= len(node.y):
                    continue
                for tt in range(int(t) + 1, int(self.spec.T)):
                    if tt < len(node.y[k]) and float(node.y[k][tt]) != 0.0:
                        return False
        return True

    def _future_yi_all_zero_after_transition(self, t: int, Y_after: np.ndarray) -> bool:
        """检查在当前 t 完成状态转移后，未来所有时刻 Yi 是否都会为 0。"""
        return int(np.asarray(Y_after).sum()) == 0 and self._future_exogenous_demand_all_zero(t)

    # ---------- windows ----------
    def _y_window_flat(self, nid: NodeId, t: int, n: int) -> List[float]:
        """y(i,k,t_n) 窗口 flatten 到 n*K。

        约定（与你描述一致）：
        - dt=0：Yi(t) = Yi(t-1) + Yi[t]，其中 Yi(t-1) 为上一步未满足需求（self._Yi），Yi[t] 为 config 中的外生需求。
        - dt>0：Yi(t+dt) 仅取 config 中该时刻的外生需求（不叠加未满足需求）。
        """
        node = self.spec.nodes[nid]
        K = self.K
        out: List[float] = []
        if node.y is None:
            return [0.0] * (n * K)

        row = self._node_row(nid)

        for dt in range(n):
            tt = min(max(t + dt, 0), self.spec.T - 1)
            for k in range(K):
                exo = 0.0
                if k < len(node.y) and tt < len(node.y[k]):
                    exo = float(node.y[k][tt])
                if dt == 0:
                    out.append(float(self._Yi[row, k]) + exo)
                else:
                    out.append(exo)
        return out

    def _y_window_flat_carried(self, nid: NodeId, t: int, n: int) -> List[float]:
        """Demand window flattened to n*K with current unmet demand carried forward."""
        node = self.spec.nodes[nid]
        K = self.K
        out: List[float] = []
        if node.y is None:
            return [0.0] * (n * K)

        row = self._node_row(nid)
        carry_now = np.asarray(self._Yi[row, :], dtype=np.float32).copy()
        for k in range(K):
            if k < len(node.y) and t < len(node.y[k]):
                carry_now[k] += float(node.y[k][t])

        future_cum = np.zeros((K,), dtype=np.float32)
        for dt in range(n):
            tt = min(max(t + dt, 0), self.spec.T - 1)
            for k in range(K):
                if dt == 0:
                    out.append(float(carry_now[k]))
                else:
                    exo = 0.0
                    if k < len(node.y) and tt < len(node.y[k]):
                        exo = float(node.y[k][tt])
                    future_cum[k] += exo
                    out.append(float(carry_now[k] + future_cum[k]))
        return out

    def _pi_window_flat(self, nid: NodeId, t: int, n: int) -> List[float]:
        """Pi(i,k,t) 窗口 (t..t+n-1) flatten 到 n*K。"""
        node = self.spec.nodes[nid]
        K = self.K
        out: List[float] = []
        for dt in range(n):
            tt = min(max(t + dt, 0), self.spec.T - 1)
            for k in range(K):
                if node.Pi is None:
                    out.append(0.5)
                else:
                    val = 0.5
                    if k < len(node.Pi) and tt < len(node.Pi[k]):
                        val = float(node.Pi[k][tt])
                    out.append(val)
        return out

    # ---------- 构造 obs ----------
    def _make_obs(self) -> Dict[str, np.ndarray]:
        spec = self.spec
        t = int(self._t)
        K = self.K
        n = self.n_obs

        # truck actor obs
        truck_obs = np.zeros((self.Et, self.truck_obs_dim), dtype=np.float32)
        for eidx, eid in enumerate(self.truck_edges):
            edge = spec.edges[eid]
            src = spec.nodes[edge.src]
            dst = spec.nodes[edge.dst]
            srow = self._node_row(edge.src)
            drow = self._node_row(edge.dst)

            feat = [
                float(src.Ui), float(src.Li), float(src.Ui_s), float(src.Li_s),
                float(dst.Ui), float(dst.Li), float(dst.Ui_s), float(dst.Li_s),
            ]
            feat.extend(self._Wi[srow, :].astype(np.float32).tolist())
            feat.extend(self._Wi[drow, :].astype(np.float32).tolist())
            feat.extend(self._y_window_flat_carried(edge.dst, t, n))      # n*K
            feat.extend(self._pi_window_flat(edge.src, t, n))     # n*K

            trunk_same_dst = int(self._truck_dst_count.get(edge.dst, 0) - 1)
            trunk_same_src = int(self._truck_src_count.get(edge.src, 0) - 1)

            feat.append(float(max(trunk_same_dst, 0)))
            feat.append(float(max(trunk_same_src, 0)))
            feat.append(float(getattr(dst, "Mj_truck_unload", 0.0)))
            feat.append(float(self._t) / float(max(spec.T, 1)))
            truck_obs[eidx, :] = np.asarray(feat, dtype=np.float32)

        # belt actor obs
        belt_obs = np.zeros((self.B, self.belt_obs_dim), dtype=np.float32)
        for cidx, cid in enumerate(self.belt_combos):
            combo = spec.belt_combos[cid]
            e = spec.edges[combo.edge_id]
            src = spec.nodes[e.src]
            dst = spec.nodes[e.dst]
            srow = self._node_row(e.src)
            drow = self._node_row(e.dst)

            feat = []

            # 上一时刻原始 belt 动作输出(K+1)
            feat.extend(self._last_belt_action[cidx, :].astype(np.float32).tolist())
            # 上一时刻是否被 overlap 置零：1=正常，0=置零
            feat.append(float(self._last_belt_flag_overlap[cidx]))

            feat.extend([
                float(src.Ui), float(src.Li), float(src.Ui_s), float(src.Li_s),
                float(dst.Ui), float(dst.Li), float(dst.Ui_s), float(dst.Li_s),
            ])
            feat.extend(self._Wi[srow, :].astype(np.float32).tolist())
            feat.extend(self._Wi[drow, :].astype(np.float32).tolist())
            feat.extend(self._y_window_flat_carried(e.dst, t, self.n_obs))      # n*K
            feat.extend(self._pi_window_flat(e.src, t, self.n_obs))     # n*K

            feat.append(float(combo.capacity))

            same_dst = int(self._belt_dst_count.get(e.dst, 0) - 1)
            same_src = int(self._belt_src_count.get(e.src, 0) - 1)
            overlap_cnt = int(len(self._combo_overlap.get(cid, [])))   # 不包括自身（构建时已跳过 i==j）

            feat.append(float(max(same_dst, 0)))
            feat.append(float(max(same_src, 0)))
            feat.append(float(max(overlap_cnt, 0)))
            feat.append(float(self._t) / float(max(spec.T, 1)))

            belt_obs[cidx, :] = np.asarray(feat, dtype=np.float32)

        # critic
        critic_feat: List[float] = []

        for nid in self.node_ids:
            nnode = spec.nodes[nid]
            critic_feat.extend([float(nnode.Ui), float(nnode.Li), float(nnode.Ui_s), float(nnode.Li_s)])

        for m in spec.materials:
            critic_feat.extend([float(m.Uk), float(m.Lk)])

        critic_feat.extend(self._Wi.astype(np.float32).reshape(-1).tolist())

        for nid in self.node_ids:      # N
            critic_feat.extend(self._y_window_flat_carried(nid, t, n))   # 没有 y 自动全 0

        for nid in self.node_ids:      # N
            critic_feat.extend(self._pi_window_flat(nid, t, n))  # 没有 Pi 自动全 0     # n*K

        critic_feat.append(float(self._f_switch))
        critic_feat.extend(self._last_decision_seq.astype(np.float32).tolist())
        critic_feat.extend(self._last_belt_flag_overlap.astype(np.float32).tolist())
        critic_feat.append(float(self._t) / float(max(spec.T, 1)))

        critic = np.asarray(critic_feat, dtype=np.float32)
        return {"truck_actor": truck_obs, "belt_actor": belt_obs, "critic": critic}

    def _window_matrix_y(self, nid: NodeId, t: int, n: int) -> List[List[float]]:
        out: List[List[float]] = []
        flat = self._y_window_flat_carried(nid, t, n)
        for dt in range(n):
            st = dt * self.K
            out.append([float(v) for v in flat[st: st + self.K]])
        return out

    def _window_matrix_pi(self, nid: NodeId, t: int, n: int) -> List[List[float]]:
        out: List[List[float]] = []
        flat = self._pi_window_flat(nid, t, n)
        for dt in range(n):
            st = dt * self.K
            out.append([float(v) for v in flat[st: st + self.K]])
        return out

    def get_debug_snapshot(self) -> Dict[str, Any]:
        """返回当前时刻（step 前）的可读 debug 状态快照。

        用于训练时打印“每个 iter 的第一个完整回合”调试信息。
        输出是结构化 dict，train.py 中会压成单行 JSON 打印。
        """
        spec = self.spec
        t = int(self._t)
        n = int(self.n_obs)

        truck_rows: List[Dict[str, Any]] = []
        for eidx, eid in enumerate(self.truck_edges):
            edge = spec.edges[eid]
            src = spec.nodes[edge.src]
            dst = spec.nodes[edge.dst]
            srow = self._node_row(edge.src)
            drow = self._node_row(edge.dst)
            truck_rows.append({
                'edge_idx': int(eidx),
                'edge_id': str(self.spec.edge_raw_ids[int(eid)]),
                'src': int(edge.src),
                'dst': int(edge.dst),
                'src_caps': {'Ui': float(src.Ui), 'Li': float(src.Li), 'Ui_s': float(src.Ui_s), 'Li_s': float(src.Li_s)},
                'dst_caps': {'Ui': float(dst.Ui), 'Li': float(dst.Li), 'Ui_s': float(dst.Ui_s), 'Li_s': float(dst.Li_s)},
                'Wi_t': {
                    str(edge.src): self._Wi[srow, :].astype(np.int32).tolist(),
                    str(edge.dst): self._Wi[drow, :].astype(np.int32).tolist(),
                },
                'Yi_future_dst': self._window_matrix_y(edge.dst, t, n),
                'Pi_future_src': self._window_matrix_pi(edge.src, t, n),
                'Mj': float(getattr(dst, 'Mj_truck_unload', 0.0) or 0.0),
                'trunk_same_dst': int(max(self._truck_dst_count.get(edge.dst, 0) - 1, 0)),
                'trunk_same_src': int(max(self._truck_src_count.get(edge.src, 0) - 1, 0)),
            })

        belt_rows: List[Dict[str, Any]] = []
        for cidx, cid in enumerate(self.belt_combos):
            combo = spec.belt_combos[cid]
            edge = spec.edges[combo.edge_id]
            src = spec.nodes[edge.src]
            dst = spec.nodes[edge.dst]
            srow = self._node_row(edge.src)
            drow = self._node_row(edge.dst)
            belt_rows.append({
                'combo_idx': int(cidx),
                'combo_id': str(self.spec.belt_combo_raw_ids[int(cid)]),
                'edge_id': str(self.spec.edge_raw_ids[int(combo.edge_id)]),
                'src': int(edge.src),
                'dst': int(edge.dst),
                'src_caps': {'Ui': float(src.Ui), 'Li': float(src.Li), 'Ui_s': float(src.Ui_s), 'Li_s': float(src.Li_s)},
                'dst_caps': {'Ui': float(dst.Ui), 'Li': float(dst.Li), 'Ui_s': float(dst.Ui_s), 'Li_s': float(dst.Li_s)},
                'Wi_t': {
                    str(edge.src): self._Wi[srow, :].astype(np.int32).tolist(),
                    str(edge.dst): self._Wi[drow, :].astype(np.int32).tolist(),
                },
                'Yi_future_dst': self._window_matrix_y(edge.dst, t, n),
                'Pi_future_src': self._window_matrix_pi(edge.src, t, n),
                'Mij_b': float(combo.capacity),
                'Bij_t-1(raw act)': self._last_belt_action[cidx, :].astype(np.float32).tolist(),
                'Flag_overlap': float(self._last_belt_flag_overlap[cidx]),
                'combos_same_dst': int(max(self._belt_dst_count.get(edge.dst, 0) - 1, 0)),
                'combos_same_src': int(max(self._belt_src_count.get(edge.src, 0) - 1, 0)),
                'overlap_cnt': int(len(self._combo_overlap.get(cid, []))),
            })

        node_caps: Dict[str, Dict[str, float]] = {}
        node_wi: Dict[str, List[int]] = {}
        node_yi: Dict[str, List[List[float]]] = {}
        node_pi: Dict[str, List[List[float]]] = {}
        for r, nid in enumerate(self.node_ids):
            node = spec.nodes[nid]
            node_caps[str(nid)] = {
                'Ui': float(node.Ui),
                'Li': float(node.Li),
                'Ui_s': float(node.Ui_s),
                'Li_s': float(node.Li_s),
            }
            node_wi[str(nid)] = self._Wi[r, :].astype(np.int32).tolist()
            node_yi[str(nid)] = self._window_matrix_y(nid, t, n)
            node_pi[str(nid)] = self._window_matrix_pi(nid, t, n)

        critic_state = {
            'node_caps': node_caps,
            'material_caps': {
                str(k): {'Uk': float(m.Uk), 'Lk': float(m.Lk)}
                for k, m in enumerate(spec.materials)
            },
            'Wi_t': node_wi,
            'Yi_future': node_yi,
            'Pi_future': node_pi,
            'f_t': int(self._f_switch),
            'Decision_seq_t-1': self._last_decision_seq.astype(np.float32).tolist(),
            'Flag_overlap_t-1': self._last_belt_flag_overlap.astype(np.float32).tolist(),
            'time_norm': float(self._t) / float(max(spec.T, 1)),
        }

        return {
            't': int(t),
            'truck_actor': truck_rows,
            'belt_actor': belt_rows,
            'critic': critic_state,
        }

    # ---------- belt order ----------
    def _belt_order_from_action(self, belt_order_action: np.ndarray) -> List[ComboId]:
        if self.spec.belt_order_mode == "manual":
            return self.spec.belt_combo_order_default()

        x = np.asarray(belt_order_action)
        if x.dtype.kind in ("i", "u"):
            idx = x.astype(np.int64).tolist()
            if sorted(idx) == list(range(self.B)):
                return [self.belt_combos[i] for i in idx]
            return self.spec.belt_combo_order_default()

        # fallback：按值排序
        x = x.reshape(-1)
        if x.shape[0] != self.B:
            return self.spec.belt_combo_order_default()
        order_idx = np.argsort(-x).tolist()
        return [self.belt_combos[i] for i in order_idx]

    def _project_executed_flows(
        self,
        truck_amounts: Dict[EdgeId, np.ndarray],
        belt_amounts: Dict[ComboId, np.ndarray],
        Wi_before: np.ndarray,
    ) -> Tuple[Dict[EdgeId, np.ndarray], Dict[ComboId, np.ndarray]]:
        spec = self.spec
        truck_proj: Dict[EdgeId, np.ndarray] = {
            eid: np.asarray(amt, dtype=np.int32).copy()
            for eid, amt in truck_amounts.items()
        }
        belt_proj: Dict[ComboId, np.ndarray] = {
            cid: np.asarray(amt, dtype=np.int32).copy()
            for cid, amt in belt_amounts.items()
        }

        def compute_node_flows() -> Tuple[np.ndarray, np.ndarray]:
            out_flow = np.zeros((self.N, self.K), dtype=np.int32)
            in_flow = np.zeros((self.N, self.K), dtype=np.int32)

            for eid, amt in truck_proj.items():
                edge = spec.edges[eid]
                src_row = self._node_id_to_row[edge.src]
                dst_row = self._node_id_to_row[edge.dst]
                out_flow[src_row, :] = out_flow[src_row, :] + amt
                in_flow[dst_row, :] = in_flow[dst_row, :] + amt

            for cid, amt in belt_proj.items():
                combo = spec.belt_combos[cid]
                edge = spec.edges[combo.edge_id]
                src_row = self._node_id_to_row[edge.src]
                dst_row = self._node_id_to_row[edge.dst]
                out_flow[src_row, :] = out_flow[src_row, :] + amt
                in_flow[dst_row, :] = in_flow[dst_row, :] + amt

            return out_flow, in_flow

        max_iter = max(8, 2 * self.N * max(self.K, 1))
        for _ in range(max_iter):
            changed = False
            out_flow, in_flow = compute_node_flows()

            # Joint source-side inventory projection:
            # for each node/material, total outgoing flow cannot exceed
            # step-start inventory plus same-step accepted inflow.
            for nid in self.node_ids:
                row = self._node_id_to_row[nid]
                for k in range(self.K):
                    out_total = int(out_flow[row, k])
                    if out_total <= 0:
                        continue

                    available = float(Wi_before[row, k]) + float(in_flow[row, k])
                    if float(out_total) <= available + 1e-9:
                        continue

                    scale = 0.0 if available <= 0.0 else (available / float(out_total))
                    for eid in self._truck_out_edges_by_src.get(nid, []):
                        old_val = int(truck_proj[eid][k])
                        new_val = int(_floor_int(float(old_val) * scale))
                        if new_val != old_val:
                            truck_proj[eid][k] = new_val
                            changed = True
                    for cid in self._belt_out_combos_by_src.get(nid, []):
                        old_val = int(belt_proj[cid][k])
                        new_val = int(_floor_int(float(old_val) * scale))
                        if new_val != old_val:
                            belt_proj[cid][k] = new_val
                            changed = True

            out_flow, in_flow = compute_node_flows()

            # Destination-side total-capacity projection:
            # accepted inflow must keep the node's end-of-step total inventory within Ui.
            for nid in self.node_ids:
                row = self._node_id_to_row[nid]
                in_total = int(in_flow[row, :].sum())
                if in_total <= 0:
                    continue

                out_total = int(out_flow[row, :].sum())
                total_before = float(Wi_before[row, :].sum())
                allowed_in = max(float(spec.nodes[nid].Ui) - total_before + float(out_total), 0.0)
                if float(in_total) <= allowed_in + 1e-9:
                    continue

                scale = 0.0 if allowed_in <= 0.0 else (allowed_in / float(in_total))
                for eid in self._truck_in_edges_by_dst.get(nid, []):
                    old_amt = truck_proj[eid].copy()
                    new_amt = _floor_int(old_amt.astype(np.float32) * scale).astype(np.int32)
                    if not np.array_equal(new_amt, old_amt):
                        truck_proj[eid] = new_amt
                        changed = True
                for cid in self._belt_in_combos_by_dst.get(nid, []):
                    old_amt = belt_proj[cid].copy()
                    new_amt = _floor_int(old_amt.astype(np.float32) * scale).astype(np.int32)
                    if not np.array_equal(new_amt, old_amt):
                        belt_proj[cid] = new_amt
                        changed = True

            if not changed:
                break

        return truck_proj, belt_proj

    def step(self, action: Dict[str, np.ndarray]):
        spec = self.spec
        t = int(self._t)
        done = False
        reason = "running"

        # unpack
        a_truck = np.asarray(action["truck"], dtype=np.float32).reshape(self.Et, self.K)
        a_belt = np.asarray(action["belt"], dtype=np.float32).reshape(self.B, self.K + 1)
        belt_order = self._belt_order_from_action(np.asarray(action["belt_order"]))

        # belt 动作先用于执行裁剪（Pi / overlap）
        X = process_truck_actions(spec, self.truck_edges, t, a_truck)
        belt_res = process_belt_actions(spec, self.belt_combos, self._combo_overlap, t, a_belt, belt_order)
        Wi_before = self._Wi.copy()
        X.edge_amounts, belt_res.combo_amounts = self._project_executed_flows(
            X.edge_amounts,
            belt_res.combo_amounts,
            Wi_before,
        )
        B_amt = belt_res.combo_amounts
        belt_res.selected_material = {
            cid: (mat if int(B_amt[cid].sum()) > 0 else None)
            for cid, mat in belt_res.selected_material.items()
        }

        # 保存“原始输入”的 belt 动作 Bij(t)（供下一时刻观测）
        # 注意：这里记录的是 env 收到的 action["belt"]，不再是约束/overlap 处理后的最终执行动作
        self._last_belt_action = a_belt.copy()

        # overlap 标志仍然按最终执行结果计算（1=正常，0=因 overlap 被忽略）
        flag_overlap = np.ones((self.B,), dtype=np.float32)
        ignored_set = set(belt_res.ignored_due_to_overlap)
        for cidx, cid in enumerate(self.belt_combos):
            if cid in ignored_set:
                flag_overlap[cidx] = 0.0
        self._last_belt_flag_overlap = flag_overlap
        self._last_decision_seq = np.asarray([self._combo_id_to_idx[cid] for cid in belt_order], dtype=np.float32)

        # 计算切换次数 f(t+1)
        # 1) 任意皮带组合前后时刻运输不同物料（同一 combo 对比） -> f += 1
        # 2) 任意皮带组合切换至与其存在重叠子皮带的皮带组合运输（跨 combo 对比） -> f += 1
        curr_sel = np.full((self.B,), -1, dtype=np.int32)
        for cidx, cid in enumerate(self.belt_combos):
            k = belt_res.selected_material.get(cid, None)
            curr_sel[cidx] = -1 if k is None else int(k)

        prev_sel = self._last_belt_selected
        prev_active = prev_sel >= 0
        curr_active = curr_sel >= 0

        # (1) 同一 combo：前后时刻物料不同
        mat_sw = prev_active & curr_active & (prev_sel != curr_sel)

        # (2) 跨 combo：当前启用的 combo 与“上时刻启用的某个重叠 combo”之间发生切换
        cid_to_idx = {cid: i for i, cid in enumerate(self.belt_combos)}
        overlap_sw = np.zeros((self.B,), dtype=np.bool_)
        for cid in self.belt_combos:
            i = cid_to_idx[cid]
            if not curr_active[i]:
                continue
            for cid2 in self._combo_overlap.get(cid, []):
                j = cid_to_idx[cid2]
                if prev_active[j]:
                    overlap_sw[i] = True
                    break

        f_before = int(self._f_switch)
        f_after = int(f_before + int(mat_sw.sum()) + int(overlap_sw.sum()))
        last_sel_after = curr_sel

        # f 的状态转移提前：让 StepTrace 反映“当前时刻做出动作后”的切换次数
        self._f_switch = int(f_after)

        # 更新 Wi/Yi（保持原逻辑：truck + belt amounts 进入系统）

        # 需求转接逻辑：
        # - self._Yi 存的是“上一时刻结束后”的未满足需求 Yi(t-1)
        # - 当前时刻外生需求 Yi[t] 进入后，得到 Yi(t)=Yi(t-1)+Yi[t]（用于本时刻满足/观测）
        Yi_before = self._Yi.copy()
        for nid in self.demand_nodes:
            node = spec.nodes[nid]
            row = self._node_row(nid)
            if node.y is None:
                continue
            for k in range(self.K):
                exo = 0
                if k < len(node.y) and t < len(node.y[k]):
                    exo = int(node.y[k][t])
                Yi_before[row, k] = int(Yi_before[row, k]) + int(exo)

        Pi_now = self._current_pi_matrix(t)
        W_after = self._Wi.copy()

        # 记录“本时刻流入各需求节点”的量：
        # 只允许这部分流入扣减 Yi；历史库存/期初库存不参与需求满足
        inflow_for_demand = np.zeros((self.N, self.K), dtype=np.int32)
        demand_row_set = {self._node_row(nid) for nid in self.demand_nodes}

        # truck: edge_amounts（dst 增加，同时 src 扣减）
        for eid, amt in X.edge_amounts.items():
            edge = spec.edges[eid]
            src_row = self._node_row(edge.src)
            dst_row = self._node_row(edge.dst)
            amt_i32 = amt.astype(np.int32)

            W_after[src_row, :] = W_after[src_row, :] - amt_i32
            W_after[dst_row, :] = W_after[dst_row, :] + amt_i32

            # 仅统计“流入到需求节点”的当步流入量
            if dst_row in demand_row_set:
                inflow_for_demand[dst_row, :] = inflow_for_demand[dst_row, :] + amt_i32

        # belt: combo_amounts（dst 增加，同时 src 扣减）
        for cid, amt in B_amt.items():
            combo = spec.belt_combos[cid]
            e = spec.edges[combo.edge_id]
            src_row = self._node_row(e.src)
            dst_row = self._node_row(e.dst)
            amt_i32 = amt.astype(np.int32)

            W_after[src_row, :] = W_after[src_row, :] - amt_i32
            W_after[dst_row, :] = W_after[dst_row, :] + amt_i32

            # 仅统计“流入到需求节点”的当步流入量
            if dst_row in demand_row_set:
                inflow_for_demand[dst_row, :] = inflow_for_demand[dst_row, :] + amt_i32

        # 满足需求：只允许“本时刻流入量”扣减需求；
        Y_after = Yi_before.copy()
        for nid in self.demand_nodes:
            row = self._node_row(nid)
            for k in range(self.K):
                sat = min(int(inflow_for_demand[row, k]), int(Y_after[row, k]))
                if sat > 0:
                    Y_after[row, k] = int(Y_after[row, k]) - int(sat)

        # reward（先算当步每步奖励）
        rp_step = compute_reward(spec, Wi_before, W_after, Yi_before, Y_after, f_before, f_after, self._Y_exo_total)
        rp = rp_step
        t_star_before = self._t_star

        # T* 是首次满足“当前剩余需求为 0，且未来外生需求也全为 0”的时刻。
        if self._t_star is None and self._future_yi_all_zero_after_transition(t, Y_after):
            self._t_star = int(t)

        # termination（在本步完成状态转移 & 需求结转后判定）
        node_ids = self.node_ids  # 与 Wi/Yi 的行顺序一致（sorted(spec.nodes.keys()))
        node_totals = W_after.sum(axis=1).astype(np.float32)

        # (3) 异常终止判定：
        # - 任意节点 Vi 总库存超 Ui
        # - 二/三级节点“全网”某物料总量超 Uk
        # - 任意节点任意物料量 < 0
        over_ui_rows = []
        for r, nid in enumerate(node_ids):
            if float(node_totals[r]) > float(spec.nodes[nid].Ui):
                over_ui_rows.append(int(r))

        neg_inventory = bool((W_after < 0).any())

        if len(self._lvl23_rows) > 0:
            lvl23_material_totals = W_after[self._lvl23_rows, :].sum(axis=0).astype(np.float32)
        else:
            lvl23_material_totals = np.zeros((self.K,), dtype=np.float32)

        over_uk_mats = []
        for k in range(self.K):
            if float(lvl23_material_totals[k]) > float(spec.materials[k].Uk):
                over_uk_mats.append(int(k))

        if neg_inventory or (len(over_ui_rows) > 0) or (len(over_uk_mats) > 0):
            done = True
            reason = "abnormal"
            self._t_star = t_star_before
            rp_term = compute_terminal_reward(
                spec,
                node_ids=node_ids,
                lvl1_rows=self._lvl1_rows,
                lvl23_rows=self._lvl23_rows,
                W_after=W_after,
                Y_after=Y_after,
                t=t,
                f_switch=f_after,
                reason=reason,
                Y_exo_total=self._Y_exo_total,
                neg_inventory=neg_inventory,
                over_ui_rows=over_ui_rows,
                lvl23_material_totals=lvl23_material_totals,
                over_uk_mats=over_uk_mats,
            )
            rp = rp_step + rp_term

        if (not done) and t_star_before is None and self._t_star is not None:
            # R_t_star 是首次命中 T* 的一次性步奖励，只在首次进入 T* 时发放。
            rp_t_star = compute_t_star_reward(spec, self._t_star)
            rp = rp + rp_t_star

        # (1)/(2) 仅在最终时刻判定 success / demand_not_fulfill：
        # - success：t+1 == T，且状态转移后所有节点 Yi 均为 0；
        # - demand_not_fulfill：t+1 == T，但仍存在未满足需求。
        if (not done) and self._t + 1 == self.spec.T:
            done = True
            if int(Y_after.sum()) == 0:
                reason = "success"
            else:
                reason = "demand_not_fulfill"
            rp_term = compute_terminal_reward(
                spec,
                node_ids=node_ids,
                lvl1_rows=self._lvl1_rows,
                lvl23_rows=self._lvl23_rows,
                W_after=W_after,
                Y_after=Y_after,
                t=t,
                f_switch=f_after,
                reason=reason,
                Y_exo_total=self._Y_exo_total,
                T_star=self._t_star,
            )
            rp = rp + rp_term

        # 按统一公式缩放总奖励
        scaled_reward = float(self.reward_scale * rp.total)

        terminate_detail: Optional[List[str]] = None
        if reason == "abnormal":
            terminate_detail = []

            for r in over_ui_rows:
                nid = node_ids[r]
                node = spec.nodes[nid]
                total = float(node_totals[r])
                ui = float(node.Ui)
                exceed = total - ui
                terminate_detail.append(
                    f"node_total_over_ui: 节点{node.name}(node_id={nid})当前总库存={total:.3f} > Ui={ui:.3f}, 超出={exceed:.3f}"
                )

            for k in over_uk_mats:
                total_k = float(lvl23_material_totals[k])
                uk = float(spec.materials[k].Uk)
                exceed = total_k - uk
                material_name = getattr(spec.materials[k], "name", f"material_{k}")
                terminate_detail.append(
                    f"level23_material_total_over_uk: 物料{material_name}(k={k})在二三级节点当前总量={total_k:.3f} > Uk={uk:.3f}, 超出={exceed:.3f}"
                )

            neg_rows, neg_cols = np.where(W_after < 0)
            for r, k in zip(neg_rows.tolist(), neg_cols.tolist()):
                nid = node_ids[r]
                node = spec.nodes[nid]
                val = float(W_after[r, k])
                material_name = getattr(spec.materials[k], "name", f"material_{k}")
                terminate_detail.append(
                    f"negative_inventory: 节点{node.name}(node_id={nid})物料{material_name}(k={k})库存={val:.3f} < 0, 低于0={abs(val):.3f}"
                )

        # trace（按约定字段顺序记录；奖励均记录缩放前的原值）
        re_trace: Optional[Dict[str, Any]] = None
        terminate_reason_trace: Optional[str] = None
        terminate_detail_trace: Optional[List[str]] = None
        if done:
            terminate_reason_trace = str(reason)
            terminate_detail_trace = terminate_detail
            if reason == "success":
                re_trace = {
                    "R_base": float(rp.R_base),
                    "R_T_left": float(rp.R_T_left),
                    "R_f_penalty": float(rp.R_f_penalty),
                    "R_level1": float(rp.R_level1),
                    "T_star": None if self._t_star is None else int(self._t_star),
                }
            elif reason == "demand_not_fulfill":
                re_trace = {
                    "R_base": float(rp.R_base),
                    "R_f_penalty": float(rp.R_f_penalty),
                    "R_demand_not_fulfilled": float(rp.R_demand_not_fulfilled),
                }
            elif reason == "abnormal":
                re_trace = {
                    "R_base": float(rp.R_base),
                    "R_f_penalty": float(rp.R_f_penalty),
                    "R_ultra_limit": float(rp.R_ultra_limit),
                }

        self.trace.append(
            StepTrace(
                t=t,
                Wi=Wi_before.astype(np.int32).copy(),
                Yi_before=Yi_before.astype(np.int32).copy(),
                Y_after=Y_after.astype(np.int32).copy(),
                Pi=Pi_now.copy(),
                X={
                    self.spec.edge_raw_ids[int(eid)]: amt.astype(np.int32).tolist()
                    for eid, amt in X.edge_amounts.items()
                },
                B={
                    self.spec.belt_combo_raw_ids[int(cid)]: amt.astype(np.int32).tolist()
                    for cid, amt in B_amt.items()
                },
                f=int(self._f_switch),
                Decision_seq=[self.spec.belt_combo_raw_ids[int(cid)] for cid in belt_order],
                R_demand=float(rp.R_demand),
                R_level23_penalty=float(rp.R_level23_penalty),
                R_step=float(rp.R_step),
                R_t_star=float(rp.R_t_star) if float(rp.R_t_star) != 0.0 else None,
                Re=re_trace,
                terminate_reason=terminate_reason_trace,
                terminate_detail=terminate_detail_trace,
            )
        )

        # 写回状态
        self._Wi = W_after.astype(np.int32)
        self._Yi = Y_after.astype(np.int32)
        self._last_belt_selected = last_sel_after
        self._t += 1

        obs = self._make_obs()

        info = {
            "terminate_reason": reason,
            "terminate_detail": terminate_detail,
            "R_base": float(rp.R_base),
            "R_demand": float(rp.R_demand),
            "R_t_star": float(rp.R_t_star),
            "R_level23_penalty": float(rp.R_level23_penalty),
            "R_step": float(rp.R_step),
            "R_T_left": float(rp.R_T_left),
            "R_f_penalty": float(rp.R_f_penalty),
            "R_level1": float(rp.R_level1),
            "R_demand_not_fulfilled": float(rp.R_demand_not_fulfilled),
            "R_over_Ui": float(rp.R_over_Ui),
            "R_over_Uk": float(rp.R_over_Uk),
            "R_below_0": float(rp.R_below_0),
            "R_ultra_limit": float(rp.R_ultra_limit),
            "Re": float(rp.Re),
            "reward_scale": float(self.reward_scale),
            "reward_raw": float(rp.total),
            "f": int(self._f_switch),
            "T_star": None if self._t_star is None else int(self._t_star),
            "ignored_overlap": list(belt_res.ignored_due_to_overlap),
        }
        return obs, scaled_reward, bool(done), False, info


# ==================================
# 5) YAML -> EnvSpec
# ==================================
def load_yaml(path: str) -> Dict[str, Any]:
    """Load yaml config as dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_env_spec(cfg: Dict[str, Any]) -> EnvSpec:
    """Build EnvSpec from the yaml dict (compatible with default.yaml)."""
    env_cfg = cfg.get("env", cfg)

    T = int(env_cfg.get("T", 1))
    n_obs = int(env_cfg.get("n_obs", 1))
    belt_order_mode = str(env_cfg.get("belt_order_mode", "manual"))

    # materials
    materials: List[MaterialSpec] = []
    for m in env_cfg.get("materials", []):
        materials.append(MaterialSpec(Uk=float(m.get("Uk", 0.0)), Lk=float(m.get("Lk", 0.0)), name=str(m.get("name", ""))))

    # nodes
    nodes: Dict[NodeId, NodeSpec] = {}
    for nd in env_cfg.get("nodes", []):
        nid = int(nd.get("node_id"))
        Ui = float(nd.get("Ui", 0.0))
        Ui_s = float(nd.get("Ui_s", Ui))
        Li_s = float(nd.get("Li_s", 0.0))
        level = int(nd.get("level", 0))
        name = str(nd.get("name", ""))
        Li = float(nd.get("Li", 0.0))
        Mj = float(nd.get("Mj_truck_unload", 0.0))

        W0 = list(nd.get("W0", [0 for _ in range(len(materials))]))
        y = nd.get("Y_exo", None)
        Pi = nd.get("Pi", None)

        nodes[nid] = NodeSpec(
            Ui=Ui,
            Ui_s=Ui_s,
            Li_s=Li_s,
            level=level,
            name=name,
            Li=Li,
            Mj_truck_unload=Mj,
            W0=W0,
            y=y,
            Pi=Pi,
        )

    # edges: map raw edge_id -> internal integer id
    edges_list = env_cfg.get("edges", [])
    edge_id_map: Dict[Any, EdgeId] = {}
    edges: Dict[EdgeId, EdgeSpec] = {}
    edge_raw_ids: List[str] = []

    for i, ed in enumerate(edges_list):
        raw_id = str(ed.get("edge_id"))
        eid = int(i)
        edge_id_map[raw_id] = eid
        edge_raw_ids.append(raw_id)
        edges[eid] = EdgeSpec(
            src=int(ed.get("src")),
            dst=int(ed.get("dst")),
            mode=str(ed.get("mode", "belt")),
        )

    # belt combos: map raw combo_id -> internal integer id
    combos_list = env_cfg.get("belt_combos", [])
    combo_id_map: Dict[Any, ComboId] = {}
    belt_combos: Dict[ComboId, BeltComboSpec] = {}
    belt_combo_raw_ids: List[str] = []

    for i, cd in enumerate(combos_list):
        raw_cid = str(cd.get("combo_id"))
        cid = int(i)
        combo_id_map[raw_cid] = cid
        belt_combo_raw_ids.append(raw_cid)

        raw_eid = cd.get("edge_id")
        if raw_eid not in edge_id_map:
            raise KeyError(f"belt_combos.edge_id {raw_eid} not found in edges")
        eid_int = edge_id_map[raw_eid]

        belt_combos[cid] = BeltComboSpec(
            edge_id=eid_int,
            capacity=float(cd.get("capacity", 0.0)),
            sub_belts=list(cd.get("sub_belts", [])),
        )

    # belt manual decision sequence (optional)
    belt_decision_seq_raw = env_cfg.get("belt_decision_seq", None)
    belt_decision_seq: Optional[List[ComboId]] = None
    if belt_decision_seq_raw is not None:
        belt_decision_seq = [combo_id_map[c] for c in belt_decision_seq_raw]

    # truck edges: include edges with mode in {truck,both}
    default_truck_capacity = float(env_cfg.get("default_truck_capacity", 30.0))
    truck_edges: List[TruckEdgeSpec] = []
    for raw_id, eid in edge_id_map.items():
        mode = str(edges[eid].mode).lower()
        if mode in ("truck", "both"):
            dst = edges[eid].dst
            cap = float(getattr(nodes.get(dst), "Mj_truck_unload", 0.0) or 0.0)
            if cap <= 0:
                cap = default_truck_capacity
            truck_edges.append(TruckEdgeSpec(edge_id=eid, capacity=cap))

    return EnvSpec(
        nodes=nodes,
        edges=edges,
        truck_edges=truck_edges,
        belt_combos=belt_combos,
        edge_raw_ids=edge_raw_ids,
        belt_combo_raw_ids=belt_combo_raw_ids,
        materials=materials,
        T=T,
        belt_order_mode=belt_order_mode,
        belt_decision_seq=belt_decision_seq,
        n_obs=n_obs,
    )
