#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ConesColorSubscriber (Angle-Constrained Stepper, Fixed-Length)
- 좌/우 콘만으로 벽(폴리라인) 구성 (RED/기타는 벽·중점 계산에서 제외)
- 헝가리안으로 좌/우 페어를 뽑아 중점 생성(있으면 사용)
- 경로 생성: 점-단위 스테퍼(각도 제약, 총 길이 고정 PATH_LEN)
  · 매 스텝 고정거리 전진(마지막 스텝은 PATH_LEN에 맞춰 자동 축소)
  · 코리더(좌/우 벽 + 안전마진) 하드 제약
  · 진행각과 벽 접선의 코사인 임계치 하드 제약
  · 콘에 대한 하드 최소거리
  · 위 제약 통과 후보 중, 중점 추종 + 조향변화(부드러움) 코스트 최소 선택
- RViz 표시:
  · 'path'            : 좌/우 벽(그리디) 라인
  · 'pair_midpoints'  : 좌/우 페어 중점
  · 'pair_lines'      : 좌/우 페어 연결선 (옵션)
  · 'opt_path'        : 최종 경로
  · 'cones_left'      : 모든 좌측 콘(노란색)
  · 'cones_right'     : 모든 우측 콘(파란색)
  · 'cones_red'       : 빨간 콘
  · 'cones_other'     : 기타 콘(회색)
"""

import math
from typing import List, Tuple, Dict, Optional

import numpy as np
from scipy.spatial import KDTree
from scipy.optimize import linear_sum_assignment

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from builtin_interfaces.msg import Duration
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA

from custom_interface.msg import TrackedConeArray


# ──────────────────────────────────────────────────────────────────────────────
# 기본 토픽/프레임
# ──────────────────────────────────────────────────────────────────────────────
INPUT_TOPIC = '/fused_sorted_cones_ukf'
OUTPUT_TOPIC = '/cones_marker_array'
USE_HEADER_FRAME = True
FALLBACK_FRAME = 'os_sensor'
FORCE_FALLBACK_FRAME = True

PROCESS_HZ = 10.0
REPUBLISH_IDLE_SEC = 0.5

# ──────────────────────────────────────────────────────────────────────────────
# 그래프(좌/우 벽 그리디용)
# ──────────────────────────────────────────────────────────────────────────────
EDGE_LEN_TH = 3.5#4.0
MAX_NEIGHBORS_PER_NODE = 12
MAX_STEPS = 1000

START_X_RANGE = (-0.5, 5.0)
START_Y_LEFT = (0.1, 5.0)
START_Y_RIGHT = (-5.0, -0.1)

PARENT_DEG_TIERS = [65.0, 110.0]
GRAND_DEG_TIERS  = [35.0, 90.0]
GLOBAL_DEG_TIERS = [90.0]
MAX_RELAX_LEVELS = 2
X_BACKTRACK_MARGIN = 0.3
PREFER_GAIN = 0.25

# ──────────────────────────────────────────────────────────────────────────────
# 색상 키/표시 색
# ──────────────────────────────────────────────────────────────────────────────
LEFT_KEYS  = {'yellow', 'yellowcone'}
RIGHT_KEYS = {'blue', 'bluecone'}
RED_KEYS   = {'red', 'redcone'}

LEFT_COLOR   = (1.0, 1.0, 0.0, 1.0)
RIGHT_COLOR  = (0.0, 0.3, 1.0, 1.0)
OTHER_COLOR  = (0.6, 0.6, 0.6, 0.9)
RED_COLOR    = (1.0, 0.0, 0.0, 1.0)
OPT_PATH_COLOR = (0.1, 1.0, 0.1, 0.95)

# ──────────────────────────────────────────────────────────────────────────────
# 마커/표시
# ──────────────────────────────────────────────────────────────────────────────
MARKER_SCALE = 0.45
PAIR_MID_SCALE = 0.35
PAIR_MID_COLOR = (1.0, 1.0, 1.0, 0.95)
PAIR_LINE_COLOR = (0.2, 1.0, 0.2, 0.55)
LIFETIME_SEC = 0.5
MIN_PATH_LEN = 3
SHOW_START_MARKERS = True
SHOW_PAIR_MIDPOINTS = True
SHOW_PAIR_LINES = True  # 원하면 False

# 페어링
ENABLE_PAIRING = True
PAIR_MAX_DIST = 5.5
PAIR_MAX_MATRIX_ELEMS = 8000

# 기타
DEDUP_EPS = 1e-3
ORIGIN_Z = 0.0

# ──────────────────────────────────────────────────────────────────────────────
# 스텝퍼(각도 제약) + 비용 항
# ──────────────────────────────────────────────────────────────────────────────
PATH_LEN = 15#20.0          # (m) 전체 경로 길이 (고정)
STEP_LEN = 1.2#1.0#0.8#1.0           # (m) 기본 스텝 길이 (마지막 스텝은 자동 축소)
MAX_DTHETA_DEG = 35.0    # (deg) 스텝당 최대 조향 변화
N_THETA = 13             # 후보 각도 샘플 수(홀수 권장)
FORWARD_EPS = 0.05       # x 전진 최소량(역주행/정지 방지) - 스텝에 맞춰 축소 적용

# 코너/코리더 제약
WALL_MARGIN = 1.5        # (m) 벽 안전마진
DIR_THRESH_LEFT_DEG = 5#15#35.0   # 벽 접선과 진행방향 최소 코사인 각(좌)
DIR_THRESH_RIGHT_DEG = 5#15#35.0  # (우)

# 장애물(콘)
OBS_HARD_RADIUS = 0.5    # (m) 하드 최소거리
OBS_RADIUS = 2.3         # (m) 소프트 반경
SOFTPLUS_BETA = 4.0
W_OBS = 1.0#3.0#1.0#2.0              # 소프트 장애물 가중치

# 목적함수
W_TRACK = 1.5#2.0#2.5#4.0            # 중점 추종
W_CURV  = 2.0#4.0#2.5#1.2            # 조향 변화 벌점


def _normalize_color_key(s: str) -> str:
    if not s:
        return 'unknown'
    return ''.join(s.lower().replace('_', '').split())


def _euclid(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


class ConesColorSubscriber(Node):
    def __init__(self):
        super().__init__('cones_color_subscriber')

        # 토픽 파라미터
        self.declare_parameter('input_topic', INPUT_TOPIC)
        self.declare_parameter('output_topic', OUTPUT_TOPIC)
        self.input_topic = self.get_parameter('input_topic').value
        self.output_topic = self.get_parameter('output_topic').value

        # QoS
        qos_pub = QoSProfile(depth=1)
        qos_pub.reliability = ReliabilityPolicy.RELIABLE
        qos_pub.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.pub = self.create_publisher(MarkerArray, self.output_topic, qos_pub)

        qos_sub = QoSProfile(depth=1)
        qos_sub.reliability = ReliabilityPolicy.BEST_EFFORT
        qos_sub.history = HistoryPolicy.KEEP_LAST
        qos_sub.durability = DurabilityPolicy.VOLATILE
        self.sub = self.create_subscription(
            TrackedConeArray, self.input_topic, self._cb, qos_sub)

        self.get_logger().info(
            f"Subscribed: {self.input_topic} → Publishing: {self.output_topic}")

        self._latest_msg: Optional[TrackedConeArray] = None
        self._last_msg_time = self.get_clock().now()
        self._last_arr: Optional[MarkerArray] = None

        self.create_timer(1.0 / max(0.5, PROCESS_HZ), self._on_timer)

        # 그리디용 코사인 티어
        self.parent_cos = [math.cos(math.radians(d)) for d in PARENT_DEG_TIERS]
        self.grand_cos  = [math.cos(math.radians(d)) for d in GRAND_DEG_TIERS]
        self.global_cos = [math.cos(math.radians(d)) for d in GLOBAL_DEG_TIERS]
        self.life = Duration(sec=int(LIFETIME_SEC),
                             nanosec=int((LIFETIME_SEC - int(LIFETIME_SEC))*1e9))

        # 이전 최적 경로(필요시)
        self._prev_path_xy: Optional[np.ndarray] = None

    # ── 콜백/타이머
    def _cb(self, msg: TrackedConeArray):
        self._latest_msg = msg
        self._last_msg_time = self.get_clock().now()

    def _on_timer(self):
        now = self.get_clock().now()
        if self._latest_msg is None:
            if self._last_arr and (now - self._last_msg_time).nanoseconds/1e9 >= REPUBLISH_IDLE_SEC:
                ts = now.to_msg()
                for m in self._last_arr.markers:
                    m.header.stamp = ts
                self.pub.publish(self._last_arr)
            return
        try:
            arr = self._process(self._latest_msg)
            self._last_arr = arr
            self.pub.publish(arr)
        except Exception as e:
            self.get_logger().error(f"[Process] exception: {e}")

    # ── 유틸
    def _sanitize(self, pts, zs, keys):
        mask = []
        uniq = set()
        for i, (p, z, k) in enumerate(zip(pts, zs, keys)):
            x, y = p
            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
                continue
            gx, gy = round(x / DEDUP_EPS), round(y / DEDUP_EPS)
            if (gx, gy) in uniq:
                continue
            uniq.add((gx, gy))
            mask.append(i)
        return mask

    def _build_graph_grid(self, pts: List[Tuple[float, float]]):
        r = max(1e-6, EDGE_LEN_TH)
        inv = 1.0 / r
        cell: Dict[Tuple[int, int], List[int]] = {}
        for i, (x, y) in enumerate(pts):
            cx, cy = int(math.floor(x * inv)), int(math.floor(y * inv))
            cell.setdefault((cx, cy), []).append(i)
        graph: Dict[int, List[int]] = {i: [] for i in range(len(pts))}
        for (cx, cy), idxs in cell.items():
            for i in idxs:
                x, y = pts[i]
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        n_idxs = cell.get((cx + dx, cy + dy))
                        if not n_idxs:
                            continue
                        for j in n_idxs:
                            if j <= i:
                                continue
                            if _euclid((x, y), pts[j]) <= r:
                                graph[i].append(j)
                                graph[j].append(i)
        if MAX_NEIGHBORS_PER_NODE > 0:
            for u, nbrs in graph.items():
                if len(nbrs) > MAX_NEIGHBORS_PER_NODE:
                    nbrs.sort(key=lambda v: _euclid(pts[u], pts[v]))
                    graph[u] = nbrs[:MAX_NEIGHBORS_PER_NODE]
        return graph

    def _tier(self, arr: List[float], i: int) -> float:
        if not arr:
            return -1.0
        return arr[i] if i < len(arr) else arr[-1]

    def _get_start_node(self, pts: List[Tuple[float, float]],
                        keys: List[str], allow_keys: set,
                        y_range: Tuple[float, float]):
        try:
            cands = [(i, x) for i, (x, y) in enumerate(pts)
                     if (keys[i] in allow_keys)
                     and (START_X_RANGE[0] <= x < START_X_RANGE[1])
                     and (y_range[0] <= y < y_range[1])]
            return min(cands, key=lambda it: it[1])[0] if cands else None
        except Exception:
            return None

    def _greedy_path(self, start, graph, pts, keys, allow_keys, prefer_keys) -> List[int]:
        if start is None:
            return []
        path = [start]
        seen = {start}
        maxx = max(p[0] for p in pts) if pts else 0.0
        steps = 0
        use_global = len(self.global_cos) > 0
        max_levels = min(
            MAX_RELAX_LEVELS,
            max(len(self.parent_cos), len(self.grand_cos), (len(self.global_cos) if len(self.global_cos)>0 else 1))
        )
        while steps < MAX_STEPS:
            steps += 1
            u = path[-1]
            parent = path[-2] if len(path) >= 2 else None
            grand = path[-3] if len(path) >= 3 else None
            px, py = pts[u]
            p1 = None
            if parent is not None:
                dx1, dy1 = px - pts[parent][0], py - pts[parent][1]
                n1 = math.hypot(dx1, dy1)
                p1 = (dx1/n1, dy1/n1) if n1 > 0 else None
            p2 = None
            if grand is not None:
                dx2, dy2 = px - pts[grand][0], py - pts[grand][1]
                n2 = math.hypot(dx2, dy2)
                p2 = (dx2/n2, dy2/n2) if n2 > 0 else None

            best = None
            for tier in range(max_levels):
                gcos = self._tier(self.global_cos, tier)
                pcos = self._tier(self.parent_cos, tier)
                qcos = self._tier(self.grand_cos, tier)
                cand = None
                cand_score = 1e18
                for v in graph.get(u, []):
                    if v == parent or v in seen:
                        continue
                    # 색상 필터: 해당 사이드만 허용 (RED/기타 제외)
                    if (keys[v] not in allow_keys) or (keys[u] not in allow_keys):
                        continue
                    vx, vy = pts[v][0] - px, pts[v][1] - py
                    dist = math.hypot(vx, vy)
                    if dist <= 0:
                        continue
                    ux, uy = vx/dist, vy/dist
                    if use_global and gcos > -1.0:
                        if ux < gcos:
                            continue
                    elif not use_global:
                        if pts[v][0] < px - X_BACKTRACK_MARGIN:
                            continue
                    if p1 and pcos > -1.0 and (p1[0]*ux + p1[1]*uy) < pcos:
                        continue
                    if p2 and qcos > -1.0 and (p2[0]*ux + p2[1]*uy) < qcos:
                        continue
                    score = dist * (1.0 - (PREFER_GAIN if keys[v] in prefer_keys else 0.0)) - 1e-3 * (pts[v][0] - px)
                    if score < cand_score:
                        cand_score = score
                        cand = v
                if cand is not None:
                    best = cand
                    break
            if best is None:
                break
            path.append(best)
            seen.add(best)
            if pts[best][0] >= maxx - 1e-6:
                break
        return path

    def _pair_cones(self, pts, zs, keys):
        if not ENABLE_PAIRING:
            return []
        left_idx = [i for i, k in enumerate(keys) if k in LEFT_KEYS]
        right_idx = [i for i, k in enumerate(keys) if k in RIGHT_KEYS]
        if not left_idx or not right_idx:
            return []
        L, R = len(left_idx), len(right_idx)
        if L*R > PAIR_MAX_MATRIX_ELEMS:
            treeR = KDTree([pts[j] for j in right_idx])
            used = set()
            pairs = []
            for i in left_idx:
                d, jn = treeR.query(pts[i])
                jn = int(jn)
                if float(d) <= PAIR_MAX_DIST and jn not in used:
                    used.add(jn)
                    j = right_idx[jn]
                    mx = 0.5*(pts[i][0]+pts[j][0]); my = 0.5*(pts[i][1]+pts[j][1]); mz = 0.5*(zs[i]+zs[j])
                    pairs.append((i, j, mx, my, mz))
            return pairs
        Lpts = np.asarray([pts[i] for i in left_idx], dtype=np.float32)
        Rpts = np.asarray([pts[j] for j in right_idx], dtype=np.float32)
        diff = Lpts[:, None, :] - Rpts[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        cost = dist.copy()
        cost[dist > PAIR_MAX_DIST] = 1e6
        try:
            rows, cols = linear_sum_assignment(cost)
        except Exception:
            return []
        pairs = []
        for r, c in zip(rows.tolist(), cols.tolist()):
            d = float(dist[r, c])
            if not math.isfinite(d) or d > PAIR_MAX_DIST:
                continue
            i = left_idx[r]; j = right_idx[c]
            mx = 0.5*(pts[i][0]+pts[j][0]); my = 0.5*(pts[i][1]+pts[j][1]); mz = 0.5*(zs[i]+zs[j])
            pairs.append((i, j, mx, my, mz))
        return pairs

    # ────────────────────────────────────────────────────────────────────────
    # 보간/유틸
    # ────────────────────────────────────────────────────────────────────────
    def _interp_mid_y(self, mids_xy: np.ndarray, x_grid: np.ndarray,
                      eps_x: float = 1e-3,   # x-중복 병합 임계 (m)
                      sigma_x: float = 1.5,  # 가중치 가우시안 폭 (m)
                      w_floor: float = 0.1   # 최소 가중치
                      ) -> Tuple[np.ndarray, np.ndarray]:
        M = x_grid.shape[0]
        if mids_xy.size == 0:
            return np.zeros(M, dtype=np.float64), np.full(M, w_floor, dtype=np.float64)
        mids = mids_xy[np.argsort(mids_xy[:, 0])]
        xs, ys = mids[:, 0].astype(np.float64), mids[:, 1].astype(np.float64)
        uniq_x, uniq_y = [], []
        i = 0; N = len(xs)
        while i < N:
            j = i + 1; acc_y = ys[i]; cnt = 1
            while j < N and abs(xs[j] - xs[i]) <= eps_x:
                acc_y += ys[j]; cnt += 1; j += 1
            uniq_x.append(xs[i]); uniq_y.append(acc_y / cnt); i = j
        ux = np.asarray(uniq_x, dtype=np.float64)
        uy = np.asarray(uniq_y, dtype=np.float64)
        y_ref = np.empty_like(x_grid, dtype=np.float64)
        if len(ux) == 1:
            y_ref.fill(uy[0])
        else:
            y_ref[:] = np.interp(x_grid, ux, uy)
            left_mask = x_grid < ux[0]
            if np.any(left_mask):
                s0 = (uy[1] - uy[0]) / max(1e-9, (ux[1] - ux[0]))
                y_ref[left_mask] = uy[0] + s0 * (x_grid[left_mask] - ux[0])
            right_mask = x_grid > ux[-1]
            if np.any(right_mask):
                sl = (uy[-1] - uy[-2]) / max(1e-9, (ux[-1] - ux[-2]))
                y_ref[right_mask] = uy[-1] + sl * (x_grid[right_mask] - ux[-1])
        idx = np.searchsorted(ux, x_grid, side='left')
        d_left = np.where(idx > 0, np.abs(x_grid - ux[np.clip(idx-1, 0, len(ux)-1)]), np.inf)
        d_right = np.where(idx < len(ux), np.abs(ux[np.clip(idx, 0, len(ux)-1)] - x_grid), np.inf)
        dmin = np.minimum(d_left, d_right)
        w = np.exp(- (dmin / max(1e-9, sigma_x))**2)
        w = np.maximum(w, w_floor)
        return y_ref, w

    def _resample_poly_to_y_slope(self, poly_xy: np.ndarray, x_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        M = x_grid.shape[0]
        y_grid = np.zeros(M, dtype=np.float64)
        s_grid = np.zeros(M, dtype=np.float64)
        has_mask = np.zeros(M, dtype=bool)
        if poly_xy.size < 2:
            return y_grid, s_grid, has_mask
        poly = poly_xy[np.argsort(poly_xy[:, 0])]
        xs = poly[:, 0].astype(np.float64)
        ys = poly[:, 1].astype(np.float64)
        ux = [xs[0]]; uy = [ys[0]]
        for i in range(1, len(xs)):
            if abs(xs[i] - ux[-1]) <= 1e-9:
                uy[-1] = 0.5*(uy[-1] + ys[i])
            else:
                ux.append(xs[i]); uy.append(ys[i])
        ux = np.asarray(ux); uy = np.asarray(uy)
        if len(ux) < 2:
            y_grid.fill(uy[0]); s_grid.fill(0.0); has_mask[:] = True
            return y_grid, s_grid, has_mask
        slopes = (uy[1:] - uy[:-1]) / np.maximum(1e-9, (ux[1:] - ux[:-1]))
        idx = np.searchsorted(ux, x_grid, side='right') - 1
        idx = np.clip(idx, 0, len(ux)-2)
        x0 = ux[idx]; y0 = uy[idx]; m = slopes[idx]
        y_grid = y0 + m * (x_grid - x0)
        s_grid = m.copy()
        has_mask[:] = True
        return y_grid, s_grid, has_mask

    @staticmethod
    def _softplus(z: np.ndarray) -> np.ndarray:
        return np.where(z > 0, z + np.log1p(np.exp(-z)), np.log1p(np.exp(z)))

    # ────────────────────────────────────────────────────────────────────────
    # 각도 제약 기반 스텝퍼 (총 길이 고정)
    # ────────────────────────────────────────────────────────────────────────
    def _grow_path_with_angle_constraints(self,
                                          mids_xy: np.ndarray,
                                          left_poly: np.ndarray,
                                          right_poly: np.ndarray,
                                          obs_xy: np.ndarray) -> np.ndarray:
        def _mid_at(x: float) -> Tuple[float, float]:
            xg = np.array([x], dtype=np.float64)
            yv, wv = self._interp_mid_y(mids_xy, xg)
            return float(yv[0]), float(wv[0])

        def _walls_at(x: float) -> Tuple[float, float, bool, float, float, bool]:
            xg = np.array([x], dtype=np.float64)
            yL, sL, mL = self._resample_poly_to_y_slope(left_poly, xg)
            yR, sR, mR = self._resample_poly_to_y_slope(right_poly, xg)
            return float(yL[0]), float(sL[0]), bool(mL[0]), float(yR[0]), float(sR[0]), bool(mR[0])

        tree = KDTree(obs_xy) if obs_xy.size > 0 else None

        # 초기 상태 (0,0), 초기 진행각은 벽 기울기 평균(가능하면)
        p = np.array([0.0, 0.0], dtype=np.float64)
        yL0, sL0, hL0, yR0, sR0, hR0 = _walls_at(0.0)
        s0 = (0.5*(sL0 + sR0) if (hL0 and hR0) else (sL0 if hL0 else (sR0 if hR0 else 0.0)))
        theta = math.atan2(s0, 1.0)

        max_d = math.radians(MAX_DTHETA_DEG)
        cos_th_L = math.cos(math.radians(DIR_THRESH_LEFT_DEG))
        cos_th_R = math.cos(math.radians(DIR_THRESH_RIGHT_DEG))

        pts = [p.copy()]
        acc_len = 0.0  # 누적 이동거리

        while acc_len < PATH_LEN - 1e-9:
            # 남은 길이에 맞춰 스텝 자동 축소
            step_d = min(STEP_LEN, PATH_LEN - acc_len)
            forward_eps = min(FORWARD_EPS, 0.5 * step_d)

            thetas = np.linspace(theta - max_d, theta + max_d, N_THETA)
            best = None; best_cost = 1e18
            for th in thetas:
                dx = step_d * math.cos(th); dy = step_d * math.sin(th)
                pn = np.array([p[0] + dx, p[1] + dy])

                # 전진성
                if pn[0] < p[0] + forward_eps:
                    continue

                # 벽/코리더
                yL, sL, hL, yR, sR, hR = _walls_at(pn[0])
                if hL or hR:
                    y_upper = max(yL, yR); y_lower = min(yL, yR)
                    if (pn[1] > y_upper - WALL_MARGIN) or (pn[1] < y_lower + WALL_MARGIN):
                        continue

                # 방향 코사인 제약
                d = np.array([math.cos(th), math.sin(th)])
                if hL:
                    dL = np.array([1.0, sL]); dL /= (np.linalg.norm(dL) + 1e-12)
                    if float(np.dot(d, dL)) < cos_th_L:
                        continue
                if hR:
                    dR = np.array([1.0, sR]); dR /= (np.linalg.norm(dR) + 1e-12)
                    if float(np.dot(d, dR)) < cos_th_R:
                        continue

                # 장애물 하드 거리
                if tree is not None:
                    dd, _ = tree.query(pn, k=1)
                    if float(dd) < OBS_HARD_RADIUS:
                        continue

                # 비용: 중점 추종 + 회전 변화 + (선택) 소프트 장애물
                y_mid, w_mid = _mid_at(pn[0])
                J = W_TRACK * w_mid * (pn[1] - y_mid)**2
                J += W_CURV * (th - theta)**2

                if tree is not None and obs_xy.size > 0:
                    idxs = tree.query_ball_point(pn, r=OBS_RADIUS)
                    for j in (idxs or []):
                        v = pn - obs_xy[j]
                        a = (OBS_RADIUS**2) - float(v[0]**2 + v[1]**2)
                        if a > 0:
                            J += (W_OBS / SOFTPLUS_BETA) * float(self._softplus(SOFTPLUS_BETA * a))

                if J < best_cost:
                    best_cost = J; best = (pn, th)

            # 후보 없음 → 각도창 넓혀 완화 1회 (step_d 유지)
            if best is None:
                thetas = np.linspace(theta - 2.0*max_d, theta + 2.0*max_d, N_THETA)
                for th in thetas:
                    dx = step_d * math.cos(th); dy = step_d * math.sin(th)
                    pn = np.array([p[0] + dx, p[1] + dy])
                    if pn[0] < p[0] + forward_eps:
                        continue
                    yL, sL, hL, yR, sR, hR = _walls_at(pn[0])
                    if hL or hR:
                        y_upper = max(yL, yR); y_lower = min(yL, yR)
                        pn[1] = min(max(pn[1], y_lower + WALL_MARGIN), y_upper - WALL_MARGIN)
                    best = (pn, th); break
                if best is None:
                    break  # 정말 불능 시 조기 종료

            p, theta = best
            pts.append(p.copy())
            acc_len += step_d

        return np.asarray(pts, dtype=np.float64)

    # ────────────────────────────────────────────────────────────────────────
    # (옵션) 호 길이 등간격 리샘플: 점 간격까지 고정하고 싶으면 사용
    # ────────────────────────────────────────────────────────────────────────
    def _resample_by_arclen(self, poly: np.ndarray, ds: float, L_target: float) -> np.ndarray:
        if poly is None or poly.shape[0] < 2:
            return poly
        diffs = np.diff(poly, axis=0)
        seg = np.hypot(diffs[:,0], diffs[:,1])
        s = np.concatenate(([0.0], np.cumsum(seg)))
        L = float(s[-1])
        L_use = min(L, L_target)
        n = max(2, int(round(L_use / ds)) + 1)
        target_s = np.linspace(0.0, L_use, n)
        x = np.interp(target_s, s, poly[:,0])
        y = np.interp(target_s, s, poly[:,1])
        return np.stack([x, y], axis=1)

    # ────────────────────────────────────────────────────────────────────────
    # 시각화 헬퍼
    # ────────────────────────────────────────────────────────────────────────
    def _make_opt_path_marker(self, path_xy: np.ndarray, frame_id, stamp) -> Marker:
        mk = Marker()
        mk.header.frame_id = frame_id
        mk.header.stamp = stamp
        mk.ns = 'opt_path'
        mk.id = 600000
        mk.type = Marker.LINE_STRIP
        mk.action = Marker.ADD
        mk.pose.orientation.w = 1.0
        mk.scale.x = 0.18
        mk.color.r, mk.color.g, mk.color.b, mk.color.a = OPT_PATH_COLOR
        mk.lifetime = self.life
        for x, y in path_xy:
            mk.points.append(Point(x=float(x), y=float(y), z=ORIGIN_Z))
        return mk

    def _add_cones_markers(self, arr: MarkerArray, frame_id, stamp,
                           pts, zs, keys):
        # 좌/우/빨강/기타 각각 SPHERE_LIST로 전체 콘 표시
        def add_group(ns, color, idxs, mid):
            if not idxs:
                return
            mk = Marker()
            mk.header.frame_id = frame_id
            mk.header.stamp = stamp
            mk.ns = ns
            mk.id = mid
            mk.type = Marker.SPHERE_LIST
            mk.action = Marker.ADD
            mk.pose.orientation.w = 1.0
            mk.scale.x = mk.scale.y = mk.scale.z = MARKER_SCALE
            mk.color.r, mk.color.g, mk.color.b, mk.color.a = color
            mk.lifetime = self.life
            for i in idxs:
                mk.points.append(Point(x=pts[i][0], y=pts[i][1], z=zs[i]))
            arr.markers.append(mk)

        left_idx  = [i for i,k in enumerate(keys) if k in LEFT_KEYS]
        right_idx = [i for i,k in enumerate(keys) if k in RIGHT_KEYS]
        red_idx   = [i for i,k in enumerate(keys) if k in RED_KEYS]
        other_idx = [i for i,k in enumerate(keys)
                     if (k not in LEFT_KEYS and k not in RIGHT_KEYS and k not in RED_KEYS)]

        add_group('cones_left',  LEFT_COLOR,  left_idx,  500000)
        add_group('cones_right', RIGHT_COLOR, right_idx, 500001)
        add_group('cones_red',   RED_COLOR,   red_idx,   500002)
        add_group('cones_other', OTHER_COLOR, other_idx, 500003)

    # ── 메인 처리
    def _process(self, msg: TrackedConeArray) -> MarkerArray:
        frame_id = (FALLBACK_FRAME if FORCE_FALLBACK_FRAME else
                    (msg.header.frame_id if (USE_HEADER_FRAME and msg.header.frame_id) else FALLBACK_FRAME))
        stamp = self.get_clock().now().to_msg()

        raw_pts: List[Tuple[float, float]] = []
        raw_zs: List[float] = []
        raw_keys: List[str] = []
        for c in msg.cones:
            x, y, z = float(c.position.x), float(c.position.y), float(c.position.z)
            raw_pts.append((x, y)); raw_zs.append(z)
            raw_keys.append(_normalize_color_key(getattr(c, 'color', '')))

        arr = MarkerArray()

        mask = self._sanitize(raw_pts, raw_zs, raw_keys)
        if not mask:
            return arr
        pts  = [raw_pts[i] for i in mask]
        zs   = [raw_zs[i] for i in mask]
        keys = [raw_keys[i] for i in mask]

        # 입력 콘들 “모두” 먼저 시각화 (좌/우/빨강/기타)
        self._add_cones_markers(arr, frame_id, stamp, pts, zs, keys)

        # 그래프 & 좌/우 벽(색상 필터 강제)
        graph = self._build_graph_grid(pts) if len(pts) >= 2 else {}
        left_start  = self._get_start_node(pts, keys, LEFT_KEYS, START_Y_LEFT)
        right_start = self._get_start_node(pts, keys, RIGHT_KEYS, START_Y_RIGHT)
        left_path = self._greedy_path(left_start, graph, pts, keys, LEFT_KEYS, LEFT_KEYS) if (graph and left_start is not None) else []
        right_path = self._greedy_path(right_start, graph, pts, keys, RIGHT_KEYS, RIGHT_KEYS) if (graph and right_start is not None) else []
        if len(left_path) < MIN_PATH_LEN:  left_path = []
        if len(right_path) < MIN_PATH_LEN: right_path = []

        # 헝가리안 페어(좌/우만 사용) 및 중점
        pairs = self._pair_cones(pts, zs, keys) if (ENABLE_PAIRING and len(pts) > 1) else []

        # 좌/우 벽 라인 시각화
        if left_path:
            mk = Marker(); mk.header.frame_id = frame_id; mk.header.stamp = stamp
            mk.ns = 'path'; mk.id = 200000
            mk.type = Marker.LINE_STRIP; mk.action = Marker.ADD
            mk.pose.orientation.w = 1.0; mk.scale.x = 0.12
            mk.color.r, mk.color.g, mk.color.b, mk.color.a = LEFT_COLOR
            mk.points = [Point(x=pts[i][0], y=pts[i][1], z=zs[i]) for i in left_path]
            mk.lifetime = self.life; arr.markers.append(mk)
        if right_path:
            mk = Marker(); mk.header.frame_id = frame_id; mk.header.stamp = stamp
            mk.ns = 'path'; mk.id = 200001
            mk.type = Marker.LINE_STRIP; mk.action = Marker.ADD
            mk.pose.orientation.w = 1.0; mk.scale.x = 0.12
            mk.color.r, mk.color.g, mk.color.b, mk.color.a = RIGHT_COLOR
            mk.points = [Point(x=pts[i][0], y=pts[i][1], z=zs[i]) for i in right_path]
            mk.lifetime = self.life; arr.markers.append(mk)

        # 시작 마커
        if SHOW_START_MARKERS:
            if left_path:
                i = left_path[0]
                mk = Marker(); mk.header.frame_id = frame_id; mk.header.stamp = stamp
                mk.ns = 'start'; mk.id = 300000
                mk.type = Marker.CUBE; mk.action = Marker.ADD
                mk.pose.position.x, mk.pose.position.y, mk.pose.position.z = pts[i][0], pts[i][1], zs[i]
                mk.pose.orientation.w = 1.0
                mk.scale.x = mk.scale.y = mk.scale.z = MARKER_SCALE * 0.6
                mk.color.r, mk.color.g, mk.color.b, mk.color.a = LEFT_COLOR
                mk.lifetime = self.life; arr.markers.append(mk)
            if right_path:
                i = right_path[0]
                mk = Marker(); mk.header.frame_id = frame_id; mk.header.stamp = stamp
                mk.ns = 'start'; mk.id = 300001
                mk.type = Marker.CUBE; mk.action = Marker.ADD
                mk.pose.position.x, mk.pose.position.y, mk.pose.position.z = pts[i][0], pts[i][1], zs[i]
                mk.pose.orientation.w = 1.0
                mk.scale.x = mk.scale.y = mk.scale.z = MARKER_SCALE * 0.6
                mk.color.r, mk.color.g, mk.color.b, mk.color.a = RIGHT_COLOR
                mk.lifetime = self.life; arr.markers.append(mk)

        # 페어 중점/라인
        if pairs:
            if SHOW_PAIR_MIDPOINTS:
                mk = Marker(); mk.header.frame_id = frame_id; mk.header.stamp = stamp
                mk.ns = 'pair_midpoints'; mk.id = 400001
                mk.type = Marker.SPHERE_LIST; mk.action = Marker.ADD
                mk.pose.orientation.w = 1.0
                mk.scale.x = mk.scale.y = mk.scale.z = PAIR_MID_SCALE
                mk.color.r, mk.color.g, mk.color.b, mk.color.a = PAIR_MID_COLOR
                mk.lifetime = self.life
                for _li, _ri, mx, my, mz in pairs:
                    mk.points.append(Point(x=mx, y=my, z=mz))
                arr.markers.append(mk)
            if SHOW_PAIR_LINES:
                mk = Marker(); mk.header.frame_id = frame_id; mk.header.stamp = stamp
                mk.ns = 'pair_lines'; mk.id = 400000
                mk.type = Marker.LINE_LIST; mk.action = Marker.ADD
                mk.pose.orientation.w = 1.0; mk.scale.x = 0.03
                mk.color.r, mk.color.g, mk.color.b, mk.color.a = PAIR_LINE_COLOR
                mk.lifetime = self.life
                for li, ri, *_ in pairs:
                    mk.points.append(Point(x=pts[li][0], y=pts[li][1], z=zs[li]))
                    mk.points.append(Point(x=pts[ri][0], y=pts[ri][1], z=zs[ri]))
                arr.markers.append(mk)

        # ── 스텝퍼 입력
        mids_list = [(mx, my) for *_ij, mx, my, _mz in (pairs or [])
                     if math.isfinite(mx) and math.isfinite(my) and mx >= -1e-6]
        mids_xy = (np.asarray(mids_list, dtype=np.float64)
                   if len(mids_list) > 0 else np.zeros((0, 2), dtype=np.float64))
        obs_list = [pts[i] for i, k in enumerate(keys) if (k in LEFT_KEYS or k in RIGHT_KEYS)]
        obs_xy = (np.asarray(obs_list, dtype=np.float64)
                  if len(obs_list) > 0 else np.zeros((0, 2), dtype=np.float64))
        left_poly  = (np.asarray([pts[i] for i in left_path], dtype=np.float64)
                      if left_path  else np.zeros((0, 2), dtype=np.float64))
        right_poly = (np.asarray([pts[i] for i in right_path], dtype=np.float64)
                      if right_path else np.zeros((0, 2), dtype=np.float64))

        # ── 경로 생성 (스텝퍼, 총 길이 고정)
        opt_path = self._grow_path_with_angle_constraints(mids_xy, left_poly, right_poly, obs_xy)

        # (선택) 점 간격까지 고정하고 싶다면 주석 해제
        # opt_path = self._resample_by_arclen(opt_path, STEP_LEN, PATH_LEN)

        arr.markers.append(self._make_opt_path_marker(opt_path, frame_id, stamp))

        # 이전 경로 저장(원하면 사용)
        self._prev_path_xy = opt_path.copy()
        return arr


def main(args=None):
    rclpy.init(args=args)
    node = ConesColorSubscriber()
    try:
        execu = SingleThreadedExecutor()
        execu.add_node(node)
        execu.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

