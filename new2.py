#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compact ConesColorSubscriber
- 최소 파라미터(토픽만), 나머지는 하드코딩된 합리적 기본값
- Grid 기반 근사 반경 그래프(빠름)
- 계층(티어) 각도 게이트 + Greedy 경로 유지
- 좌/우 콘 헝가리안 매칭 + 중점 시각화 (Greedy 폴백 포함)
- RViz 시각화는 경량 모드만 유지
"""

import math
from typing import List, Tuple, Set, Dict, Optional

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
# 하드코딩 기본값 (필요 최소만 조절하고 싶으면 여기 숫자만 바꾸세요)
# ──────────────────────────────────────────────────────────────────────────────
INPUT_TOPIC = '/cone/lidar/ukf'
OUTPUT_TOPIC = '/cones_marker_array'
USE_HEADER_FRAME = True
FALLBACK_FRAME = 'os_sensor'
FORCE_FALLBACK_FRAME = True

PROCESS_HZ = 10.0
REPUBLISH_IDLE_SEC = 0.5
MAX_PROC_MS = 30.0

# Grid 그래프 파라미터
EDGE_LEN_TH = 3.5  # 반경 (m)
MAX_NEIGHBORS_PER_NODE = 12
MAX_STEPS = 1000

# 시작 영역
START_X_RANGE = (-0.5, 5.0)
START_Y_LEFT = (0.1, 5.0)
START_Y_RIGHT = (-5.0, -0.1)

# 각도 티어(간소화)
PARENT_DEG_TIERS = [65.0, 110.0]
GRAND_DEG_TIERS  = [35.0, 90.0]
GLOBAL_DEG_TIERS = [90.0]  # 비우면 x_backtrack 사용
MAX_RELAX_LEVELS = 2
X_BACKTRACK_MARGIN = 0.3
PREFER_GAIN = 0.25  # 기대 색상 선호 가중

# 색상 키/표시 색
LEFT_KEYS  = {'yellow', 'yellowcone'}
RIGHT_KEYS = {'blue', 'bluecone'}
RED_KEYS   = {'red', 'redcone'}
LEFT_COLOR  = (1.0, 1.0, 0.0, 1.0)
RIGHT_COLOR = (0.0, 0.3, 1.0, 1.0)
OTHER_COLOR = (0.6, 0.6, 0.6, 0.9)
RED_COLOR   = (1.0, 0.0, 0.0, 1.0)

# 마커
MARKER_SCALE = 0.45
LIFETIME_SEC = 0.5
MIN_PATH_LEN = 3
SHOW_START_MARKERS = False#True

# 페어링(헝가리안) 간소화 파라미터
ENABLE_PAIRING = True
PAIR_MAX_DIST = 6.0
PAIR_MAX_MATRIX_ELEMS = 8000
SHOW_PAIR_MIDPOINTS = True
SHOW_PAIR_LINES = False
PAIR_MID_SCALE = 0.35
PAIR_MID_COLOR = (1.0, 1.0, 1.0, 0.95)
PAIR_LINE_COLOR = (0.2, 1.0, 0.2, 0.55)

# 기타 가드
DEDUP_EPS = 1e-3


def _normalize_color_key(s: str) -> str:
    if not s:
        return 'unknown'
    return ''.join(s.lower().replace('_', '').split())


def _euclid(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


class ConesColorSubscriber(Node):
    def __init__(self):
        super().__init__('cones_color_subscriber')

        # 토픽만 파라미터 허용 (나머지는 하드코딩)
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

        # 상태
        self._latest_msg: Optional[TrackedConeArray] = None
        self._last_msg_time = self.get_clock().now()
        self._last_arr: Optional[MarkerArray] = None

        self.create_timer(1.0 / max(0.5, PROCESS_HZ), self._on_timer)

        # 미리 계산된 코사인 티어
        self.parent_cos = [math.cos(math.radians(d)) for d in PARENT_DEG_TIERS]
        self.grand_cos  = [math.cos(math.radians(d)) for d in GRAND_DEG_TIERS]
        self.global_cos = [math.cos(math.radians(d)) for d in GLOBAL_DEG_TIERS]
        self.life = Duration(sec=int(LIFETIME_SEC),
                             nanosec=int((LIFETIME_SEC - int(LIFETIME_SEC))*1e9))

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
        # 이웃 수를 거리 기준으로 제한
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

    def _greedy_path(self, start, graph, pts, keys, prefer_keys) -> List[int]:
        if start is None:
            return []
        path = [start]
        seen = {start}
        maxx = max(p[0] for p in pts) if pts else 0.0
        steps = 0
        use_global = len(self.global_cos) > 0
        max_levels = min(MAX_RELAX_LEVELS, max(len(self.parent_cos), len(self.grand_cos), len(self.global_cos) or [1]))
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
                    vx, vy = pts[v][0] - px, pts[v][1] - py
                    dist = math.hypot(vx, vy)
                    if dist <= 0:
                        continue
                    ux, uy = vx/dist, vy/dist
                    # 글로벌 헤딩 또는 x-backtrack
                    if use_global and gcos > -1.0:
                        if ux < gcos:
                            continue
                    elif not use_global:
                        if pts[v][0] < px - X_BACKTRACK_MARGIN:
                            continue
                    # parent/grand 게이트
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

    def _get_start_node(self, pts: List[Tuple[float, float]], y_range: Tuple[float, float]):
        try:
            cands = [(i, x) for i, (x, y) in enumerate(pts)
                     if START_X_RANGE[0] <= x < START_X_RANGE[1] and y_range[0] <= y < y_range[1]]
            return min(cands, key=lambda it: it[1])[0] if cands else None
        except Exception:
            return None

    def _pair_cones(self, pts, zs, keys):
        if not ENABLE_PAIRING:
            return []
        left_idx = [i for i, k in enumerate(keys) if k in LEFT_KEYS]
        right_idx = [i for i, k in enumerate(keys) if k in RIGHT_KEYS]
        if not left_idx or not right_idx:
            return []
        L, R = len(left_idx), len(right_idx)
        # 매트릭스가 크면 Greedy 폴백
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
        # Hungarian
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

        graph = self._build_graph_grid(pts) if len(pts) >= 2 else {}

        # 시작점 & 경로
        left_start  = self._get_start_node(pts, START_Y_LEFT)
        right_start = self._get_start_node(pts, START_Y_RIGHT)
        left_path = self._greedy_path(left_start, graph, pts, keys, LEFT_KEYS) if graph and left_start is not None else []
        right_path = self._greedy_path(right_start, graph, pts, keys, RIGHT_KEYS) if graph and right_start is not None else []
        if len(left_path) < MIN_PATH_LEN:
            left_path = []
        if len(right_path) < MIN_PATH_LEN:
            right_path = []

        #print("l:", left_start)
        #print("r:", right_start)

        # 페어링 + 중점
        pairs = self._pair_cones(pts, zs, keys) if (ENABLE_PAIRING and len(pts) > 1) else []

        # ── 시각화 (경량)
        # 경로 라인
        if left_path:
            mk = Marker()
            mk.header.frame_id = frame_id; mk.header.stamp = stamp
            mk.ns = 'path'; mk.id = 200000
            mk.type = Marker.LINE_STRIP; mk.action = Marker.ADD
            mk.pose.orientation.w = 1.0
            mk.scale.x = 0.12
            mk.color.r, mk.color.g, mk.color.b, mk.color.a = LEFT_COLOR
            mk.points = [Point(x=pts[i][0], y=pts[i][1], z=zs[i]) for i in left_path]
            mk.lifetime = self.life
            arr.markers.append(mk)
        if right_path:
            mk = Marker()
            mk.header.frame_id = frame_id; mk.header.stamp = stamp
            mk.ns = 'path'; mk.id = 200001
            mk.type = Marker.LINE_STRIP; mk.action = Marker.ADD
            mk.pose.orientation.w = 1.0
            mk.scale.x = 0.12
            mk.color.r, mk.color.g, mk.color.b, mk.color.a = RIGHT_COLOR
            mk.points = [Point(x=pts[i][0], y=pts[i][1], z=zs[i]) for i in right_path]
            mk.lifetime = self.life
            arr.markers.append(mk)

        # 경로 위 콘만 구체로 표시(색상 유지 + red override)
        on_path = set(left_path) | set(right_path)
        if on_path:
            mk = Marker()
            mk.header.frame_id = frame_id; mk.header.stamp = stamp
            mk.ns = 'cones_on_path'; mk.id = 1
            mk.type = Marker.SPHERE_LIST; mk.action = Marker.ADD
            mk.pose.orientation.w = 1.0
            mk.scale.x = mk.scale.y = mk.scale.z = MARKER_SCALE
            mk.lifetime = self.life
            for i in sorted(on_path):
                mk.points.append(Point(x=pts[i][0], y=pts[i][1], z=zs[i]))
                if keys[i] in RED_KEYS:
                    c = RED_COLOR
                else:
                    c = LEFT_COLOR if i in set(left_path) else RIGHT_COLOR
                mk.colors.append(ColorRGBA(r=c[0], g=c[1], b=c[2], a=c[3]))
            arr.markers.append(mk)

        # if SHOW_START_MARKERS:
        #     if left_path:
        #         i = left_path[0]
        #         mk = Marker(); mk.header.frame_id = frame_id; mk.header.stamp = stamp
        #         mk.ns = 'start'; mk.id = 300000
        #         mk.type = Marker.CUBE; mk.action = Marker.ADD
        #         mk.pose.position.x, mk.pose.position.y, mk.pose.position.z = pts[i][0], pts[i][1], zs[i]
        #         mk.pose.orientation.w = 1.0
        #         mk.scale.x = mk.scale.y = mk.scale.z = MARKER_SCALE * 0.6
        #         c = RED_COLOR if keys[i] in RED_KEYS else LEFT_COLOR
        #         mk.color.r, mk.color.g, mk.color.b, mk.color.a = c
        #         mk.lifetime = self.life
        #         arr.markers.append(mk)
        #     if right_path:
        #         i = right_path[0]
        #         mk = Marker(); mk.header.frame_id = frame_id; mk.header.stamp = stamp
        #         mk.ns = 'start'; mk.id = 300001
        #         mk.type = Marker.CUBE; mk.action = Marker.ADD
        #         mk.pose.position.x, mk.pose.position.y, mk.pose.position.z = pts[i][0], pts[i][1], zs[i]
        #         mk.pose.orientation.w = 1.0
        #         mk.scale.x = mk.scale.y = mk.scale.z = MARKER_SCALE * 0.6
        #         c = RED_COLOR if keys[i] in RED_KEYS else RIGHT_COLOR
        #         mk.color.r, mk.color.g, mk.color.b, mk.color.a = c
        #         mk.lifetime = self.life
        #         arr.markers.append(mk)

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
                mk.pose.orientation.w = 1.0
                mk.scale.x = 0.03
                mk.color.r, mk.color.g, mk.color.b, mk.color.a = PAIR_LINE_COLOR
                mk.lifetime = self.life
                for li, ri, *_ in pairs:
                    mk.points.append(Point(x=pts[li][0], y=pts[li][1], z=zs[li]))
                    mk.points.append(Point(x=pts[ri][0], y=pts[ri][1], z=zs[ri]))
                arr.markers.append(mk)

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

