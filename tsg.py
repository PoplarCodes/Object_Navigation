# -*- coding: utf-8 -*-
"""
TinyTSG: 轻量级长期语义图，仅用于记录，不干预导航决策
"""

from dataclasses import dataclass, asdict, field
from typing import Dict, Tuple, List, Optional, Any
import os
import json
import time
import numpy as np

# 尝试导入更好的形态学库；若不可用则退化
try:
    import cv2
    _HAS_CV2 = True
except Exception:  # pragma: no cover - 环境可能没有安装
    _HAS_CV2 = False

try:
    from skimage import morphology as sk_morph, measure as sk_measure
    _HAS_SK = True
except Exception:  # pragma: no cover
    _HAS_SK = False


@dataclass
class RoomNode:
    """房间节点，记录房间范围及语义统计信息"""
    room_id: int
    bbox: Tuple[int, int, int, int]                 # (min_r, max_r, min_c, max_c)
    visits: int = 0
    obj_hits: Dict[int, int] = field(default_factory=dict)   # 15 类语义像素计数
    obj_conf: Dict[int, float] = field(default_factory=dict) # EMA 累计值
    room_type: str = "unknown"                      # 简易房间类型
    portals: List[Tuple[int, int]] = field(default_factory=list)  # 入口像素（可选）


@dataclass
class PortalEdge:
    """房间之间的边，记录跨房间次数及入口"""
    u: int
    v: int
    traversable: bool = True
    cross_count: int = 0
    portal_rc: List[Tuple[int, int]] = field(default_factory=list)


class TinyTSG:
    """轻量 TSG：只做记录，不用于导航决策"""
    def __init__(self, ema: float = 0.6):
        self.rooms: Dict[int, RoomNode] = {}
        self.edges: Dict[Tuple[int, int], PortalEdge] = {}
        self.cell2room: Optional[np.ndarray] = None
        self.prev_room: Optional[int] = None
        self._ema = float(ema)

    # ---------- 房间类型启发式 ----------
    @staticmethod
    def _infer_room_type(obj_hits: Dict[int, int]) -> str:
        """根据房间内出现的物体粗略推断房间类型"""
        has = lambda i: obj_hits.get(i, 0) > 0
        if has(3): return "bedroom"                        # bed
        if has(4) or has(8): return "bathroom"             # toilet/sink
        if has(7) or has(9) or has(8) or has(6): return "kitchen"  # oven/fridge/sink/table
        if has(1) or has(5): return "living_room"          # couch/tv
        if has(6) and has(0): return "dining_room"         # table+chair
        return "unknown"

    # ---------- 房间重建（基于 full_map 的 occ/exp） ----------
    def rebuild_rooms_from_fullmap(self, occ_map: np.ndarray, exp_map: np.ndarray):
        """根据全局占据与探索图划分房间"""
        H, W = occ_map.shape
        free = (occ_map == 0) & (exp_map == 1)

        if _HAS_CV2:
            # 使用形态学操作断开狭窄门洞
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            free_smooth = cv2.morphologyEx(free.astype(np.uint8), cv2.MORPH_OPEN, k)
            free_erode = cv2.erode(free_smooth, k, iterations=2)
            num, labels = cv2.connectedComponents(free_erode.astype(np.uint8), connectivity=8)
        elif _HAS_SK:
            se = sk_morph.disk(2)
            free_smooth = sk_morph.opening(free, se)
            free_erode = sk_morph.erosion(sk_morph.erosion(free_smooth, sk_morph.disk(1)))
            labels = sk_measure.label(free_erode.astype(np.uint8), connectivity=2)
        else:
            # 无依赖库时退化为单房间
            labels = np.zeros_like(free, dtype=np.int32)
            if free.any():
                labels[free] = 1

        self.cell2room = labels

        # 更新/新建房间节点
        new_rooms: Dict[int, RoomNode] = {}
        unique_rids = np.unique(labels)
        for rid in unique_rids:
            if rid == 0:
                continue
            ys, xs = np.where(labels == rid)
            if ys.size == 0:
                continue
            bbox = (int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max()))
            if rid in self.rooms:
                node = self.rooms[rid]
                node.bbox = bbox
                new_rooms[rid] = node
            else:
                new_rooms[rid] = RoomNode(room_id=int(rid), bbox=bbox)
        self.rooms = new_rooms

    # ---------- 定位 agent 所在房间 ----------
    def locate_room(self, r: int, c: int) -> Optional[int]:
        """根据栅格坐标定位当前房间"""
        L = self.cell2room
        if L is None:
            return None
        if r < 0 or c < 0 or r >= L.shape[0] or c >= L.shape[1]:
            return None
        rid = int(L[r, c])
        return rid if rid != 0 else None

    # ---------- 语义累计 ----------
    def integrate_semantics(self, rid: Optional[int], sem_local: np.ndarray, thr: float = 0.0):
        """将局部语义图累积到对应房间"""
        if rid is None or rid not in self.rooms:
            return
        node = self.rooms[rid]
        arr = sem_local
        if arr.ndim == 3 and arr.shape[0] == 15:
            arr = np.transpose(arr, (1, 2, 0))  # -> (H,W,15)
        H, W, C = arr.shape
        C = min(C, 15)
        for k in range(C):
            v = float((arr[..., k] > thr).sum())
            if v <= 0:
                continue
            node.obj_hits[k] = int(node.obj_hits.get(k, 0) + v)
            prev = float(node.obj_conf.get(k, 0.0))
            node.obj_conf[k] = float(self._ema * prev + (1.0 - self._ema) * v)
        node.room_type = self._infer_room_type(node.obj_hits)

    # ---------- 连通性（记录跨房间切换） ----------
    def update_connectivity_on_transition(self, curr_rid: Optional[int], rc: Tuple[int, int]):
        """跨房间移动时记录边及入口"""
        if curr_rid is None:
            return
        node = self.rooms.get(curr_rid)
        if node:
            node.visits += 1

        if self.prev_room is None:
            self.prev_room = curr_rid
            return

        if curr_rid != self.prev_room:
            u, v = sorted([self.prev_room, curr_rid])
            key = (int(u), int(v))
            if key not in self.edges:
                self.edges[key] = PortalEdge(
                    u=int(u),
                    v=int(v),
                    traversable=True,
                    cross_count=1,
                    portal_rc=[(int(rc[0]), int(rc[1]))],
                )
            else:
                e = self.edges[key]
                e.cross_count += 1
                e.portal_rc.append((int(rc[0]), int(rc[1])))
            self.prev_room = curr_rid

    # ---------- 序列化 ----------
    def to_dict(self) -> Dict[str, Any]:
        """将 TSG 转为可 JSON 序列化的字典"""
        rooms_out = {}
        for rid, node in self.rooms.items():
            d = asdict(node)
            d["room_id"] = int(d["room_id"])
            d["bbox"] = tuple(int(x) for x in d["bbox"])
            d["obj_hits"] = {int(k): int(v) for k, v in d["obj_hits"].items()}
            d["obj_conf"] = {int(k): float(v) for k, v in d["obj_conf"].items()}
            d["portals"] = [(int(r), int(c)) for (r, c) in d["portals"]]
            rooms_out[int(rid)] = d

        edges_out = []
        for (u, v), e in self.edges.items():
            edges_out.append({
                "u": int(u),
                "v": int(v),
                "traversable": bool(e.traversable),
                "cross_count": int(e.cross_count),
                "portal_rc": [(int(r), int(c)) for (r, c) in e.portal_rc],
            })

        return {
            "meta": {"version": "tinytsg-1.0", "time": int(time.time())},
            "rooms": rooms_out,
            "edges": edges_out,
        }

    def save(self, path: str):
        """将当前 TSG 保存为 JSON 文件"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
