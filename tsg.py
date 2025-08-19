# -*- coding: utf-8 -*-
"""
TinyTSG: 轻量级的长期语义图结构，用于在导航过程中记忆房间、物体与连通性。
集成到 agents/utils 下，方便被智能体调用。
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional
import numpy as np
import skimage.morphology as morph
import skimage.measure as measure

# ---- 房间与连通边 ---------------------------------------------------------
@dataclass
class RoomNode:
    """图中每个房间的节点信息"""
    room_id: int
    bbox: Tuple[int, int, int, int]                 # (min_r, max_r, min_c, max_c)
    visits: int = 0
    obj_hits: Dict[int, int] = field(default_factory=dict)  # 15类物体的计数
    obj_conf: Dict[int, float] = field(default_factory=dict) # 置信度累计（可用像素和或EMA）
    room_type: str = "unknown"                      # 简易启发式分类
    portals: List[Tuple[int, int]] = field(default_factory=list)  # 门洞/入口像素（若有）

@dataclass
class PortalEdge:
    """房间之间连通边的信息"""
    u: int
    v: int
    traversable: bool = True
    cross_count: int = 0
    portal_rc: List[Tuple[int, int]] = field(default_factory=list)  # 入口像素簇（可选）

# ---- 整体图 ---------------------------------------------------------------
class TinyTSG:
    """
    一个“够用”的长期记忆图：
      - 房间划分：连通域 + 简单门洞分割
      - 物体记忆：累计每个房间的 15 类目标出现像素
      - 房间类型：根据 obj_hits 的启发式投票
      - 连通性：当 agent 从 A 跨到 B，就在 (A,B) 上加边
    """
    def __init__(self, door_min_px: int = 7, door_max_px: int = 20, ema: float = 0.6):
        # 存储房间和连通信息的核心数据结构
        self.rooms: Dict[int, RoomNode] = {}
        self.edges: Dict[Tuple[int, int], PortalEdge] = {}
        self.cell2room: Optional[np.ndarray] = None
        self.prev_room: Optional[int] = None
        self._ema = ema
        self._door_min_px = door_min_px
        self._door_max_px = door_max_px

    # --------- 房间类型启发式（可按需改） ---------
    @staticmethod
    def _infer_room_type(obj_hits: Dict[int, int]) -> str:
        """根据物体出现情况推断房间类型"""
        # 物体类别 id 约定：0..14（chair..bottle）
        # 简易规则：出现次数>0 即视为存在
        has = lambda i: obj_hits.get(i, 0) > 0
        # 卧室：bed
        if has(3): return "bedroom"
        # 卫生间：toilet 或 sink
        if has(4) or has(8): return "bathroom"
        # 厨房：oven/ fridge/ sink/ dining-table
        if has(7) or has(9) or has(8) or has(6): return "kitchen"
        # 客厅：couch/ tv（+book/vase 可弱提示）
        if has(1) or has(5): return "living_room"
        # 餐厅：dining-table + chairs
        if has(6) and has(0): return "dining_room"
        return "unknown"

    # --------- 基于全图（downsample 后）做房间标注 ----------
    def rebuild_rooms_from_fullmap(self, occ_map: np.ndarray, exp_map: np.ndarray):
        """
        根据全局占据图和探索图重新划分房间
        occ_map: 0=可通行(空地), 1=障碍
        exp_map: 0=未知, 1=已探索
        """
        h, w = occ_map.shape
        free = (occ_map == 0) & (exp_map == 1)

        # 关键：把狭窄通道磨掉，房间自然被断开
        se = morph.disk(2)
        free_smooth = morph.opening(free, se)
        # 腐蚀若干次，抹去小缝/门洞；次数与门洞像素宽度近似相关
        free_erode = morph.erosion(free_smooth, morph.disk(1))
        free_erode = morph.erosion(free_erode, morph.disk(1))

        labels = measure.label(free_erode, connectivity=2)  # 0=背景，1..K 房间
        self.cell2room = labels

        # 更新 rooms 字典（bbox 变了要刷新；已有 obj 统计先保留）
        new_rooms: Dict[int, RoomNode] = {}
        for rid in np.unique(labels):
            if rid == 0: continue
            ys, xs = np.where(labels == rid)
            bbox = (ys.min(), ys.max(), xs.min(), xs.max())
            if rid in self.rooms:
                node = self.rooms[rid]
                node.bbox = bbox
                new_rooms[rid] = node
            else:
                new_rooms[rid] = RoomNode(room_id=rid, bbox=bbox)
        self.rooms = new_rooms

    # --------- 当前 agent 所在房间 id ----------
    def locate_room(self, r: int, c: int) -> Optional[int]:
        """根据栅格坐标定位 agent 所在房间"""
        if self.cell2room is None: return None
        if r < 0 or c < 0 or r >= self.cell2room.shape[0] or c >= self.cell2room.shape[1]:
            return None
        rid = int(self.cell2room[r, c])
        return rid if rid != 0 else None

    # --------- 观测融合：把当前 local_map 的语义像素打到账户 ----------
    def integrate_semantics(self, rid: Optional[int], sem_local: np.ndarray, thr: float = 0.0):
        """
        将局部语义图整合到指定房间的统计量中
        sem_local: (H, W, 15) 或者 (15, H, W) 都可
        thr: 像素置信度阈值
        """
        if rid is None or rid not in self.rooms: return
        node = self.rooms[rid]
        if sem_local.ndim == 3 and sem_local.shape[0] == 15:  # (15,H,W) -> (H,W,15)
            sem_local = np.transpose(sem_local, (1,2,0))
        H, W, C = sem_local.shape
        assert C >= 15

        # 用像素和作为“命中/置信累计”
        for k in range(15):
            v = float((sem_local[..., k] > thr).sum())
            if v <= 0: continue
            node.obj_hits[k] = node.obj_hits.get(k, 0) + int(v)
            prev = node.obj_conf.get(k, 0.0)
            node.obj_conf[k] = self._ema * prev + (1 - self._ema) * v

        # 每次融合后尝试更新房间类型
        node.room_type = self._infer_room_type(node.obj_hits)

    # --------- 连通性更新：当房间切换时加边 ----------
    def update_connectivity_on_transition(self, curr_rid: Optional[int], rc: Tuple[int, int]):
        """根据房间切换更新连通图"""
        if curr_rid is None: return
        node = self.rooms.get(curr_rid)
        if node: node.visits += 1

        if self.prev_room is None:
            self.prev_room = curr_rid
            return

        if curr_rid != self.prev_room:
            u, v = sorted([self.prev_room, curr_rid])
            key = (u, v)
            if key not in self.edges:
                self.edges[key] = PortalEdge(u=u, v=v, traversable=True, cross_count=0, portal_rc=[rc])
            else:
                self.edges[key].cross_count += 1
                self.edges[key].portal_rc.append(rc)
            self.prev_room = curr_rid

    # --------- 基于 TSG 为目标类别选择房间 ----------
    def pick_room_for_goal(self, goal_cat_id: int) -> Optional[Tuple[int, Tuple[int, int]]]:
        """根据目标物体类别选择最有可能的房间及入口"""
        if not self.rooms: return None
        # 简单打分：obj_conf[k] * log(1+visits)（见过越多越可信）
        best = None
        best_score = -1
        for rid, node in self.rooms.items():
            conf = node.obj_conf.get(goal_cat_id, 0.0)
            score = conf * np.log1p(node.visits)
            if score > best_score:
                best_score, best = score, rid
        if best is None or best_score <= 0:
            return None
        # 入口 fallback：若没有记录入口，就用房间 bbox 中心
        entry = None
        # 找与其它房间边上“被穿越最多”的 Portal 当成入口
        cand = [(e.cross_count, e.portal_rc, e.u, e.v) for (u,v), e in self.edges.items() if u==best or v==best]
        if cand:
            cand.sort(reverse=True, key=lambda x: x[0])
            if cand[0][1]:
                entry = cand[0][1][0]
        if entry is None:
            r1,r2,c1,c2 = self.rooms[best].bbox
            entry = ((r1+r2)//2, (c1+c2)//2)
        return best, entry

# --------------------- 前沿（frontier）目标：最大未知区域中心 ---------------------
def pick_frontier_goal(local_map: np.ndarray) -> np.ndarray:
    """
    输入 local_map 的 3 通道：map(障碍), exp(已探索), curr/past 无需
    输出 goal_map (H, W) 二值图：最大未知连通块中心打 1
    """
    # channel 0: map / obstacles  (1=障碍, 0=空地)
    # channel 1: explored (1=已探索, 0=未知)
    occ = (local_map[0] > 0.5)
    exp = (local_map[1] > 0.5)
    unknown_free = (~exp) & (~occ)

    # 取最大连通片
    lbl = measure.label(unknown_free.astype(np.uint8), connectivity=2)
    if lbl.max() == 0:
        return np.zeros_like(local_map[0])   # 没有未知了
    areas = [(lbl==i).sum() for i in range(1, lbl.max()+1)]
    best_id = 1 + int(np.argmax(areas))
    mask = (lbl == best_id)

    # 取该片内的质心（用坐标平均）
    ys, xs = np.where(mask)
    cy, cx = int(np.mean(ys)), int(np.mean(xs))
    goal = np.zeros_like(local_map[0], dtype=np.float32)
    goal[cy, cx] = 1.0
    return goal
