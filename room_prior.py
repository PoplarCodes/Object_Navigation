# -*- coding: utf-8 -*-
"""
OnlineRoomInfer: 在线启发式房间分割 + 房型（Room Type）投票
适配 Object_Navigation 项目（15 类语义），为长期目标（global goal）提供房间先验热力图。

依赖：
- 必选：numpy
- 可选：cv2、skimage 或 scipy.ndimage（若都缺失，退化为连通域分割，不做“窄通道切割”）

输出：
- self.room_id_map: int32 [H, W]，>=1 表示房间ID，0 表示未知/未探索
- self.room_type_scores: List[Dict]，每个房间一个字典，含 type_probs（7 维）、obj_hits 等
- build_goal_prior(target_obj_id) -> float32 [H, W]，归一化的先验热力图（和目标类别相关）

房型集合：
  0: bedroom
  1: bathroom
  2: kitchen
  3: dining_room
  4: living_room
  5: corridor
  6: other

15 个对象类别（项目内顺序）:
  0 chair, 1 couch, 2 potted_plant, 3 bed, 4 toilet,
  5 tv, 6 dining_table, 7 oven, 8 sink, 9 refrigerator,
  10 book, 11 clock, 12 vase, 13 cup, 14 bottle
"""
import json  # 引入 json 将打分结果写入文件
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import os  # 引入 os 以保存房型概率供可视化
import logging  # 引入 logging 用于输出警告
try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

try:
    from skimage import morphology as sk_morph, measure as sk_measure
    _HAS_SK = True
except Exception:
    _HAS_SK = False

try:
    from scipy import ndimage as ndi  # type: ignore  # 尝试导入 scipy.ndimage 以提供形态学操作
    _HAS_ND = True
except Exception:
    _HAS_ND = False

logger = logging.getLogger(__name__)  # 获取当前模块的 logger

@dataclass
class RoomCfg:
    resolution_m: float = 0.05      # 每像素米数（例：5cm/px）
    door_min_width_m: float = 0.7   # 常见门洞最小宽度
    door_max_width_m: float = 1.0   # 常见门洞最大宽度
    robot_radius_m: float = 0.18    # 规划用机器人半径
    min_room_area_m2: float = 2.0   # 过小区域并回收
    vote_temp: float = 1.0          # 温度参数（平滑房型 softmax）
    decay_explored_in_prior: float = 0.0  # 为 0~1，>0 时会降低已充分探索房间的先验
    default_type_logits: Tuple[float, float, float, float, float, float, float] = (
        0.2, 0.1, 0.15, 0.1, 0.2, 0.15, 0.1
    )  # 无对象证据时的房型默认分布


# 15x7 对象→房型先验权重矩阵（行：对象；列：房型）
# 经验权重，可按数据集微调；要求每行非负，内部再归一化。
_OBJ2ROOM = np.array([
#  bedrm, bath,  kitch, dining, living, corr, other
    [0.05,   0.00,  0.10,  0.35,   0.40,  0.05, 0.05],  # chair
    [0.05,   0.00,  0.05,  0.05,   0.80,  0.02, 0.03],  # couch
    [0.20,   0.00,  0.05,  0.05,   0.50,  0.05, 0.15],  # potted_plant
    [0.90,   0.00,  0.00,  0.00,   0.05,  0.00, 0.05],  # bed
    [0.00,   0.95,  0.00,  0.00,   0.00,  0.00, 0.05],  # toilet
    [0.30,   0.00,  0.05,  0.00,   0.55,  0.00, 0.10],  # tv
    [0.00,   0.00,  0.10,  0.80,   0.05,  0.00, 0.05],  # dining_table
    [0.00,   0.00,  0.95,  0.00,   0.00,  0.00, 0.05],  # oven
    [0.00,   0.45,  0.45,  0.05,   0.00,  0.00, 0.05],  # sink
    [0.00,   0.00,  0.95,  0.00,   0.00,  0.00, 0.05],  # refrigerator
    [0.40,   0.00,  0.05,  0.05,   0.40,  0.00, 0.10],  # book
    [0.15,   0.00,  0.30,  0.05,   0.35,  0.00, 0.15],  # clock
    [0.30,   0.00,  0.10,  0.10,   0.40,  0.00, 0.10],  # vase
    [0.05,   0.00,  0.60,  0.25,   0.05,  0.00, 0.05],  # cup
    [0.05,   0.00,  0.60,  0.25,   0.05,  0.00, 0.05],  # bottle
], dtype=np.float32)


@dataclass
class RoomInfo:
    room_id: int
    pixels: np.ndarray  # 该房间像素坐标的布尔掩码 [H,W]
    area_px: int
    type_probs: np.ndarray  # [7,]
    obj_hits: Dict[int, float] = field(default_factory=dict)
    explored_ratio: float = 0.0  # 可选：若能估计房内已探索比例


class OnlineRoomInfer:
    def __init__(self, cfg: RoomCfg, n_obj_classes: int = 15, dump_dir: str = "tmp"):
        self.cfg = cfg
        self.n_obj = n_obj_classes
        self.dump_dir = dump_dir  # 数据输出的根目录
        self.room_id_map: Optional[np.ndarray] = None   # int32 [H,W]
        self.rooms: List[RoomInfo] = []
        # 归一化对象→房型先验
        row_sum = _OBJ2ROOM.sum(axis=1, keepdims=True) + 1e-6
        self.obj2room = _OBJ2ROOM / row_sum
        # Episode 级房型概率缓存
        self._room_prob_buffers: Dict[int, List[Dict[str, Any]]] = {}
        # Episode 缓冲：按环境记录当前 Episode 的打分链路
        self._episode_buffers: Dict[int, List[Dict]] = {}
        # Episode 计数：用于输出文件命名
        self._episode_ids: Dict[int, int] = {}
        # 记录每个环境上一次写入的环境步，用于防止重复导出
        self._last_env_step_dumped: Dict[int, int] = {}
    # ------------------------- 对外主接口 -------------------------
    def update(self,
               traversible: np.ndarray,   # bool/0-1 可行走
               explored: np.ndarray,      # bool/0-1 已探索
               sem_probs: np.ndarray,     # float [15,H,W] 语义概率/置信度
               env_id: int = 0,  # 环境编号
               env_step: int = 0,  # 环境时间步
               explored_ratio_map: Optional[np.ndarray] = None  # 0~1（可缺省）
               ) -> None:
        """基于当前栅格状态与语义置信度，更新房间分割与房型概率。
        注意：需保证输入尺寸一致，且与长期目标使用的坐标一致。
        """
        # env_step 为 0 时意味着新的 Episode 开始，先将上一 Episode 的缓存写盘
        if env_id not in self._episode_buffers:
            # 第一次见到该环境，初始化缓存与计数
            self._episode_buffers[env_id] = []
            self._episode_ids[env_id] = 0
            self._room_prob_buffers[env_id] = []
        elif env_step == 0:
            # 新 Episode，写出上一 Episode 的概率序列与打分链路
            self._flush_room_probs(env_id)
            self._flush_episode_json(env_id)

        H, W = traversible.shape
        assert explored.shape == (H, W)
        assert sem_probs.shape[1:] == (H, W)
        free = (traversible.astype(np.uint8) > 0) & (explored.astype(np.uint8) > 0)
        # 1) 房间分割
        room_id_map = self._segment_rooms(free)
        self.room_id_map = room_id_map.astype(np.int32)

        # 2) 统计各房间的对象证据
        self.rooms = []
        n_rooms = int(room_id_map.max())
        if n_rooms == 0:
            return

        # 预先把语义概率阈值化/平滑
        sem_soft = np.clip(sem_probs, 0.0, 1.0).astype(np.float32)


        # 预备保存对象到房型打分链路的中间结果
        json_rooms: List[Dict] = []  # 收集所有房间的打分信息

        # 计算用于对象证据统计的膨胀核半径（米→像素），覆盖靠墙/障碍的物体
        dil_r = max(int(0.4 / self.cfg.resolution_m), 1)
        if _HAS_CV2:
            k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                             (2 * dil_r + 1, 2 * dil_r + 1))
        elif _HAS_SK:
            k_dil = sk_morph.disk(dil_r)
        elif _HAS_ND:
            k_dil = np.ones((2 * dil_r + 1, 2 * dil_r + 1), dtype=bool)
        else:
            k_dil = None  # 若无可用库则不做膨胀

        # 计算每个房间的 hits（对象出现证据）
        for rid in range(1, n_rooms + 1):
            mask = (room_id_map == rid)
            area_px = int(mask.sum())
            if area_px == 0:
                continue

            # 将房间地面掩码向外膨胀 ~0.4m，统计贴墙的对象像素
            if k_dil is not None:
                if _HAS_CV2:
                    mask_for_hits = cv2.dilate(mask.astype(np.uint8), k_dil) > 0
                elif _HAS_SK:
                    mask_for_hits = sk_morph.binary_dilation(mask, k_dil)
                else:  # _HAS_ND
                    mask_for_hits = ndi.binary_dilation(mask, structure=k_dil)
            else:
                mask_for_hits = mask

            obj_hits = {}  # 记录每个对象在房间内的出现证据
            for k in range(self.n_obj):
                # 使用膨胀后的掩码累积语义置信度作为对象证据
                v = float((sem_soft[k] * mask_for_hits).sum())
                if v > 0:
                    obj_hits[k] = v

            # 3) 对象投票 → 房型概率
            type_logits = np.zeros(7, dtype=np.float32)
            for k, v in obj_hits.items():
                contrib = float(v) * self.obj2room[k]
                type_logits += contrib  # 累积对象证据转化的房型贡献

            # 4) 几何启发（走廊倾向）
            corr_boost = self._corridor_score(mask)
            if corr_boost > 0:
                type_logits[5] += 0.2 * corr_boost * type_logits.max()

            # 无任何对象证据时，使用预设的默认房型分布作为先验
            if type_logits.sum() <= 0:
                type_logits = np.array(self.cfg.default_type_logits, dtype=np.float32)

            # softmax
            t = max(self.cfg.vote_temp, 1e-3)
            z = type_logits / t
            z = z - z.max()  # 减去最大值，防止 np.exp 溢出
            ex = np.exp(z)
            type_probs = ex / (ex.sum() + 1e-6)
            # 将可能出现的 NaN/Inf 转成 0，避免污染后续计算
            type_probs = np.nan_to_num(type_probs, nan=0.0, posinf=0.0, neginf=0.0)

            explored_ratio = 0.0
            if explored_ratio_map is not None:
                explored_ratio = float((explored_ratio_map * mask).sum() / (mask.sum() + 1e-6))

            self.rooms.append(RoomInfo(
                room_id=rid,
                pixels=mask,
                area_px=area_px,
                type_probs=type_probs,
                obj_hits=obj_hits,
                explored_ratio=explored_ratio,
            ))

            # 将当前房间的打分信息加入列表，用于后续写入 JSON
            json_rooms.append({
                "room_id": rid,
                "obj_hits": {str(k): v for k, v in obj_hits.items()},  # 各对象在房间内的出现证据
                "type_probs": type_probs.tolist(),  # 房型概率分布
            })

        if json_rooms:
            buf = self._episode_buffers.setdefault(env_id, [])
            cur_step = int(env_step)
            # 若与上一次写入的环境步相同，则跳过，防止重复记录
            if self._last_env_step_dumped.get(env_id) != cur_step:
                buf.append({
                    "env_step": cur_step,  # 记录当前环境步编号
                    "rooms": json_rooms,  # 当前步的所有房间得分信息
                })
                # 记录最新写入的环境步
                self._last_env_step_dumped[env_id] = cur_step

        # 保存房间分割结果并缓存房型概率
        if len(self.rooms) > 0:
            type_probs_all = np.stack([r.type_probs for r in self.rooms], axis=0)
            # 缓存当前步的房型概率供 Episode 结束时统一写盘
            self._room_prob_buffers.setdefault(env_id, []).append({"env_step": int(env_step), "probs": type_probs_all})
            ep = self._episode_ids.get(env_id, 0)  # 当前Episode编号
            # 目录结构：<dump_dir>/room_map/thread_<env_id>/eps_<ep+1>/room_map_step<env_step>.npy
            base = os.path.join(self.dump_dir, "room_map", f"thread_{env_id}", f"eps_{ep + 1}")  # Episode编号从1开始
            os.makedirs(base, exist_ok=True)  # 递归创建目录
            np.save(os.path.join(base, f"room_map_step{env_step}.npy"), self.room_id_map)

    def dump_episode_json(self, env_id: int) -> None:
        """主动将某环境当前 Episode 的缓存写入 JSON 文件。"""
        self._flush_room_probs(env_id)  # 先写出房型概率序列
        self._flush_episode_json(env_id)

    def _flush_room_probs(self, env_id: int) -> None:
        """内部工具：写出某环境缓存的房型概率并清空。
        目录结构：<dump_dir>/room_map/thread_<env_id>/eps_<ep+1>/room_probs.npz
        """
        buf = self._room_prob_buffers.get(env_id, [])
        if not buf:
            return  # 缓存为空直接返回
        ep = self._episode_ids.get(env_id, 0)  # 当前 Episode 编号
        base = os.path.join(
            "tmp",
            "room_map",
            os.path.basename(self.dump_dir),
            f"thread_{env_id}",
            f"eps_{ep + 1}",
        )  # 构建输出目录，Episode编号从1开始
        os.makedirs(base, exist_ok=True)  # 确保目录存在
        env_steps = np.array([b["env_step"] for b in buf])  # 收集环境步序列
        # 各步的房型概率矩阵可能因房间数不同而形状不一，需先统一尺寸
        probs_list = [b["probs"] for b in buf]  # 提取概率矩阵列表
        # 计算所有矩阵在各维度上的最大形状
        max_shape = np.max([p.shape for p in probs_list], axis=0)
        padded = []
        for p in probs_list:
            # 按最大形状在缺失维度补零，确保可堆叠
            pad_width = [(0, max_shape[i] - p.shape[i]) for i in range(len(max_shape))]
            padded.append(np.pad(p, pad_width, mode="constant"))
        probs = np.stack(padded, axis=0)  # 统一形状后堆叠概率数组
        np.savez(os.path.join(base, "room_probs.npz"), env_steps=env_steps, probs=probs)  # 写入 npz
        self._room_prob_buffers[env_id] = []  # 清空缓存


    def _flush_episode_json(self, env_id: int) -> None:
        """内部工具：按Episode存盘并清空缓存。
        目录结构：<dump_dir>/episodes/thread_<env_id>/eps_<ep+1>/room_scores.json
        """
        buf = self._episode_buffers.get(env_id, [])
        if not buf:
            return
        ep = self._episode_ids.get(env_id, 0)
        base = os.path.join(
            self.dump_dir,
            "episodes",
            f"thread_{env_id}",
            f"eps_{ep + 1}",
        )  # 按线程和Episode组织目录，Episode编号从1开始
        os.makedirs(base, exist_ok=True)  # 递归创建目录
        path = os.path.join(base, "room_scores.json")  # 固定文件名存储房间得分
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(buf, f, ensure_ascii=False, indent=2)  # 写入Episode数据
        self._episode_buffers[env_id] = []  # 清空当前线程缓存
        self._episode_ids[env_id] = ep + 1  # Episode计数递增
        # 清除最近环境步记录，确保新 Episode 可从 env_step0 重新计数
        if env_id in self._last_env_step_dumped:
            del self._last_env_step_dumped[env_id]

    def build_goal_prior(self, target_obj_id: int) -> np.ndarray:
        """根据目标对象类别，生成整图的房型先验热力图 [H,W]，已归一化。
        规则：目标→房型偏好（obj2room[target]），与各房间的 type_probs 做点乘，
             得到每个房间的权重，再把权重均匀分配到该房间像素。
        可选：对已充分探索的房间施加衰减（decay_explored_in_prior）。
        """
        if self.room_id_map is None or len(self.rooms) == 0:
            return np.zeros((1, 1), dtype=np.float32)  # 调用方应判空
        H, W = self.room_id_map.shape
        prior = np.zeros((H, W), dtype=np.float32)
        pref = self.obj2room[int(target_obj_id)].astype(np.float32)  # [7]
        decay = float(self.cfg.decay_explored_in_prior)

        for r in self.rooms:
            w = float(np.dot(pref, r.type_probs))  # 房间权重
            if decay > 0:
                w = w * (1.0 - decay * np.clip(r.explored_ratio, 0.0, 1.0))
            if w <= 0:
                continue
            mass = w / (r.area_px + 1e-6)
            prior[r.pixels] += mass

        s = prior.sum()
        # 将 NaN/Inf 归零后再归一化，避免污染整图
        prior = np.nan_to_num(prior, nan=0.0, posinf=0.0, neginf=0.0)
        # s = prior.sum()
        if s > 0:
            prior /= s
        return prior

    # ----------------------- 内部：房间分割 ------------------------
    def _segment_rooms(self, free_mask: np.ndarray) -> np.ndarray:
        """把可行走 + 已探索区域切成“房间”。
        方法：
          - 形态学闭运算平滑噪声
          - 以 door_max/2 为半径做一次腐蚀，打断狭窄通道连接
          - 连通域标记得到房间原型；再膨胀回去以贴近原自由空间
          - 过滤过小区域并并入相邻房间
        若无 cv2/skimage/scipy，则退化为单次连通域标记。
        """
        H, W = free_mask.shape
        fm = (free_mask.astype(np.uint8) > 0)

        r_max = max(int(0.5 * self.cfg.door_max_width_m / self.cfg.resolution_m), 1)
        r_min = max(int(0.5 * self.cfg.door_min_width_m / self.cfg.resolution_m), 1)
        min_area_px = int(self.cfg.min_room_area_m2 / (self.cfg.resolution_m ** 2))

        if _HAS_CV2:
            fm_u8 = (fm * 255).astype(np.uint8)
            # 闭运算
            k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            fm_u8 = cv2.morphologyEx(fm_u8, cv2.MORPH_CLOSE, k_close)
            # 腐蚀打断狭窄连接
            k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_max + 1, 2 * r_max + 1))
            eroded = cv2.erode(fm_u8, k_erode)
            # 连通域
            n, labels = cv2.connectedComponents((eroded > 0).astype(np.uint8), connectivity=4)
            # 膨胀回去
            dilated_labels = np.zeros_like(labels, dtype=np.int32)
            k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_max + 1, 2 * r_max + 1))
            for lid in range(1, n):
                mask = (labels == lid).astype(np.uint8) * 255
                dm = cv2.dilate(mask, k_dil) > 0
                dilated_labels[dm] = lid
            labels = dilated_labels
        elif _HAS_SK:
            fm_bool = fm
            fm_bool = sk_morph.binary_closing(fm_bool, sk_morph.disk(1))  # skimage 闭运算
            eroded = sk_morph.binary_erosion(fm_bool, sk_morph.disk(r_max))  # skimage 腐蚀
            labels = sk_measure.label(eroded, background=0, connectivity=1)
            # 膨胀回去
            out = np.zeros_like(labels, dtype=np.int32)
            dil = sk_morph.disk(r_max)
            for lid in range(1, labels.max() + 1):
                dm = sk_morph.binary_dilation(labels == lid, dil)
                out[dm] = lid
            labels = out
        elif _HAS_ND:
            fm_bool = fm
            fm_bool = ndi.binary_closing(fm_bool, structure=np.ones((3, 3), bool))  # scipy 闭运算
            k = np.ones((2 * r_max + 1, 2 * r_max + 1), bool)  # 方形结构元近似门宽
            eroded = ndi.binary_erosion(fm_bool, structure=k)  # scipy 腐蚀切断窄通道
            labels, n = ndi.label(eroded)  # 连通域标记
            labels = labels.astype(np.int32)
            # 膨胀回去
            out = np.zeros_like(labels, dtype=np.int32)
            for lid in range(1, n + 1):
                dm = ndi.binary_dilation(labels == lid, structure=k)
                out[dm] = lid
            labels = out
        else:
            logger.warning("cv2/skimage/scipy 均不可用，房间分割退化，狭窄通道无法切断")  # 警告缺少依赖
            # 退化：直接连通域（不做门洞切割）
            labels = self._label_cc(fm)

        # 过滤过小区域并并入邻居
        labels = self._merge_small_regions(labels, min_area_px)
        # 重新压缩 ID
        labels = self._relabel_compact(labels)
        return labels.astype(np.int32)

    # ---------------------- 内部：几何启发 ------------------------
    def _corridor_score(self, mask: np.ndarray) -> float:
        """粗略走廊评分：细长且窄 -> 分数高。0~1。
        """
        ys, xs = np.where(mask)
        if len(xs) < 25:
            return 0.0
        h = xs.max() - xs.min() + 1
        w = ys.max() - ys.min() + 1
        bbox_area = float(h * w)
        area = float(mask.sum())
        fill_ratio = area / (bbox_area + 1e-6)
        aspect = max(h, w) / (min(h, w) + 1e-6)
        # 细长（aspect 大）且填充率低（像通道）
        score = np.clip((aspect - 2.0) / 6.0, 0.0, 1.0) * np.clip((0.5 - fill_ratio) / 0.5, 0.0, 1.0)
        return float(score)

    # ---------------------- 工具：连通域/压缩 ---------------------
    def _label_cc(self, fm: np.ndarray) -> np.ndarray:
        H, W = fm.shape
        labels = np.zeros((H, W), dtype=np.int32)
        cur = 0
        # 简单 BFS 四连通
        for y in range(H):
            for x in range(W):
                if fm[y, x] and labels[y, x] == 0:
                    cur += 1
                    q = [(y, x)]
                    labels[y, x] = cur
                    while q:
                        yy, xx = q.pop()
                        for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                            ny, nx = yy + dy, xx + dx
                            if 0 <= ny < H and 0 <= nx < W and fm[ny, nx] and labels[ny, nx] == 0:
                                labels[ny, nx] = cur
                                q.append((ny, nx))
        return labels

    def _merge_small_regions(self, labels: np.ndarray, min_area_px: int) -> np.ndarray:
        if labels.max() == 0:
            return labels
        H, W = labels.shape
        out = labels.copy()
        # 统计面积
        areas = np.bincount(out.ravel())
        small_ids = [i for i in range(1, len(areas)) if areas[i] > 0 and areas[i] < min_area_px]
        if not small_ids:
            return out
        # 对每个小区域，查找边界邻居并并入面积最大的邻居
        for sid in small_ids:
            m = (out == sid)
            # 邻居集合
            nbr = set()
            ys, xs = np.where(m)
            for y, x in zip(ys, xs):
                for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W:
                        lab = out[ny, nx]
                        if lab != sid and lab != 0:
                            nbr.add(int(lab))
            if nbr:
                # 选面积最大的邻居
                tgt = max(nbr, key=lambda i: areas[i])
                out[m] = tgt
            else:
                # 无邻居，直接清零
                out[m] = 0
        return out

    def _relabel_compact(self, labels: np.ndarray) -> np.ndarray:
        vals = np.unique(labels)
        vals = vals[vals != 0]
        mp = {int(v): i+1 for i, v in enumerate(vals)}
        out = np.zeros_like(labels, dtype=np.int32)
        for v, i in mp.items():
            out[labels == v] = i
        return out


# -------------------------- 便捷构造函数 --------------------------
def build_online_room_infer_from_args(args, n_obj_classes: int = 15) -> OnlineRoomInfer:
    # 注意：项目里 args.map_resolution 通常是“厘米/像素”，例如 5 表示 5cm/px
    res_m = getattr(args, 'map_resolution', 5.0) / 100.0
    cfg = RoomCfg(
        resolution_m=res_m,
        door_min_width_m=getattr(args, 'door_min_width_m', 0.7),
        door_max_width_m=getattr(args, 'door_max_width_m', 1.5),
        robot_radius_m=getattr(args, 'agent_radius', 0.18),
        min_room_area_m2=getattr(args, 'min_room_area_m2', 2.0),
        vote_temp=getattr(args, 'room_vote_temp', 1.0),
        decay_explored_in_prior=getattr(args, 'room_prior_decay', 0.0),
        default_type_logits=tuple(getattr(
            args,
            'default_type_logits',
            [0.2, 0.1, 0.15, 0.1, 0.2, 0.15, 0.1]
        )),  # 可选指定默认房型分布
    )
    dump_dir = getattr(
        args,
        'dump_dir',
        os.path.join(args.dump_location, "dump", args.exp_name)
    )  # 优先使用自定义 dump_dir，确保与图像保存路径一致
    return OnlineRoomInfer(cfg, n_obj_classes=n_obj_classes, dump_dir=dump_dir)
