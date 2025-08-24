import numpy as np
from typing import Optional, Tuple  # 引入类型注解所需的 Optional 与 Tuple

def _refine_core(ppo_point: Tuple[int, int],
                 prior: np.ndarray,
                 reachable: np.ndarray,
                 frontier: np.ndarray,
                 explore_mask: np.ndarray,
                 revisit_penalty: np.ndarray,
                 alpha: float,
                 beta: float,
                 gamma: float,
                 sigma: float,
                 radius: int,
                 door_band: Optional[np.ndarray] = None,
                 other_room: bool = False,
                 delta: float = 0.3) -> Tuple[int, int]:
    """核心实现：结合房间先验与多种掩码精炼长期目标点"""
    x, y = ppo_point

    # -- Step1: 以 (x,y) 为中心构造二维高斯图 G --
    h, w = prior.shape
    yy, xx = np.ogrid[:h, :w]
    G = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))

    # -- Step2: 生成综合掩码 M，结合可达、前沿、探索度与回访惩罚 --
    M = (reachable.astype(np.float32) *
         frontier.astype(np.float32) *
         explore_mask.astype(np.float32) *
         revisit_penalty.astype(np.float32))

    def _norm(x: np.ndarray) -> np.ndarray:
        """对输入矩阵做归一化，避免总和为零导致除零错误"""
        s = x.sum()
        return x / s if s > 0 else x

    def _search_best(H: np.ndarray) -> Tuple[int, int]:
        """在半径 r 的圆盘内寻找最大值，若无则扩大到 2r，再无则退回前沿"""
        yy2, xx2 = np.ogrid[:h, :w]

        def _argmax_in_disk(rad: int):
            mask = (xx2 - x) ** 2 + (yy2 - y) ** 2 <= rad ** 2
            cand = H * mask
            if cand.max() > 0:
                cy, cx = np.unravel_index(cand.argmax(), cand.shape)
                return int(cx), int(cy)
            return None

        best = _argmax_in_disk(radius)
        if best is None:
            best = _argmax_in_disk(radius * 2)
        if best is None:
            f_coords = np.argwhere(frontier)
            if f_coords.size > 0:
                # 取出前沿处的先验值，命名为 prior_vals 避免与外部变量重名
                prior_vals = prior[frontier]
                if prior_vals.sum() > 0:
                    idx = np.argmax(prior_vals)
                    fy, fx = f_coords[idx]
                else:
                    fy, fx = f_coords[0]
                best = (int(fx), int(fy))
            else:
                best = (int(x), int(y))  # 兜底返回 PPO 点
        return best

    # -- Step3: 计算综合得分 H，并在圆盘内寻找最优点 --
    H = _norm((G ** beta) * (prior ** alpha) * (M ** gamma))
    bx, by = _search_best(H)

    # -- Step4: 门口加成逻辑 --
    if door_band is not None and other_room:
        door_frontier = door_band.astype(bool) & frontier.astype(bool)
        if door_frontier.any() and not door_frontier[by, bx]:
            P2 = prior.copy()
            P2[door_band.astype(bool)] *= (1 + delta)
            H2 = _norm((G ** beta) * (P2 ** alpha) * (M ** gamma))
            bx, by = _search_best(H2)

    return bx, by

def refine_ltg_with_prior(point: Tuple[int, int],
                          prior: np.ndarray,
                          masks: dict,
                          room_infer_obj,
                          recent_goals,
                          alpha: float = 1.0,
                          beta: float = 1.0,
                          gamma: float = 1.0,
                          sigma: float = 8.0,
                          radius: int = 10) -> Tuple[int, int]:
    """封装接口：根据先验与多种掩码细化长期目标。

    参数:
        point:        PPO 给出的原始目标点 (x, y)
        prior:        房间先验热力图
        masks:        包含 free/explored/frontier/novelty 的掩码字典
        room_infer_obj: 房型推理器，可用于后续扩展（当前未使用）
        recent_goals:  历史目标列表，用于生成回访惩罚
        alpha,beta,gamma,sigma,radius: 调节各项权重与搜索范围

    返回:
        细化后的 (x, y) 坐标
    """

    free = masks.get('free')
    explored = masks.get('explored')
    frontier = masks.get('frontier')
    novelty = masks.get('novelty')

    reachable = free.astype(np.bool_)
    explore_mask = (~explored).astype(np.float32)
    revisit_penalty = novelty.astype(np.float32)

    bx, by = _refine_core(point, prior, reachable, frontier,
                          explore_mask, revisit_penalty,
                          alpha, beta, gamma, sigma, radius)
    return bx, by