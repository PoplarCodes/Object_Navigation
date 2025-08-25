import numpy as np
from typing import Optional, Tuple  # 引入类型注解所需的 Optional 与 Tuple

def _refine_core(ppo_point: Tuple[int, int],
                 prior: np.ndarray,
                 reachable: np.ndarray,
                 frontier: np.ndarray,
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

    # -- Step2: 生成综合掩码 M，结合可达、前沿与回访惩罚 --
    #    frontier 本身已位于未探索区域，无需额外乘 (~explored)
    M = (reachable.astype(np.float32) *
         frontier.astype(np.float32) *
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
            # 进一步在“可达前沿”中挑选备选点
            fr_coords = np.argwhere(frontier & reachable)
            if fr_coords.size > 0:
                fr_prior = prior[frontier & reachable]
                if fr_prior.sum() > 0:
                    # 有先验时选取先验最大的前沿点
                    idx = np.argmax(fr_prior)
                    fy, fx = fr_coords[idx]
                else:
                    # 先验全零时，选距离 (x,y) 最近的前沿点
                    dists = ((fr_coords - np.array([y, x])) ** 2).sum(axis=1)
                    fy, fx = fr_coords[np.argmin(dists)]
                best = (int(fx), int(fy))
                best = (int(fx), int(fy))

        if best is None:
            # 找不到前沿时退化为搜索最近的可达栅格
            reachable_coords = np.argwhere(reachable)
            if reachable_coords.shape[0] > 0:
                diff_mask = ~((reachable_coords[:, 0] == y) &
                              (reachable_coords[:, 1] == x))
                reachable_diff = reachable_coords[diff_mask]
                if reachable_diff.shape[0] > 0:
                    dists = ((reachable_diff - np.array([y, x])) ** 2).sum(axis=1)
                    ry, rx = reachable_diff[np.argmin(dists)]
                    best = (int(rx), int(ry))
        if best is None:
            # 若仍无备选，只能返回原点以防解包错误（理论上极少出现）
            best = (int(x), int(y))
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

    # -- Step5: 若选中点不在前沿上，则吸附到最近的可达前沿 --
    if not frontier[by, bx]:
        fr_coords = np.argwhere(frontier & reachable)
        if fr_coords.shape[0] > 0:
            dists = ((fr_coords - np.array([by, bx])) ** 2).sum(axis=1)
            by2, bx2 = fr_coords[np.argmin(dists)]
            bx, by = int(bx2), int(by2)
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
                          radius: int = 20,
                          revisit_radius: float = 5.0) -> Tuple[int, int]:
    """封装接口：根据先验与多种掩码细化长期目标。

    参数:
        point:        PPO 给出的原始目标点 (x, y)
        prior:        房间先验热力图
        masks:        包含 free/explored/frontier 的掩码字典
        room_infer_obj: 房型推理器，可用于后续扩展（当前未使用）
        recent_goals:  历史目标列表，用于生成回访惩罚
        alpha,beta,gamma,sigma,radius: 调节各项权重与搜索范围

    返回:
        细化后的 (x, y) 坐标
    """

    free = masks.get('free')
    explored = masks.get('explored')
    frontier = masks.get('frontier')
    # -- 根据历史目标生成回访惩罚 --
    novelty = np.ones_like(free, dtype=np.float32)
    if len(recent_goals) > 0:
        yy, xx = np.ogrid[:free.shape[0], :free.shape[1]]
        for px, py in recent_goals:
            # R(x,y) = 0 当与历史目标距离不超过 revisit_radius，避免目标反复跳动
            mask = (xx - px) ** 2 + (yy - py) ** 2 <= (revisit_radius ** 2)
            novelty[mask] = 0.0

    reachable = free.astype(np.bool_)
    # 回访惩罚项取值 [0,1]，取 0 时表示该区域近期已访问
    revisit_penalty = novelty.astype(np.float32)

    bx, by = _refine_core(point, prior, reachable, frontier,
                          revisit_penalty,
                          alpha, beta, gamma, sigma, radius)
    return bx, by