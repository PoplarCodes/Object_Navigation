import numpy as np


def refine_ltg_with_prior(ppo_point: tuple,
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
                          door_band: np.ndarray | None = None,
                          other_room: bool = False,
                          delta: float = 0.3):
    """结合房间先验与多种掩码精炼长期目标点。

    参数:
        ppo_point:    PPO 给出的初始目标点 (x, y)
        prior:        房间先验热力图 [H,W]
        reachable:    可达掩码
        frontier:     前沿掩码
        explore_mask: 探索度掩码（未探索区域权重大）
        revisit_penalty: 回访惩罚掩码
        alpha, beta, gamma: 各项权重指数
        sigma:        高斯分布标准差
        radius:       初始搜索半径
        door_band:    门口环带掩码
        other_room:   房型推理是否指向其他房间
        delta:        门口加成倍率

    返回:
        (x, y) 处理后的长期目标点坐标
    """

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

    def _search_best(H: np.ndarray) -> tuple[int, int]:
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
                w = prior[frontier]
                if w.sum() > 0:
                    idx = np.argmax(w)
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