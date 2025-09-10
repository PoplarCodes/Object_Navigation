# -*- coding: utf-8 -*-
"""可视化 goal_prior_step.npy 文件"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


def main() -> None:
    """脚本入口"""
    # 创建参数解析器，接收待可视化的 .npy 文件路径
    parser = argparse.ArgumentParser(description="可视化 goal_prior_step.npy 文件")
    parser.add_argument("input", help="goal_prior_step.npy 文件路径")
    parser.add_argument("--save", action="store_true", help="是否将图像保存为 PNG 文件")
    args = parser.parse_args()

    # 读取目标先验概率矩阵
    goal_prior = np.load(args.input)

    # 将 NaN 和 无穷值替换为 0，避免热力图失效
    goal_prior = np.nan_to_num(goal_prior, nan=0.0, posinf=0.0, neginf=0.0)

    # 若概率存在负值，整体平移至非负区间
    min_val = goal_prior.min()
    if min_val < 0:
        goal_prior = goal_prior - min_val

    # 对概率矩阵进行归一化，防止数值过小导致显示全橙
    total = goal_prior.sum()
    if total > 0:
        goal_prior = goal_prior / total

    # 找到概率最大的单元格坐标，作为长期目标点
    goal_row, goal_col = np.unravel_index(np.argmax(goal_prior), goal_prior.shape)

    # 为可视化创建副本，并按最大值缩放至 [0,1]
    display_map = goal_prior.copy()
    max_val = display_map.max()
    if max_val > 0:
        display_map = display_map / max_val

    # 用热力图展示概率分布，并锁定显示范围 [0,1]
    plt.imshow(display_map, cmap="hot", vmin=0, vmax=1)
    plt.colorbar(label="probility")
    plt.title("Goal Prior")
    plt.xlabel("x")
    plt.ylabel("y")
    # 在图中用红点标出长期目标位置
    plt.scatter(goal_col, goal_row, c="red", s=50, marker="o", label="LTG")
    plt.legend(loc="upper right")

    # 根据参数决定展示或保存图像
    if args.save:
        # 如果选择保存，则在同目录下以相同文件名保存 PNG 图
        out_path = os.path.splitext(args.input)[0] + ".png"
        plt.savefig(out_path)
    else:
        # 默认行为为直接显示窗口
        plt.show()


if __name__ == "__main__":
    main()