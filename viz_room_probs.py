"""可视化 room_probs.npz 文件"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from constants import ROOM_TYPES  # 引入统一的房型标签

def main() -> None:
    """脚本入口"""
    # 创建参数解析器，接收待可视化的 .npz 文件路径
    parser = argparse.ArgumentParser(description="可视化 room_probs.npz 文件")
    parser.add_argument("input", help="room_probs.npz 文件路径")
    parser.add_argument("--save", action="store_true", help="是否将图像保存为 PNG 文件")
    args = parser.parse_args()

    # 读取 npz 文件，包含步数序列和对应的房间概率矩阵
    data = np.load(args.input)
    env_steps = data["env_steps"]
    probs = data["probs"]

    # 判断概率矩阵的维度以兼容不同版本
    if probs.ndim == 3:
        # 新版格式：[步数, 房间数, 房型数]
        for i in range(probs.shape[1]):
            # 根据最后一步的概率取最大房型作为标签
            type_idx = int(np.argmax(probs[-1, i]))
            room_probs = probs[:, i, type_idx]
            # 将小于 0.5 的概率替换为 NaN，避免绘制这些点
            filtered = np.where(room_probs >= 0.5, room_probs, np.nan)
            # 仅当该房型存在大于 0.5 的概率时才绘制折线
            if np.any(~np.isnan(filtered)):
                plt.plot(env_steps, filtered, label=ROOM_TYPES[type_idx])
    else:
        # 旧版格式：[步数, 房间数]
        for i in range(probs.shape[1]):
            room_probs = probs[:, i]
            filtered = np.where(room_probs >= 0.5, room_probs, np.nan)
            if np.any(~np.isnan(filtered)):
                plt.plot(env_steps, filtered, label=f"room{i}")

    plt.xlabel("step")
    plt.ylabel("probability")
    plt.title("Room Probabilities")
    plt.legend()

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