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
        for t in range(probs.shape[2]):
            # 在所有房间中取该房型的最大概率，忽略房间编号
            type_probs = np.max(probs[:, :, t], axis=1)
            # 将低于 0.5 的概率置为 NaN，过滤掉噪声
            filtered = np.where(type_probs >= 0.5, type_probs, np.nan)
            # 若存在有效概率则绘制并仅以房型命名
            if np.any(~np.isnan(filtered)):
                plt.plot(env_steps, filtered, label=ROOM_TYPES[t])
    else:
        # 旧版格式：[步数, 房间数]
        # 取所有房间的最大概率，无法区分房型
        max_probs = np.max(probs, axis=1)
        filtered = np.where(max_probs >= 0.5, max_probs, np.nan)
        if np.any(~np.isnan(filtered)):
            plt.plot(env_steps, filtered, label="room")

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