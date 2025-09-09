"""可视化 room_map_step.npy 文件"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def main() -> None:
    """脚本入口"""
    # 创建参数解析器，接收待可视化的 .npy 文件路径
    parser = argparse.ArgumentParser(description="可视化 room_map_step.npy 文件")
    parser.add_argument("input", help="room_map_step.npy 文件路径")
    parser.add_argument("--save", action="store_true", help="是否将图像保存为 PNG 文件")
    args = parser.parse_args()

    # 读取房间映射矩阵，矩阵中的值代表不同的房间编号
    room_map = np.load(args.input)

    # 获取房间编号范围，用于构建颜色映射
    num_colors = int(np.max(room_map)) + 1

    # 构造自定义颜色表：将编号0的颜色改为灰白色作为背景
    base_colors = plt.cm.get_cmap("tab20", num_colors)(np.arange(num_colors))
    base_colors[0] = (0.9, 0.9, 0.9, 1.0)
    cmap = ListedColormap(base_colors)

    # 使用自定义颜色映射展示房间分布，使未标记区域呈灰白色
    plt.imshow(room_map, cmap=cmap, vmin=0, vmax=num_colors - 1)
    plt.colorbar(label="room number")
    plt.title("Room Map")

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