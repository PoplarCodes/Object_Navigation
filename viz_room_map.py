"""可视化 room_map_step.npy 文件"""

import argparse
import os
import re  # 正则解析步数
import json  # 读取房间标签
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from constants import ROOM_TYPES  # 房型标签统一来源于 constants 模块


def main() -> None:
    """脚本入口"""
    # 创建参数解析器，接收待可视化的 .npy 文件路径
    parser = argparse.ArgumentParser(description="可视化 room_map_step.npy 文件")
    parser.add_argument("input", help="room_map_step.npy 文件路径")
    parser.add_argument("--scores", help="同步的 room_scores.json 路径，可用于显示房型标签", default=None)
    parser.add_argument("--save", action="store_true", help="是否将图像保存为 PNG 文件")
    args = parser.parse_args()

    # 读取房间映射矩阵，矩阵中的值代表不同的房间编号
    room_map = np.load(args.input)

    # 尝试从 room_scores.json 读取对应步的房型标签
    room_labels = {}
    if args.scores and os.path.exists(args.scores):
        # 从文件名解析出当前环境步数，兼容带 env 的文件名
        # 正则兼容前缀中可能出现的 env 信息
        m = re.search(r"room_map_(?:env\d+_)?step(\d+)", os.path.basename(args.input))  # 兼容含 env 前缀的文件名
        step = int(m.group(1)) if m else None  # 若未匹配到则为 None
        with open(args.scores, "r", encoding="utf-8") as f:
            score_data = json.load(f)
        if step is not None:
            # 在 JSON 中查找对应 env_step（可能为字符串），未匹配则忽略
            target = None
            for entry in score_data:
                es = entry.get("env_step", entry.get("step"))
                if es is not None and int(es) == step:
                    target = entry
                    break
            if target:
                for r in target.get("rooms", []):
                    rid = int(r.get("room_id", -1))
                    type_idx = int(r.get("type_label", -1))
                    if rid > 0 and 0 <= type_idx < len(ROOM_TYPES):
                        # 标签采用“编号:类型”格式，便于对照
                        room_labels[rid] = f"{rid}:{ROOM_TYPES[type_idx]}"
    # 获取房间编号范围，用于构建颜色映射
    num_colors = int(np.max(room_map)) + 1

    # 构造自定义颜色表：将编号0的颜色改为灰白色作为背景
    # 使用新的 matplotlib API 获取颜色映射，避免弃用警告
    base_colors = plt.get_cmap("tab20", num_colors)(np.arange(num_colors))
    base_colors[0] = (0.9, 0.9, 0.9, 1.0)
    cmap = ListedColormap(base_colors)

    # 使用自定义颜色映射展示房间分布，使未标记区域呈灰白色
    plt.imshow(room_map, cmap=cmap, vmin=0, vmax=num_colors - 1)
    plt.colorbar(label="room number")
    plt.title("Room Map")

    # 在房间中心位置标注房型标签
    for rid, label in room_labels.items():
        ys, xs = np.where(room_map == rid)
        if ys.size == 0:
            continue
        y = ys.mean()
        x = xs.mean()
        plt.text(
            x,
            y,
            f"{rid}:{label}",  # 同时显示房间编号和房型
            color="black",
            fontsize=8,
            ha="center",
            va="center",
        )

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