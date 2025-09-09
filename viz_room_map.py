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
    room_types = {}   # 房间编号到房型索引的映射，后续用于按照房型着色
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
                        # 仅记录房间编号对应的房型索引，后续不再使用房间编号
                        room_types[rid] = type_idx


    if room_types:
        # 若获取到房型信息，则按房型绘制
        type_map = np.zeros_like(room_map)  # 保存房型索引，0 为背景
        for rid, t_idx in room_types.items():
            # 将房间编号转换为房型索引 +1，避免与背景 0 冲突
            type_map[room_map == rid] = t_idx + 1

        num_colors = len(ROOM_TYPES) + 1  # 包含背景颜色
        base_colors = plt.get_cmap("tab20", num_colors)(np.arange(num_colors))
        base_colors[0] = (0.9, 0.9, 0.9, 1.0)
        cmap = ListedColormap(base_colors)

        # 显示房型分布图，仅关注房型
        plt.imshow(type_map, cmap=cmap, vmin=0, vmax=num_colors - 1)
        plt.colorbar(label="room type")
        # 统计出现的房型索引（跳过 0），只按房型整体标注一次，避免同一房型多处重复标注
        for t in np.unique(type_map):
            if t == 0:
                continue  # 0 为背景，不标注
            ys, xs = np.where(type_map == t)
            if ys.size == 0:
                continue
            y = ys.mean()
            x = xs.mean()
            plt.text(
                x,
                y,
                ROOM_TYPES[t - 1],  # 使用房型名称作为标签
                color="black",
                fontsize=8,
                ha="center",
                va="center",
            )
    else:
        # 若没有房型信息，仍按房间编号展示
        num_colors = int(np.max(room_map)) + 1
        base_colors = plt.get_cmap("tab20", num_colors)(np.arange(num_colors))
        base_colors[0] = (0.9, 0.9, 0.9, 1.0)
        cmap = ListedColormap(base_colors)
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