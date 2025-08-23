from collections import deque, defaultdict
import os
import logging
import time
import json
import gym
import torch.nn as nn
import torch
import numpy as np

from model import RL_Policy, Semantic_Mapping
from utils.storage import GlobalRolloutStorage
from envs import make_vec_envs
from arguments import get_args
import algo
import matplotlib.pyplot as plt  # 导入 Matplotlib 库
from skimage.morphology import binary_dilation, binary_erosion, disk  # 形态学操作用于前沿提取
from room_prior import build_online_room_infer_from_args  # 引入房间先验推理器构建函数

os.environ["OMP_NUM_THREADS"] = "1"

def sample_goal_by_room(prior: np.ndarray, frontier: np.ndarray, room_infer_obj,
                        fallback_goal, last_room_id: int, hold_steps: int,
                        min_hold_steps: int, switch_ratio: float, topk: int,
                        band_radius_m: float):
    """依据房间先验、前沿与持有策略选择长期目标点。
    参数:
        prior:         归一化先验热力图
        frontier:      当前的前沿掩码
        room_infer_obj:房间推理器对象，提供房间像素集合
        fallback_goal: 无法使用房间信息时回退的目标
        last_room_id:  上一次选择的房间编号，-1 表示无
        hold_steps:    当前房间已持续的步数
        min_hold_steps:在此步数前若新房间优势不明显则保持旧房间
        switch_ratio:  新房间权重需超过旧房间的倍数阈值
        topk:          仅考虑概率最高的前 k 个房间，0 表示全部
                band_radius_m: 膨胀半径（米），用于在房间边界附近寻找前沿

    返回:
        (x, y), 选定房间编号, 更新后的 hold_steps
    """
    if room_infer_obj is None or len(room_infer_obj.rooms) == 0:
        return fallback_goal, last_room_id, hold_steps

    # 解锁阈值：三倍最小持有步数或 60 步，取较大值
    unlock_thresh = max(3 * min_hold_steps, 60)
    # 若仅存在 1 个房间且已停留超过阈值，则直接回退到全局策略
    if len(room_infer_obj.rooms) <= 1 and hold_steps >= unlock_thresh:
        return fallback_goal, last_room_id, 0


    # 计算各房间的累计概率作为权重
    room_probs = np.array([
        prior[r.pixels].sum() for r in room_infer_obj.rooms
    ], dtype=np.float32)

    # 仅保留概率最高的 topk 个房间，其余权重置零
    if topk > 0 and len(room_probs) > topk:
        top_idx = np.argsort(room_probs)[::-1][:topk]
        mask_topk = np.zeros_like(room_probs)
        mask_topk[top_idx] = 1
        room_probs = room_probs * mask_topk

    total = float(room_probs.sum())
    if total <= 0:
        return fallback_goal, last_room_id, hold_steps

    # 选择概率最大的房间作为候选
    best_rid = int(room_probs.argmax())
    chosen_rid = best_rid

    # 判断是否需要切换房间
    if last_room_id is not None and last_room_id >= 0 and last_room_id < len(room_probs):
        cur_w = room_probs[last_room_id]
        new_w = room_probs[best_rid]
        if best_rid != last_room_id and new_w <= cur_w * switch_ratio \
                and hold_steps < min_hold_steps:
            chosen_rid = last_room_id  # 持有旧房间
        else:
            hold_steps = 0  # 触发切换，重置计数

    mask = room_infer_obj.rooms[chosen_rid].pixels

    # 1) 构造“门宽尺度”的边界环带
    r_min_px = max(int(0.5 * room_infer_obj.cfg.door_min_width_m /
                       room_infer_obj.cfg.resolution_m), 1)
    inner = binary_erosion(mask, disk(r_min_px))
    outer = binary_dilation(mask, disk(r_min_px))
    door_band = outer & (~inner)

    # 2) 优先在门宽带与前沿的交集中采样
    door_frontier = frontier & door_band
    if door_frontier.any():
        w = prior[door_frontier]
        coords = np.argwhere(door_frontier)
        w_sum = w.sum()
        if w_sum > 0:
            # 按先验权重采样门口前沿点
            w = w / w_sum
            y, x = coords[np.random.choice(len(coords), p=w)]
        else:
            # 若权重全为 0，则均匀采样
            y, x = coords[np.random.choice(len(coords))]
    else:
        # 兜底：回到原来的膨胀带前沿采样
        dil_px = max(int(band_radius_m / room_infer_obj.cfg.resolution_m), 1)
        band_frontier = frontier & binary_dilation(mask, disk(dil_px))
        if band_frontier.any():
            w = prior[band_frontier]
            coords = np.argwhere(band_frontier)
            w_sum = w.sum()
            if w_sum > 0:
                # 按先验权重采样膨胀带前沿点
                w = w / w_sum
                y, x = coords[np.random.choice(len(coords), p=w)]
            else:
                # 若权重全为 0，则均匀采样
                y, x = coords[np.random.choice(len(coords))]
        else:
            # 再兜底：房间内部最大值/质心
            sub_prior = prior * mask
            if sub_prior.sum() > 0:
                y, x = np.unravel_index(sub_prior.argmax(), sub_prior.shape)
            else:
                coords = np.argwhere(mask)
                if coords.size == 0:
                    return fallback_goal, chosen_rid, hold_steps
                y, x = coords.mean(axis=0).astype(int)

    return (int(x), int(y)), chosen_rid, hold_steps

def main():
    args = get_args()


    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Setup Logging
    log_dir = "{}/models/{}/".format(args.dump_location, args.exp_name)
    dump_dir = "{}/dump/{}/".format(args.dump_location, args.exp_name)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    room_prior_dir = os.path.join(dump_dir, "room_map")
    os.makedirs(room_prior_dir, exist_ok=True)

    logging.basicConfig(
        filename=log_dir + 'train.log',
        level=logging.INFO)
    print("Dumping at {}".format(log_dir))
    print(args)
    logging.info(args)

    # 单独的信息日志文件（存放在当前实验目录下），用于记录每个 Episode 的统计
    info_log_file = os.path.join(log_dir, 'info.log')  # tmp/models/实验名/info.log
    os.makedirs(os.path.dirname(info_log_file), exist_ok=True)
    total_episodes = 0  # 已运行的 Episode 总数
    success_episodes = 0  # 成功的 Episode 数

    # Logging and loss variables
    num_scenes = args.num_processes  # 并行场景的数量
    num_episodes = int(args.num_eval_episodes)
    device = args.device = torch.device("cuda:0" if args.cuda else "cpu")

    g_masks = torch.ones(num_scenes).float().to(device)  # 全 1 的张量，用于记录每个场景的掩码信息

    best_g_reward = -np.inf

    # 为1表示处于评估模式
    if args.eval:
        episode_success = []
        episode_spl = []
        episode_dist = []
        for _ in range(args.num_processes):
            episode_success.append(deque(maxlen=num_episodes))
            episode_spl.append(deque(maxlen=num_episodes))
            episode_dist.append(deque(maxlen=num_episodes))

    # false 0为训练模式
    else:
        episode_success = deque(maxlen=1000)
        episode_spl = deque(maxlen=1000)
        episode_dist = deque(maxlen=1000)


    # 进程完成状态
    finished = np.zeros((args.num_processes))
    wait_env = np.zeros((args.num_processes))

    g_episode_rewards = deque(maxlen=1000)

    g_value_losses = deque(maxlen=1000)
    g_action_losses = deque(maxlen=1000)
    g_dist_entropies = deque(maxlen=1000)

    per_step_g_rewards = deque(maxlen=1000)

    # 记录长期目标所选房间及其持续步数
    last_room_ids = [-1 for _ in range(num_scenes)]  # 上一次的房间编号
    goal_hold_steps = [0 for _ in range(num_scenes)]  # 当前房间已持有的步数

    g_process_rewards = np.zeros((num_scenes))

    # 初始化房间先验推理器，每个环境一个实例
    room_infer = [build_online_room_infer_from_args(args, n_obj_classes=15)
                  for _ in range(num_scenes)]
    """
    开始环境：创建并行的模拟环境，加载场景和智能体
    返回环境对象 envs，其初始状态包括观察数据 obs 和环境信息 infos
    """
    # Starting environments
    torch.set_num_threads(1)
    envs = make_vec_envs(args)
    obs, infos = envs.reset()

    torch.set_grad_enabled(False)

    # Initialize map variables:
    # Full map consists of multiple channels containing the following:
    # 1. Obstacle Map
    # 2. Exploread Area
    # 3. Current Agent Location
    # 4. Past Agent Locations
    # 5,6,7,.. : Semantic Categories
    nc = args.num_sem_categories + 4  # num channels

    # Calculating full and local map sizes
    map_size = args.map_size_cm // args.map_resolution
    full_w, full_h = map_size, map_size
    local_w = int(full_w / args.global_downscaling)
    local_h = int(full_h / args.global_downscaling)

    # Initializing full and local map
    full_map = torch.zeros(num_scenes, nc, full_w, full_h).float().to(device)
    local_map = torch.zeros(num_scenes, nc, local_w,
                            local_h).float().to(device)

    # Initial full and local pose
    full_pose = torch.zeros(num_scenes, 3).float().to(device)
    local_pose = torch.zeros(num_scenes, 3).float().to(device)

    # Origin of local map
    origins = np.zeros((num_scenes, 3))

    # Local Map Boundaries
    lmb = np.zeros((num_scenes, 4)).astype(int)

    # Planner pose inputs has 7 dimensions
    # 1-3 store continuous global agent location
    # 4-7 store local map boundaries
    planner_pose_inputs = np.zeros((num_scenes, 7))

    # 计算局部地图边界
    def get_local_map_boundaries(agent_loc, local_sizes, full_sizes):
        loc_r, loc_c = agent_loc
        local_w, local_h = local_sizes
        full_w, full_h = full_sizes

        if args.global_downscaling > 1:
            gx1, gy1 = loc_r - local_w // 2, loc_c - local_h // 2
            gx2, gy2 = gx1 + local_w, gy1 + local_h
            if gx1 < 0:
                gx1, gx2 = 0, local_w
            if gx2 > full_w:
                gx1, gx2 = full_w - local_w, full_w

            if gy1 < 0:
                gy1, gy2 = 0, local_h
            if gy2 > full_h:
                gy1, gy2 = full_h - local_h, full_h
        else:
            gx1, gx2, gy1, gy2 = 0, full_w, 0, full_h

        return [gx1, gx2, gy1, gy2]

    # 初始化完整地图和完整姿态
    def init_map_and_pose():
        full_map.fill_(0.)
        full_pose.fill_(0.)
        full_pose[:, :2] = args.map_size_cm / 100.0 / 2.0  # 智能体初始位置为地图的中心

        locs = full_pose.cpu().numpy()
        planner_pose_inputs[:, :3] = locs  # 将智能体的完整姿态信息的前三维（x,y,朝向）给规划器的输入信息
        for e in range(num_scenes):
            r, c = locs[e, 1], locs[e, 0]  # 提取当前场景中智能体的xy
            loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                            int(c * 100.0 / args.map_resolution)]

            full_map[e, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0

            lmb[e] = get_local_map_boundaries((loc_r, loc_c),
                                              (local_w, local_h),
                                              (full_w, full_h))

            planner_pose_inputs[e, 3:] = lmb[e]
            origins[e] = [lmb[e][2] * args.map_resolution / 100.0,
                          lmb[e][0] * args.map_resolution / 100.0, 0.]

        for e in range(num_scenes):
            local_map[e] = full_map[e, :,
                           lmb[e, 0]:lmb[e, 1],
                           lmb[e, 2]:lmb[e, 3]]
            local_pose[e] = full_pose[e] - \
                            torch.from_numpy(origins[e]).to(device).float()

    def init_map_and_pose_for_env(e):
        full_map[e].fill_(0.)
        full_pose[e].fill_(0.)
        full_pose[e, :2] = args.map_size_cm / 100.0 / 2.0

        locs = full_pose[e].cpu().numpy()
        planner_pose_inputs[e, :3] = locs
        r, c = locs[1], locs[0]
        loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                        int(c * 100.0 / args.map_resolution)]

        full_map[e, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0

        lmb[e] = get_local_map_boundaries((loc_r, loc_c),
                                          (local_w, local_h),
                                          (full_w, full_h))

        planner_pose_inputs[e, 3:] = lmb[e]
        origins[e] = [lmb[e][2] * args.map_resolution / 100.0,
                      lmb[e][0] * args.map_resolution / 100.0, 0.]

        local_map[e] = full_map[e, :, lmb[e, 0]:lmb[e, 1], lmb[e, 2]:lmb[e, 3]]
        local_pose[e] = full_pose[e] - \
                        torch.from_numpy(origins[e]).to(device).float()

    # 更新内在奖励
    def update_intrinsic_rew(e):
        prev_explored_area = full_map[e, 1].sum(1).sum(0)  # 计算之前探索的区域面积，网格数
        full_map[e, :, lmb[e, 0]:lmb[e, 1], lmb[e, 2]:lmb[e, 3]] = \
            local_map[e]  # 环境的局部地图更新到完整地图对应的局部区域，以反映智能体在局部区域的最新探索情况
        curr_explored_area = full_map[e, 1].sum(1).sum(0)  # 当前探索的区域总网格数

        # 内在奖励定义为当前探索区域面积减去之前探索的区域面积
        intrinsic_rews[e] = curr_explored_area - prev_explored_area
        intrinsic_rews[e] *= (args.map_resolution / 100.) ** 2  # to m^2  内在奖励从网格数转换为实际的面积

    # 调用函数初始化
    init_map_and_pose()

    # 定义全局策略的观测空间
    # 智能体在与环境交互时，会从这个观测空间中获取信息，以便做出决策
    ngc = 8 + args.num_sem_categories  # 通道数 num_sem_categories = 16
    es = 2
    g_observation_space = gym.spaces.Box(0, 1,
                                         (ngc,
                                          local_w,
                                          local_h), dtype='uint8')

    # Global policy action space
    g_action_space = gym.spaces.Box(low=0.0, high=0.99,
                                    shape=(2,), dtype=np.float32)

    # 设置全局策略中循环层的隐藏层大小
    g_hidden_size = args.global_hidden_size  # global_hidden_size = 256

    # 初始化语义地图模块
    sem_map_module = Semantic_Mapping(args).to(device)
    sem_map_module.eval()  # 设置为评估模式

    # Global policy
    g_policy = RL_Policy(g_observation_space.shape, g_action_space,
                         model_type=1,
                         base_kwargs={'recurrent': args.use_recurrent_global,
                                      'hidden_size': g_hidden_size,
                                      'num_sem_categories': ngc - 8
                                      }).to(device)
    g_agent = algo.PPO(g_policy, args.clip_param, args.ppo_epoch,
                       args.num_mini_batch, args.value_loss_coef,
                       args.entropy_coef, lr=args.lr, eps=args.eps,
                       max_grad_norm=args.max_grad_norm)

    global_input = torch.zeros(num_scenes, ngc, local_w, local_h)
    global_orientation = torch.zeros(num_scenes, 1).long()
    intrinsic_rews = torch.zeros(num_scenes).to(device)
    extras = torch.zeros(num_scenes, 2)

    # Storage
    g_rollouts = GlobalRolloutStorage(args.num_global_steps,
                                      num_scenes, g_observation_space.shape,
                                      g_action_space, g_policy.rec_state_size,
                                      es).to(device)

    if args.load != "0":
        print("Loading model {}".format(args.load))
        state_dict = torch.load(args.load,
                                map_location=lambda storage, loc: storage)
        g_policy.load_state_dict(state_dict)

    if args.eval:
        g_policy.eval()

    # Predict semantic map from frame 1
    poses = torch.from_numpy(np.asarray(
        [infos[env_idx]['sensor_pose'] for env_idx in range(num_scenes)])
    ).float().to(device)

    _, local_map, _, local_pose = \
        sem_map_module(obs, poses, local_map, local_pose)

    # Compute Global policy input
    locs = local_pose.cpu().numpy()
    global_input = torch.zeros(num_scenes, ngc, local_w, local_h)
    global_orientation = torch.zeros(num_scenes, 1).long()

    for e in range(num_scenes):
        r, c = locs[e, 1], locs[e, 0]
        loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                        int(c * 100.0 / args.map_resolution)]

        local_map[e, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.
        global_orientation[e] = int((locs[e, 2] + 180.0) / 5.)

    global_input[:, 0:4, :, :] = local_map[:, 0:4, :, :].detach()
    global_input[:, 4:8, :, :] = nn.MaxPool2d(args.global_downscaling)(
        full_map[:, 0:4, :, :])
    global_input[:, 8:, :, :] = local_map[:, 4:, :, :].detach()
    #
    # 修改
    goal_cat_id = torch.from_numpy(np.asarray(
        [infos[env_idx]['goal_cat_id'] for env_idx
         in range(num_scenes)]))
    goal_cat_id_np = goal_cat_id.cpu().numpy()
    # 更新房间先验推理器，保证与长期目标使用的坐标一致
    for e in range(num_scenes):
        traversible = (local_map[e, 0].cpu().numpy() == 0)
        explored = (local_map[e, 1].cpu().numpy() > 0)
        sem_probs = local_map[e, 4:4 + args.num_sem_categories].cpu().numpy()
        explored_ratio_map = local_map[e, 1].cpu().numpy()  # 传入已探索比例地图，供房间先验衰减使用
        room_infer[e].update(traversible, explored, sem_probs,
                             env_id=e,
                             env_step=int(infos[e]['time']),
                             explored_ratio_map=explored_ratio_map)  # 使用环境时间步记录，保存房型概率供可视化

    extras = torch.zeros(num_scenes, 2)
    extras[:, 0] = global_orientation[:, 0]
    extras[:, 1] = goal_cat_id

    g_rollouts.obs[0].copy_(global_input)
    g_rollouts.extras[0].copy_(extras)

    # Run Global Policy (global_goals = Long-Term Goal)
    g_value, g_action, g_action_log_prob, g_rec_states = \
        g_policy.act(
            g_rollouts.obs[0],
            g_rollouts.rec_states[0],
            g_rollouts.masks[0],
            extras=g_rollouts.extras[0],
            deterministic=False
        )

    cpu_actions = nn.Sigmoid()(g_action).cpu().numpy()
    global_goals = [[int(action[0] * local_w), int(action[1] * local_h)]
                    for action in cpu_actions]
    global_goals = [[min(x, int(local_w - 1)), min(y, int(local_h - 1))]
                    for x, y in global_goals]

    goal_maps = [np.zeros((local_w, local_h)) for _ in range(num_scenes)]
    found_goal = [0 for _ in range(num_scenes)]

    for e in range(num_scenes):
        cn = infos[e]['goal_cat_id'] + 4
        if local_map[e, cn, :, :].sum() != 0.:
            cat_semantic_map = local_map[e, cn, :, :].cpu().numpy()
            cat_semantic_scores = cat_semantic_map
            cat_semantic_scores[cat_semantic_scores > 0] = 1.
            goal_maps[e] = cat_semantic_scores
            found_goal[e] = 1
        elif getattr(args, 'use_room_prior', False):
            prior = room_infer[e].build_goal_prior(int(goal_cat_id_np[e]))


            # 计算前沿掩码：free & dilate(explored) & ~explored
            free = (local_map[e, 0].cpu().numpy() == 0)
            explored = (local_map[e, 1].cpu().numpy() > 0)
            frontier = free & binary_dilation(explored, disk(1)) & (~explored)

            # 先对全局策略输出加入扰动，作为像素级采样的后备方案
            gx, gy = global_goals[e]
            shift = np.random.randint(-1, 2, size=2)
            gx = int(np.clip(gx + shift[0], 0, local_w - 1))
            gy = int(np.clip(gy + shift[1], 0, local_h - 1))
            fallback = (gx, gy)
            if prior.shape == goal_maps[e].shape and prior.sum() > 0:
                # 房间级采样：考虑房间持有策略与前沿
                (gx, gy), last_room_ids[e], goal_hold_steps[e] = \
                    sample_goal_by_room(prior, frontier, room_infer[e], fallback,
                                        last_room_ids[e], goal_hold_steps[e],
                                        args.min_goal_hold_steps,
                                        args.goal_switch_ratio,
                                        args.room_prior_topk,
                                        args.frontier_band_radius_m)
            else:
                gx, gy = fallback
            goal_maps[e][:, :] = 0
            goal_maps[e][gx, gy] = 1
            goal_hold_steps[e] += 1
        else:
            gx, gy = global_goals[e]
            # 对全局策略输出位置加入随机扰动进行概率采样，鼓励探索
            shift = np.random.randint(-1, 2, size=2)
            gx = int(np.clip(gx + shift[0], 0, local_w - 1))
            gy = int(np.clip(gy + shift[1], 0, local_h - 1))
            goal_maps[e][gx, gy] = 1

    planner_inputs = [{} for e in range(num_scenes)]
    for e, p_input in enumerate(planner_inputs):
        p_input['map_pred'] = local_map[e, 0, :, :].cpu().numpy()
        p_input['exp_pred'] = local_map[e, 1, :, :].cpu().numpy()
        p_input['pose_pred'] = planner_pose_inputs[e]
        p_input['goal'] = goal_maps[e]  # global_goals[e]
        # p_input['new_goal'] = 1  # 原逻辑：初始化阶段总是触发新目标
        p_input['new_goal'] = False  # 初始化时不触发新目标
        p_input['found_goal'] = found_goal[e]
        p_input['wait'] = wait_env[e] or finished[e]
        # 将房型推理器传入环境，便于在可视化阶段更新房间信息
        p_input['room_infer'] = room_infer[e]
        if args.visualize or args.print_images:
            local_map[e, -1, :, :] = 1e-5
            p_input['sem_map_pred'] = local_map[e, 4:, :, :].argmax(0).cpu().numpy()
    # 执行计划、动作与预处理，同时得到包含前一轮与下一轮信息的字典
    obs, _, done, infos = envs.plan_act_and_preprocess(planner_inputs)

    start = time.time()
    g_reward = 0

    torch.set_grad_enabled(False)
    spl_per_category = defaultdict(list)
    success_per_category = defaultdict(list)

    # 用于存储奖励数据
    global_eps_rewards = []

    for step in range(args.num_training_frames // args.num_processes + 1):
        if finished.sum() == args.num_processes:
            break

        g_step = (step // args.num_local_steps) % args.num_global_steps
        l_step = step % args.num_local_steps

        # ------------------------------------------------------------------
        # Reinitialize variables when episode ends
        l_masks = torch.FloatTensor([0 if x else 1
                                     for x in done]).to(device)
        g_masks *= l_masks

        # 根据 done 标记处理已结束的 episode，并使用 infos[e]['final'] 记录的旧信息
        for e, x in enumerate(done):
            if x:
                final_info = infos[e]['final']
                spl = final_info['spl']
                success = final_info['success']
                dist = final_info['distance_to_goal']
                spl_per_category[final_info['goal_name']].append(spl)
                success_per_category[final_info['goal_name']].append(success)

                # 将本次 Episode 的关键统计写入 info.log
                episode_data = {
                    'thread_id': int(final_info.get('thread_id', e)),
                    # reset 后的 episode_id 指向下一轮，因此需减 1 才是刚结束的编号
                    'episode_id': int(final_info.get('episode_id', 0)) - 1,
                    'scene': final_info.get('scene'),
                    'goal_category': final_info.get('goal_name'),
                    'success': int(success),
                    'distance_to_goal': float(dist),
                    'stop_called': bool(final_info.get('stop_called', False)),
                    'steps': int(final_info.get('time', 0)),
                    'spl': float(spl)
                }
                if not success:
                    # 根据是否调用 stop 判断失败原因
                    if final_info.get('stop_called', False):
                        episode_data['failure_reason'] = 'stop_before_goal'
                    else:
                        episode_data['failure_reason'] = 'timeout'

                # 打开日志文件，将 Episode 信息和当前汇总信息同时写入
                with open(info_log_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(episode_data, ensure_ascii=False) + '\n')
                    # 先更新累计统计
                    total_episodes += 1
                    if success:
                        success_episodes += 1
                    # 紧接着写入最新的汇总统计
                    summary = {
                        'total_episodes': total_episodes,
                        'success_episodes': success_episodes
                    }
                    f.write(json.dumps(summary, ensure_ascii=False) + '\n')


                if args.eval:
                    episode_success[e].append(success)
                    episode_spl[e].append(spl)
                    episode_dist[e].append(dist)
                    if len(episode_success[e]) == num_episodes:
                        finished[e] = 1
                else:
                    episode_success.append(success)
                    episode_spl.append(spl)
                    episode_dist.append(dist)

                wait_env[e] = 1.
                update_intrinsic_rew(e)
                init_map_and_pose_for_env(e)
                # 重置房间持有状态，避免跨 episode 影响
                last_room_ids[e] = -1
                goal_hold_steps[e] = 0
                # 环境重置后立即调用房间推理器，写出上一轮数据并初始化新episode
                traversible = (local_map[e, 0].cpu().numpy() == 0)
                explored = (local_map[e, 1].cpu().numpy() > 0)
                sem_probs = local_map[e, 4:4 + args.num_sem_categories].cpu().numpy()
                explored_ratio_map = local_map[e, 1].cpu().numpy()  # 当前探索比例图用于衰减
                room_infer[e].update(traversible, explored, sem_probs,
                                     env_id=e,
                                     env_step=int(final_info['time']),
                                     explored_ratio_map=explored_ratio_map)  # 使用环境时间步，0表示新episode开始
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Semantic Mapping Module
        # 从 info['next'] 中读取当前位置姿态
        poses = torch.from_numpy(np.asarray(
            [infos[env_idx]['next']['sensor_pose'] for env_idx
             in range(num_scenes)])
        ).float().to(device)

        _, local_map, _, local_pose = \
            sem_map_module(obs, poses, local_map, local_pose)

        locs = local_pose.cpu().numpy()
        planner_pose_inputs[:, :3] = locs + origins
        local_map[:, 2, :, :].fill_(0.)  # Resetting current location channel
        for e in range(num_scenes):
            r, c = locs[e, 1], locs[e, 0]
            loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                            int(c * 100.0 / args.map_resolution)]
            local_map[e, 2:4, loc_r - 2:loc_r + 3, loc_c - 2:loc_c + 3] = 1.

        # ------------------------------------------------------------------
        # 当前目标类别ID列表，用于后续策略与房间先验（来自 info['next']）
        goal_cat_ids = np.asarray([infos[env_idx]['next']['goal_cat_id']
                                   for env_idx in range(num_scenes)])

        # 每个环境步都更新房间推理器，按环境步记录对象-房型得分
        for e in range(num_scenes):
            # 构造可行走与已探索栅格，以及当前语义置信度
            traversible = (local_map[e, 0].cpu().numpy() == 0)
            explored = (local_map[e, 1].cpu().numpy() > 0)
            sem_probs = local_map[e, 4:4 + args.num_sem_categories].cpu().numpy()
            explored_ratio_map = local_map[e, 1].cpu().numpy()  # 探索比例用于先验衰减
            # 使用 info['next'] 提供的真实时间步 env_step 记录房型打分
            room_infer[e].update(traversible, explored, sem_probs,
                                 env_id=e,
                                 env_step=int(infos[e]['next']['time']),
                                 explored_ratio_map=explored_ratio_map)

        # ------------------------------------------------------------------
        # Global Policy
        if l_step == args.num_local_steps - 1:
            # For every global step, update the full and local maps
            for e in range(num_scenes):
                if wait_env[e] == 1:  # New episode
                    wait_env[e] = 0.
                else:
                    update_intrinsic_rew(e)

                full_map[e, :, lmb[e, 0]:lmb[e, 1], lmb[e, 2]:lmb[e, 3]] = \
                    local_map[e]
                full_pose[e] = local_pose[e] + \
                               torch.from_numpy(origins[e]).to(device).float()

                locs = full_pose[e].cpu().numpy()
                r, c = locs[1], locs[0]
                loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                                int(c * 100.0 / args.map_resolution)]

                lmb[e] = get_local_map_boundaries((loc_r, loc_c),
                                                  (local_w, local_h),
                                                  (full_w, full_h))

                planner_pose_inputs[e, 3:] = lmb[e]
                origins[e] = [lmb[e][2] * args.map_resolution / 100.0,
                              lmb[e][0] * args.map_resolution / 100.0, 0.]

                local_map[e] = full_map[e, :,
                               lmb[e, 0]:lmb[e, 1],
                               lmb[e, 2]:lmb[e, 3]]
                local_pose[e] = full_pose[e] - \
                                torch.from_numpy(origins[e]).to(device).float()

            locs = local_pose.cpu().numpy()
            for e in range(num_scenes):
                global_orientation[e] = int((locs[e, 2] + 180.0) / 5.)
            global_input[:, 0:4, :, :] = local_map[:, 0:4, :, :]
            global_input[:, 4:8, :, :] = \
                nn.MaxPool2d(args.global_downscaling)(
                    full_map[:, 0:4, :, :])
            global_input[:, 8:, :, :] = local_map[:, 4:, :, :].detach()

            goal_cat_id = torch.from_numpy(goal_cat_ids)

            extras[:, 0] = global_orientation[:, 0]
            extras[:, 1] = goal_cat_id

            # Get exploration reward and metrics
            # 从 info['next'] 读取探索奖励信息
            g_reward = torch.from_numpy(np.asarray(
                [infos[env_idx]['next']['g_reward'] for env_idx in range(num_scenes)])
            ).float().to(device)
            g_reward += args.intrinsic_rew_coeff * intrinsic_rews.detach()

            g_process_rewards += g_reward.cpu().numpy()
            g_total_rewards = g_process_rewards * \
                              (1 - g_masks.cpu().numpy())
            g_process_rewards *= g_masks.cpu().numpy()
            per_step_g_rewards.append(np.mean(g_reward.cpu().numpy()))

            if np.sum(g_total_rewards) != 0:
                for total_rew in g_total_rewards:
                    if total_rew != 0:
                        g_episode_rewards.append(total_rew)
                # 记录每个episode的平均奖励
                global_eps_rewards.append(np.mean(g_total_rewards))

            # Add samples to global policy storage
            if step == 0:
                g_rollouts.obs[0].copy_(global_input)
                g_rollouts.extras[0].copy_(extras)
            else:
                g_rollouts.insert(
                    global_input, g_rec_states,
                    g_action, g_action_log_prob, g_value,
                    g_reward, g_masks, extras
                )

            # Sample long-term goal from global policy
            g_value, g_action, g_action_log_prob, g_rec_states = \
                g_policy.act(
                    g_rollouts.obs[g_step + 1],
                    g_rollouts.rec_states[g_step + 1],
                    g_rollouts.masks[g_step + 1],
                    extras=g_rollouts.extras[g_step + 1],
                    deterministic=False
                )
            cpu_actions = nn.Sigmoid()(g_action).cpu().numpy()
            global_goals = [[int(action[0] * local_w),
                             int(action[1] * local_h)]
                            for action in cpu_actions]
            global_goals = [[min(x, int(local_w - 1)),
                             min(y, int(local_h - 1))]
                            for x, y in global_goals]

            g_reward = 0
            g_masks = torch.ones(num_scenes).float().to(device)

        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Update long-term goal if target object is found
        found_goal = [0 for _ in range(num_scenes)]
        goal_maps = [np.zeros((local_w, local_h)) for _ in range(num_scenes)]

        for e in range(num_scenes):
            # 使用 info['next'] 中的目标类别更新语义地图
            cn = infos[e]['next']['goal_cat_id'] + 4
            if local_map[e, cn, :, :].sum() != 0.:
                cat_semantic_map = local_map[e, cn, :, :].cpu().numpy()
                cat_semantic_scores = cat_semantic_map
                cat_semantic_scores[cat_semantic_scores > 0] = 1.
                goal_maps[e] = cat_semantic_scores
                found_goal[e] = 1
            elif getattr(args, 'use_room_prior', False):
                prior = room_infer[e].build_goal_prior(int(goal_cat_ids[e]))
                # np.save(os.path.join(room_prior_dir, f'room_prior_env{e}_step{g_step}.npy'), prior)  # 保存先验热力图以检查房型推理效果

                # 计算前沿掩码，鼓励向未探索区域前进
                free = (local_map[e, 0].cpu().numpy() == 0)
                explored = (local_map[e, 1].cpu().numpy() > 0)
                frontier = free & binary_dilation(explored, disk(1)) & (~explored)

                # 预先对全局策略输出加入扰动，作为像素级采样的后备方案
                gx, gy = global_goals[e]
                shift = np.random.randint(-1, 2, size=2)
                gx = int(np.clip(gx + shift[0], 0, local_w - 1))
                gy = int(np.clip(gy + shift[1], 0, local_h - 1))
                fallback = (gx, gy)
                if prior.shape == goal_maps[e].shape and prior.sum() > 0:
                    (gx, gy), last_room_ids[e], goal_hold_steps[e] = \
                        sample_goal_by_room(prior, frontier, room_infer[e], fallback,
                                            last_room_ids[e], goal_hold_steps[e],
                                            args.min_goal_hold_steps,
                                            args.goal_switch_ratio,
                                            args.room_prior_topk,
                                            args.frontier_band_radius_m)
                else:
                    gx, gy = fallback
                goal_maps[e][:, :] = 0
                goal_maps[e][gx, gy] = 1
                goal_hold_steps[e] += 1
            else:
                gx, gy = global_goals[e]
                # 对全局策略输出位置加入随机扰动进行概率采样，鼓励探索
                shift = np.random.randint(-1, 2, size=2)
                gx = int(np.clip(gx + shift[0], 0, local_w - 1))
                gy = int(np.clip(gy + shift[1], 0, local_h - 1))
                goal_maps[e][gx, gy] = 1
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Take action and get next observation
        planner_inputs = [{} for e in range(num_scenes)]
        for e, p_input in enumerate(planner_inputs):
            p_input['map_pred'] = local_map[e, 0, :, :].cpu().numpy()
            p_input['exp_pred'] = local_map[e, 1, :, :].cpu().numpy()
            p_input['pose_pred'] = planner_pose_inputs[e]
            p_input['goal'] = goal_maps[e]  # global_goals[e]
            if l_step == args.num_local_steps - 1:
                p_input['new_goal'] = True  # 局部周期结束，触发新目标
            else:
                p_input['new_goal'] = False  # 非周期末，不触发新目标
            p_input['found_goal'] = found_goal[e]
            p_input['wait'] = wait_env[e] or finished[e]
            # 将房型推理器传入环境，便于在 _visualize 内更新房型打分
            p_input['room_infer'] = room_infer[e]
            if args.visualize or args.print_images:
                local_map[e, -1, :, :] = 1e-5
                p_input['sem_map_pred'] = local_map[e, 4:, :,
                                          :].argmax(0).cpu().numpy()

        # 调用环境继续执行并获得下一步的观测与信息
        obs, _, done, infos = envs.plan_act_and_preprocess(planner_inputs)
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Training
        torch.set_grad_enabled(True)
        if g_step % args.num_global_steps == args.num_global_steps - 1 \
                and l_step == args.num_local_steps - 1:
            if not args.eval:
                g_next_value = g_policy.get_value(
                    g_rollouts.obs[-1],
                    g_rollouts.rec_states[-1],
                    g_rollouts.masks[-1],
                    extras=g_rollouts.extras[-1]
                ).detach()

                g_rollouts.compute_returns(g_next_value, args.use_gae,
                                           args.gamma, args.tau)
                g_value_loss, g_action_loss, g_dist_entropy = \
                    g_agent.update(g_rollouts)
                g_value_losses.append(g_value_loss)
                g_action_losses.append(g_action_loss)
                g_dist_entropies.append(g_dist_entropy)
            g_rollouts.after_update()

        torch.set_grad_enabled(False)
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Logging
        if step % args.log_interval == 0:
            end = time.time()
            time_elapsed = time.gmtime(end - start)
            log = " ".join([
                "Time: {0:0=2d}d".format(time_elapsed.tm_mday - 1),
                "{},".format(time.strftime("%Hh %Mm %Ss", time_elapsed)),
                "num timesteps {},".format(step * num_scenes),
                "FPS {},".format(int(step * num_scenes / (end - start)))
            ])

            log += "\n\tRewards:"

            if len(g_episode_rewards) > 0:
                log += " ".join([
                    " Global step mean/med rew:",
                    "{:.4f}/{:.4f},".format(
                        np.mean(per_step_g_rewards),
                        np.median(per_step_g_rewards)),
                    " Global eps mean/med/min/max eps rew:",
                    "{:.3f}/{:.3f}/{:.3f}/{:.3f},".format(
                        np.mean(g_episode_rewards),
                        np.median(g_episode_rewards),
                        np.min(g_episode_rewards),
                        np.max(g_episode_rewards))
                ])

            if args.eval:
                total_success = []
                total_spl = []
                total_dist = []
                for e in range(args.num_processes):
                    for acc in episode_success[e]:
                        total_success.append(acc)
                    for dist in episode_dist[e]:
                        total_dist.append(dist)
                    for spl in episode_spl[e]:
                        total_spl.append(spl)

                if len(total_spl) > 0:
                    log += " ObjectNav succ/spl/dtg:"
                    log += " {:.3f}/{:.3f}/{:.3f}({:.0f}),".format(
                        np.mean(total_success),
                        np.mean(total_spl),
                        np.mean(total_dist),
                        len(total_spl))
            else:
                if len(episode_success) > 100:
                    log += " ObjectNav succ/spl/dtg:"
                    log += " {:.3f}/{:.3f}/{:.3f}({:.0f}),".format(
                        np.mean(episode_success),
                        np.mean(episode_spl),
                        np.mean(episode_dist),
                        len(episode_spl))

            log += "\n\tLosses:"
            if len(g_value_losses) > 0 and not args.eval:
                log += " ".join([
                    " Policy Loss value/action/dist:",
                    "{:.3f}/{:.3f}/{:.3f},".format(
                        np.mean(g_value_losses),
                        np.mean(g_action_losses),
                        np.mean(g_dist_entropies))
                ])

            print(log)
            logging.info(log)
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Save best models
        if (step * num_scenes) % args.save_interval < \
                num_scenes:
            if len(g_episode_rewards) >= 1000 and \
                    (np.mean(g_episode_rewards) >= best_g_reward) \
                    and not args.eval:
                torch.save(g_policy.state_dict(),
                           os.path.join(log_dir, "model_best.pth"))
                best_g_reward = np.mean(g_episode_rewards)

        # Save periodic models
        if (step * num_scenes) % args.save_periodic < \
                num_scenes:
            total_steps = step * num_scenes
            if not args.eval:
                torch.save(g_policy.state_dict(),
                           os.path.join(dump_dir,
                                        "periodic_{}.pth".format(total_steps)))
        # ------------------------------------------------------------------
    # 遍历所有房间推理器实例，写出剩余缓存，防止最后一个 Episode 丢失
    for e in range(num_scenes):
        room_infer[e].dump_episode_json(e)  # 写出剩余缓存

    # Print and save model performance numbers during evaluation
    if args.eval:
        print("Dumping eval details...")

        total_success = []
        total_spl = []
        total_dist = []
        for e in range(args.num_processes):
            for acc in episode_success[e]:
                total_success.append(acc)
            for dist in episode_dist[e]:
                total_dist.append(dist)
            for spl in episode_spl[e]:
                total_spl.append(spl)

        if len(total_spl) > 0:
            log = "Final ObjectNav succ/spl/dtg:"
            log += " {:.3f}/{:.3f}/{:.3f}({:.0f}),".format(
                np.mean(total_success),
                np.mean(total_spl),
                np.mean(total_dist),
                len(total_spl))

        print(log)
        logging.info(log)

        # Save the spl per category
        log = "Success | SPL per category\n"
        for key in success_per_category:
            log += "{}: {} | {}\n".format(key,
                                          sum(success_per_category[key]) /
                                          len(success_per_category[key]),
                                          sum(spl_per_category[key]) /
                                          len(spl_per_category[key]))

        print(log)
        logging.info(log)

        with open('{}/{}_spl_per_cat_pred_thr.json'.format(
                dump_dir, args.split), 'w') as f:
            json.dump(spl_per_category, f)

        with open('{}/{}_success_per_cat_pred_thr.json'.format(
                dump_dir, args.split), 'w') as f:
            json.dump(success_per_category, f)

    # 写入汇总统计：总 Episode 数与成功数
    summary = {
        'total_episodes': total_episodes,
        'successful_episodes': success_episodes
    }
    with open(info_log_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(summary, ensure_ascii=False) + '\n')

    # 绘制奖励曲线
    plt.plot(global_eps_rewards, label="Global eps mean rew")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
