import numpy as np
import envs.utils.depth_utils as du
from constants import NUM_ROOM_CATEGORIES  # 引入房间类别数，构建房间先验地图


class MapBuilder(object):
    def __init__(self, params):
        self.params = params
        frame_width = params['frame_width']
        frame_height = params['frame_height']
        fov = params['fov']
        self.camera_matrix = du.get_camera_matrix(
            frame_width,
            frame_height,
            fov)
        self.vision_range = params['vision_range']

        self.map_size_cm = params['map_size_cm']
        self.resolution = params['resolution']
        agent_min_z = params['agent_min_z']
        agent_max_z = params['agent_max_z']
        self.z_bins = [agent_min_z, agent_max_z]
        self.du_scale = params['du_scale']
        self.visualize = params['visualize']
        self.obs_threshold = params['obs_threshold']

        self.map = np.zeros((self.map_size_cm // self.resolution,
                             self.map_size_cm // self.resolution,
                             len(self.z_bins) + 1), dtype=np.float32)

        # 新增房间先验地图，用于记录各栅格属于不同房间的概率
        self.room_map = np.zeros((self.map_size_cm // self.resolution,
                                  self.map_size_cm // self.resolution,
                                  NUM_ROOM_CATEGORIES), dtype=np.float32)

        self.agent_height = params['agent_height']
        self.agent_view_angle = params['agent_view_angle']
        return

    def update_map(self, depth, current_pose):
        with np.errstate(invalid="ignore"):
            depth[depth > self.vision_range * self.resolution] = np.NaN
        point_cloud = du.get_point_cloud_from_z(depth, self.camera_matrix,
                                                scale=self.du_scale)

        agent_view = du.transform_camera_view(point_cloud,
                                              self.agent_height,
                                              self.agent_view_angle)

        shift_loc = [self.vision_range * self.resolution // 2, 0, np.pi / 2.0]
        agent_view_centered = du.transform_pose(agent_view, shift_loc)

        agent_view_flat = du.bin_points(
            agent_view_centered,
            self.vision_range,
            self.z_bins,
            self.resolution)

        agent_view_cropped = agent_view_flat[:, :, 1]
        agent_view_cropped = agent_view_cropped / self.obs_threshold
        agent_view_cropped[agent_view_cropped >= 0.5] = 1.0
        agent_view_cropped[agent_view_cropped < 0.5] = 0.0

        agent_view_explored = agent_view_flat.sum(2)
        agent_view_explored[agent_view_explored > 0] = 1.0

        geocentric_pc = du.transform_pose(agent_view, current_pose)

        geocentric_flat = du.bin_points(
            geocentric_pc,
            self.map.shape[0],
            self.z_bins,
            self.resolution)

        self.map = self.map + geocentric_flat

        map_gt = self.map[:, :, 1] / self.obs_threshold
        map_gt[map_gt >= 0.5] = 1.0
        map_gt[map_gt < 0.5] = 0.0

        explored_gt = self.map.sum(2)
        explored_gt[explored_gt > 1] = 1.0

        return agent_view_cropped, map_gt, agent_view_explored, explored_gt

    def update_room_map(self, depth, room_heatmap, current_pose):
        """将图像坐标的房间概率热图投影到 BEV 地图"""
        with np.errstate(invalid="ignore"):
            depth[depth > self.vision_range * self.resolution] = np.NaN
        # 生成点云并转换到地理坐标系
        point_cloud = du.get_point_cloud_from_z(depth, self.camera_matrix,
                                                scale=self.du_scale)
        agent_view = du.transform_camera_view(point_cloud,
                                              self.agent_height,
                                              self.agent_view_angle)
        geocentric_pc = du.transform_pose(agent_view, current_pose)
        # 调用 bin_points 以获得与占用图一致的栅格离散化
        du.bin_points(geocentric_pc, self.map.shape[0],
                      self.z_bins, self.resolution)

        flat_pc = geocentric_pc.reshape(-1, 3)
        # 将房间热图展平，方便与点云一一对应
        flat_probs = room_heatmap.reshape(NUM_ROOM_CATEGORIES, -1)
        isnotnan = ~np.isnan(flat_pc[:, 0])
        x_bin = np.round(flat_pc[:, 0] / self.resolution).astype(np.int32)
        y_bin = np.round(flat_pc[:, 1] / self.resolution).astype(np.int32)
        valid = (x_bin >= 0) & (x_bin < self.room_map.shape[1]) & \
                (y_bin >= 0) & (y_bin < self.room_map.shape[0]) & isnotnan
        x_bin = x_bin[valid]
        y_bin = y_bin[valid]
        # 按照概率将像素贡献累加到对应栅格
        for r in range(NUM_ROOM_CATEGORIES):
            weights = flat_probs[r, valid]
            np.add.at(self.room_map[:, :, r], (y_bin, x_bin), weights)

    def get_room_map(self):
        """返回当前维护的房间先验地图"""
        return self.room_map

    def get_st_pose(self, current_loc):
        loc = [- (current_loc[0] / self.resolution
                  - self.map_size_cm // (self.resolution * 2)) /
               (self.map_size_cm // (self.resolution * 2)),
               - (current_loc[1] / self.resolution
                  - self.map_size_cm // (self.resolution * 2)) /
               (self.map_size_cm // (self.resolution * 2)),
               90 - np.rad2deg(current_loc[2])]
        return loc

    def reset_map(self, map_size):
        self.map_size_cm = map_size

        self.map = np.zeros((self.map_size_cm // self.resolution,
                             self.map_size_cm // self.resolution,
                             len(self.z_bins) + 1), dtype=np.float32)

        # 同时重置房间先验地图
        self.room_map = np.zeros((self.map_size_cm // self.resolution,
                                  self.map_size_cm // self.resolution,
                                  NUM_ROOM_CATEGORIES), dtype=np.float32)

    def get_map(self):
        return self.map
