# neural_map/neural_map.py
import torch
import torch.nn.functional as F
from .model import NeuralMapModel


class NeuralMap:
    """
    High-level NeuralMap manager that handles updating the map with new observations
    and decoding local maps for usage.
    """

    def __init__(self, map_size_m=20.0, coarse_resolution=0.2, fine_resolution=0.05,
                 num_semantic_classes=0, sensor_height=1.5, hfov=90.0, img_width=256, img_height=256):
        """
        Args:
            map_size_m (float): Map area side length in meters.
            coarse_resolution (float): Coarse feature plane resolution (m).
            fine_resolution (float): Fine feature plane resolution (m).
            num_semantic_classes (int): Number of semantic classes (excluding background).
            sensor_height (float): Camera sensor height from ground (m).
            hfov (float): Horizontal field of view of the camera (degrees).
            img_width, img_height (int): Sensor image resolution (pixels).
        """
        self.map_model = NeuralMapModel(map_size_m, coarse_resolution, fine_resolution, num_semantic_classes)
        # Optimizer (initially for coarse features + decoder only; fine features added later for progressive training)
        # We will use two-phase optimization: coarse+decoder first, then fine.
        params_coarse = [p for name, p in self.map_model.named_parameters() if "fine_plane" not in name]
        params_fine = [p for name, p in self.map_model.named_parameters() if "fine_plane" in name]
        # Initially freeze fine plane parameters for progressive optimization
        for p in params_fine:
            p.requires_grad_(False)
        self.optimizer = torch.optim.Adam(params_coarse, lr=1e-3)
        self.progressive_enabled = False  # will switch to True when fine-plane training begins
        self.progressive_step_threshold = 100  # e.g., after 100 frames, enable fine features

        # Camera intrinsics for projection
        self.img_width = img_width
        self.img_height = img_height
        # Principal point (assuming center of image)
        self.cx = img_width / 2.0
        self.cy = img_height / 2.0
        # Focal length (assuming square pixels) derived from horizontal FOV
        # hfov (deg) to rad:
        fov_rad = hfov * 3.1415926535 / 180.0
        # fx = fy = (width/2) / tan(hfov/2)
        self.fx = (img_width / 2.0) / (torch.tan(torch.tensor(fov_rad / 2.0)))
        self.fy = self.fx  # assume same for vertical FOV given square pixels (approx)
        # Sensor (camera) height in world (for determining floor)
        self.sensor_height = sensor_height

        # Counters
        self.frame_count = 0

    def reset(self):
        """Re-initialize the map model and optimizer for a new episode/environment."""
        map_size = self.map_model.map_size_m
        coarse_res = self.map_model.coarse_res
        fine_res = self.map_model.fine_res
        num_sem_classes = self.map_model.num_classes - 1 if self.map_model.has_semantics else 0
        # Re-create the model to reset all parameters
        self.map_model = NeuralMapModel(map_size, coarse_res, fine_res, num_semantic_classes=num_sem_classes)
        # Reset optimizer similarly (coarse first, fine later)
        params_coarse = [p for name, p in self.map_model.named_parameters() if "fine_plane" not in name]
        params_fine = [p for name, p in self.map_model.named_parameters() if "fine_plane" in name]
        for p in params_fine:
            p.requires_grad_(False)
        self.optimizer = torch.optim.Adam(params_coarse, lr=1e-3)
        self.progressive_enabled = False
        self.frame_count = 0

    def update(self, agent_pose, rgb_image, depth_image, semantic_mask=None):
        """
        Update the NeuralMap using a new observation.
        Args:
            agent_pose (tuple): (x, y, theta) agent's current world pose.
                                x, y in world coordinates (meters), theta in radians.
            rgb_image (ndarray or tensor): (H, W, 3) RGB image from the agent's camera.
            depth_image (ndarray or tensor): (H, W) Depth image (metric depth in meters for each pixel).
            semantic_mask (ndarray or tensor, optional): (H, W) semantic class ID for each pixel (0 for background).
        """
        # Convert inputs to torch tensors on CPU (we'll move to GPU if available for model)
        device = next(self.map_model.parameters()).device  # get model device (CPU or GPU)
        if not torch.is_tensor(depth_image):
            depth = torch.from_numpy(depth_image.astype('float32'))
        else:
            depth = depth_image.clone().float()
        depth = depth.to(device)
        # Also get semantic as tensor if provided
        sem_mask_tensor = None
        if semantic_mask is not None:
            if not torch.is_tensor(semantic_mask):
                sem_mask_tensor = torch.from_numpy(semantic_mask.astype('int64')).to(device)
            else:
                sem_mask_tensor = semantic_mask.clone().long().to(device)
        # If RGB image is needed for color supervision:
        rgb_tensor = None
        if rgb_image is not None:
            if not torch.is_tensor(rgb_image):
                rgb_tensor = torch.from_numpy(rgb_image.astype('float32')) / 255.0  # scale to [0,1]
            else:
                rgb_tensor = rgb_image.clone().float() / 255.0
            rgb_tensor = rgb_tensor.to(device)

        # Get agent pose components
        ax, ay, ath = agent_pose  # assuming pose is (x, y, yaw) in world coordinates
        # We'll project depth points into world coordinates.
        H, W = depth.shape
        # Create a meshgrid of pixel coordinates
        # (We'll vectorize by creating arrays of u and v for all pixels or a subset)
        us = torch.arange(0, W, device=device)
        vs = torch.arange(0, H, device=device)
        u_grid, v_grid = torch.meshgrid(us, vs,
                                        indexing='xy')  # u: width direction (x-axis), v: height direction (y-axis)
        u_grid = u_grid.flatten()  # 1-D tensor of length N (N=H*W)
        v_grid = v_grid.flatten()
        depth_flat = depth.flatten()
        # Filter out invalid depth (e.g., zeros or max range if any)
        # Assume depth==0 indicates no return, or very large values beyond sensor range.
        # Also filter points with depth >= some max (if sensor has a max range).
        max_range = 0.0
        if depth_flat.numel() > 0:
            max_range = depth_flat.max().item()
        # We consider depth values equal to max_range as possibly no-hit (depending on sensor).
        # If the sensor provides a known max range, use that. For safety, treat depth==0 or depth >= max_range as no hit.
        valid_mask = (depth_flat > 0)
        if max_range > 0:
            valid_mask &= (depth_flat < max_range * 0.99)  # consider anything very close to max range as no hit
        # Coordinates for valid depth points
        u_valid = u_grid[valid_mask]
        v_valid = v_grid[valid_mask]
        depth_valid = depth_flat[valid_mask]

        # Compute camera coordinates for valid points (in agent's camera coordinate system):
        # Using pinhole projection:
        # X_cam = (u - cx)/fx * Z_cam, Y_cam = -(v - cy)/fy * Z_cam, Z_cam = Z_cam (depth)
        Z_cam = depth_valid
        X_cam = (u_valid - self.cx) / self.fx * Z_cam
        Y_cam = -(v_valid - self.cy) / self.fy * Z_cam  # negative because v down is -Y in world
        # Now rotate these cam coords to world coords (assuming camera forward aligned with agent's yaw, and no camera tilt):
        cos_th = torch.cos(torch.tensor(ath, device=device))
        sin_th = torch.sin(torch.tensor(ath, device=device))
        # Agent is at (ax, ay) on ground with orientation ath, camera at height sensor_height.
        # So agent base position (ax, ay), we assume ground plane z=0. If agent is standing, camera world height = sensor_height.
        # Convert camera coordinate to world:
        # world_x = ax + X_cam * cos(th) - Z_cam * sin(th)
        # world_y = ay + X_cam * sin(th) + Z_cam * cos(th)
        # world_z = sensor_height + Y_cam  (vertical position)
        world_x = ax + X_cam * cos_th - Z_cam * sin_th
        world_y = ay + X_cam * sin_th + Z_cam * cos_th
        world_z = self.sensor_height + Y_cam

        # Determine which points correspond to obstacles vs free space:
        occ_points = []
        occ_labels = []  # semantic labels for occupied points (with 0 for background if unknown)
        occ_colors = []  # color values for occupied points
        free_points = []
        # We will sample one free-space point along each ray just before the obstacle.
        # Also for rays with no hit (depth beyond range), sample a far free point.
        # Create a dictionary to store nearest depth per ray if needed (here each pixel is a ray).

        # Process each valid depth point
        N = depth_valid.shape[0]
        # Note: We vectorize by processing all points.
        # Determine obstacle criteria for each:
        is_obstacle = torch.zeros(N, dtype=torch.bool, device=device)
        semantic_id = None
        if sem_mask_tensor is not None:
            # Get semantic class for each valid point
            # Need to index sem_mask at those pixel coords
            semantic_flat = sem_mask_tensor.flatten()
            sem_valid = semantic_flat[valid_mask]  # semantic classes corresponding to each depth point
            semantic_id = sem_valid
        # If semantic info is available, mark obstacles if the semantic label is a valid object (non-background).
        if semantic_id is not None:
            # By convention, let's consider label 0 as background (floor/wall) and >=1 as objects of interest.
            is_object = semantic_id > 0
            is_obstacle |= is_object
        # Use geometry (height) to mark obstacles: anything significantly above ground is obstacle (like wall or furniture).
        # Define threshold for floor height (e.g., 0.3m).
        floor_thresh = 0.3
        above_floor = world_z > (floor_thresh)
        is_obstacle |= above_floor  # mark any point higher than threshold as obstacle
        # However, we do not want to mark actual floor points as obstacle. If semantic indicates floor (maybe label 0), we want to ignore those.
        # If semantic provided and indicates floor explicitly (not sure if available), we could exclude them.
        # (We assume background class includes floor/wall, which above rule may mark as obstacle if wall (height>thresh) or floor (height ~0? which won't be above thresh, so floor remains not obstacle).

        # Now separate points into occ vs free based on is_obstacle
        occ_mask = is_obstacle
        free_mask = ~is_obstacle
        # For each obstacle point, we'll add a corresponding free point slightly before it along the ray.
        # Compute a free point at 90% of the depth distance (10% closer to agent than the obstacle)
        t = 0.9  # factor for free point placement
        X_cam_free = X_cam[occ_mask] * t
        Y_cam_free = Y_cam[occ_mask] * t
        Z_cam_free = Z_cam[occ_mask] * t
        if X_cam_free.numel() > 0:
            fx = X_cam_free
            fy = Y_cam_free
            fz = Z_cam_free
            wx_free = ax + fx * cos_th - fz * sin_th
            wy_free = ay + fx * sin_th + fz * cos_th
            # For free points, we don't need vertical coordinate (they are along ray, not on ground necessarily, but we consider them free space in line of sight)
            free_pts = torch.stack([wx_free, wy_free], dim=1)
            free_points.append(free_pts)
        # For rays with no hit (if any depth were max range, we consider far free point):
        # Determine rays with no hit: we earlier filtered valid_mask to exclude depth >= max_range*0.99.
        # So "no hit" rays are those originally present in pixel grid but not in valid_mask.
        # We can derive them:
        total_pixels = H * W
        all_idx = torch.arange(total_pixels, device=device)
        invalid_mask = ~valid_mask  # these indices were not processed, meaning depth == 0 or ~max range
        if invalid_mask.any():
            u_invalid = u_grid[invalid_mask]
            v_invalid = v_grid[invalid_mask]
            # For those rays, use max_range (or a defined sensor range) for free point
            if max_range == 0:
                max_range = 10.0  # assume some max range if not provided
            # We'll sample at 0.9 * max_range as free
            Z_far = torch.tensor(max_range * 0.9, device=device)
            X_far = (u_invalid - self.cx) / self.fx * Z_far
            Y_far = -(v_invalid - self.cy) / self.fy * Z_far
            wx_far = ax + X_far * cos_th - Z_far * sin_th
            wy_far = ay + X_far * sin_th + Z_far * cos_th
            far_free_pts = torch.stack([wx_far, wy_far], dim=1)
            free_points.append(far_free_pts)

        # Now collate all occ points
        if occ_mask.any():
            wx_occ = world_x[occ_mask]
            wy_occ = world_y[occ_mask]
            occ_pts = torch.stack([wx_occ, wy_occ], dim=1)
            occ_points.append(occ_pts)
            # Prepare semantic labels for those points
            if semantic_id is not None:
                labels = semantic_id[occ_mask].clone()
                # Convert any background label (0 or negative) to 0
                labels[labels < 0] = 0
                # We already treat 0 as background class index
                occ_labels.append(labels)
            else:
                # If no semantic, label all as background (0)
                occ_labels.append(torch.zeros(wx_occ.shape, dtype=torch.long, device=device))
            # Prepare color for occ points
            if rgb_tensor is not None:
                # Get color of each pixel that led to these occ points
                rgb_flat = rgb_tensor.view(-1, 3)
                rgb_valid = rgb_flat[valid_mask]  # colors for each depth point
                occ_colors.append(rgb_valid[occ_mask])

        # Concatenate lists to single tensors
        if occ_points:
            occ_points = torch.cat(occ_points, dim=0).to(device)
        else:
            occ_points = torch.zeros((0, 2), device=device)
        if free_points:
            free_points = torch.cat(free_points, dim=0).to(device)
        else:
            free_points = torch.zeros((0, 2), device=device)
        if occ_labels:
            occ_labels = torch.cat(occ_labels, dim=0).to(device)
        else:
            occ_labels = torch.zeros((0,), dtype=torch.long, device=device)
        if occ_colors:
            occ_colors = torch.cat(occ_colors, dim=0).to(device)
        else:
            occ_colors = torch.zeros((0, 3), device=device)

        # Prepare training data for the NeuralMap model:
        num_occ = occ_points.shape[0]
        num_free = free_points.shape[0]
        if num_occ == 0 and num_free == 0:
            return  # nothing to update (no valid depth data)
        # Build batch of query coordinates
        all_points = []
        occupancy_targets = []
        semantic_targets = []
        color_targets = []
        if num_occ > 0:
            all_points.append(occ_points)
            occupancy_targets.append(torch.ones(num_occ, device=device))  # occupied = 1
            if self.map_model.has_semantics:
                semantic_targets.append(occ_labels)  # class indices (including 0 for background)
            if self.map_model.has_color:
                color_targets.append(occ_colors)
        if num_free > 0:
            all_points.append(free_points)
            occupancy_targets.append(torch.zeros(num_free, device=device))  # free = 0
            if self.map_model.has_semantics:
                # For free points, we don't supervise semantics (no object present).
                # We can either skip or assign them a "background" label with a mask to ignore in loss.
                # Here, we will assign background (0) and handle by ignoring in loss calculation.
                free_labels = torch.zeros(num_free, dtype=torch.long, device=device)
                semantic_targets.append(free_labels)
            if self.map_model.has_color:
                # For free points, color supervision not applicable (no surface).
                # We will not supervise color on free points. We can append dummy or handle by mask similarly.
                dummy_colors = torch.zeros((num_free, 3), device=device)
                color_targets.append(dummy_colors)
        all_points = torch.cat(all_points, dim=0)
        occupancy_targets = torch.cat(occupancy_targets, dim=0)
        if self.map_model.has_semantics:
            semantic_targets = torch.cat(semantic_targets, dim=0)
        if self.map_model.has_color:
            color_targets = torch.cat(color_targets, dim=0)

        # Run the model on all_points
        occ_logits, sem_logits, color_preds = self.map_model(all_points)
        # Compute losses
        # Occupancy: binary cross-entropy with logits
        occ_loss = F.binary_cross_entropy_with_logits(occ_logits, occupancy_targets)
        # Semantic: cross-entropy (only for points that are occupied or where an object is present)
        sem_loss = torch.tensor(0.0, device=device)
        if self.map_model.has_semantics and sem_logits is not None:
            # We will ignore free-space points in semantic loss. These have target 0 (background).
            # It's actually okay to include them as background class to encourage map to predict background where nothing,
            # but there is a risk it confuses floor vs wall since background covers both. We keep them to minimal influence.
            sem_loss = F.cross_entropy(sem_logits, semantic_targets)
        # Color: MSE loss on color for occupied points only
        color_loss = torch.tensor(0.0, device=device)
        if self.map_model.has_color and color_preds is not None and num_occ > 0:
            # Only consider first num_occ entries which correspond to occ points
            pred_cols = color_preds[:num_occ]
            true_cols = color_targets[:num_occ]
            color_loss = F.mse_loss(pred_cols, true_cols)
        # Total loss (we can weight them if needed; for simplicity use equal weight)
        total_loss = occ_loss + sem_loss + color_loss

        # Backpropagate and optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Progressive optimization: enable fine-plane training after certain number of frames
        self.frame_count += 1
        if (not self.progressive_enabled) and self.frame_count >= self.progressive_step_threshold:
            # Unfreeze fine feature plane and include in optimizer
            for p in self.map_model.fine_plane.parameters() if hasattr(self.map_model, "fine_plane") else []:
                p.requires_grad_(True)
            # Reinitialize optimizer to include fine_plane parameters (keeping old parameters' states is complex, we reset optimizer for simplicity)
            self.optimizer = torch.optim.Adam(self.map_model.parameters(), lr=1e-3)
            self.progressive_enabled = True

    def get_local_map(self, agent_pose, agent_orientation, map_size_m=5.0, output_resolution=0.1):
        """解码智能体周围的局部占用与语义地图"""
        import numpy as np
        ax, ay = agent_pose  # agent's world coordinates (x, y)
        yaw = agent_orientation
        # Determine number of cells for local map grid
        grid_dim = round(map_size_m / output_resolution)
        # 这里不再强制将网格维度设为奇数，保持与导航模块中局部地图
        # 的大小一致，避免生成 241×241 网格时与 240×240 的局部地图
        # 写入操作发生维度不匹配
        half = grid_dim // 2
        cos_th, sin_th = np.cos(yaw), np.sin(yaw)
        xv, yv = np.meshgrid(np.arange(grid_dim), np.arange(grid_dim))
        xv = xv.reshape(-1)
        yv = yv.reshape(-1)
        # Local coordinates relative to agent
        x_local = (xv - half) * output_resolution
        y_local = (half - yv) * output_resolution
        world_x = ax + x_local * cos_th - y_local * sin_th
        world_y = ay + x_local * sin_th + y_local * cos_th
        coords = torch.tensor(np.stack([world_x, world_y], axis=1), dtype=torch.float32,
                              device=next(self.map_model.parameters()).device)
        occ_logits, sem_logits, _ = self.map_model(coords)
        occ_probs = torch.sigmoid(occ_logits).detach().cpu().numpy()
        if sem_logits is not None:
            sem_preds = torch.argmax(sem_logits, dim=1).detach().cpu().numpy()
        else:
            sem_preds = np.zeros_like(occ_probs, dtype=np.int64)
        # Fill the local grids
        occ_grid_flat = occ_probs.reshape(grid_dim * grid_dim)
        sem_grid_flat = sem_preds.reshape(grid_dim * grid_dim)
        # For semantics, set free spaces' class to 0 and use predicted class for occupied if occupancy > 0.5
        # We can threshold occupancy probability to determine occupied/free cells.
        occ_binary = occ_grid_flat > 0.5
        # Assign occupancy grid (as probability or binary; here we'll output probability map)
        occupancy_grid = occ_probs.reshape(grid_dim, grid_dim)
        # Assign semantic grid
        semantic_grid = np.zeros((grid_dim, grid_dim), dtype=np.int64)
        sem_grid_flat = sem_grid_flat  # (grid_dim*grid_dim,) predicted classes including background
        # Only keep semantic predictions where occupancy is high (occupied). Free cells remain 0.
        sem_grid_flat[~occ_binary] = 0
        semantic_grid = sem_grid_flat.reshape(grid_dim, grid_dim)
        return occupancy_grid, semantic_grid
