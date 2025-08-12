# neural_map/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralMapModel(nn.Module):
    """
    Continuous feature-plane based map representation with multi-scale feature planes
    and decoding network for occupancy, semantics, and color.
    """

    def __init__(self, map_size_m=20.0, coarse_resolution=0.2, fine_resolution=0.05,
                 num_semantic_classes=0):
        """
        Args:
            map_size_m (float): Length of one side of the square map area (meters).
            coarse_resolution (float): Grid cell size for coarse feature plane (m).
            fine_resolution (float): Grid cell size for fine feature plane (m).
            num_semantic_classes (int): Number of semantic classes (excluding background).
                                        If >0, the model will output semantic logits.
        """
        super().__init__()
        # Map dimensions
        self.map_size_m = map_size_m
        self.coarse_res = coarse_resolution
        self.fine_res = fine_resolution
        # Number of cells for each plane
        self.coarse_dim = int(map_size_m / coarse_resolution)  # e.g., 20/0.2 = 100
        self.fine_dim = int(map_size_m / fine_resolution)  # e.g., 20/0.05 = 400
        # Ensure odd dimensions if needed (for center alignment)
        # (Optional: can adjust to ensure center pixel corresponds exactly to origin)

        # Feature dimensions for planes
        self.feat_dim = 16  # feature channels per plane (you can adjust)
        # Initialize feature planes as learnable parameters (coarse and fine)
        # Shape: (feat_dim, H, W)
        self.coarse_plane = nn.Parameter(torch.zeros(self.feat_dim, self.coarse_dim, self.coarse_dim))
        self.fine_plane = nn.Parameter(torch.zeros(self.feat_dim, self.fine_dim, self.fine_dim))

        # Decoder network: input = concatenated coarse+fine feature vector, output occupancy + semantics + color
        input_dim = self.feat_dim * 2
        hidden_dim = 64
        output_dim = 0
        # 1 for occupancy
        output_dim += 1
        # semantic classes (including background class 0) if applicable
        self.has_semantics = num_semantic_classes > 0
        if self.has_semantics:
            self.num_classes = num_semantic_classes + 1  # add background class as class 0
            output_dim += self.num_classes
        else:
            self.num_classes = 0
        # 3 for RGB color
        self.has_color = True
        if self.has_color:
            output_dim += 3

        # Define a simple MLP for decoding
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

        # Initialize weights (optional: small random init to break symmetry)
        nn.init.normal_(self.coarse_plane, mean=0.0, std=0.01)
        nn.init.normal_(self.fine_plane, mean=0.0, std=0.01)
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.constant_(self.fc_out.bias, 0)

    def forward(self, coords):
        """
        Query the NeuralMap at given world coordinates.
        Args:
            coords (Tensor): shape (N, 2), each row is (x_world, y_world) coordinates in meters relative to map origin.
        Returns:
            Tuple of Tensors: (occupancy_logits, semantic_logits (if any), color_preds (if any))
        """
        # coords are in continuous world units (origin at center of map). We convert to feature plane indices.
        # Normalize coordinates to [0, grid_dim] for coarse and fine
        # Map covers [-map_size/2, +map_size/2] in both x and y.
        # So origin (0,0) should map to center index of each plane.
        device = coords.device
        N = coords.shape[0]
        # Split input coords
        x_world = coords[:, 0]  # [N]
        y_world = coords[:, 1]  # [N]
        # Compute normalized indices in each grid
        # For coarse plane:
        # e.g., x_idx_coarse = (x_world / coarse_resolution) + coarse_dim/2
        x_idx_coarse = x_world / self.coarse_res + 0.5 * self.coarse_dim
        y_idx_coarse = y_world / self.coarse_res + 0.5 * self.coarse_dim
        # For fine plane:
        x_idx_fine = x_world / self.fine_res + 0.5 * self.fine_dim
        y_idx_fine = y_world / self.fine_res + 0.5 * self.fine_dim

        # Clamp indices to within bounds [0, dim-1] to avoid sampling outside
        x_idx_coarse = torch.clamp(x_idx_coarse, 0, self.coarse_dim - 1)
        y_idx_coarse = torch.clamp(y_idx_coarse, 0, self.coarse_dim - 1)
        x_idx_fine = torch.clamp(x_idx_fine, 0, self.fine_dim - 1)
        y_idx_fine = torch.clamp(y_idx_fine, 0, self.fine_dim - 1)

        # Compute integer base indices and interpolation weights for coarse
        x0c = torch.floor(x_idx_coarse).long()
        y0c = torch.floor(y_idx_coarse).long()
        x1c = torch.clamp(x0c + 1, max=self.coarse_dim - 1)
        y1c = torch.clamp(y0c + 1, max=self.coarse_dim - 1)
        wx_c = (x_idx_coarse - x0c.float()).unsqueeze(0)  # shape (1, N)
        wy_c = (y_idx_coarse - y0c.float()).unsqueeze(0)  # shape (1, N)

        # Compute integer base indices and interpolation weights for fine
        x0f = torch.floor(x_idx_fine).long()
        y0f = torch.floor(y_idx_fine).long()
        x1f = torch.clamp(x0f + 1, max=self.fine_dim - 1)
        y1f = torch.clamp(y0f + 1, max=self.fine_dim - 1)
        wx_f = (x_idx_fine - x0f.float()).unsqueeze(0)
        wy_f = (y_idx_fine - y0f.float()).unsqueeze(0)

        # Gather corner features from coarse plane
        # coarse_plane: (feat_dim, Hc, Wc)
        feat_cc00 = self.coarse_plane[:, y0c, x0c]  # shape (feat_dim, N)
        feat_cc01 = self.coarse_plane[:, y0c, x1c]
        feat_cc10 = self.coarse_plane[:, y1c, x0c]
        feat_cc11 = self.coarse_plane[:, y1c, x1c]
        # bilinear interpolate
        coarse_feats = (feat_cc00 * (1 - wx_c) * (1 - wy_c) +
                        feat_cc01 * (wx_c) * (1 - wy_c) +
                        feat_cc10 * (1 - wx_c) * (wy_c) +
                        feat_cc11 * (wx_c) * (wy_c))
        coarse_feats = coarse_feats.t()  # shape (N, feat_dim)

        # Gather corner features from fine plane
        feat_ff00 = self.fine_plane[:, y0f, x0f]  # (feat_dim, N)
        feat_ff01 = self.fine_plane[:, y0f, x1f]
        feat_ff10 = self.fine_plane[:, y1f, x0f]
        feat_ff11 = self.fine_plane[:, y1f, x1f]
        fine_feats = (feat_ff00 * (1 - wx_f) * (1 - wy_f) +
                      feat_ff01 * (wx_f) * (1 - wy_f) +
                      feat_ff10 * (1 - wx_f) * (wy_f) +
                      feat_ff11 * (wx_f) * (wy_f))
        fine_feats = fine_feats.t()  # shape (N, feat_dim)

        # Concatenate coarse and fine features for each point
        feat_vec = torch.cat([coarse_feats, fine_feats], dim=1)  # (N, 2*feat_dim)

        # Decode through MLP
        h = F.relu(self.fc1(feat_vec))
        h = F.relu(self.fc2(h))
        out = self.fc_out(h)  # shape (N, output_dim)
        # Split outputs
        # Occupancy logit:
        occ_logit = out[:, 0]  # (N,)
        idx = 1
        sem_logit = None
        if self.has_semantics:
            sem_logit = out[:, idx: idx + self.num_classes]  # (N, num_classes)
            idx += self.num_classes
        color_pred = None
        if self.has_color:
            color_pred = out[:, idx: idx + 3]  # (N, 3)
            # Apply sigmoid to color_pred to bound it [0,1]
            color_pred = torch.sigmoid(color_pred)
        return occ_logit, sem_logit, color_pred
