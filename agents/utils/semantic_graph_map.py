import numpy as np
from collections import deque
import heapq


class SemanticGraphMap:
    """Semantic Graph Map storing place, image and object nodes."""

    def __init__(self):
        self.place_nodes = []
        self.image_nodes = []
        self.object_nodes = []
        self.obj_cat_to_idx = {}

        self.A_pi = np.zeros((0, 0), dtype=np.int32)  # place-image
        self.A_im = np.zeros((0, 0), dtype=np.int32)  # image-image
        self.A_io = np.zeros((0, 0), dtype=np.int32)  # image-object
        self.last_image_idx = None

    def update(self, obs, sem_map, pose):
        """Update graph with a new observation.

        Args:
            obs: Current observation.
            sem_map: Semantic map prediction.
            pose: Agent pose.
        """
        # Add image node
        i_idx = len(self.image_nodes)
        self.image_nodes.append({'pose': pose})
        self.A_im = self._expand_square(self.A_im)
        if self.last_image_idx is not None:
            self.A_im[self.last_image_idx, i_idx] = 1

        # Ensure image-object adjacency has a row for the new image
        self.A_io = self._expand_rect(self.A_io, len(self.image_nodes), len(self.object_nodes))

        # Add place node and connect to image node
        p_idx = len(self.place_nodes)
        self.place_nodes.append({'pose': pose})
        self.A_pi = self._expand_rect(self.A_pi, len(self.place_nodes), len(self.image_nodes))
        self.A_pi[p_idx, i_idx] = 1

        # Add object nodes detected in semantic map
        if sem_map is not None:
            for obj_cat in np.unique(sem_map):
                if obj_cat <= 0:
                    continue
                if obj_cat not in self.obj_cat_to_idx:
                    o_idx = len(self.object_nodes)
                    self.object_nodes.append({'category': int(obj_cat)})
                    self.obj_cat_to_idx[obj_cat] = o_idx
                    # expand for the new object column
                    self.A_io = self._expand_rect(
                        self.A_io, len(self.image_nodes), len(self.object_nodes)
                    )
                o_idx = self.obj_cat_to_idx[obj_cat]
                self.A_io[i_idx, o_idx] = 1

        self.last_image_idx = i_idx

    def place_place_accessibility(self):
        """Return reachability matrix between place nodes.

        Two place nodes are considered connected if they belong to the same
        connected component induced by the image-image edges.
        """
        n_places = len(self.place_nodes)
        if n_places == 0:
            return np.zeros((0, 0), dtype=np.int32)

        # Map each image to its place index
        img_to_place = np.argmax(self.A_pi, axis=0) if self.A_pi.size else []

        # Build adjacency between places based on image-image edges
        adj = np.zeros((n_places, n_places), dtype=np.int32)
        for i in range(len(self.image_nodes)):
            for j in np.where(self.A_im[i] > 0)[0]:
                p_i = img_to_place[i]
                p_j = img_to_place[j]
                adj[p_i, p_j] = 1
                adj[p_j, p_i] = 1

        # Compute connected components using BFS
        visited = -np.ones(n_places, dtype=np.int32)
        comp = 0
        for idx in range(n_places):
            if visited[idx] != -1:
                continue
            queue = deque([idx])
            visited[idx] = comp
            while queue:
                u = queue.popleft()
                for v in np.where(adj[u] > 0)[0]:
                    if visited[v] == -1:
                        visited[v] = comp
                        queue.append(v)
            comp += 1

        connectivity = np.zeros((n_places, n_places), dtype=np.int32)
        for cid in range(comp):
            nodes = np.where(visited == cid)[0]
            connectivity[nodes[:, None], nodes[None, :]] = 1
        return connectivity

    def place_object_matrix(self, num_categories):
        """Compute place-object category connections for this scene."""
        n_places = len(self.place_nodes)
        R = np.zeros((n_places, num_categories), dtype=np.int32)
        if n_places == 0 or len(self.object_nodes) == 0:
            return R

        A_po = self.A_pi @ self.A_io  # place-object nodes
        for o_idx, obj in enumerate(self.object_nodes):
            cat = int(obj.get("category", -1))
            if 0 <= cat < num_categories:
                R[:, cat] = np.logical_or(R[:, cat], A_po[:, o_idx])
        return R.astype(np.int32)

    def semantic_shortest_path(self, gamma, start_idx, goal_idx):
        """Compute semantic shortest path between two places using Γ.

        Edge weights are defined as ``-log Γ[i, j]``. Uses Dijkstra's
        algorithm to find the minimum-cost path.

        Args:
            gamma (ndarray): Place-Place reachability matrix.
            start_idx (int): Index of the start place node.
            goal_idx (int): Index of the goal place node.

        Returns:
            list[int]: Sequence of place indices forming the path. Empty if no
            path exists or inputs are invalid.
        """

        if gamma.size == 0:
            return []
        n = gamma.shape[0]
        if start_idx >= n or goal_idx >= n:
            return []

        dist = [float("inf")] * n
        prev = [-1] * n
        dist[start_idx] = 0.0
        pq = [(0.0, start_idx)]

        while pq:
            d, u = heapq.heappop(pq)
            if u == goal_idx:
                break
            if d > dist[u]:
                continue
            for v, conn in enumerate(gamma[u]):
                if conn <= 0:
                    continue
                w = -np.log(max(conn, 1e-6))
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))

        if dist[goal_idx] == float("inf"):
            return []

        path = [goal_idx]
        while path[-1] != start_idx:
            path.append(prev[path[-1]])
        path.reverse()
        return path

    @staticmethod
    def _expand_square(mat):
        n = mat.shape[0]
        new_mat = np.zeros((n + 1, n + 1), dtype=mat.dtype)
        new_mat[:n, :n] = mat
        return new_mat

    @staticmethod
    def _expand_rect(mat, rows, cols):
        cur_rows, cur_cols = mat.shape
        new_mat = np.zeros((rows, cols), dtype=mat.dtype)
        new_mat[:cur_rows, :cur_cols] = mat
        return new_mat


def aggregate_graph_statistics(sgms, num_categories):
    """Aggregate training statistics across multiple SGMs.

    Args:
        sgms: list of SemanticGraphMap instances.
        num_categories: total number of object categories.

    Returns:
        gamma: Place-Place accessibility matrix averaged over scenes.
        R: Place-Object connection counts.
    """
    if not sgms:
        return np.zeros((0, 0), dtype=np.float32), np.zeros(
            (0, num_categories), dtype=np.float32
        )

    num_places = max(len(sgm.place_nodes) for sgm in sgms)
    gamma = np.zeros((num_places, num_places), dtype=np.float32)
    R = np.zeros((num_places, num_categories), dtype=np.float32)

    for sgm in sgms:
        pp = sgm.place_place_accessibility()
        n_p = pp.shape[0]
        gamma[:n_p, :n_p] += pp

        po = sgm.place_object_matrix(num_categories)
        R[:po.shape[0], :] += po

    gamma /= len(sgms)
    return gamma, R