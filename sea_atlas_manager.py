# file: sea_atlas_manager.py
# Manages the Semantic Environment Atlas built from multiple training environments.
import numpy as np

class SEAAtlasManager:
    def __init__(self):
        # Place-place reachability matrix (R) and place-object connection matrix (C)
        self.place_reachability = None  # shape [P, P]
        self.place_object = None        # shape [P, M] (M = number of object categories)
        self.num_places = 0
        self.num_objects = 0
        self.place_ids = []             # List of global place cluster ids
        self.object_ids = []            # List of object categories (indexed)
        self.threshold = 0.5            # Threshold for deciding new connections (tunable)

    def build_atlas(self, semantic_graphs):
        """
        Build global SEA atlas from a list of SemanticGraph instances from training scenes.
        - semantic_graphs: list of SemanticGraph objects (one per training environment).
        """
        # Collect unique place clusters and object categories across graphs
        global_place_set = set()
        global_object_set = set()
        for sg in semantic_graphs:
            for p in sg.place_nodes:
                global_place_set.add(p)
            for (_, cat) in sg.object_nodes:
                global_object_set.add(cat)
        self.place_ids = sorted(list(global_place_set))
        self.object_ids = sorted(list(global_object_set))
        P = len(self.place_ids)
        M = len(self.object_ids)
        self.num_places, self.num_objects = P, M
        # Initialize matrices
        R = np.zeros((P, P))
        C = np.zeros((P, M))
        # Map place/object to index
        place_to_idx = {p: i for i, p in enumerate(self.place_ids)}
        obj_to_idx = {cat: i for i, cat in enumerate(self.object_ids)}
        # Aggregate reachability and object counts
        env_count = np.zeros(P)  # count of environments each place appears
        for sg in semantic_graphs:
            # Determine unique places in this scene
            scene_places = set(sg.place_nodes)
            for p in scene_places:
                env_count[place_to_idx[p]] += 1
            # Place-place reachability from this scene
            local_adj = sg.get_place_adjacency()
            for i, p in enumerate(self.place_ids):
                for j, q in enumerate(self.place_ids):
                    if p != q and p in sg.place_nodes and q in sg.place_nodes:
                        # If any image connection indicates reachability
                        if local_adj[sg.place_nodes.index(p), sg.place_nodes.index(q)] > 0:
                            R[i, j] += 1
            # Place-object connections
            # For each image-object edge, link through its place
            for (img_id, obj_id), affinity in sg.image_object_affinity.items():
                if img_id in sg.place_cluster:
                    p = sg.place_cluster[img_id]
                    if p in place_to_idx:
                        i = place_to_idx[p]
                        # Find object category index
                        cat = None
                        for (_id, c) in sg.object_nodes:
                            if _id == obj_id:
                                cat = c; break
                        if cat is not None:
                            j = obj_to_idx[cat]
                            C[i, j] += 1
        # Normalize reachability: binary if >0 in any scene
        R_bin = (R > 0).astype(float)
        # Average reachability over the number of scenes where both places appear
        # Avoid division by zero
        for i in range(P):
            for j in range(P):
                if R_bin[i, j] > 0:
                    # Normalize by min(env_count[i], env_count[j])
                    denom = max(env_count[i], 1)
                    R[i, j] = R[i, j] / denom
        self.place_reachability = R
        self.place_object = C  # raw counts

    def localize(self, observed_objects):
        """
        Implicit localization: Given observed object categories (list or set), produce belief over places.
        - observed_objects: list of detected object categories in current observation.
        Returns:
            belief: np.array of length num_places, probability distribution over place clusters.
        """
        if self.place_object is None or self.place_reachability is None:
            # Atlas not built yet
            return np.ones(self.num_places) / self.num_places
        # Compute a simple belief: P(place | obs) proportional to sum over objects of P(object|place)
        # Here place_object holds counts; convert to probabilities P(object|place)
        place_object_prob = self.place_object.copy().astype(float)
        # Normalize counts to probabilities for each place cluster
        place_object_prob = place_object_prob / (np.sum(place_object_prob, axis=1, keepdims=True) + 1e-6)
        # Initialize uniform prior
        belief = np.ones(self.num_places) / self.num_places
        # Update belief for each observed object category
        for obj in observed_objects:
            if obj in self.object_ids:
                j = self.object_ids.index(obj)
                # Likelihood of object given place is place_object_prob[:, j]
                likelihood = place_object_prob[:, j]
                belief = belief * (likelihood + 1e-6)
        # Normalize to get posterior
        if belief.sum() > 0:
            belief = belief / belief.sum()
        else:
            # If no matching observations, keep uniform
            belief = np.ones(self.num_places) / self.num_places
        return belief

    def select_subgoal(self, goal_category, belief):
        """
        Select a subgoal place cluster based on target category and current belief distribution.
        - goal_category: target object category.
        - belief: current belief distribution over places (np.array length num_places).
        Returns:
            subgoal_place: index of the chosen place cluster as subgoal.
        """
        # Determine target place cluster: P(place|goal) proportional to place_object[:, goal]
        if goal_category not in self.object_ids or self.place_object is None:
            # No info about goal or atlas empty: fallback to most probable current place
            return np.argmax(belief)
        j = self.object_ids.index(goal_category)
        # Compute probability P(place | goal) proportional to place_object[:, j]
        p_place_given_goal = self.place_object[:, j].astype(float)
        if p_place_given_goal.sum() > 0:
            p_place_given_goal = p_place_given_goal / (p_place_given_goal.sum() + 1e-6)
        else:
            p_place_given_goal = np.ones(self.num_places) / self.num_places
        # Choose target place cluster (highest probability)
        target_place = np.argmax(p_place_given_goal)
        # Determine current place: max of belief
        current_place = np.argmax(belief)
        # If current_place is the target, return it (no further subgoal needed)
        if current_place == target_place:
            return target_place
        # Otherwise, select a neighboring place of current that moves towards target
        neighbors = np.nonzero(self.place_reachability[current_place] > self.threshold)[0]
        if len(neighbors) == 0:
            # No immediate neighbors recorded: go directly to target
            return target_place
        # Among neighbors, pick the one with highest reachability to target
        best_neighbor = current_place
        best_score = -1
        for n in neighbors:
            score = self.place_reachability[n, target_place]
            if score > best_score:
                best_score = score
                best_neighbor = n
        return best_neighbor
