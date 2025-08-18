# file: semantic_graph.py
# Module to build and maintain a semantic graph of the environment (nodes: places, images, objects; edges: relationships).
import numpy as np

class SemanticGraph:
    def __init__(self):
        # Lists of graph nodes
        self.place_nodes = []   # each place can be a cluster id or coordinate
        self.image_nodes = []   # each image node corresponds to a viewpoint
        self.object_nodes = []  # each object node corresponds to a detected object category
        # Adjacency relations
        self.place_image_affinity = {}  # key: (place_id, image_id), value: affinity score
        self.image_object_affinity = {} # key: (image_id, object_id), value: affinity score
        self.image_adj = {}             # key: (image_id, neighbor_image_id), value: 1 if connected
        self.place_cluster = {}         # mapping image_id -> place_id
        self.next_place_id = 0
        self.next_image_id = 0
        self.next_object_id = 0

    def add_place(self, place_feature=None):
        """Create a new place node (cluster) in the graph."""
        place_id = self.next_place_id
        self.next_place_id += 1
        # Optionally store features or coordinates
        self.place_nodes.append(place_id)
        return place_id

    def add_image(self, image_feature=None, place_id=None):
        """Add an image node (viewpoint) to the graph, and link to a place."""
        image_id = self.next_image_id
        self.next_image_id += 1
        self.image_nodes.append(image_id)
        if place_id is not None:
            # Connect image to place with initial affinity (e.g., 1.0)
            self.place_image_affinity[(place_id, image_id)] = 1.0
            self.place_cluster[image_id] = place_id
        return image_id

    def add_object(self, object_category, image_id):
        """Add an object node with given category, and link to the image node."""
        # Check if object category already has a node
        for obj_id, cat in self.object_nodes:
            if cat == object_category:
                # Link existing object node to image
                self.image_object_affinity[(image_id, obj_id)] = 1.0
                return obj_id
        # Else create a new object node
        object_id = self.next_object_id
        self.next_object_id += 1
        self.object_nodes.append((object_id, object_category))
        self.image_object_affinity[(image_id, object_id)] = 1.0
        return object_id

    def connect_images(self, img_id1, img_id2):
        """Connect two image nodes (e.g., when agent moves between viewpoints)."""
        self.image_adj[(img_id1, img_id2)] = 1
        self.image_adj[(img_id2, img_id1)] = 1

    def update(self, observation):
        """
        Update the semantic graph given a new observation.
        - observation may include current image features and detected objects.
        """
        # Pseudocode outline (needs actual implementation):
        # 1. Extract image features and object detections from observation.
        # 2. Compare current image to previous image nodes (cosine similarity).
        # 3. If new place (image similarity below threshold): add new image node & connect to a new place cluster.
        # 4. Else, update existing image node or link accordingly.
        # 5. For each detected object: add to graph if new, or update link strength.
        pass

    def get_place_adjacency(self):
        """Return adjacency of place clusters inferred from connected image nodes."""
        # Build place-place adjacency: if two places share an image connectivity, mark reachable.
        place_adj = np.zeros((self.next_place_id, self.next_place_id))
        for (img1, img2), conn in self.image_adj.items():
            if img1 in self.place_cluster and img2 in self.place_cluster:
                p1 = self.place_cluster[img1]
                p2 = self.place_cluster[img2]
                if p1 != p2:
                    place_adj[p1, p2] = 1
                    place_adj[p2, p1] = 1
        return place_adj

    def get_place_object_counts(self):
        """Return matrix of counts of object nodes connected to each place cluster."""
        # Count number of object connections per place cluster.
        counts = np.zeros((self.next_place_id, ))  # flatten for simplicity
        for (img_id, obj_id), affinity in self.image_object_affinity.items():
            if img_id in self.place_cluster:
                p = self.place_cluster[img_id]
                counts[p] += 1  # or affinity
        return counts
