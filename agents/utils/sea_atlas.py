import numpy as np


class SemanticEnvironmentAtlas:
    """Lightweight container for semantic features observed over time."""

    def __init__(self, update_interval: int = 1):
        self.update_interval = max(1, update_interval)
        self._step = 0
        self.observations = []
        self.poses = []

    def update(self, observation, pose):
        """Store observation and pose at a fixed interval.

        Args:
            observation: Current observation tensor/array.
            pose: Agent pose corresponding to the observation.
        """
        self._step += 1
        if self._step % self.update_interval == 0:
            self.observations.append(np.array(observation))
            self.poses.append(np.array(pose))

    def query(self, goal_category):
        """Return stored features for planning.

        Args:
            goal_category: Category index for which features are requested.

        Returns:
            ndarray or None: Features corresponding to the goal.
        """
        if not self.observations:
            return None
        # For now simply return last observation as placeholder feature map.
        return self.observations[-1]