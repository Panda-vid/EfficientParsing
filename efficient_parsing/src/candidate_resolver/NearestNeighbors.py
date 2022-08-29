import numpy as np
from sklearn.neighbors import KDTree


class NearestNeighbors:
    def __init__(self, examples: np.ndarray, radius: float = 400.0):
        self.radius = radius
        self.examples = examples
        self.performant_implementation = None if len(self.examples.shape) >= 3 else \
            KDTree(self.examples, leaf_size=2, metric="euclidean")

        if len(self.examples.shape) < 3:
            self.performant_implementation = KDTree(self.examples)

    def query_radius(self, query: np.ndarray):
        indices, distances = self.query_radius_brute(query) if self.performant_implementation is None \
            else self.performant_implementation.query_radius(
            query[0] if len(query.shape) == 3 else query, self.radius, return_distance=True, sort_results=True
        )
        return (indices.flatten()[0], distances.flatten()[0]) if self.performant_implementation is not None \
            else (indices.flatten(), distances.flatten())

    def query_radius_brute(self, query: np.ndarray):
        # noinspection PyTypeChecker
        distances = np.linalg.norm(self.examples - query, axis=(2, 3)).flatten()
        distances = distances[distances < self.radius]
        indices = np.argsort(distances)
        return indices, distances[indices]

