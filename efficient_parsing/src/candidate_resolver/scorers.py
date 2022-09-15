"""
This module provides different scoring functions for the nearest neighbor search.
"""
import numpy as np


def euclidean_distance(x, y) -> np.ndarray:
    return np.linalg.norm(np.array(x) - np.array(y))


def normalized_distance(x, y, normalization_parameter: float = 1.0) -> float:
    return normalize_distance(euclidean_distance(x, y), normalization_parameter)


def normalize_distance(distance: np.ndarray, normalization_parameter: float):
    return 2 / (1 + distance * normalization_parameter) - 1
