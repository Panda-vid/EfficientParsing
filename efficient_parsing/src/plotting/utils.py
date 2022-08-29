import numpy as np
import pandas as pd

from math import sqrt
from sklearn import manifold


def smallest_divisor(number: int):
    i = 2
    if number % i == 0:
        return i
    i += 1
    while i <= sqrt(number):
        if number % i == 0:
            return i
        i += 2


def tsne_reduce(X, metric="cosine", number_of_components=3):
    return manifold.TSNE(
        n_components=number_of_components,
        early_exaggeration=20.0,
        perplexity=1.0,
        init="pca",
        metric=metric
    ).fit_transform(X)


def compute_vmin(column: pd.Series) -> float:
    return np.min(np.array([np.min(distances) for distances in column]))
