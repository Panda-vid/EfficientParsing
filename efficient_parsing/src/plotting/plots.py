import itertools
from pathlib import Path

import pandas as pd
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

from typing import Tuple, List
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

from src.candidate_resolver.CandidateResolver import CandidateResolver
from src.candidate_resolver.embedding.embedding_utils import compute_cosine_similarities
from src.plotting.utils import compute_vmin, smallest_divisor, tsne_reduce


FIGURES_PATH = Path(__file__).parents[3] / "res" / "figures"


def plot_function_of_candidate_resolver_models_by_distance(
        plot_name: str,
        models: List[CandidateResolver],
        model_labels: List[str],
        max_distance: float = 50):
    if len(models) != len(model_labels):
        raise ValueError("There has to be a label in model_labels for each model in models.\n" +
                         f"Given: {len(models)} models and {len(model_labels)} labels.")

    x = np.linspace([0], [max_distance], 10000)
    fig, ax = plt.subplots(figsize=(20, 10))
    for i, model in enumerate(models):
        y = np.array(model.regressor(x))
        ax.plot(x.flatten(), y.flatten(), label=model_labels[i])
        ax.set_xlabel("distance", fontsize=20)
        ax.set_ylabel("regressor_output", fontsize=20)
    ax.legend(loc="upper right", fontsize=15)
    ax.tick_params(labelsize=15)
    plt.savefig(str(FIGURES_PATH / (plot_name + ".pdf")))
    plt.show()


def create_distance_histplot(
        plot_name: str,
        data: pd.DataFrame,
        feature_column_name: str,
        label_column_name: str,
        scorer,
):
    feature_vectors = data[feature_column_name]
    labels = data[label_column_name]
    same_class_distances = []
    different_class_distances = []
    sns.set(font_scale=1.0)
    for i, j in itertools.combinations(range(len(feature_vectors)), 2):
        feature_vec1 = feature_vectors[i]
        feature_vec2 = feature_vectors[j]
        label1 = labels[i]
        label2 = labels[j]
        if label1 == label2:
            same_class_distances.append(scorer(feature_vec1, feature_vec2))
        else:
            different_class_distances.append(scorer(feature_vec1, feature_vec2))

    plt.style.use('seaborn-deep')
    fig, ax = plt.subplots()
    ax.hist([different_class_distances, same_class_distances],
            label=[
                 "between instances with different label",
                 "between instances with same label"
             ])
    ax.set_xlabel("distance")
    ax.set_ylabel("numbber of occurances")
    ax.legend(loc="upper right")
    plt.savefig(str(FIGURES_PATH / (plot_name + ".pdf")))
    plt.show()


# noinspection PyTypeChecker
def create_three_dimensional_scatter_of(
        plot_name: str,
        dataframe: pd.DataFrame,
        feature_column_name: str,
        label_column_name: str,
        plot_title: str):
    reduced_vectors = reduce_from_dataframe(dataframe, feature_column_name, output_dim=3)
    xs = reduced_vectors[:, 0]
    ys = reduced_vectors[:, 1]
    zs = reduced_vectors[:, 2]
    ax = three_dim_scatter_of_classes(xs, ys, zs, dataframe, label_column_name)
    ax.set_title(plot_title, fontsize=20)
    ax.xaxis.get_label().set_fontsize(20)
    ax.yaxis.get_label().set_fontsize(20)
    ax.zaxis.get_label().set_fontsize(20)
    ax.tick_params(labelsize=20)
    plt.legend()
    plt.savefig(str(FIGURES_PATH / (plot_name + ".pdf")))
    plt.show()


def create_heatmaps_of_class_internal_distances(
        plot_name: str,
        data: pd.DataFrame,
        distances_column_name: str,
        class_ax_format_string: str,
        super_title: str,
        axes_title_padding: int = 17,
        **subplots_adjust_kwargs):

    classes = data["class"].unique()
    sns.set(font_scale=1.4)
    axes, cbar_ax = setup_figure(super_title, len(classes))
    for i, ax in enumerate(axes):
        ax.set_title(class_ax_format_string.format(classes[i]), pad=axes_title_padding)
        sns.heatmap(
            data[distances_column_name][i],
            ax=ax, cbar=(i == 0),
            vmin=compute_vmin(data[distances_column_name]),
            vmax=1,
            cbar_ax=None if i else cbar_ax
        )
    plt.subplots_adjust(**subplots_adjust_kwargs)
    plt.savefig(str(FIGURES_PATH / (plot_name + ".pdf")))


def create_similarity_heatmap_across_all_feature_vectors_divided_by_class(
        plot_name: str,
        data: pd.DataFrame,
        input_column_name: str,
        super_title: str,
        **subplots_adjust_kwargs):
    all_vectors = np.concatenate(data[input_column_name].to_numpy(), axis=0)
    all_dists = compute_cosine_similarities(all_vectors)
    classes = data["class"].unique()
    sns.set(font_scale=1.4)
    axes, cbar_ax = setup_figure(super_title, len(classes), figsize=(10, 20), separate_cbar_ax=True, sharex=True)
    vmin = np.min(all_dists)
    xticklabels = get_xticklabels(data, input_column_name)

    feature_index, class_index = data.loc[data["class"] == classes[0]][input_column_name].to_numpy()[0].shape[0], 1
    create_subheatmap(
        axes[0], xticklabels, classes[0],
        all_dists[0:feature_index, :], vmin, 1.0,
        True, cbar_ax,
        labelbottom=False, labelleft=False
    )

    while feature_index < len(all_vectors) and class_index < len(classes):
        len_class = data.loc[data["class"] == classes[class_index]][input_column_name].to_numpy()[0].shape[0]
        class_begin = feature_index
        class_end = class_begin + len_class
        create_subheatmap(
            axes[class_index], xticklabels, classes[class_index],
            all_dists[class_begin:class_end, :], vmin, 1.0,
            False, None,
            labelbottom=(class_index + 2 >= len(classes)), labelleft=False
        )
        feature_index = class_end
        class_index += 1
    plt.subplots_adjust(**subplots_adjust_kwargs)
    plt.savefig(str(FIGURES_PATH / (plot_name + ".pdf")))


def three_dim_scatter_of_classes(
        xs: np.ndarray,
        ys: np.ndarray,
        zs: np.ndarray,
        dataframe: pd.DataFrame,
        class_column: str):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')
    colors = mcolors.TABLEAU_COLORS
    colors = [color_name for color_name in colors.keys()]
    for i, cls in enumerate(dataframe[class_column].unique()):
        indices = dataframe.index[dataframe[class_column] == cls].tolist()
        ax.scatter(xs[indices], ys[indices], zs[indices], c=colors[i], label=cls)
    return ax


def reduce_from_dataframe(dataframe: pd.DataFrame, column_name: str, output_dim="auto"):
    vectors = dataframe[column_name]
    input_rank = tf.rank(vectors[0])
    if input_rank == 1:
        vectors = vectors.apply(lambda vector: vector[tf.newaxis, :])
    if input_rank > 2:
        raise ValueError(f"The input can have a maximum rank of 2.")
    input_shape = vectors[0].shape
    if not isinstance(output_dim, int):
        if output_dim.lower() == "auto":
            output_dim = smallest_divisor(input_shape[-1])
    vectors = np.vstack(vectors)
    return tsne_reduce(vectors, number_of_components=output_dim)


def setup_figure(super_title: str,
                 number_of_subplots: int,
                 figsize: Tuple[int, int] = (15, 20),
                 separate_cbar_ax: bool = True,
                 sharex: bool = False):
    fig = plt.figure(figsize=figsize)
    plt.suptitle(super_title, fontsize=20)
    number_of_rows = int(number_of_subplots / 2) + 1
    fig, axes = create_axes(fig, number_of_rows, number_of_subplots, sharex)
    return axes, fig.add_axes([.93, .3, .02, .3]) if separate_cbar_ax else axes


def create_axes(fig, number_of_rows: int, number_of_subplots: int, sharex: bool):
    axes = []
    for i in range(number_of_subplots):
        ax = fig.add_subplot(number_of_rows, 2, i + 1, sharex=None if not (sharex and i > 0) else axes[0])
        axes.append(ax)
    return fig, axes


def create_subheatmap(
        ax,
        xticklabels: np.ndarray, ylabel: str,
        distances: np.ndarray,
        vmin: float, vmax: float,
        cbar: bool, cbar_ax,
        **tick_params_kwargs):
    ax.tick_params(**tick_params_kwargs)
    sns.heatmap(
        distances,
        ax=ax,
        vmin=vmin, vmax=vmax,
        cbar_kws={'label': 'cosine similarity'}, cbar=cbar, cbar_ax=cbar_ax
    )
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(xticklabels, rotation="vertical")


def get_xticklabels(data: pd.DataFrame, input_column_name: str):
    return np.concatenate([
        np.repeat(data["class"][i], len(data[input_column_name][i]))
        for i in range(len(data.index))
    ])


def draw_sphere(x, y, z, radius, color, ax):
    u, v = np.mgrid[0: 2*np.pi:20j, 0:np.pi:10j]
    x = x + radius * (np.cos(u) * np.sin(v))
    y = y + radius * (np.sin(u) * np.sin(v))
    z = z + radius * np.cos(v)
    ax.plot_surface(x, y, z, color=color, alpha=0.1)
