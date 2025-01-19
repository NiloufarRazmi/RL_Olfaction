"""Define all the plotting functions."""

import itertools
import shutil
from collections import OrderedDict
from functools import partial

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
import torch

# from curlyBrace import curlyBrace
from imojify import imojify
from matplotlib.gridspec import GridSpec
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

# from utils import get_location_count
from .environment import Actions, TriangleState

sns.set_theme(font_scale=1.5)
# plt.style.use("ggplot")
# print(shutil.which("latex"))
USETEX = bool(shutil.which("latex"))
mpl.rcParams["text.usetex"] = USETEX
if USETEX:
    mpl.rcParams["font.family"] = ["serif"]
else:
    mpl.rcParams["font.family"] = ["sans-serif"]
    mpl.rcParams["font.sans-serif"] = [
        "Fira Sans",
        "Computer Modern Sans Serif",
        "DejaVu Sans",
        "Verdana",
        "Arial",
        "Helvetica",
    ]


def plot_steps_and_rewards(df, n_runs=1, figpath=None, logger=None):
    """Plot the steps and rewards from dataframes."""
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    sns.lineplot(data=df, x="Episodes", y="Rewards", ax=ax[0], color="black")
    ax[0].set(
        ylabel=f"Rewards\naveraged over {n_runs} runs" if n_runs > 1 else "Rewards"
    )

    sns.lineplot(data=df, x="Episodes", y="Steps", ax=ax[1], color="black")
    ax[1].set(
        ylabel=(
            f"Steps number\naveraged over {n_runs} runs"
            if n_runs > 1
            else "Steps number"
        )
    )

    for axi in ax:
        axi.set_facecolor("0.9")
    fig.tight_layout()
    fig.patch.set_alpha(0)
    fig.patch.set_facecolor("white")
    if figpath:
        figfullpath = figpath / "steps-and-rewards.png"
        fig.savefig(figfullpath, bbox_inches="tight")
        if logger:
            msg = f"Saved figure to: {figfullpath.absolute()}"
            print(msg)
            logger.info(msg)
    # plt.show()


def plot_steps_and_rewards_dist(df, figpath=None, logger=None):
    """Plot the steps and rewards distributions from dataframes."""
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    sns.histplot(data=df, x="Rewards", ax=ax[0], color="black")
    sns.histplot(data=df, x="Steps", ax=ax[1], color="black")
    for axi in ax:
        axi.set_facecolor("0.9")
    fig.tight_layout()
    fig.patch.set_alpha(0)
    fig.patch.set_facecolor("white")
    if figpath:
        figfullpath = figpath / "steps-and-rewards-distrib.png"
        fig.savefig(figfullpath, bbox_inches="tight")
        if logger:
            msg = f"Saved figure to: {figfullpath.absolute()}"
            print(msg)
            logger.info(msg)
    # plt.show()


def qtable_directions_map(qtable, rows, cols):
    """Get the best learned action & map it to arrows."""
    qtable_val_max = qtable.max(axis=1).values.reshape(rows, cols)
    qtable_best_action = qtable.argmax(axis=1).reshape(rows, cols)
    directions = {
        Actions.UP: "↑",
        Actions.DOWN: "↓",
        Actions.LEFT: "←",
        Actions.RIGHT: "→",
    }
    qtable_directions = np.empty(qtable_best_action.flatten().shape, dtype=str)
    eps = torch.finfo(torch.float64).eps  # Minimum float number on the machine
    for idx, val in enumerate(qtable_best_action.flatten()):
        if qtable_val_max.flatten()[idx] > eps:
            # Assign an arrow only if a minimal Q-value has been learned as best action
            # otherwise since 0 is a direction, it also gets mapped on the tiles where
            # it didn't actually learn anything
            qtable_directions[idx] = directions[Actions(val.item())]
    qtable_directions = qtable_directions.reshape(rows, cols)
    return qtable_val_max, qtable_directions


# def plot_q_values_map(qtable, rows, cols):
#     """
#     Plot the heatmap of the Q-values.

#     Also plot the best action's direction with arrows.
#     """
#     # TODO: Remove function as it doesn't seem to be used anymore?
#     qtable_val_max, qtable_directions = qtable_directions_map(qtable, rows, cols)

#     # font_name = "DejaVu Math TeX Gyre"
#     # mpl.rcParams["font.family"] = font_name
#     f, ax = plt.subplots()
#     sns.heatmap(
#         qtable_val_max,
#         annot=qtable_directions,
#         fmt="",
#         ax=ax,
#         cmap=sns.color_palette("Blues", as_cmap=True),
#         linewidths=0.7,
#         linecolor="black",
#         xticklabels=[],
#         yticklabels=[],
#         annot_kws={"fontsize": "xx-large"},
#     ).set(title="Learned Q-values\nArrows represent best action")
#     for _, spine in ax.spines.items():
#         spine.set_visible(True)
#         spine.set_linewidth(0.7)
#         spine.set_color("black")
#     plt.show()


# def plot_heatmap(matrix, title=None, xlabel=None, ylabel=None, braces=[]):
#     """Plot heatmap."""
#     fig, ax = plt.subplots(figsize=(9, 9))
#     cmap = sns.light_palette("seagreen", as_cmap=True)
#     # cmap = sns.color_palette("light:b", as_cmap=True)
#     chart = sns.heatmap(matrix, cmap=cmap, ax=ax)
#     if title:
#         chart.set(title=title)
#     if xlabel:
#         chart.set(xlabel=xlabel)
#     if ylabel:
#         chart.set(ylabel=ylabel)
#     ax.tick_params(axis="x", rotation=90)
#     ax.tick_params(axis="y", rotation=0)
#     ax.tick_params(left=True)
#     ax.xaxis.tick_top()
#     if braces:
#         for _, brace in enumerate(braces):
#             curlyBrace(
#                 fig=fig,
#                 ax=ax,
#                 p1=brace["p1"],
#                 p2=brace["p2"],
#                 k_r=0.05,
#                 bool_auto=False,
#                 str_text=brace["str_text"],
#                 color="black",
#                 lw=2,
#                 int_line_num=2,
#             )
#     plt.show()


def plot_tiles_locations(tiles_list, rows, cols, title=None):
    """Plot the states/tiles numbers."""
    tiles_annot = np.reshape(list(tiles_list), (rows, cols))
    tiles_val = np.zeros_like(tiles_annot)

    f, ax = plt.subplots()
    chart = sns.heatmap(
        tiles_val,
        annot=tiles_annot,
        fmt="",
        ax=ax,
        cbar=False,
        # cmap=sns.color_palette("Blues", as_cmap=True),
        linewidths=0.7,
        linecolor="white",
        xticklabels=[],
        yticklabels=[],
        annot_kws={"fontsize": "xx-large"},
    )
    if title:
        chart.set(title=title)
    else:
        chart.set(title="Tiles numbers")
    # for _, spine in ax.spines.items():
    #     spine.set_visible(True)
    #     spine.set_linewidth(0.7)
    #     spine.set_color("black")
    f.set_facecolor("white")
    plt.show()


def plot_q_values_maps(qtable, rows, cols, labels):
    """
    Plot the heatmap of the Q-values.

    Also plot the best action's direction with arrows.
    """
    # font_name = "DejaVu Math TeX Gyre"
    # mpl.rcParams["font.family"] = font_name

    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    maps = [
        np.arange(0, rows * cols),
        np.arange(rows * cols, 2 * rows * cols),
        np.arange(2 * rows * cols, 3 * rows * cols),
        np.arange(3 * rows * cols, 4 * rows * cols),
    ]
    for idx, cue in enumerate(labels):
        qtable_val_max, qtable_directions = qtable_directions_map(
            qtable[maps[idx], :], rows, cols
        )
        sns.heatmap(
            qtable_val_max,
            annot=qtable_directions,
            fmt="",
            ax=ax.flatten()[idx],
            cmap=sns.color_palette("Blues", as_cmap=True),
            linewidths=0.7,
            linecolor="black",
            xticklabels=[],
            yticklabels=[],
            annot_kws={"fontsize": "xx-large"},
            cbar_kws={"label": "Q-value"},
        ).set(title=labels[cue])
        for _, spine in ax.flatten()[idx].spines.items():
            spine.set_visible(True)
            spine.set_linewidth(0.7)
            spine.set_color("black")

        # Annotate the ports names
        bbox = {
            "facecolor": "black",
            "edgecolor": "none",
            "boxstyle": "round",
            "alpha": 0.1,
        }
        ax.flatten()[idx].text(
            x=4.7,
            y=0.3,
            s="N",
            bbox=bbox,
            color="white",
        )
        ax.flatten()[idx].text(
            x=0.05,
            y=4.9,
            s="S",
            bbox=bbox,
            color="white",
        )
        ax.flatten()[idx].text(
            x=4.7,
            y=4.9,
            s="E",
            bbox=bbox,
            color="white",
        )
        ax.flatten()[idx].text(
            x=0.05,
            y=0.3,
            s="W",
            bbox=bbox,
            color="white",
        )

    # Make background transparent
    fig.patch.set_alpha(0)
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    plt.show()


def add_emoji(coords, emoji, ax):
    """Add emoji as image at absolute coordinates."""
    img = plt.imread(imojify.get_img_path(emoji))
    im = OffsetImage(img, zoom=0.08)
    im.image.axes = ax
    ab = AnnotationBbox(
        im, (coords[0], coords[1]), frameon=False, pad=0, annotation_clip=False
    )
    ax.add_artist(ab)


def plot_policy_emoji(qtable, rows, cols, label, emoji):
    """
    Plot the heatmap of the Q-values.

    Also plot the best action's direction with arrows.
    """
    grid_spec = {"width_ratios": (0.9, 0.05)}
    fig, (ax, cbar_ax) = plt.subplots(1, 2, gridspec_kw=grid_spec, figsize=(10, 8))

    qtable_val_max, qtable_directions = qtable_directions_map(qtable, rows, cols)
    chart = sns.heatmap(
        qtable_val_max,
        annot=qtable_directions,
        fmt="",
        ax=ax,
        cmap=sns.color_palette("Blues", as_cmap=True),
        linewidths=0.7,
        linecolor="black",
        xticklabels=[],
        yticklabels=[],
        annot_kws={"fontsize": "xx-large"},
        cbar_ax=cbar_ax,
        cbar_kws={"label": "Q-value"},
    )
    chart.set_title(label=label, fontsize=40)
    cbar_ax.yaxis.label.set_size(30)
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
        spine.set_color("black")
    for _, emo in enumerate(emoji):
        add_emoji(emo["coords"], emo["emoji"], ax)

    fig.patch.set_alpha(0)
    fig.patch.set_facecolor("white")
    plt.show()


# def plot_states_actions_distribution(states, actions):
#     """Plot the distributions of states and actions."""
#     fig, ax = plt.subplots(2, 1, figsize=(12, 5))
#     sns.histplot(data=states, ax=ax[0], kde=True)
#     ax[0].set_title("States")
#     sns.histplot(data=actions, ax=ax[1])
#     ax[1].set_title("Actions")
#     fig.tight_layout()
#     check_plots()
#     plt.savefig(PLOTS_PATH / "states+actions.png", bbox_inches="tight")
#     plt.show()


def plot_rotated_q_values_maps(qtable, rows, cols, labels):
    """Plot 45° rotated Q-values."""
    # See https://stackoverflow.com/q/12848581/4129062

    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    maps = [
        np.arange(0, rows * cols),
        np.arange(rows * cols, 2 * rows * cols),
        np.arange(2 * rows * cols, 3 * rows * cols),
        np.arange(3 * rows * cols, 4 * rows * cols),
    ]
    for idx, cue in enumerate(labels):
        qtable_val_max, qtable_directions = qtable_directions_map(
            qtable[maps[idx], :], rows, cols
        )

        # im = ax.imshow(qtable_val_max, cmap="Blues")
        # im = ax.pcolormesh(np.flip(qtable_val_max, axis=0), cmap="Blues")

        def pcolormesh_45deg(C):
            n = C.shape[0]
            # create rotation/scaling matrix
            t = np.array([[1, 0.5], [-1, 0.5]])
            # create coordinate matrix and transform it
            A = np.dot(
                np.array(
                    [
                        (i[1], i[0])
                        for i in itertools.product(range(n, -1, -1), range(0, n + 1, 1))
                    ]
                ),
                t,
            )
            # plot
            return (
                A[:, 1].reshape(n + 1, n + 1),
                A[:, 0].reshape(n + 1, n + 1),
                np.flipud(C),
            )

        X, Y, C = pcolormesh_45deg(qtable_val_max)
        im = ax.flatten()[idx].pcolormesh(X, Y, C, cmap="Blues")
        ax.flatten()[idx].figure.colorbar(im, ax=ax.flatten()[idx])

        # Loop over data dimensions and create text annotations.
        for i in range(rows):
            for j in range(cols):
                color = (
                    "white" if qtable_val_max[i, j] > qtable_val_max.mean() else "black"
                )
                ax.flatten()[idx].text(
                    # j,
                    # i,
                    np.flipud(X)[i, j] + 0.5,
                    np.flipud(Y)[i, j],
                    qtable_directions[i, j],
                    ha="center",
                    va="center",
                    color=color,
                    rotation=45,
                )

        ax.flatten()[idx].set_title(labels[cue])
        # ax.set_xticks(np.arange(qtable_val_max.shape[1] + 1) - 0.5, minor=True)
        # ax.set_yticks(np.arange(qtable_val_max.shape[0] + 1) - 0.5, minor=True)
        # ax.grid(which="minor", color="black", linestyle="-", linewidth=0.7)
        ax.flatten()[idx].set_xticks([])
        ax.flatten()[idx].set_yticks([])
        # for _, spine in ax.spines.items():
        #     spine.set_visible(True)
        #     spine.set_linewidth(0.7)
        #     spine.set_color("black")

    fig.tight_layout()
    plt.show()


# def plot_location_count(
#     all_state_composite, tiles_locations, cols, rows, cues=None, contexts_labels=None
# ):
#     """Plot location counts heatmap."""
#     # cmap = sns.color_palette("Blues", as_cmap=True)
#     cmap = sns.color_palette("rocket_r", as_cmap=True)

#     if cues and contexts_labels:
#         fig, ax = plt.subplots(1, 3, figsize=(12, 4))
#         for idx, cue in enumerate(cues):
#             location_count = get_location_count(
#                 all_state_composite=all_state_composite,
#                 tiles_locations=tiles_locations,
#                 cols=cols,
#                 rows=rows,
#                 cue=cue,
#             )
#             chart = sns.heatmap(location_count, cmap=cmap, ax=ax.flatten()[idx])
#             chart.set(title=contexts_labels[cue])
#             ax.flatten()[idx].set_xticks([])
#             ax.flatten()[idx].set_yticks([])
#         fig.suptitle("Locations counts during training")  # , fontsize="xx-large")

#     else:  # Plot everything
#         location_count = get_location_count(
#             all_state_composite,
#             tiles_locations=tiles_locations,
#             cols=cols,
#             rows=rows,
#             cue=cues,
#         )
#         fig, ax = plt.subplots(figsize=(10, 8))
#         chart = sns.heatmap(location_count, cmap=cmap, ax=ax)
#         chart.set(title="Locations count during training")
#         ax.set_xticks([])
#         ax.set_yticks([])
#     fig.tight_layout()
#     plt.show()


def plot_weights_biases_distributions(
    weights_df, biases_df, label=None, figpath=None, logger=None
):
    """Plot weights biases distributions."""
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(13, 5))

    ax[0].set_title("Weights")
    if label:
        ax[0].set_xlabel(label)
    else:
        ax[0].set_xlabel("Values")
    palette = sns.color_palette()[0 : len(weights_df.Layer.unique())]
    sns.histplot(
        data=weights_df,
        x="Val",
        hue="Layer",
        # kde=True,
        log_scale=True,  # if label == "Gradients" else False,
        palette=palette,
        ax=ax[0],
    )

    ax[1].set_title("Biases")
    if label:
        ax[1].set_xlabel(label)
    else:
        ax[1].set_xlabel("Values")
    eps = torch.finfo(torch.float64).eps
    palette = sns.color_palette()[
        0 : len(biases_df[biases_df.Val > eps].Layer.unique())
    ]
    sns.histplot(
        data=biases_df[biases_df.Val > eps],
        x="Val",
        hue="Layer",
        # kde=True,
        log_scale=True,
        palette=palette,
        ax=ax[1],
    )

    for axi in ax:
        axi.set_facecolor("0.9")
    # fig.tight_layout()
    fig.patch.set_alpha(0)
    fig.patch.set_facecolor("white")
    if figpath:
        figfullpath = figpath / f"weights-biases-distrib-{label}.png"
        fig.savefig(figfullpath, bbox_inches="tight")
        if logger:
            msg = f"Saved figure to: {figfullpath.absolute()}"
            print(msg)
            logger.info(msg)
    # plt.show()


def plot_weights_biases_stats(
    weights_stats, biases_stats, label=None, figpath=None, logger=None
):
    """Plot weights biases stats."""
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(13, 8))

    if label:
        ax[0, 0].set_title("Weights " + label)
    else:
        ax[0, 0].set_title("Weights")
    ax[0, 0].set_xlabel("Episodes")
    palette = sns.color_palette()[0 : len(weights_stats.Layer.unique())]
    sns.lineplot(
        data=weights_stats,
        x="Index",
        y="Std",
        hue="Layer",
        palette=palette,
        ax=ax[0, 0],
    )
    ax[0, 0].set(yscale="log")

    if label:
        ax[0, 1].set_title("Weights " + label)
    else:
        ax[0, 1].set_title("Weights")
    ax[0, 1].set_xlabel("Episodes")
    palette = sns.color_palette()[0 : len(weights_stats.Layer.unique())]
    sns.lineplot(
        data=weights_stats,
        x="Index",
        y="Avg",
        hue="Layer",
        palette=palette,
        ax=ax[0, 1],
    )
    ax[0, 1].set(yscale="log")

    if label:
        ax[1, 0].set_title("Biases " + label)
    else:
        ax[1, 0].set_title("Biases")
    ax[1, 0].set_xlabel("Steps")
    palette = sns.color_palette()[0 : len(biases_stats.Layer.unique())]
    sns.lineplot(
        data=biases_stats,
        x="Index",
        y="Std",
        hue="Layer",
        palette=palette,
        ax=ax[1, 0],
    )
    ax[1, 0].set(yscale="log")

    if label:
        ax[1, 1].set_title("Biases " + label)
    else:
        ax[1, 1].set_title("Biases")
    ax[1, 1].set_xlabel("Steps")
    palette = sns.color_palette()[0 : len(biases_stats.Layer.unique())]
    sns.lineplot(
        data=biases_stats,
        x="Index",
        y="Avg",
        hue="Layer",
        palette=palette,
        ax=ax[1, 1],
    )
    ax[1, 1].set(yscale="log")

    fig.tight_layout()
    fig.patch.set_alpha(0)
    fig.patch.set_facecolor("white")
    if figpath:
        figfullpath = figpath / f"weights-biases-stats-{label}.png"
        fig.savefig(figfullpath, bbox_inches="tight")
        if logger:
            msg = f"Saved figure to: {figfullpath.absolute()}"
            print(msg)
            logger.info(msg)
    # plt.show()


def plot_loss(loss_df, n_runs=1, figpath=None, logger=None):
    """Plot the training loss."""
    fig, ax = plt.subplots()
    sns.lineplot(data=loss_df, x="Steps", y="Loss", ax=ax, color="black")
    if USETEX:
        ax.set(
            ylabel=(
                f"$Log_{{10}}(\mathrm{{Loss}})$\naveraged over {n_runs} runs"
                if n_runs > 1
                else "$Log_{10}(\mathrm{Loss})$"
            )
        )
    else:
        ax.set(
            ylabel=(
                f"$Log_{{10}}(\\text{{Loss}})$\naveraged over {n_runs} runs"
                if n_runs > 1
                else "$Log_{10}(\\text{Loss})$"
            )
        )
    ax.set(xlabel="Steps")
    ax.set(yscale="log")
    ax.set_facecolor("0.9")
    fig.tight_layout()
    fig.patch.set_alpha(0)
    fig.patch.set_facecolor("white")
    if figpath:
        figfullpath = figpath / "loss.png"
        fig.savefig(figfullpath, bbox_inches="tight")
        if logger:
            msg = f"Saved figure to: {figfullpath.absolute()}"
            print(msg)
            logger.info(msg)
    # plt.show()


def plot_exploration_rate(epsilons, steps, figpath=None, logger=None):
    """Plot the exploration rate."""
    fig, ax = plt.subplots()
    sns.lineplot(epsilons, color="black")
    ax.set(ylabel="Epsilon")
    ax.set(xlabel="Steps")

    # Second x axis
    sec_x = ax.secondary_xaxis(location="top")
    sec_x.set(xlabel="Episodes")
    cumsteps = steps.flatten().cumsum()
    # qindex = np.quantile(index, np.array([1/4, 1/2, 3/4, 1]))
    qindex = np.percentile(cumsteps, np.arange(0, 100, 10))
    sec_x.xaxis.set_major_locator(ticker.FixedLocator(qindex))

    ax.set_facecolor("0.9")
    fig.tight_layout()
    fig.patch.set_alpha(0)
    fig.patch.set_facecolor("white")
    if figpath:
        figfullpath = figpath / "exploration-rate.png"
        fig.savefig(figfullpath, bbox_inches="tight")
        if logger:
            msg = f"Saved figure to: {figfullpath.absolute()}"
            print(msg)
            logger.info(msg)
    # plt.show()


def plot_actions_distribution(actions, figpath=None, logger=None):
    """Plot the distributions of states and actions."""
    fig, ax = plt.subplots()
    flatten_actions = flattenarray(actions, dtype=str)
    sns.histplot(data=flatten_actions, ax=ax, color="black")
    ax.set_xticks(
        [item.value for item in Actions], labels=[item.name for item in Actions]
    )
    ax.set_title("Actions")
    ax.set_facecolor("0.9")
    fig.tight_layout()
    fig.patch.set_alpha(0)
    fig.patch.set_facecolor("white")
    if figpath:
        figfullpath = figpath / "actions-distribution.png"
        fig.savefig(figfullpath, bbox_inches="tight")
        if logger:
            msg = f"Saved figure to: {figfullpath.absolute()}"
            print(msg)
            logger.info(msg)
    # plt.show()


def flattenarray(arr, dtype):
    """Flatten array of 1D lists into single 1D array."""
    flatten_length = np.sum([len(run) for run in arr])
    res = np.empty(flatten_length, dtype=dtype)
    for idx, val in enumerate(arr):
        if idx == 0:
            res[0 : len(val)] = val
            previous_run_len = 0
        else:
            previous_run_len += len(arr[idx - 1])
            res[previous_run_len : previous_run_len + len(val)] = val
    return res


# def plot_policies(q_values, labels, n_rows, n_cols, figpath=None, logger=None):
#     """
#     Plot the heatmap of the Q-values.

#     Also plot the best action's direction with arrows.
#     """
#     fig, ax = plt.subplots(2, len(labels), figsize=(13, 8))
#     for tri_i, tri_v in enumerate(TriangleState):
#         for cue_i, cue_v in enumerate(labels):
#             qtable_val_max, qtable_directions = qtable_directions_map(
#                 qtable=q_values[:, cue_i, :], rows=n_rows, cols=n_cols
#             )
#             if tri_v == TriangleState.upper:
#                 qtable_val_max = torch.triu(qtable_val_max)
#                 qtable_directions = np.triu(qtable_directions)
#             elif tri_v == TriangleState.lower:
#                 qtable_val_max = torch.tril(qtable_val_max)
#                 qtable_directions = np.tril(qtable_directions)
#             sns.heatmap(
#                 qtable_val_max.cpu(),
#                 annot=qtable_directions,
#                 fmt="",
#                 ax=ax[tri_i, cue_i],
#                 cmap=sns.color_palette("Blues", as_cmap=True),
#                 linewidths=0.7,
#                 linecolor="black",
#                 xticklabels=[],
#                 yticklabels=[],
#                 annot_kws={"fontsize": "xx-large"},
#                 cbar_kws={"label": "Q-value"},
#             ).set(title=labels[cue_v])
#             for _, spine in ax[tri_i, cue_i].spines.items():
#                 spine.set_visible(True)
#                 spine.set_linewidth(0.7)
#                 spine.set_color("black")

#             # Annotate the ports names
#             bbox = {
#                 "facecolor": "black",
#                 "edgecolor": "none",
#                 "boxstyle": "round",
#                 "alpha": 0.1,
#             }
#             ax[tri_i, cue_i].text(
#                 x=4.7,
#                 y=0.3,
#                 s="N",
#                 bbox=bbox,
#                 color="white",
#             )
#             ax[tri_i, cue_i].text(
#                 x=0.05,
#                 y=4.9,
#                 s="S",
#                 bbox=bbox,
#                 color="white",
#             )
#             ax[tri_i, cue_i].text(
#                 x=4.7,
#                 y=4.9,
#                 s="E",
#                 bbox=bbox,
#                 color="white",
#             )
#             ax[tri_i, cue_i].text(
#                 x=0.05,
#                 y=0.3,
#                 s="W",
#                 bbox=bbox,
#                 color="white",
#             )

#     # Make background transparent
#     fig.patch.set_alpha(0)
#     fig.patch.set_facecolor("white")
#     fig.tight_layout()
#     if figpath:
#         figfullpath = figpath / "policy.png"
#         fig.savefig(figfullpath, bbox_inches="tight")
#         if logger:
#             msg = f"Saved figure to: {figfullpath.absolute()}"
#             print(msg)
#             logger.info(msg)
#     # plt.show()


def plot_weights_matrices(
    weights_untrained, weights_trained, figpath=None, logger=None
):
    """Plot heatmap weights before and after training."""
    msg = "Plotting weights matrices..."
    print(msg)
    if logger:
        logger.info(msg)

    fig = plt.figure(layout="constrained", figsize=(12, 17))
    subfigs = fig.subfigures(nrows=1, ncols=2)
    ax = []
    for subf in subfigs:
        ax.append(
            subf.subplots(
                nrows=round(len(weights_trained) / 2),
                ncols=2,
                width_ratios=[10, 1],
            )
        )
    subfigs[0].suptitle("Before training")
    subfigs[1].suptitle("After training")
    # subfigs[0].colorbar(pc, shrink=0.6, ax=axsLeft, location='bottom')
    # subfigs[1].colorbar(pc, shrink=0.6, ax=axsRight)
    # fig.suptitle('Weights')

    for idx, (w_untrained, w_trained) in enumerate(
        zip(weights_untrained, weights_trained)
    ):
        # cmap = "bwr"
        cmap = "coolwarm"

        plot_row = int(np.floor(idx / 2))  # Row index to lay out the plots

        if len(w_trained.shape) < 2:  # Biases
            b_untrained_current = w_untrained.unsqueeze(-1).cpu().detach().numpy()
            b_trained_current = w_trained.unsqueeze(-1).cpu().detach().numpy()
            sns.heatmap(b_untrained_current, ax=ax[0][plot_row, 1], cmap=cmap)
            sns.heatmap(b_trained_current, ax=ax[1][plot_row, 1], cmap=cmap)
            for axi in ax:
                axi[plot_row, 1].xaxis.set_major_locator(mpl.ticker.NullLocator())

        else:  # Weights
            w_untrained_current = w_untrained.cpu().detach().numpy()
            w_trained_current = w_trained.cpu().detach().numpy()
            sns.heatmap(w_untrained_current, ax=ax[0][plot_row, 0], cmap=cmap)
            sns.heatmap(w_trained_current, ax=ax[1][plot_row, 0], cmap=cmap)
            for axi in ax:
                axi[plot_row, 0].tick_params(labelbottom=False, labeltop=True)
                axi[plot_row, 0].xaxis.set_major_locator(
                    mpl.ticker.LinearLocator(numticks=3)
                )
                for axj in axi.flatten():
                    axj.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%d"))

    for axlr in ax:
        for axi in axlr:
            for axj in axi:
                axj.yaxis.set_major_locator(mpl.ticker.LinearLocator(numticks=3))
                axj.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%d"))

        # fig.tight_layout()
        fig.patch.set_alpha(0)
        fig.patch.set_facecolor("white")
    if figpath:
        figfullpath = figpath / "weights-matrices.png"
        fig.savefig(figfullpath, bbox_inches="tight")
        if logger:
            msg = f"Saved figure to: {figfullpath.absolute()}"
            print(msg)
            logger.info(msg)
    # plt.show()


def plot_activations(
    activations_layer_df, input_cond, labels, layer_inspected, figpath=None, logger=None
):
    """Plot activations learned in `layer_inspected`."""
    msg = "Plotting activations learned..."
    print(msg)
    if logger:
        logger.info(msg)

    # Create a categorical palette to identify the clusters
    # cluster_palette = sns.color_palette("Pastel2")
    cluster_palette = sns.color_palette("Accent")
    cluster_colors = dict(zip(list(labels.values()), cluster_palette))
    row_colors = [cluster_colors[cond.split(" | ")[0]] for cond in input_cond]
    row_colors_serie = pd.Series(row_colors)
    row_colors_serie = row_colors_serie.set_axis(list(input_cond.keys()))

    # cmap = "mako"
    # cmap = "rocket"
    # cmap = "magma"
    cmap = "viridis"
    chart = sns.clustermap(activations_layer_df, cmap=cmap, row_colors=row_colors_serie)
    chart.ax_heatmap.set_xlabel(f"Neurons activations in layer {layer_inspected}")

    for label, col_val in cluster_colors.items():
        chart.ax_col_dendrogram.bar(0, 0, color=col_val, label=label, linewidth=0)
    chart.ax_col_dendrogram.legend(loc="center", bbox_to_anchor=(1.1, 0.7))  # , ncol=6)

    chart.fig.patch.set_alpha(0)
    chart.fig.patch.set_facecolor("white")
    if figpath:
        chart.savefig(figpath / "activations-learned.png", bbox_inches="tight")
    if figpath:
        figfullpath = figpath / "activations-learned.png"
        chart.savefig(figfullpath, bbox_inches="tight")
        if logger:
            msg = f"Saved figure to: {figfullpath.absolute()}"
            print(msg)
            logger.info(msg)
    # plt.show()


def arrow_right(x, y):
    """Return relative coordinates for DOWN arrow in a dataframe."""
    x_tail = x - 0.25 * 0.75
    x_head = x + 0.25 * 0.75
    y_tail = y
    y_head = y
    res = pd.DataFrame(
        {
            "x_tail": [x_tail],
            "y_tail": [y_tail],
            "x_head": [x_head],
            "y_head": [y_head],
        }
    )
    return res


def arrow_down(x, y):
    """Return relative coordinates for DOWN arrow in a dataframe."""
    x_tail = x
    x_head = x
    y_tail = y + 0.25 * 0.75
    y_head = y - 0.25 * 0.75
    res = pd.DataFrame(
        {
            "x_tail": [x_tail],
            "y_tail": [y_tail],
            "x_head": [x_head],
            "y_head": [y_head],
        }
    )
    return res


def arrow_up(x, y):
    """Return relative coordinates for UP arrow in a dataframe."""
    x_tail = x
    x_head = x
    y_tail = y - 0.25 * 0.75
    y_head = y + 0.25 * 0.75
    res = pd.DataFrame(
        {
            "x_tail": [x_tail],
            "y_tail": [y_tail],
            "x_head": [x_head],
            "y_head": [y_head],
        }
    )
    return res


def arrow_left(x, y):
    """Return relative coordinates for LEFT arrow in a dataframe."""
    x_tail = x + 0.25 * 0.75
    x_head = x - 0.25 * 0.75
    y_tail = y
    y_head = y
    res = pd.DataFrame(
        {
            "x_tail": [x_tail],
            "y_tail": [y_tail],
            "x_head": [x_head],
            "y_head": [y_head],
        }
    )
    return res


def map_action_to_direction(action, head_direction):
    """Map action to its corresponding allocentric arrow direction."""
    arrow_directions = {
        "up": partial(arrow_up),
        "down": partial(arrow_down),
        "left": partial(arrow_left),
        "right": partial(arrow_right),
    }
    direction = None
    if head_direction == 0:
        if action == Actions.forward:
            direction = arrow_directions["up"]
        elif action == Actions.right:
            direction = arrow_directions["right"]
        elif action == Actions.left:
            direction = arrow_directions["left"]
        elif action == Actions.backward:
            direction = arrow_directions["down"]

    elif head_direction == 90:
        if action == Actions.forward:
            direction = arrow_directions["right"]
        elif action == Actions.right:
            direction = arrow_directions["down"]
        elif action == Actions.left:
            direction = arrow_directions["up"]
        elif action == Actions.backward:
            direction = arrow_directions["left"]

    elif head_direction == 180:
        if action == Actions.forward:
            direction = arrow_directions["down"]
        elif action == Actions.right:
            direction = arrow_directions["left"]
        elif action == Actions.left:
            direction = arrow_directions["right"]
        elif action == Actions.backward:
            direction = arrow_directions["up"]

    elif head_direction == 270:
        if action == Actions.forward:
            direction = arrow_directions["left"]
        elif action == Actions.right:
            direction = arrow_directions["up"]
        elif action == Actions.left:
            direction = arrow_directions["down"]
        elif action == Actions.backward:
            direction = arrow_directions["right"]

    if direction is None:
        raise ValueError("Impossible action-head direction combination")
    return direction


def qvalues_directions_map(q_values, env, cues):
    """Get the best learned action & map it to directions for arrows."""
    q_val_best = OrderedDict()
    for cue_i, cue_v in enumerate(cues):
        q_val_best[cue_v] = OrderedDict()
        for x_i, x_v in enumerate(env.tiles_locations["x"]):
            x_v = x_v.item()
            for y_i, y_v in enumerate(env.tiles_locations["y"]):
                y_v = y_v.item()
                q_val_best[cue_v][(x_v, y_v)] = OrderedDict()
                for direction_i, direction_v in enumerate(env.head_angle_space):
                    direction_v = direction_v.item()
                    q_val_best[cue_v][(x_v, y_v)][direction_v] = OrderedDict()
                    q_val_best[cue_v][(x_v, y_v)][direction_v]["q_max"] = q_values[
                        cue_i, x_i, y_i, direction_i, :
                    ].max()
                    best_action = Actions(
                        q_values[cue_i, x_i, y_i, direction_i, :].argmax().item()
                    )
                    q_val_best[cue_v][(x_v, y_v)][direction_v]["best_action"] = (
                        map_action_to_direction(best_action, direction_v)
                    )
    return q_val_best


def convert_row_origin(row, rows):
    """
    Convert row and column for plotting.

    In the environment, row and col starts in the top left corner,
    but for plotting the origin is in the bottom left corner.
    """
    max_row_idx = rows - 1
    conv_row = max_row_idx - row
    return conv_row


def draw_single_tile_arrows(row, col, arrows_tile, ax):
    """Plot arrows for all 4 angles for a single tile."""
    coord_center_arrows = pd.DataFrame(
        {
            "x": [0.5 + col, 0.75 + col, 0.5 + col, 0.25 + col],
            "y": [0.75 + row, 0.5 + row, 0.25 + row, 0.5 + row],
            "angle": arrows_tile.keys(),
        }
    ).set_index("angle")

    arrows_coords = pd.DataFrame(
        {"x_tail": [], "y_tail": [], "x_head": [], "y_head": [], "angle": []}
    )

    for angle in arrows_tile:
        direction = arrows_tile[angle]["direction"]  # Get the `partial` function
        x = coord_center_arrows.loc[angle].x
        y = coord_center_arrows.loc[angle].y
        direction_df = direction(x, y)  # Apply the `partial` function
        direction_df["angle"] = angle
        arrows_coords = pd.concat(
            [arrows_coords, direction_df],
            ignore_index=True,
        )
    arrows_coords["angle"] = coord_center_arrows.index
    arrows_coords.set_index("angle", inplace=True)

    for angle, coord in arrows_coords.iterrows():
        arrow = mpatches.FancyArrowPatch(
            (coord.x_tail, coord.y_tail),
            (coord.x_head, coord.y_head),
            mutation_scale=10,
            color=arrows_tile[angle]["color"],
        )
        ax.add_patch(arrow)


def plot_policies_by_head_directions(
    q_values, labels, env, cues, figpath=None, logger=None
):
    """
    Plot the heatmap of the Q-values.

    And plot the best action with arrows for each head direction.
    """
    q_val_best = qvalues_directions_map(q_values, env, cues)

    cmap = mpl.colormaps["PuRd"]
    norm = mpl.colors.Normalize(vmin=q_values.min(), vmax=q_values.max())

    fig = plt.figure(figsize=(13, 8))
    gs = GridSpec(len(TriangleState), len(labels) + 1, width_ratios=[10, 10, 10, 1])
    ax = np.empty(gs.get_geometry(), dtype=object)
    for idx in range(gs.get_geometry()[0]):
        for jdx in range(gs.get_geometry()[1]):
            ax[idx, jdx] = fig.add_subplot(gs[idx, jdx])
    ax_clb = fig.add_subplot(gs[:, 3])  # Last column for the colorbar
    # Remove ticks for the colorbar
    for tri_i, _ in enumerate(TriangleState):
        ax[tri_i, 3].axes.xaxis.set_ticklabels([])
        ax[tri_i, 3].axes.yaxis.set_ticklabels([])

    half_tile = env.tile_step / 2
    for tri_i, tri_v in enumerate(TriangleState):
        for cue_i, cue_v in enumerate(labels):
            ax[tri_i, cue_i].set_yticks(
                [env.rangeY["min"] - half_tile, env.rangeY["max"] + half_tile],
                minor=True,
            )
            ax[tri_i, cue_i].set_xticks(
                [env.rangeX["min"] - half_tile, env.rangeX["max"] + half_tile],
                minor=True,
            )
            ax[tri_i, cue_i].xaxis.set_major_locator(
                ticker.MultipleLocator(
                    base=env.tile_step, offset=env.rangeX["min"] - half_tile
                )
            )
            ax[tri_i, cue_i].yaxis.set_major_locator(
                ticker.MultipleLocator(
                    base=env.tile_step, offset=env.rangeY["min"] - half_tile
                )
            )
            ax[tri_i, cue_i].yaxis.grid(True, which="major")
            ax[tri_i, cue_i].xaxis.grid(True, which="major")
            ax[tri_i, cue_i].axes.xaxis.set_ticklabels([])
            ax[tri_i, cue_i].axes.yaxis.set_ticklabels([])
            ax[tri_i, cue_i].set_title(labels[cue_v], pad=10)

            for _, x_v in enumerate(env.tiles_locations["x"]):
                x_v = x_v.item()
                # conv_row = convert_row_origin(x_v, rows)
                for _, y_v in enumerate(env.tiles_locations["y"]):
                    y_v = y_v.item()

                    if (
                        tri_v == TriangleState.upper
                        and x_v + y_v >= 0
                        or tri_v == TriangleState.lower
                        and x_v + y_v <= 0
                    ):  # Plot only upper or lower triangle
                        # Get the data to plot for each angle
                        arrows_tile = OrderedDict()
                        for _, angle in enumerate(env.head_angle_space):
                            angle = angle.item()
                            arrows_tile[angle] = {}
                            arrows_tile[angle]["direction"] = q_val_best[cue_v][
                                (x_v, y_v)
                            ][angle]["best_action"]
                            q_max = q_val_best[cue_v][(x_v, y_v)][angle]["q_max"]
                            arrows_tile[angle]["color"] = cmap(norm(q_max.cpu()))

                        # Plot arrows for all 4 angles for a single tile
                        draw_single_tile_arrows(
                            # row=conv_row,
                            row=y_v - half_tile,
                            col=x_v - half_tile,
                            arrows_tile=arrows_tile,
                            ax=ax[tri_i, cue_i],
                        )

            # Annotate the ports names
            bbox = {
                "facecolor": "black",
                "edgecolor": "none",
                "boxstyle": "round",
                "alpha": 0.1,
            }
            ax[tri_i, cue_i].text(
                x=env.rangeX["max"] + half_tile - 0.3,
                y=env.rangeY["min"] - half_tile,
                s="E",
                bbox=bbox,
                color="white",
            )
            ax[tri_i, cue_i].text(
                x=env.rangeX["min"] - half_tile + 0.05,
                y=env.rangeY["max"] + half_tile - 0.1,
                s="W",
                bbox=bbox,
                color="white",
            )
            ax[tri_i, cue_i].text(
                x=env.rangeX["max"] + half_tile - 0.3,
                y=env.rangeY["max"] + half_tile - 0.1,
                s="N",
                bbox=bbox,
                color="white",
            )
            ax[tri_i, cue_i].text(
                x=env.rangeX["min"] - half_tile + 0.05,
                y=env.rangeY["min"] - half_tile,
                s="S",
                bbox=bbox,
                color="white",
            )

    clb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax_clb)
    clb.ax.set_title("Q-value", pad=10)

    # Make background transparent
    fig.patch.set_alpha(0)
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    if figpath:
        figfullpath = figpath / "policy.png"
        fig.savefig(figfullpath, bbox_inches="tight")
        if logger:
            msg = f"Saved figure to: {figfullpath.absolute()}"
            print(msg)
            logger.info(msg)
    # plt.show()
