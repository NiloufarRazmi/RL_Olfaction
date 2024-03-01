import itertools
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

# from curlyBrace import curlyBrace
from imojify import imojify
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

# from utils import get_location_count

sns.set_theme(font_scale=1.5)
mpl.rcParams["font.family"] = ["sans-serif"]
mpl.rcParams["font.sans-serif"] = [
    "Fira Sans",
    "Computer Modern Sans Serif",
    "DejaVu Sans",
    "Verdana",
    "Arial",
    "Helvetica",
]
# plt.rcParams['text.usetex'] = True


ROOT_PATH = Path(__file__).parent
PLOTS_PATH = ROOT_PATH / "plots"


def check_plots():
    if not PLOTS_PATH.exists():
        os.mkdir(PLOTS_PATH)


def plot_steps_and_rewards(df, n_runs=0, log=None):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

    # Rewards
    sns.lineplot(data=df, x="Episodes", y="Rewards", ax=ax[0])

    # Steps
    sns.lineplot(data=df, x="Episodes", y="Steps", ax=ax[1])
    if log:
        ax[1].set_yscale("log")

    if n_runs > 1:
        ax[0].set(ylabel=f"Rewards\naveraged over {n_runs} runs")
        ax[1].set(ylabel=f"Steps number\naveraged over {n_runs} runs")
    else:
        ax[0].set(ylabel="Rewards")
        ax[1].set(ylabel="Steps number")

    # Transparent background
    fig.patch.set_alpha(0)
    fig.patch.set_facecolor("white")

    fig.tight_layout()
    check_plots()
    plt.savefig(PLOTS_PATH / "rew+steps.png", bbox_inches="tight")
    plt.show()


def qtable_directions_map(qtable, rows, cols):
    """Get the best learned action & map it to arrows."""
    qtable_val_max = qtable.max(axis=1).reshape(rows, cols)
    qtable_best_action = np.argmax(qtable, axis=1).reshape(rows, cols)
    directions = {0: "↑", 1: "↓", 2: "←", 3: "→"}
    qtable_directions = np.empty(qtable_best_action.flatten().shape, dtype=str)
    eps = np.finfo(float).eps  # Minimum float number on the machine
    for idx, val in enumerate(qtable_best_action.flatten()):
        if qtable_val_max.flatten()[idx] > eps:
            # Assign an arrow only if a minimal Q-value has been learned as best action
            # otherwise since 0 is a direction, it also gets mapped on the tiles where
            # it didn't actually learn anything
            qtable_directions[idx] = directions[val]
    qtable_directions = qtable_directions.reshape(rows, cols)
    return qtable_val_max, qtable_directions


def plot_q_values_map(qtable, rows, cols):
    """Plot the heatmap of the Q-values.

    Also plot the best action's direction with arrows."""
    # TODO: Remove function as it doesn't seem to be used anymore?
    qtable_val_max, qtable_directions = qtable_directions_map(qtable, rows, cols)

    # font_name = "DejaVu Math TeX Gyre"
    # mpl.rcParams["font.family"] = font_name
    f, ax = plt.subplots()
    sns.heatmap(
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
    ).set(title="Learned Q-values\nArrows represent best action")
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
        spine.set_color("black")
    plt.show()


def plot_heatmap(matrix, title=None, xlabel=None, ylabel=None, braces=[]):
    fig, ax = plt.subplots(figsize=(9, 9))
    cmap = sns.light_palette("seagreen", as_cmap=True)
    # cmap = sns.color_palette("light:b", as_cmap=True)
    chart = sns.heatmap(matrix, cmap=cmap, ax=ax)
    if title:
        chart.set(title=title)
    if xlabel:
        chart.set(xlabel=xlabel)
    if ylabel:
        chart.set(ylabel=ylabel)
    ax.tick_params(axis="x", rotation=90)
    ax.tick_params(axis="y", rotation=0)
    ax.tick_params(left=True)
    ax.xaxis.tick_top()
    if braces:
        for idx, brace in enumerate(braces):
            curlyBrace(
                fig=fig,
                ax=ax,
                p1=brace["p1"],
                p2=brace["p2"],
                k_r=0.05,
                bool_auto=False,
                str_text=brace["str_text"],
                color="black",
                lw=2,
                int_line_num=2,
            )
    plt.show()


def plot_tiles_locations(tiles_list, rows, cols, title=None):
    """Simple plot to show the states/tiles numbers."""
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
    """Plot the heatmap of the Q-values.

    Also plot the best action's direction with arrows."""

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
    """Plot the heatmap of the Q-values.

    Also plot the best action's direction with arrows."""
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


def plot_states_actions_distribution(states, actions):
    """Plot the distributions of states and actions."""
    fig, ax = plt.subplots(2, 1, figsize=(12, 5))
    sns.histplot(data=states, ax=ax[0], kde=True)
    ax[0].set_title("States")
    sns.histplot(data=actions, ax=ax[1])
    ax[1].set_title("Actions")
    fig.tight_layout()
    check_plots()
    plt.savefig(PLOTS_PATH / "states+actions.png", bbox_inches="tight")
    plt.show()


def plot_rotated_q_values_maps(qtable, rows, cols, labels):
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

        def arrow_color(table, val):
            if val > table.mean():
                color = "white"
            else:
                color = "black"
            return color

        # Loop over data dimensions and create text annotations.
        for i in range(rows):
            for j in range(cols):
                ax.flatten()[idx].text(
                    # j,
                    # i,
                    np.flipud(X)[i, j] + 0.5,
                    np.flipud(Y)[i, j],
                    qtable_directions[i, j],
                    ha="center",
                    va="center",
                    color=arrow_color(qtable_val_max, qtable_val_max[i, j]),
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


def plot_location_count(
    all_state_composite, tiles_locations, cols, rows, cues=None, contexts_labels=None
):
    # cmap = sns.color_palette("Blues", as_cmap=True)
    cmap = sns.color_palette("rocket_r", as_cmap=True)

    if cues and contexts_labels:
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        for idx, cue in enumerate(cues):
            location_count = get_location_count(
                all_state_composite=all_state_composite,
                tiles_locations=tiles_locations,
                cols=cols,
                rows=rows,
                cue=cue,
            )
            chart = sns.heatmap(location_count, cmap=cmap, ax=ax.flatten()[idx])
            chart.set(title=contexts_labels[cue])
            ax.flatten()[idx].set_xticks([])
            ax.flatten()[idx].set_yticks([])
        fig.suptitle("Locations counts during training")  # , fontsize="xx-large")

    else:  # Plot everything
        location_count = get_location_count(
            all_state_composite,
            tiles_locations=tiles_locations,
            cols=cols,
            rows=rows,
            cue=cues,
        )
        fig, ax = plt.subplots(figsize=(10, 8))
        chart = sns.heatmap(location_count, cmap=cmap, ax=ax)
        chart.set(title="Locations count during training")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    plt.show()


def plot_weights_biases_distributions(weights_df, biases_df, label=None, figpath=None):
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

    fig.tight_layout()
    fig.patch.set_alpha(0)
    fig.patch.set_facecolor("white")
    if figpath:
        fig.savefig(
            figpath / f"weights-biases-distrib-{label}.png",
            bbox_inches="tight",
        )
    plt.show()
