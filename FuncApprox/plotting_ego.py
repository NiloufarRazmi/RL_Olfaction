from collections import OrderedDict
from functools import partial

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from environment_ego import Actions
from matplotlib.gridspec import GridSpec


def arrow_right(x, y):
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


def arrow_up(x, y):
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


def arrow_left(x, y):
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


def draw_single_tile_arrows(row, col, arrows_tile, ax):
    coord_center_arrows = pd.DataFrame(
        {
            "x": [0.5 + col, 0.25 + col, 0.5 + col, 0.75 + col],
            "y": [0.25 + row, 0.5 + row, 0.75 + row, 0.5 + row],
        }
    )

    arrows_coords = pd.DataFrame(
        {"x_tail": [], "y_tail": [], "x_head": [], "y_head": []}
    )
    for angle_idx, angle in enumerate(arrows_tile):
        direction = arrows_tile[angle]["direction"]
        arrows_coords = pd.concat(
            [
                arrows_coords,
                direction(
                    coord_center_arrows.x[angle_idx],
                    coord_center_arrows.y[angle_idx],
                ),
            ],
            ignore_index=True,
        )
    arrows_coords["angle"] = arrows_tile.keys()

    for jdx, coord in arrows_coords.iterrows():
        arrow = mpatches.FancyArrowPatch(
            (coord.x_tail, coord.y_tail),
            (coord.x_head, coord.y_head),
            mutation_scale=10,
            color=arrows_tile[coord.angle]["color"],
        )
        ax.add_patch(arrow)


def map_action_to_direction(action, head_direction):
    arrow_directions = {
        "up": partial(arrow_up),
        "down": partial(arrow_down),
        "left": partial(arrow_left),
        "right": partial(arrow_right),
    }
    direction = None
    if head_direction == 0:
        if action == Actions.FORWARD:
            direction = arrow_directions["up"]
        elif action == Actions.RIGHT:
            direction = arrow_directions["right"]
        elif action == Actions.LEFT:
            direction = arrow_directions["left"]

    elif head_direction == 90:
        if action == Actions.FORWARD:
            direction = arrow_directions["right"]
        elif action == Actions.RIGHT:
            direction = arrow_directions["down"]
        elif action == Actions.LEFT:
            direction = arrow_directions["up"]

    elif head_direction == 180:
        if action == Actions.FORWARD:
            direction = arrow_directions["down"]
        elif action == Actions.RIGHT:
            direction = arrow_directions["left"]
        elif action == Actions.LEFT:
            direction = arrow_directions["right"]

    elif head_direction == 270:
        if action == Actions.FORWARD:
            direction = arrow_directions["left"]
        elif action == Actions.RIGHT:
            direction = arrow_directions["up"]
        elif action == Actions.LEFT:
            direction = arrow_directions["down"]

    if direction is None:
        raise ValueError("Impossible action-head direction combination")
    return direction


def qtable_directions_map_ego(qtable, rows, cols, states):
    """Get the best learned action & map it to directions for arrows."""
    q_val_best = OrderedDict()
    for cue in states:
        q_val_best[cue] = OrderedDict()
        for angle in states[cue]:
            q_val_best[cue][angle] = {}
            q_val_best[cue][angle]["q_max"] = (
                np.empty(qtable[states[cue][angle]].shape[0]) * np.nan
            )
            q_val_best[cue][angle]["best_action"] = []
            for st, flat_state in enumerate(states[cue][angle]):
                q_val_best[cue][angle]["q_max"][st] = qtable[flat_state, :].max()
                best_action = Actions(np.argmax(qtable[flat_state, :]))
                q_val_best[cue][angle]["best_action"].append(
                    map_action_to_direction(best_action, angle)
                )
    return q_val_best


def plot_ego_q_values_maps(qtable, rows, cols, labels, q_val_best):
    """Plot the heatmap of the Q-values.

    Also plot the best action's direction with arrows."""

    cmap = sns.color_palette("Blues", as_cmap=True)
    norm = mpl.colors.Normalize(vmin=qtable.min(), vmax=qtable.max())

    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 3, width_ratios=[10, 10, 1])
    ax = []
    ax.append(fig.add_subplot(gs[0, 0]))
    ax.append(fig.add_subplot(gs[0, 1]))
    ax.append(fig.add_subplot(gs[1, 0]))
    ax.append(fig.add_subplot(gs[1, 1]))
    ax_clb = fig.add_subplot(gs[:, 2])
    for idx, cue in enumerate(labels):
        # ax[idx].set_yticks([2.5, 7.5], minor=False)
        ax[idx].set_yticks([0, 5], minor=True)
        ax[idx].yaxis.grid(True, which="major")
        # ax.flatten()[idx].yaxis.grid(True, which="minor")
        ax[idx].set_xticks([0, 5], minor=True)
        ax[idx].xaxis.grid(True, which="major")
        ax[idx].axes.xaxis.set_ticklabels([])
        ax[idx].axes.yaxis.set_ticklabels([])
        ax[idx].set_title(labels[cue])

        for row in range(rows):
            for col in range(cols):
                arrows_tile = {}
                for angle in q_val_best[cue]:
                    arrows_tile[angle] = {}
                    arrows_tile[angle]["direction"] = np.array(
                        q_val_best[cue][angle]["best_action"]
                    ).reshape(rows, cols)[row, col]
                    q_max = np.array(q_val_best[cue][angle]["q_max"]).reshape(
                        rows, cols
                    )[row, col]
                    arrows_tile[angle]["color"] = cmap(norm(q_max))
                draw_single_tile_arrows(
                    row=row,
                    col=col,
                    arrows_tile=arrows_tile,
                    ax=ax[idx],
                )

    clb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax_clb)
    clb.ax.set_title('Q-value')
    plt.show()
