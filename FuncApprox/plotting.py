import itertools

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme()


def plot_steps_and_rewards(df):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    # ax[0].set(ylabel="Cummulated rewards")
    sns.lineplot(data=df, x="Episodes", y="Rewards", ax=ax[0])
    # ax[0].set(ylabel="Cumulated rewards")

    sns.lineplot(data=df, x="Episodes", y="Steps", ax=ax[1])
    # ax[1].set(ylabel="Averaged steps number")

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


def plot_rotated_q_values_map(qtable, rows, cols):
    # See https://stackoverflow.com/q/12848581/4129062
    qtable_val_max, qtable_directions = qtable_directions_map(qtable, rows, cols)

    fig, ax = plt.subplots()
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
    im = ax.pcolormesh(X, Y, C, cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    def arrow_color(table, val):
        if val > table.mean():
            color = "white"
        else:
            color = "black"
        return color

    # Loop over data dimensions and create text annotations.
    for i in range(rows):
        for j in range(cols):
            ax.text(
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

    ax.set_title("Learned Q-values\nArrows represent best action")
    # ax.set_xticks(np.arange(qtable_val_max.shape[1] + 1) - 0.5, minor=True)
    # ax.set_yticks(np.arange(qtable_val_max.shape[0] + 1) - 0.5, minor=True)
    # ax.grid(which="minor", color="black", linestyle="-", linewidth=0.7)
    ax.set_xticks([])
    ax.set_yticks([])
    # for _, spine in ax.spines.items():
    #     spine.set_visible(True)
    #     spine.set_linewidth(0.7)
    #     spine.set_color("black")

    fig.tight_layout()
    plt.show()


def plot_heatmap(matrix, title=None):
    fig, ax = plt.subplots(figsize=(9, 9))
    cmap = sns.light_palette("seagreen", as_cmap=True)
    # cmap = sns.color_palette("light:b", as_cmap=True)
    chart = sns.heatmap(matrix, cmap=cmap, ax=ax)
    if title:
        chart.set(title=title)
    ax.tick_params(axis="x", rotation=90)
    ax.tick_params(axis="y", rotation=0)
    ax.tick_params(left=True)
    ax.xaxis.tick_top()
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

    f, ax = plt.subplots(2, 2, figsize=(12, 10))
    maps = [
        np.arange(0, rows * cols),
        np.arange(rows * cols, 2 * rows * cols),
        np.arange(2 * rows * cols, 3 * rows * cols),
        np.arange(3 * rows * cols, 4 * rows * cols),
    ]
    for idx, title in enumerate(labels):
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
        ).set(title=title)
        for _, spine in ax.flatten()[idx].spines.items():
            spine.set_visible(True)
            spine.set_linewidth(0.7)
            spine.set_color("black")
    plt.show()


def plot_states_actions_distribution(states, actions):
    """Plot the distributions of states and actions."""
    fig, ax = plt.subplots(2, 1, figsize=(15, 8))
    sns.histplot(data=states, ax=ax[0], kde=True)
    ax[0].set_title("States")
    sns.histplot(data=actions, ax=ax[1])
    ax[1].set_title("Actions")
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
    for idx, title in enumerate(labels):
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

        ax.flatten()[idx].set_title(title)
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
