import itertools
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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

    font_name = "DejaVu Math TeX Gyre"
    mpl.rcParams["font.family"] = font_name
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
