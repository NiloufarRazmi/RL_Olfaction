# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Setup

# %%
from collections import OrderedDict
from enum import Enum

# %%
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import torch
import numpy as np

# %%
from curlyBrace import curlyBrace

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

# %%
# Formatting & autoreload stuff
# %load_ext lab_black
# %load_ext autoreload
# %autoreload 2
# # %matplotlib ipympl

# %% [markdown]
# ## Activity heatmap

# %%
loc_rand = torch.randn((34, 34)) * 0.2
loc_rand.shape

# %%
# loc_rand2 = torch.randn((34, 34)) * 0.2
# loc_rand2.shape

# %%
blocs = torch.block_diag(
    torch.ones((4, 4)),
    torch.ones((6, 6)),
    torch.ones((4, 4)),
    torch.ones((6, 6)),
    torch.ones((5, 5)),
    torch.ones((2, 2)),
    torch.ones((7, 7)),
)
blocs.shape

# %%
# blocs2 = torch.block_diag(
#     torch.ones((6, 6)),
#     torch.ones((4, 4)),
#     torch.ones((5, 5)),
#     torch.ones((3, 3)),
#     torch.ones((7, 7)),
# )
# blocs2.shape

# %%
loc = blocs + loc_rand

# %%
# loc2 = blocs2 + loc_rand2

# %%
with plt.xkcd():
    fig, ax = plt.subplots()
    ax.matshow(loc)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()

# %%
odor_diag = torch.zeros(3, 3)
odor_diag.fill_diagonal_(1)
odor_diag

# %%
# odor_rand = torch.randn((3, 3)) * 0.2
odor_rand = torch.zeros((3, 3))
odor_rand

# %%
odor = odor_diag + odor_rand
odor

# %%
with plt.xkcd():
    fig, ax = plt.subplots()
    ax.matshow(odor)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()

# %%
odor_loc = torch.randn((34, 3))
odor_loc.T

# %%
# odor_loc2 = torch.randn((25, 3))

# %%
tmp_mat1 = torch.cat((loc, odor_loc), dim=1)
tmp_mat1.shape

# %%
tmp_mat2 = torch.cat((odor_loc.T, odor), dim=1)
tmp_mat2.shape

# %%
neural = torch.cat((tmp_mat1, tmp_mat2), dim=0)
neural.shape

# %%
# simu = torch.cat(
#     (torch.cat((loc2, odor_loc2), dim=1), torch.cat((odor_loc2.T, odor), dim=1)), dim=0
# )
# simu.shape

# %%
braces = []
braces.append(
    {
        "p1": [-2, 0],
        "p2": [-2, 19],
        "str_text": "cartesian",
    }
)
braces.append(
    {
        "p1": [-2, 20],
        "p2": [-2, 34],
        "str_text": "polar",
    }
)
braces.append(
    {
        "p1": [-2, 34],
        "p2": [-2, 36],
        "str_text": "odor",
    }
)
# braces.append(
#     {
#         "p1": [0, -2],
#         "p2": [24, -2],
#         "str_text": "location",
#     }
# )
# braces.append(
#     {
#         "p1": [25, -2],
#         "p2": [27, -2],
#         "str_text": "odor",
#     }
# )

# %%
with plt.xkcd():
    neural[0:20, :] = neural[0:20, :] / 5

    fig, ax = plt.subplots(1, 2, figsize=(13, 8))
    ax[0].matshow(neural)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    # ax.set_xlabel('time')
    # ax.set_xlabel("Neural data", fontsize=30)
    ax[0].set_title("Activity on left/right task")
    neural[0:20, :] = neural[0:20, :] * 5
    neural[21:34, :] = neural[21:34, :] / 5
    ax[1].matshow(neural)
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set_title("Activity on east/west task")

    for axi in ax:
        for spine in axi.spines.values():
            spine.set_visible(False)

        for idx, brace in enumerate(braces):
            curlyBrace(
                fig=fig,
                ax=axi,
                p1=brace["p1"],
                p2=brace["p2"],
                k_r=0.0,
                bool_auto=False,
                str_text=brace["str_text"],
                color="black",
                lw=2,
                int_line_num=2,
            )
    fig.patch.set_alpha(0)
    fig.patch.set_facecolor("white")

    plt.show()

# %% [markdown]
# ## Box plot

# %%
agents_pop_num = 10
tmp = ["no translation" for _ in range(agents_pop_num)]
tmp.extend(["cartesian\ntranslated" for _ in range(agents_pop_num)])
tmp.extend(["polar\ntranslated" for _ in range(agents_pop_num)])
df_transl = pd.DataFrame(
    {
        "task_solved": np.concatenate(
            [
                np.random.uniform(low=0.8, high=1.0, size=agents_pop_num),
                np.random.uniform(low=0.8, high=1.0, size=agents_pop_num),
                np.random.uniform(low=0.8, high=1.0, size=agents_pop_num),
            ]
        ),
        "translated_experiment": tmp,
    }
)
df_transl

# %%
tmp = ["left/right\ncorrect\nangle" for _ in range(agents_pop_num)]
tmp.extend(["left/right\nincorrect\nangle" for _ in range(agents_pop_num)])
tmp.extend(["east/west\ncorrect\nangle" for _ in range(agents_pop_num)])
tmp.extend(["east/west\nincorrect\nangle" for _ in range(agents_pop_num)])
df_wrong_angle = pd.DataFrame(
    {
        "task_solved": np.concatenate(
            [
                np.random.uniform(low=0.8, high=1.0, size=agents_pop_num),
                np.random.uniform(low=0.8, high=1.0, size=agents_pop_num),
                np.random.uniform(low=0.8, high=1.0, size=agents_pop_num),
                np.random.uniform(low=0.0, high=0.2, size=agents_pop_num),
            ]
        ),
        "incorrect_angle_experiment": tmp,
    }
)
df_wrong_angle

# %%
plt.style.use("ggplot")
with plt.xkcd():
    fig, ax = plt.subplots(1, 2, figsize=(15, 8))
    palette = sns.color_palette("tab10")

    sns.boxplot(
        data=df_transl,
        x="translated_experiment",
        y="task_solved",
        ax=ax[0],
        color=palette[0],
    )
    ax[0].set(ylim=(0, 1))

    sns.boxplot(
        data=df_wrong_angle,
        x="incorrect_angle_experiment",
        y="task_solved",
        ax=ax[1],
        color=palette[1],
    )

    fig.patch.set_alpha(0)
    fig.patch.set_facecolor("white")

    plt.show()


# %% [markdown]
# ## Weights


# %%
class Cues(Enum):
    NoOdor = 0
    OdorA = 1
    OdorB = 2


CONTEXTS_LABELS = OrderedDict(
    [
        # (LightCues.North, "Pre odor - North light"),
        # (LightCues.South, "Pre odor - South light"),
        # (OdorID.A, "Post odor - Odor A"),
        # (OdorID.B, "Post odor - Odor B"),
        (Cues.NoOdor, "Pre odor"),
        (Cues.OdorA, "Odor A"),
        (Cues.OdorB, "Odor B"),
    ]
)
rows = 5
cols = 5
tiles_locations = torch.arange(rows * cols, device=device)

# %%
# Construct input dictionnary to be fed to the network
input_cond = OrderedDict({})
for cue_obj, cue_txt in CONTEXTS_LABELS.items():
    for loc in tiles_locations:
        current_state = torch.tensor([loc, cue_obj.value], device=device)
        input_cond[f"{loc}-{cue_txt}"] = current_state.float()
input_cond

# %%
# Get the number of neurons in the layer inspected
neurons_num = 100

# Get the activations from the network
# activations_layer = (
#     torch.ones((len(input_cond), neurons_num), device=DEVICE) * torch.nan
# )
# for idx, (cond, input_val) in enumerate(input_cond.items()):
#     activations_layer[idx, :] = torch.randn((1, neurons_num))
activations_layer = torch.randn((len(input_cond), neurons_num), device=device) / 5

# %%
activations_layer = activations_layer + torch.block_diag(
    torch.ones((4, 6)),
    torch.ones((6, 7)),
    torch.ones((4, 10)),
    torch.ones((6, 7)),
    torch.ones((5, 6)),
    torch.ones((2, 3)),
    torch.ones((7, 8)),
    torch.ones((5, 6)),
    torch.ones((10, 12)),
    torch.ones((3, 4)),
    torch.ones((15, 16)),
    torch.ones((5, 6)),
    torch.ones((3, 9)),
)
activations_layer.shape

# %%
activations_layer_df = pd.DataFrame(activations_layer.cpu())  # , columns=cols)
activations_layer_df["Input"] = list(input_cond.keys())
activations_layer_df.set_index("Input", inplace=True)
activations_layer_df


# %%
def plot_activations(activations_layer_df, input_cond, labels, layer_inspected):
    # Create a categorical palette to identify the clusters
    # cluster_palette = sns.color_palette("Pastel2")
    cluster_palette = sns.color_palette("Accent")
    cluster_colors = dict(zip(list(labels.values()), cluster_palette))
    row_colors = [cluster_colors[cond.split("-")[1]] for cond in input_cond.keys()]
    row_colors_serie = pd.Series(row_colors)
    row_colors_serie = row_colors_serie.set_axis(list(input_cond.keys()))

    cmap = "viridis"
    chart = sns.clustermap(activations_layer_df, cmap=cmap, row_colors=row_colors_serie)
    chart.ax_heatmap.set_xlabel(f"Neurons activations in layer {str(layer_inspected)}")

    for label, col_val in cluster_colors.items():
        chart.ax_col_dendrogram.bar(0, 0, color=col_val, label=label, linewidth=0)
    chart.ax_col_dendrogram.legend(loc="center", bbox_to_anchor=(1.1, 0.7))  # , ncol=6)

    chart.fig.patch.set_alpha(0)
    chart.fig.patch.set_facecolor("white")
    plt.show()


# %%
plot_activations(
    activations_layer_df=activations_layer_df,
    input_cond=input_cond,
    labels=CONTEXTS_LABELS,
    layer_inspected="L",
)

# %% [markdown]
# ## Deviation from correct angle

# %%
angle_activations = torch.randn((128, 360), device=device)
angle_activations

# %%
# plt.style.use("ggplot")
with plt.xkcd():
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(angle_activations, extent=[-180, 180, 0, 128])
    ax.set_title("Mapping from angles to neural activations")
    ax.set_xlabel("Deviation from correct angle [°]")  # , fontsize=30)
    ax.set_ylabel("Activations")  # , fontsize=30)
    ax.set_yticks([])
    fig.patch.set_alpha(0)
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    plt.show()

# %%
