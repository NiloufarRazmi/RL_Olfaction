# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import matplotlib.pyplot as plt

# %%
import torch
from curlyBrace import curlyBrace

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

# %%
loc_rand = torch.randn((25, 25)) * 0.2
loc_rand.shape

# %%
loc_rand2 = torch.randn((25, 25)) * 0.2
loc_rand2.shape

# %%
blocs = torch.block_diag(
    torch.ones((4, 4)),
    torch.ones((6, 6)),
    torch.ones((4, 4)),
    torch.ones((6, 6)),
    torch.ones((5, 5)),
)
blocs.shape

# %%
blocs2 = torch.block_diag(
    torch.ones((6, 6)),
    torch.ones((4, 4)),
    torch.ones((5, 5)),
    torch.ones((3, 3)),
    torch.ones((7, 7)),
)
blocs2.shape

# %%
loc = blocs + loc_rand

# %%
loc2 = blocs2 + loc_rand2

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
odor_loc = torch.randn((25, 3))
odor_loc.T

# %%
odor_loc2 = torch.randn((25, 3))

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
simu = torch.cat(
    (torch.cat((loc2, odor_loc2), dim=1), torch.cat((odor_loc2.T, odor), dim=1)), dim=0
)
simu.shape

# %%
braces = []
braces.append(
    {
        "p1": [-2, 0],
        "p2": [-2, 24],
        "str_text": "location",
    }
)
braces.append(
    {
        "p1": [-2, 25],
        "p2": [-2, 27],
        "str_text": "odor",
    }
)
braces.append(
    {
        "p1": [0, -2],
        "p2": [24, -2],
        "str_text": "location",
    }
)
braces.append(
    {
        "p1": [25, -2],
        "p2": [27, -2],
        "str_text": "odor",
    }
)

# %%
with plt.xkcd():
    fig, ax = plt.subplots()
    ax.matshow(neural)
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_xlabel('time')
    ax.set_xlabel("Neural data", fontsize=30)
    # ax.set_title("Neural data")

    for spine in ax.spines.values():
        spine.set_visible(False)

    for idx, brace in enumerate(braces):
        curlyBrace(
            fig=fig,
            ax=ax,
            p1=brace["p1"],
            p2=brace["p2"],
            k_r=0.0,
            bool_auto=False,
            str_text=brace["str_text"],
            color="black",
            lw=2,
            int_line_num=2,
        )

    plt.show()

# %%
with plt.xkcd():
    fig, ax = plt.subplots(1, 2, figsize=(18, 7))
    ax[0].matshow(neural)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    # ax.set_ylabel('my overall health')
    # ax[0].set_title("Neural data", fontsize=40)
    ax[0].set_xlabel("Neural data", fontsize=40)
    for spine in ax[0].spines.values():
        spine.set_visible(False)

    ax[1].matshow(simu)
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    # ax[1].set_title("Simulated data", fontsize=40)
    ax[1].set_xlabel("Simulated data", fontsize=40)
    for spine in ax[1].spines.values():
        spine.set_visible(False)

    for axi in ax:
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

    plt.show()

# %%
