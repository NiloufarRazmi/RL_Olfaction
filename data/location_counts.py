# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Init

# %%
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import minmax_scale
from tqdm import tqdm

# %%
sns.set_theme(font_scale=1.5)
# %load_ext lab_black

# %%
# Global variables
rows = 5
cols = 5
tiles_locations = np.arange(rows * cols)
tiles_locations.reshape((rows, cols))
likelihood_threshold = 0.85

# %%
# day = "d1"
# day = "d23"
# mouse = "F03"

# %%
# data_folder = Path(f"{mouse}")

# %%
# file_day = data_folder / f"F03_{day}.csv"
# entries_and_time_spent = data_folder / f"F03_{day}entries_and_time_spent.csv"
# file_day, entries_and_time_spent

# %%
# day_data = pd.read_csv(file_day, index_col=0)
# day_data

# %%
# eats = pd.read_csv(entries_and_time_spent, index_col=0)
# eats

# %% [markdown]
# ## Naive locations

# %%
naive_coord_path = Path(
    "C01_d0p0_2022-01-12_15.58DLC_resnet50_OM02Sep7shuffle1_450000.csv"
)

# %%
naive_data = pd.read_csv(naive_coord_path, header=[0, 1, 2])
naive_data

# %%
head_naive = naive_data.DLC_resnet50_OM02Sep7shuffle1_450000["head"]
head_naive

# %%
nose_naive = naive_data.DLC_resnet50_OM02Sep7shuffle1_450000.nose
nose_naive


# %%
def plot_coords(head, nose, title):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(data=head, x="x", y="y", ax=ax, alpha=0.25)
    sns.scatterplot(data=nose, x="x", y="y", ax=ax, alpha=0.25)
    ax.set_title(title, pad=20)
    plt.legend(["head", "nose"])
    plt.show()


# %%
plot_coords(
    head=head_naive,
    nose=nose_naive,
    title="Naive mouse coordinates\n" "C01_d0p0_2022-01-12_15.58",
)

# %%
head_naive[head_naive.likelihood < likelihood_threshold]

# %%
sns.scatterplot(
    data=head_naive[head_naive.likelihood > likelihood_threshold],
    x="x",
    y="y",
    alpha=0.25,
)
plt.show()

# %%
head_naive.query("x < 100 | x > 550")

# %%
sns.scatterplot(data=head_naive.query("x > 100 & x < 550"), x="x", y="y", alpha=0.25)
plt.show()

# %%
head_naive.drop(
    head_naive[head_naive.likelihood < likelihood_threshold].index, inplace=True
)
head_naive.drop(head_naive.query("x < 100 | x > 550").index, inplace=True)

# %%
head_naive

# %%
sns.scatterplot(data=head_naive, x="x", y="y", alpha=0.25)
plt.show()

# %%
head_naive


# %%
def plot_locations_count(data, title, scale=False, cols=5, rows=5):
    # Preprocess
    hist2d, xedges, yedges = np.histogram2d(x=data.x, y=data.y, bins=[cols, rows])
    hist2d_res = hist2d.T
    if scale:
        locations_scaled = minmax_scale(hist2d.T.flatten())
        locations_scaled = locations_scaled.reshape((cols, rows))
        hist2d_res = locations_scaled

    cmap = sns.color_palette("rocket_r", as_cmap=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(hist2d_res, cmap=cmap, ax=ax)
    ax.set_title(title, pad=20)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()
    return hist2d_res


# %%
plot_locations_count(
    data=head_naive,
    title="Total locations count\n" "C01_d0p0_2022-01-12_15.58 - naive",
    scale=False,
    cols=cols,
    rows=rows,
)

# %%
# locations_scaled = minmax_scale(hist2d_naive.T.flatten())
# locations_scaled = locations_scaled.reshape((cols, rows))
# locations_scaled

# %%
plot_locations_count(
    data=head_naive,
    title="Total locations count\n" "C01_d0p0_2022-01-12_15.58 - naive",
    scale=True,
    cols=cols,
    rows=rows,
)

# %% [markdown]
# ## Trained locations

# %%
trained_coord_path = Path("E02d49.csv")

# %%
trained_data = pd.read_csv(trained_coord_path)
trained_data

# %%
nose_trained = trained_data.filter(regex=("nose.*"))
nose_trained = nose_trained.rename(
    lambda name: name.replace("nose_", ""), axis="columns"
)
nose_trained

# %%
head_trained = trained_data.filter(regex=("head.*"))
head_trained = head_trained.rename(
    lambda name: name.replace("head_", ""), axis="columns"
)
head_trained

# %%
plot_coords(
    head=head_trained,
    nose=nose_trained,
    title="Trained mouse coordinates\n" f"{trained_coord_path.stem}",
)

# %%
head_trained.drop(
    head_trained[head_trained.likelihood < likelihood_threshold].index, inplace=True
)
head_trained.drop(head_trained.query("x < 80 or x > 550").index, inplace=True)
head_trained

# %%
sns.scatterplot(data=head_trained, x="x", y="y", alpha=0.1)
plt.show()

# %%
plot_locations_count(
    data=head_trained,
    title="Total locations count\n" f"{trained_coord_path.stem} - trained",
    scale=False,
    cols=cols,
    rows=rows,
)

# %%
plot_locations_count(
    data=head_trained,
    title="Total locations count\n" f"{trained_coord_path.stem} - trained",
    scale=True,
    cols=cols,
    rows=rows,
)

# %% [markdown]
# ## Extract actions

# %%
# Get the boundaries of the tiles
hist2d_naive, xedges_naive, yedges_naive = np.histogram2d(
    x=head_naive.x, y=head_naive.y, bins=[cols, rows]
)
hist2d_naive, xedges_naive, yedges_naive

# %%
# Populate the row/col numbers
col = np.full(shape=len(head_naive), fill_value=np.nan)
row = np.full(shape=len(head_naive), fill_value=np.nan)
for idr, _ in enumerate(tqdm(head_naive.index)):
    for idx, xedge in enumerate(xedges_naive[1:]):
        if head_naive.iloc[idr].x <= xedge:
            col[idr] = idx
            break
    for idy, yedge in enumerate(yedges_naive[1:]):
        if head_naive.iloc[idr].y <= yedge:
            row[idr] = idy
            break
head_naive["row"] = row.astype(np.int_)
head_naive["col"] = col.astype(np.int_)
head_naive

# %%
tiles_map = tiles_locations.reshape((rows, cols))
tiles_map

# %%
tiles_map.T

# %%
# Populate the tile number
head_naive["tile"] = np.full(shape=len(head_naive), fill_value=np.nan)
for row in range(rows):
    for col in range(cols):
        head_naive.loc[
            (head_naive.row == row) & (head_naive.col == col), "tile"
        ] = tiles_map.T[row, col]
head_naive

# %%
# Transpose everything because the north port is filmed
# in the bottom left instead of top right corner
for idx in tqdm(head_naive.index):
    row, col = np.argwhere(
        tiles_map.T
        == tiles_map[int(head_naive.loc[idx].row), int(head_naive.loc[idx].col)]
    ).flatten()
    head_naive.loc[idx, "row"] = row
    head_naive.loc[idx, "col"] = col
    head_naive.loc[idx, "tile"] = tiles_map.T[row, col]
head_naive

# %%
# Select only moves from one tile to another
tile_changes = head_naive.iloc[
    np.concatenate(
        [np.array([False]), (head_naive.tile.diff().dropna() != 0).to_numpy()]
    )
]
tile_changes

# %%
# Populate the action corresponding to each tile movement
row_conv = {-1: "UP", 1: "DOWN"}
col_conv = {1: "RIGHT", -1: "LEFT"}

head_naive["action"] = np.full(shape=len(head_naive), fill_value=np.nan)
diff_row = tile_changes.row.diff()
diff_col = tile_changes.col.diff()
for filt_idx, _ in enumerate(tile_changes.index[1:]):
    main_idx = tile_changes.index[filt_idx]
    if diff_col.iloc[filt_idx] == 0 and np.abs(diff_row.iloc[filt_idx]) == 1:
        head_naive.loc[main_idx, "action"] = row_conv[diff_row.iloc[filt_idx]]
    elif diff_row.iloc[filt_idx] == 0 and np.abs(diff_col.iloc[filt_idx]) == 1:
        head_naive.loc[main_idx, "action"] = col_conv[diff_col.iloc[filt_idx]]
    else:
        # If the move if the move is of more than one tile, there's something wrong
        warnings.warn(
            f"Issue with row: {filt_idx} - row: {diff_row.iloc[filt_idx]} - col: {diff_col.iloc[filt_idx]}"
        )
head_naive.dropna()

# %%
tiles_map.T

# %%
# # Filter incoherent actions
# head_naive["coherent"] = np.full(shape=len(head_naive), fill_value=False)
# filt_idx_prev = head_naive.dropna().index[0]
# for idx, filt_idx in enumerate(head_naive.loc[1:].dropna().index):
#     diff_row = head_naive.loc[filt_idx].row - head_naive.loc[filt_idx_prev].row
#     diff_col = head_naive.loc[filt_idx].col - head_naive.loc[filt_idx_prev].col
#     if diff_col == 0 and np.abs(diff_row) == 1:
#         head_naive.loc[filt_idx, "coherent"] = True
#     elif diff_row == 0 and np.abs(diff_col) == 1:
#         head_naive.loc[filt_idx, "coherent"] = True
#     filt_idx_prev = filt_idx
# head_naive_coherent = head_naive[head_naive.coherent == True]
# head_naive_coherent

# %%
# Save the dataset
actions_path = Path(f"{naive_coord_path.stem}_naive_actions.csv")
head_naive.to_csv(actions_path)

# %%
