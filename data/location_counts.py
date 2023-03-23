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

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import minmax_scale

# %%
sns.set_theme(font_scale=1.5)
# %load_ext lab_black

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
# ## Naive

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
head_naive[head_naive.likelihood < 0.5]

# %%
sns.scatterplot(data=head_naive[head_naive.likelihood > 0.75], x="x", y="y", alpha=0.25)
plt.show()

# %%
head_naive.query("x < 100 | x > 550")

# %%
sns.scatterplot(data=head_naive.query("x > 100 & x < 550"), x="x", y="y", alpha=0.25)
plt.show()

# %%
head_naive.drop(head_naive[head_naive.likelihood < 0.5].index, inplace=True)
head_naive.drop(head_naive.query("x < 100 | x > 550").index, inplace=True)

# %%
head_naive

# %%
sns.scatterplot(data=head_naive, x="x", y="y", alpha=0.25)
plt.show()

# %%
head_naive

# %%
rows = 5
cols = 5
tiles_locations = np.arange(rows * cols)
tiles_locations

# %%
# x_bins = np.arange(cols)
# y_bins = np.arange(rows)
# for col in range(cols):
#     x_bins[col] = (
#         head_naive.x.max() - head_naive.x.min()
#     ) / cols + col * head_naive.x.min()
# for row in range(rows):
#     y_bins[row] = (
#         head_naive.y.max() - head_naive.y.min()
#     ) / rows + row * head_naive.y.min()
# x_bins, y_bins

# %%
# head_naive["col"] = np.digitize(head_naive.x, x_bins)
# head_naive["row"] = np.digitize(head_naive.y, y_bins)
# head_naive

# %%
hist2d_naive, xedges, yedges = np.histogram2d(
    x=head_naive.x, y=head_naive.y, bins=[cols, rows]
)
hist2d_naive

# %%
hist2d_naive.T


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
locations_scaled = minmax_scale(hist2d_naive.T.flatten())
locations_scaled = locations_scaled.reshape((cols, rows))
locations_scaled

# %%
plot_locations_count(
    data=head_naive,
    title="Total locations count\n" "C01_d0p0_2022-01-12_15.58 - naive",
    scale=True,
    cols=cols,
    rows=rows,
)

# %% [markdown]
# ## Trained

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
head_trained.drop(head_trained[head_trained.likelihood < 0.5].index, inplace=True)
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
