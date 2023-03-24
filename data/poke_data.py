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

# %%
sns.set_theme(font_scale=1.5)
# %load_ext lab_black

# %%
path = Path("E02_d49p1C_18.Jul.22_15.19_PHASE1.csv")

# %%
data = pd.read_csv(path)
data.Time = pd.to_datetime(data.Time, unit="s")
data

# %%
data.Time[1] - data.Time[0]

# %%
# time_diff = np.diff(data.Time)
time_diff = np.array([t.total_seconds() for t in data.Time.diff().dropna()])
time_diff

# %%
np.unique(time_diff).shape

# %%
# time_diff_sel = time_diff[time_diff > pd.Timedelta(seconds=1)]
# time_diff_sel.shape

# %%
time_diff[time_diff > 100]

# %%
fig, ax = plt.subplots(figsize=(9, 9))
# chart = sns.histplot(time_diff_round, ax=ax)
chart = sns.scatterplot(time_diff, ax=ax)
chart.set_title("Time difference between 2 consecutive data points")
chart.set_ylabel("Seconds")
# fig.tight_layout()
plt.show()
