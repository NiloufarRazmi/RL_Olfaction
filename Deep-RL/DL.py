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
# # Deep Learning

# %% [markdown]
# 1. inputs need to reflect position in arena and odor (NOT CONJUNCTIONS)
# 2. outputs need to reflect action values
# 3. actions are selected via softmax on output neuron activity. 
# 4. RPE requires knowing value of new state
#    -- so this will require a forward pass using "new state" inputs.

# %%
# Import packages
import matplotlib as mpl

# # %matplotlib ipympl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# %%
# Load custom functions
from deep_learning import Network
from utils import Params

# %%
# Formatting & autoreload stuff
# %load_ext lab_black
# %load_ext autoreload
# %autoreload 2
sns.set_theme(font_scale=1.5)
mpl.rcParams["font.family"] = ["Fira Sans", "sans-serif"]

# %% [markdown]
# ## Main options

# %%
nTrain = 1000000
nTest = 100

# %%
nTot = nTrain + nTest
# create two normal input channels
X = np.random.multivariate_normal([0, 0], 10 * np.eye(2), size=nTot)
# X = pd.read_csv("X.csv", header=None).to_numpy()
# y is positive if X(1) & X(2) are positive, OR if X(1) and X(2) are negative.
X.shape

# %%
y = np.sign(X[:, 0] * X[:, 1]) / 2.0 + 0.5
y.shape

# %% [markdown]
# ## Choose the task parameters

# %%
# Choose the parameters for the task
p = Params(learning_rate=0.001, nLayers=6, nHiddenUnits=20)
p

# %% [markdown]
# ## Step 2: Build network

# %%
net = Network(
    nInputUnits=X.shape[1],
    nLayers=p.nLayers,
    nOutputUnits=1,
    nHiddenUnits=p.nHiddenUnits,
    initVar=1,
)

# %%
# Weights matrices shapes
[layer.shape for layer in net.wtMatrix]

# %% [markdown]
# ## Step 3: Train network

# %%
# allError = np.nan * np.ones(nTot)
# y_hat = np.nan * np.ones(nTot)

# for i in tqdm(range(nTrain)):
#     activity = net.forward_pass(nLayers=p.nLayers, X=X, obs=i)

#     # Take an action! softmax over actions or similar

#     # incorporate your model of the task, to determine where agent actually goes.

#     # Now you need to do another forward pass, to see how good the new
#     # state is so that you can compute the RPE below.

#     # your cost function will differ from the one below,
#     # should look something like this:
#     # C =  R - X(S)*W+ DISCOUNT*max(X(S')*W)

#     delta = net.backward_pass(nLayers=p.nLayers, y=y, activity=activity, obs=i)

#     # Update weight matrices according to gradients and activities:
#     for j in range(len(net.wtMatrix) - 1):
#         # net.wtMatrix[j] = (
#         #     net.wtMatrix[j]
#         #     + p.learning_rate * np.expand_dims(activity[j], axis=1) * delta[j + 1].T
#         # )
#         net.wtMatrix[j] = net.gradient_descent(
#             weight=net.wtMatrix[j],
#             learning_rate=p.learning_rate,
#             activity=activity[j],
#             delta=delta[j + 1],
#         )

#     # store error:
#     allError[i] = delta[-1]
#     y_hat[i] = activity[-1] > 0.5

# %%
allError, y_hat, delta, activity = net.backprop(
    X=X, y=y, nLayers=p.nLayers, learning_rate=p.learning_rate
)

# %%
y_hat = y_hat > 0.5
y_hat

# %%
activity

# %%
delta

# %%
allError

# %%
y_hat

# %%
Bins = np.round(np.linspace(0, len(allError), num=100)).astype(int)

meanError = np.zeros_like(Bins) * np.nan
for i in range(len(Bins) - 1):
    meanError[i] = np.nanmean(abs(allError[Bins[i] : Bins[i + 1]]))
meanError

# %%
fig, ax = plt.subplots(figsize=(8, 6))
chart = sns.lineplot(meanError, ax=ax)
ax.set_ylabel("Error")
ax.set_xlabel("Batches")
plt.show()

# %%
weights_avg = np.nan * np.empty((1, len(net.wtMatrix)))
weights_std = np.nan * np.empty((1, len(net.wtMatrix)))
for w_id, w_val in enumerate(net.wtMatrix):
    weights_avg[0, w_id] = w_val.mean()
    weights_std[0, w_id] = w_val.std()
weights_metrics_avg = pd.DataFrame(weights_avg)
weights_metrics_std = pd.DataFrame(weights_std)

# %%
weights_metrics_avg

# %%
weights_metrics_std

# %%
grads_avg = np.nan * np.empty((1, len(delta)))
grads_std = np.nan * np.empty((1, len(delta)))
for d_id, d_val in enumerate(delta):
    grads_avg[0, d_id] = d_val.mean()
    grads_std[0, d_id] = d_val.std()
grads_metrics_avg = pd.DataFrame(grads_avg)
grads_metrics_std = pd.DataFrame(grads_std)

# %%
grads_metrics_avg

# %%
grads_metrics_std

# %% [markdown]
# ## Step 4: Test Network

# %%
for obs in range((nTrain + 1), nTot):
    activity = net.forward_pass(x_obs=X[obs, :])

    # store error
    allError[obs] = delta[-1]
    y_hat[obs] = activity[-1] > 0.5

isTest = np.zeros(nTot, dtype=bool)
isTest[nTrain + 1 :] = True

# %%
activity

# %%
allError

# %%
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].set_title("Ground Truth")
sns.scatterplot(
    x=X[(y == 1) & isTest, 0],
    y=X[(y == 1) & isTest, 1],
    s=200,
    ax=ax[0],
)
sns.scatterplot(
    x=X[(y == 0) & isTest, 0],
    y=X[(y == 0) & isTest, 1],
    s=200,
    ax=ax[0],
)
ax[0].set_xlabel("Feature 1")
ax[0].set_ylabel("Feature 2")


ax[1].set_title("Model Classification")
sns.scatterplot(
    x=X[(y_hat == 1) & isTest, 0],
    y=X[(y_hat == 1) & isTest, 1],
    s=200,
    ax=ax[1],
)
sns.scatterplot(
    x=X[(y_hat == 0) & isTest, 0],
    y=X[(y_hat == 0) & isTest, 1],
    s=200,
    ax=ax[1],
)
ax[1].set_xlabel("Feature 1")
ax[1].set_ylabel("Feature 2")

fig.tight_layout()
plt.show()
