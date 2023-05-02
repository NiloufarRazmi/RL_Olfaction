# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
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
# # Deep Learning

# %% [markdown]
# 1. inputs need to reflect position in arena and odor (NOT CONJUNCTIONS)
# 2. outputs need to reflect action values
# 3. actions are selected via softmax on output neuron activity.
# 4. RPE requires knowing value of new state
#    -- so this will require a forward pass using "new state" inputs.

import matplotlib as mpl

# # %matplotlib ipympl
import matplotlib.pyplot as plt

# %%
# Import packages
import numpy as np
import seaborn as sns
from deep_learning import Network

# %%
# Load custom functions
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
# Y is positive if X(1) & X(2) are positive, OR if X(1) and X(2) are negative.
X.shape

# %%
Y = np.sign(X[:, 0] * X[:, 1]) / 2.0 + 0.5
Y

# %% [markdown]
# ## Choose the task parameters

# %%
# Choose the parameters for the task
p = Params(learning_rate=0.001, nLayers=5, nHiddenUnits=20)
p

# %% [markdown]
# ## Step 2: Build network

# %%
nn = Network(
    nInputUnits=X.shape[1],
    nLayers=p.nLayers,
    nOutputUnits=1,
    nHiddenUnits=p.nHiddenUnits,
    initVar=1,
)

# %% [markdown]
# ## Step 3: Train network

# %%
# allError = np.nan * np.ones(nTot)
# catPredict = np.nan * np.ones(nTot)

# for i in tqdm(range(nTrain)):
#     activity = nn.forward_pass(nLayers=p.nLayers, X=X, i=i)

#     # Take an action! softmax over actions or similar

#     # incorporate your model of the task, to determine where agent actually goes.

#     # Now you need to do another forward pass, to see how good the new
#     # state is so that you can compute the RPE below.

#     # your cost function will differ from the one below,
#     # should look something like this:
#     # C =  R - X(S)*W+ DISCOUNT*max(X(S')*W)

#     delta = nn.backward_pass(nLayers=p.nLayers, Y=Y, activity=activity, i=i)

#     # Update weight matrices according to gradients and activities:
#     for j in range(len(nn.wtMatrix) - 1):
#         # nn.wtMatrix[j] = (
#         #     nn.wtMatrix[j]
#         #     + p.learning_rate * np.expand_dims(activity[j], axis=1) * delta[j + 1].T
#         # )
#         nn.wtMatrix[j] = nn.gradient_descent(
#             weight=nn.wtMatrix[j],
#             learning_rate=p.learning_rate,
#             activity=activity[j],
#             delta=delta[j + 1],
#         )

#     # store error:
#     allError[i] = delta[-1]
#     catPredict[i] = activity[-1] > 0.5

# %%
allError, catPredict, delta, activity = nn.backprop(
    n_obs=nTot, X=X, Y=Y, nLayers=p.nLayers, learning_rate=p.learning_rate
)

# %%
activity

# %%
delta

# %%
allError

# %%
catPredict

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

# %% [markdown]
# ## Step 4: Test Network

# %%
for i in range((nTrain + 1), nTot):
    activity = nn.forward_pass(nLayers=p.nLayers, X=X, i=i)

    # store error
    allError[i] = delta[-1]
    catPredict[i] = activity[-1] > 0.5

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
    x=X[(Y == 1) & isTest, 0],
    y=X[(Y == 1) & isTest, 1],
    s=200,
    ax=ax[0],
)
sns.scatterplot(
    x=X[(Y == 0) & isTest, 0],
    y=X[(Y == 0) & isTest, 1],
    s=200,
    ax=ax[0],
)
ax[0].set_xlabel("Feature 1")
ax[0].set_ylabel("Feature 2")


ax[1].set_title("Model Classification")
sns.scatterplot(
    x=X[(catPredict == 1) & isTest, 0],
    y=X[(catPredict == 1) & isTest, 1],
    s=200,
    ax=ax[1],
)
sns.scatterplot(
    x=X[(catPredict == 0) & isTest, 0],
    y=X[(catPredict == 0) & isTest, 1],
    s=200,
    ax=ax[1],
)
ax[1].set_xlabel("Feature 1")
ax[1].set_ylabel("Feature 2")

fig.tight_layout()
plt.show()

# %%
