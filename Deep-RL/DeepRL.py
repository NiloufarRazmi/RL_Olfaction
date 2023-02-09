# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Deep RL

# %% [markdown]
# 1. inputs need to reflect position in arena and odor (NOT CONJUNCTIONS)
# 2. outputs need to reflect action values
# 3. actions are selected via softmax on output neuron activity.
# 4. RPE requires knowing value of new state
#    -- so this will require a forward pass using "new state" inputs.

# # Replace `%matplotlib ipympl` by `%matplotlib inline` in case you get javascript issues
# # %matplotlib ipympl
import matplotlib.pyplot as plt

# %%
# Import packages
import numpy as np
from deep_learning import Network
from tqdm import tqdm

# %%
# Load custom functions
from utils import Params, Sigmoid

# %%
# Formatting & autoreload stuff
# %load_ext lab_black
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# ## Main options

# %%
nTrain = 1000000
nTest = 100
problem = 2

# %%
nTot = nTrain + nTest
# create two normal input channels
X = np.random.multivariate_normal([0, 0], 10 * np.eye(2), size=nTot)
# X = pd.read_csv("X.csv", header=None).to_numpy()
# Y is positive if X(1) & X(2) are positive, OR if X(1) and X(2) are negative.
X.shape

# %%
Y = np.sign(X[:, 0] * X[:, 1]) / 2.0 + 0.5

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
sigmoid = Sigmoid()


allError = np.nan * np.ones(nTot)
catPredict = np.nan * np.ones(nTot)

for i in tqdm(range(nTrain)):
    # Generate model prediction (forward pass of activity through units):
    activity = [np.array([]) for _ in range(p.nLayers)]
    for j in range(p.nLayers):
        # Determine layer input:
        if j == 0:
            input = X[i, :]  # THIS WILL BE YOUR POSITION/ODOR!!!!!
        else:
            if activity[j].shape == ():  # Convert to a vector in case it is scalar
                activity[j] = activity[j, np.newaxis]
            input = activity[j - 1] @ nn.wtMatrix[j - 1]

        # Apply non-linearity
        if nn.nonLin[j]:
            activity[j] = sigmoid(input)
        else:
            activity[j] = input

        # Take an action! softmax over actions or similar

        # incorporate your model of the task, to determine where agent actually goes.

        # Now you need to do another forward pass, to see how good the new
        # state is so that you can compute the RPE below.

        # your cost function will differ from the one below, should look something like this:
        # C =  R - X(S)*W+ DISCOUNT*max(X(S')*W)

    # Backpropagate errors to compute gradients for all layers:
    delta = [np.array([]) for _ in range(p.nLayers)]
    for j in reversed(range(p.nLayers)):
        # Determine layer input:
        if j == p.nLayers - 1:
            # IF there is nonlinearity, should multiply by derivative of
            # activation with respect to input (activity.*(1-activity)) here.
            delta[j] = (Y[i] - activity[j]) * (
                sigmoid.gradient(activity[j])
            ).T  # THIS SHOULD BE REPLACED WITH YOUR COST FUNCTION!

            # doing this in RL framework means that you'll need one RPE for
            # each output neuron -- so RPE computed above should be
            # associated with the action agent took... all other RPEs
            # should be zero.

        else:
            # OK, here is the magic -- neurons in layer j share the
            # gradient (ie. prediction errors) from the next layer
            # according to their responsibility... that is to say, if I
            # project to a unit in next layer with a strong weight,
            # then i inherit the gradient (PE) of that unit.
            if delta[j + 1].shape == ():  # Convert to a vector in case it is scalar
                delta[j + 1] = delta[j + 1, np.newaxis]
            delta[j] = (
                nn.wtMatrix[j] @ delta[j + 1] * (activity[j] * (1.0 - activity[j])).T
            )

    # Update weight matrices according to gradients and activities:
    for j in range(len(nn.wtMatrix) - 1):
        nn.wtMatrix[j] = (
            nn.wtMatrix[j]
            + p.learning_rate * np.expand_dims(activity[j], axis=1) * delta[j + 1].T
        )

    # store error:
    allError[i] = delta[-1]
    catPredict[i] = activity[-1] > 0.5

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
fig2 = plt.figure(2)
fig2.clear(True)
plt.plot(meanError)
plt.ylabel("Error")
plt.xlabel("Batches")
plt.show()

# %% [markdown]
# ## Step 4: Test Network

# %%
for i in range((nTrain + 1), nTot):
    # Generate model prediction (forward pass of activity through units):
    activity = [np.array([]) for _ in range(p.nLayers)]
    for j in range(p.nLayers):
        # Determine layer input:
        if j == 0:
            input = X[i, :]  # initial layer is activated according to input
        else:
            input = activity[j - 1] @ nn.wtMatrix[j - 1]

        # Apply non-linearity
        if nn.nonLin[j]:
            activity[j] = sigmoid(input)
        else:
            activity[j] = input

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
fig3 = plt.figure(3)
fig3.clear(True)
plt.subplot(1, 2, 1)
plt.title("Ground Truth")
# plt.plot([-2, 2], [0, 0], "--k")
# plt.plot([0, 0], [-2, 2], "--k")
plt.plot(
    X[(Y == 1) & isTest, 0],
    X[(Y == 1) & isTest, 1],
    "o",
    markerfacecolor="r",
    linewidth=1,
    markeredgecolor="k",
    markersize=14,
)
plt.plot(
    X[(Y == 0) & isTest, 0],
    X[(Y == 0) & isTest, 1],
    "o",
    markerfacecolor="b",
    linewidth=1,
    markeredgecolor="k",
    markersize=14,
)
plt.ylabel("Feature 1")
plt.xlabel("Feature 2")


plt.subplot(1, 2, 2)
plt.title("Model Classification")
# plt.plot([-2, 2], [0, 0], "--k")
# plt.plot([0, 0], [-2, 2], "--k")
plt.plot(
    X[(catPredict == 1) & isTest, 0],
    X[(catPredict == 1) & isTest, 1],
    "o",
    markerfacecolor="r",
    linewidth=1,
    markeredgecolor="k",
    markersize=14,
)
plt.plot(
    X[(catPredict == 0) & isTest, 0],
    X[(catPredict == 0) & isTest, 1],
    "o",
    markerfacecolor="b",
    linewidth=1,
    markeredgecolor="k",
    markersize=14,
)
plt.ylabel("Feature 1")
plt.xlabel("Feature 2")

plt.show()
