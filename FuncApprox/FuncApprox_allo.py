# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Q-learning with function approximation - allocentric environment

# %% [markdown]
# [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/NiloufarRazmi/RL_Olfaction/HEAD?labpath=FuncApprox%2FFuncApprox_allo.ipynb)

# %% [markdown]
# ## The task

# %% [markdown]
# <img src='./img/task.png' width="400">

# %% [markdown]
# ## Initialization

import re

# %%
# Presentation figures
from collections import OrderedDict

import matplotlib.pyplot as plt

# %%
# Import packages
import numpy as np
import pandas as pd
import plotting
import seaborn as sns
from agent import EpsilonGreedy, QLearningFuncApprox
from environment_allo import (
    CONTEXTS_LABELS,
    Actions,
    LightCues,
    OdorID,
    WrappedEnvironment,
)
from plotting_ego import plot_location_count
from tqdm import tqdm

# %%
# Load custom functions
from utils import Params

sns.set(font_scale=2)

# %%
# Formatting & autoreload stuff
# %load_ext lab_black
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# ## Choose the task parameters

# %%
# Choose the parameters for the task
params = Params(epsilon=0.1, n_runs=20, numEpisodes=300, alpha=0.025)
params

# %% [markdown]
# ## Load the environment and the agent algorithms

# %%
# Load the environment
env = WrappedEnvironment(params)

# %% [markdown]
# ### Choose the features

# %% [markdown]
# Manually engineered features, optional.
# If `None`, a diagonal matrix of features will be created automatically

# %%
# Place only features
tmp1 = np.matlib.repmat(
    np.eye(len(env.tiles_locations), len(env.tiles_locations)), len(env.cues), 1
)
tmp1.shape

# %%
# 4 cues features
# Solves the task but not optimally
tmp2 = np.vstack(
    (
        np.hstack(
            (
                np.ones((len(env.tiles_locations), 1)),
                np.zeros((len(env.tiles_locations), len(env.cues) - 1)),
            )
        ),
        np.hstack(
            (
                np.zeros((len(env.tiles_locations), 1)),
                np.ones((len(env.tiles_locations), 1)),
                np.zeros((len(env.tiles_locations), len(env.cues) - 2)),
            )
        ),
        np.hstack(
            (
                np.zeros((len(env.tiles_locations), 2)),
                np.ones((len(env.tiles_locations), 1)),
                np.zeros((len(env.tiles_locations), len(env.cues) - 3)),
            )
        ),
        np.hstack(
            (
                np.zeros((len(env.tiles_locations), len(env.cues) - 1)),
                np.ones((len(env.tiles_locations), 1)),
            )
        ),
    )
)

tmp2.shape

# %%
# # 2 cues features
# # Doesn't solve the task
# tmp2 = np.vstack(
#     (
#         np.hstack(
#             (
#                 np.ones((len(env.tiles_locations) * len(OdorID), 1)),
#                 np.zeros((len(env.tiles_locations) * len(OdorID), 1)),
#             )
#         ),
#         np.hstack(
#             (
#                 np.zeros((len(env.tiles_locations) * len(LightCues), 1)),
#                 np.ones((len(env.tiles_locations) * len(LightCues), 1)),
#             )
#         ),
#     )
# )

# tmp2.shape

# %%
features = np.hstack((tmp1, tmp2))
features.shape

# %%
# # Place-light features
# tmp3 = np.matlib.repmat(
#     np.eye(
#         len(env.tiles_locations) * len(LightCues),
#         len(env.tiles_locations) * len(LightCues),
#     ),
#     len(LightCues),
#     1,
# )
# tmp3.shape

# %%
# features = np.hstack((tmp1, tmp2, tmp3))
# features.shape

# %%
# # Features == identity matrix
# features = None

# %%
# Load the agent algorithms
learner = QLearningFuncApprox(
    learning_rate=params.alpha,
    gamma=params.gamma,
    state_size=env.numStates,
    action_size=env.numActions,
    features_matrix=features,
)
explorer = EpsilonGreedy(epsilon=params.epsilon)

# %%
braces = []
for idx, cue in enumerate(CONTEXTS_LABELS):
    braces.append(
        {
            "p1": [-7, idx * len(env.tiles_locations)],
            "p2": [-7, (idx + 1) * len(env.tiles_locations)],
            "str_text": re.sub(r"^P.*?odor - ", "", CONTEXTS_LABELS[cue]),
        }
    )
braces

# %%
# braces.append(
#     {
#         "p2": [15.0, 10.0],
#         "p1": [5.0, 10.0],
#         "str_text": "Locations",
#     }
# )
# braces

# %%
plotting.plot_heatmap(
    matrix=learner.features, title="Features", ylabel="States", braces=braces
)

# %%
learner.Q_hat_table.shape, learner.weights.shape, learner.features.shape

# %% [markdown]
# ## States and actions meaning

# %%
# State space
for idx, cue in enumerate(CONTEXTS_LABELS):
    plotting.plot_tiles_locations(
        np.array(list(env.tiles_locations)) + idx * len(env.tiles_locations),
        env.rows,
        env.cols,
        title=CONTEXTS_LABELS[cue],
    )

# %% [markdown]
# ### Correspondance between flat states and (internal) composite states

# %%
state = 63
env.convert_flat_state_to_composite(state)

# %%
env.convert_composite_to_flat_state({"location": 13, "cue": LightCues.North})

# %% [markdown]
# ### Action meaning

# %%
action = 0
Actions(action)

# %% [markdown]
# ## Main loop

# %%
rewards = np.zeros((params.numEpisodes, params.n_runs))
steps = np.zeros((params.numEpisodes, params.n_runs))
episodes = np.arange(params.numEpisodes)
qtables = np.zeros((params.n_runs, *learner.Q_hat_table.shape))
all_states = []
all_actions = []

for run in range(params.n_runs):  # Run several times to account for stochasticity
    learner.reset(
        action_size=env.numActions
    )  # Reset the Q-table and the weights between runs

    for episode in tqdm(
        episodes, desc=f"Run {run+1}/{params.n_runs} - Episodes", leave=False
    ):
        state = env.reset()  # Reset the environment
        step_count = 0
        done = False
        total_rewards = 0

        while not done:
            learner.Q_hat_table = learner.Q_hat(learner.weights, learner.features)

            action = explorer.choose_action(
                action_space=env.action_space,
                state=state,
                qtable=learner.Q_hat_table,
            )

            # Record states and actions
            all_states.append(state)
            all_actions.append(Actions(action).name)

            # Take the action (a) and observe the outcome state(s') and reward (r)
            new_state, reward, done = env.step(action, state)

            learner.weights[:, action] = learner.update_weights(
                state, action, reward, new_state
            )

            total_rewards += reward
            step_count += 1

            # Our new state is state
            state = new_state

        # explorer.epsilon = explorer.update_epsilon(episode)

        rewards[episode, run] = total_rewards
        steps[episode, run] = step_count
    qtables[run, :, :] = learner.Q_hat_table

# %% [markdown]
# ## Postprocessing

# %%
res = pd.DataFrame(
    data={
        "Episodes": np.tile(episodes, reps=params.n_runs),
        "Rewards": rewards.flatten(order="F"),
        "Steps": steps.flatten(order="F"),
    }
)
res["cum_rewards"] = rewards.cumsum(axis=0).flatten(order="F")
# st = pd.DataFrame(data={"Episodes": episodes, "Steps": steps.mean(axis=1)})
qtable = qtables.mean(axis=0)  # Average the Q-table between runs

# %%
res

# %%
tmp = []
for idx, st in enumerate(tqdm(all_states)):
    tmp.append(env.convert_flat_state_to_composite(st))
all_state_composite = pd.DataFrame(tmp)
all_state_composite

# %% [markdown]
# ## Visualization

# %%
plotting.plot_heatmap(matrix=learner.weights, title="Weights")

# %%
plotting.plot_heatmap(matrix=qtable, title="Q-table")

# %%
plotting.plot_states_actions_distribution(all_states, all_actions)

# %%
plotting.plot_steps_and_rewards(res, n_runs=params.n_runs, log=True)

# %%
plotting.plot_q_values_maps(qtable, env.rows, env.cols, CONTEXTS_LABELS)

# %%
plotting.plot_rotated_q_values_maps(qtable, env.rows, env.cols, CONTEXTS_LABELS)

# %%
plot_location_count(
    all_state_composite,
    tiles_locations=env.tiles_locations,
    cols=env.cols,
    rows=env.rows,
    cues=None,
    contexts_labels=None,
)

# %%
plot_location_count(
    all_state_composite,
    tiles_locations=env.tiles_locations,
    cols=env.cols,
    rows=env.rows,
    cues=env.cues,
    contexts_labels=CONTEXTS_LABELS,
)

# %% [markdown]
# ## Presentations figures

# %%
fig, ax = plt.subplots(figsize=(12, 8))
sns.lineplot(data=res, x="Episodes", y="Steps", ax=ax)
sns.regplot(
    x="Episodes",
    y="Steps",
    data=res,
    ax=ax,
    order=5,
    scatter=False,
    # ci=None,
    # scatter_kws={"s": 80},
    line_kws={"color": "C1"},
)
ax.set(xlabel="Trial")
fig.tight_layout()
fig.patch.set_alpha(0)
fig.patch.set_facecolor("white")
plt.show()

# %%
fig, ax = plt.subplots(figsize=(12, 8))
sns.lineplot(data=res, x="Episodes", y="Rewards", ax=ax)
sns.regplot(
    x="Episodes",
    y="Rewards",
    data=res,
    ax=ax,
    order=6,
    # ci=None,
    # scatter_kws={"s": 80},
    scatter=False,
    line_kws={"color": "C1"},
)
ax.set_ylim(bottom=-0.5)
fig.tight_layout()
fig.patch.set_alpha(0)
fig.patch.set_facecolor("white")
ax.set(xlabel="Trial")
plt.show()

# %%
emoji = [
    [{"emoji": "💡", "coords": [4.5, 0.5]}],
    [{"emoji": "💡", "coords": [0.5, 4.5]}],
    [{"emoji": "💧", "coords": [0.5, 0.5]}, {"emoji": "🍌", "coords": [4, -0.25]}],
    [{"emoji": "💧", "coords": [4.5, 4.5]}, {"emoji": "🍋", "coords": [4, -0.25]}],
]

# %%
trunc_labels = OrderedDict(
    [
        (LightCues.North, "Pre odor - North light"),
        (LightCues.South, "Pre odor - South light"),
        (OdorID.A, "Post odor - "),
        (OdorID.B, "Post odor - "),
    ]
)

# %%
for idx, cue in enumerate(trunc_labels):
    current_map = np.array(list(env.tiles_locations)) + idx * len(env.tiles_locations)
    current_q_table = qtable[current_map, :]
    plotting.plot_policy_emoji(
        qtable=current_q_table,
        rows=env.rows,
        cols=env.cols,
        label=trunc_labels[cue],
        emoji=emoji[idx],
    )

# %%
fig, ax = plt.subplots(figsize=(9, 9))
# cmap = sns.light_palette("seagreen", as_cmap=True)
cmap = sns.color_palette("light:b", as_cmap=True)
chart = sns.heatmap(qtable, cmap=cmap, ax=ax, cbar=False)
chart.set_title("Actions", fontsize=40)
chart.set_ylabel("States", fontsize=40)
ax.tick_params(
    left=False, right=False, labelleft=False, labelbottom=False, bottom=False
)
fig.patch.set_alpha(0)
fig.patch.set_facecolor("white")
plt.show()

# %%
plt.plot(res.Steps.rolling(window=3).mean())

# %%
plt.plot(res.Rewards.rolling(window=3).mean())
