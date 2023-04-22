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
# # Q-learning with function approximation - egocentric environment

# %% [markdown]
# [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/NiloufarRazmi/RL_Olfaction/HEAD?labpath=FuncApprox%2FFuncApprox_ego.ipynb)

# %% [markdown]
# ## The task

# %% [markdown]
# <img src='./img/task.png' width="400">

# %% [markdown]
# ## Initialization

# %%
# Import packages
# from pprint import pprint

import re

import numpy as np
import pandas as pd
import plotting
import plotting_ego
from agent import EpsilonGreedy, QLearningFuncApprox
from environment_ego import CONTEXTS_LABELS, Actions, LightCues, WrappedEnvironment
from tqdm import tqdm

# %%
# Load custom functions
from utils import Params

# %%
# Formatting & autoreload stuff
# %load_ext lab_black
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# ## Choose the task parameters

# %%
# Choose the parameters for the task
params = Params(epsilon=0.1, n_runs=10, numEpisodes=500, alpha=0.025)
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
    np.eye(
        len(env.tiles_locations) * len(env.head_angle_space),
        len(env.tiles_locations) * len(env.head_angle_space),
    ),
    len(env.cues),
    1,
)
tmp1.shape


# %%
# # 4 cues features
# tmp2 = np.vstack(
#     (
#         np.hstack(
#             (
#                 np.ones((len(env.tiles_locations) * len(env.head_angle_space), 1)),
#                 np.zeros(
#                     (
#                         len(env.tiles_locations) * len(env.head_angle_space),
#                         len(env.cues) - 1,
#                     )
#                 ),
#             )
#         ),
#         np.hstack(
#             (
#                 np.zeros((len(env.tiles_locations) * len(env.head_angle_space), 1)),
#                 np.ones((len(env.tiles_locations) * len(env.head_angle_space), 1)),
#                 np.zeros(
#                     (
#                         len(env.tiles_locations) * len(env.head_angle_space),
#                         len(env.cues) - 2,
#                     )
#                 ),
#             )
#         ),
#         np.hstack(
#             (
#                 np.zeros((len(env.tiles_locations) * len(env.head_angle_space), 2)),
#                 np.ones((len(env.tiles_locations) * len(env.head_angle_space), 1)),
#                 np.zeros(
#                     (
#                         len(env.tiles_locations) * len(env.head_angle_space),
#                         len(env.cues) - 3,
#                     )
#                 ),
#             )
#         ),
#         np.hstack(
#             (
#                 np.zeros(
#                     (
#                         len(env.tiles_locations) * len(env.head_angle_space),
#                         len(env.cues) - 1,
#                     )
#                 ),
#                 np.ones((len(env.tiles_locations) * len(env.head_angle_space), 1)),
#             )
#         ),
#     )
# )

# tmp2.shape


# %%
def ones_custom(total_rows, total_cols, pattern_rows, pattern_cols):
    ones_pattern = np.ones((pattern_rows, pattern_cols))
    zeros_pattern = np.zeros((pattern_rows, pattern_cols))
    # mat = np.full((total_rows, total_cols), fill_value=np.nan)
    for row in range(int(total_rows / pattern_rows)):
        for col in range(total_cols):
            if col == 0:
                if row == col:
                    tmp_mat = ones_pattern
                else:
                    tmp_mat = zeros_pattern
                continue
            if row == col:
                tmp_mat = np.hstack((tmp_mat, ones_pattern))
            else:
                tmp_mat = np.hstack((tmp_mat, zeros_pattern))
        if row == 0:
            mat = tmp_mat
        else:
            mat = np.vstack((mat, tmp_mat))
    return mat


# %%
def ones_pattern_repeat(rows_repeat, cols_repeat, pattern_mat):
    zeros_pattern = np.zeros_like(pattern_mat)
    for row in range(rows_repeat):
        for col in range(cols_repeat):
            if col == 0:
                if row == col:
                    tmp_mat = pattern_mat
                else:
                    tmp_mat = zeros_pattern
                continue
            if row == col:
                tmp_mat = np.hstack((tmp_mat, pattern_mat))
            else:
                tmp_mat = np.hstack((tmp_mat, zeros_pattern))
        if row == 0:
            mat = tmp_mat
        else:
            mat = np.vstack((mat, tmp_mat))
    return mat


# %%
sub_mat = ones_custom(total_rows=100, total_cols=4, pattern_rows=25, pattern_cols=1)
sub_mat.shape

# %%
# 4 cues features x 4 head direction angles
tmp2 = ones_pattern_repeat(rows_repeat=4, cols_repeat=4, pattern_mat=sub_mat)
tmp2.shape

# %%
# # 2 cues features
# # Doesn't solve the task
# tmp2 = np.vstack(
#     (
#         np.hstack(
#             (
#                 np.ones(
#                     (
#                         len(env.tiles_locations)
#                         * len(env.head_angle_space)
#                         * len(OdorID),
#                         1,
#                     )
#                 ),
#                 np.zeros(
#                     (
#                         len(env.tiles_locations)
#                         * len(env.head_angle_space)
#                         * len(OdorID),
#                         1,
#                     )
#                 ),
#             )
#         ),
#         np.hstack(
#             (
#                 np.zeros(
#                     (
#                         len(env.tiles_locations)
#                         * len(env.head_angle_space)
#                         * len(LightCues),
#                         1,
#                     )
#                 ),
#                 np.ones(
#                     (
#                         len(env.tiles_locations)
#                         * len(env.head_angle_space)
#                         * len(LightCues),
#                         1,
#                     )
#                 ),
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
#         len(env.tiles_locations) * len(env.head_angle_space) * len(LightCues),
#         len(env.tiles_locations) * len(env.head_angle_space) * len(LightCues),
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
            "p1": [-30, idx * len(env.tiles_locations) * len(env.head_angle_space)],
            "p2": [
                -30,
                (idx + 1) * len(env.tiles_locations) * len(env.head_angle_space),
            ],
            "str_text": re.sub(r"^P.*?odor - ", "", CONTEXTS_LABELS[cue]),
        }
    )
braces

# %%
plotting.plot_heatmap(
    matrix=learner.features, title="Features", ylabel="States", braces=braces
)

# %% [markdown]
# ## States and actions meaning

# %%
env.get_states_structure()

# %%
plotting_ego.plot_tiles_locations(
    states_structure=env.get_states_structure(),
    rows=env.rows,
    cols=env.cols,
    contexts_labels=CONTEXTS_LABELS,
)

# %% [markdown]
# ### Correspondance between flat states and (internal) composite states

# %%
state = 63
env.convert_flat_state_to_composite(state)

# %%
state = {"location": 13, "direction": 90, "cue": LightCues.North}
env.convert_composite_to_flat_state(state)

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
plotting_ego.plot_ego_q_values_maps(
    qtable, env.rows, env.cols, CONTEXTS_LABELS, env.get_states_structure()
)

# %%
plotting_ego.plot_location_count(
    all_state_composite=all_state_composite,
    tiles_locations=env.tiles_locations,
    cols=env.cols,
    rows=env.rows,
)

# %%
plotting_ego.plot_location_count(
    all_state_composite=all_state_composite,
    tiles_locations=env.tiles_locations,
    cols=env.cols,
    rows=env.rows,
    cues=env.cues,
    contexts_labels=CONTEXTS_LABELS,
)

# %%
