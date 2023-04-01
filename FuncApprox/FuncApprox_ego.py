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

import numpy as np
import pandas as pd
import plotting
import plotting_ego
from agent import EpsilonGreedy, QLearningFuncApprox
from environment_ego import (
    CONTEXTS_LABELS,
    Actions,
    LightCues,
    OdorID,
    WrappedEnvironment,
)
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
params = Params(epsilon=0.1, n_runs=3, numEpisodes=1000, alpha=0.025)
params

# %% [markdown]
# ## Load the environment and the agent algorithms

# %%
# Load the environment
env = WrappedEnvironment(params)

# %%
# Manually engineered features, optional
# if `None`, a diagonal matrix of features will be created automatically
# features = np.matlib.repmat(
#     np.eye(
#         len(env.tiles_locations) * len(env.head_angle_space),
#         len(env.tiles_locations) * len(env.head_angle_space),
#     ),
#     len(LightCues) * len(OdorID),
#     len(LightCues) * len(OdorID),
# )

# %%
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
tmp2 = np.vstack(
    (
        np.hstack(
            (
                np.ones((len(env.tiles_locations) * len(env.head_angle_space), 1)),
                np.zeros(
                    (
                        len(env.tiles_locations) * len(env.head_angle_space),
                        len(env.cues) - 1,
                    )
                ),
            )
        ),
        np.hstack(
            (
                np.zeros((len(env.tiles_locations) * len(env.head_angle_space), 1)),
                np.ones((len(env.tiles_locations) * len(env.head_angle_space), 1)),
                np.zeros(
                    (
                        len(env.tiles_locations) * len(env.head_angle_space),
                        len(env.cues) - 2,
                    )
                ),
            )
        ),
        np.hstack(
            (
                np.zeros((len(env.tiles_locations) * len(env.head_angle_space), 2)),
                np.ones((len(env.tiles_locations) * len(env.head_angle_space), 1)),
                np.zeros(
                    (
                        len(env.tiles_locations) * len(env.head_angle_space),
                        len(env.cues) - 3,
                    )
                ),
            )
        ),
        np.hstack(
            (
                np.zeros(
                    (
                        len(env.tiles_locations) * len(env.head_angle_space),
                        len(env.cues) - 1,
                    )
                ),
                np.ones((len(env.tiles_locations) * len(env.head_angle_space), 1)),
            )
        ),
    )
)

tmp2.shape

# %%
features = np.hstack((tmp1, tmp2))
features.shape

# %%
features = None

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
plotting.plot_heatmap(matrix=learner.features, title="Features", ylabel="States")

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
    learner.reset_Q_hat_table()  # Reset the Q-table between runs

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
        "Rewards": rewards.flatten(),
        "Steps": steps.flatten(),
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
plotting.plot_steps_and_rewards(res)

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
