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
features = np.matlib.repmat(
    np.eye(
        len(env.tiles_locations) * len(env.head_angle_space),
        len(env.tiles_locations) * len(env.head_angle_space),
    ),
    len(LightCues) * len(OdorID),
    len(LightCues) * len(OdorID),
)
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
plotting.plot_heatmap(matrix=learner.features, title="Features")

# %% [markdown]
# ## States and actions meaning

# %%
env.get_states_structure()

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

# %% tags=[]
tmp = []
for idx, st in enumerate(tqdm(all_states)):
    tmp.append(env.convert_flat_state_to_composite(st))
all_state_composite = pd.DataFrame(tmp)
all_state_composite


# %% tags=[]
def get_location_count(all_state_composite, cue=None):
    """Count the occurences for each tile location.

    Optionally filter by `cue`"""
    location_count = np.zeros(len(env.tiles_locations))
    for tile in env.tiles_locations:
        if cue:  # Select based on chosen cue
            location_count[tile] = len(
                all_state_composite[
                    (all_state_composite.location == tile)
                    & (all_state_composite.cue == cue)
                ]
            )
        else:  # Select
            location_count[tile] = len(
                all_state_composite[all_state_composite.location == tile]
            )
    location_count = location_count.reshape((env.rows, env.cols))
    return location_count


# %% tags=[]
get_location_count(all_state_composite, cue=OdorID.B)

# %% [markdown]
# ## Visualization

import matplotlib as mpl

# %% tags=[]
import matplotlib.pyplot as plt


def plot_location_count(all_state_composite, cues=None):
    # cmap = sns.color_palette("Blues", as_cmap=True)
    cmap = sns.color_palette("rocket_r", as_cmap=True)

    if cues:
        fig, ax = plt.subplots(2, 2, figsize=(10, 8))
        for idx, cue in enumerate(cues):
            location_count = get_location_count(all_state_composite, cue=cue)
            chart = sns.heatmap(location_count, cmap=cmap, ax=ax.flatten()[idx])
            chart.set(title=CONTEXTS_LABELS[cue])
            ax.flatten()[idx].set_xticks([])
            ax.flatten()[idx].set_yticks([])
        fig.suptitle("Locations counts during training", fontsize="xx-large")

    else:  # Plot everything
        location_count = get_location_count(all_state_composite, cue=cues)
        fig, ax = plt.subplots()
        chart = sns.heatmap(location_count, cmap=cmap, ax=ax)
        chart.set(title="Total locations count during training")
        ax.set_xticks([])
        ax.set_yticks([])
    # fig.tight_layout()
    plt.show()


# %% tags=[]
plot_location_count(all_state_composite)

# %% tags=[]
plot_location_count(all_state_composite, cues=env.cues)

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
