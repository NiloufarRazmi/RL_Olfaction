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
# # Function approximation notebook

# %% [markdown]
# ## Initialization

# %%
# Import packages
import numpy as np
import pandas as pd
from agent import EpsilonGreedy, QLearningFuncApprox
from environment import Environment
from plotting import (
    plot_features,
    plot_q_values_map,
    plot_rotated_q_values_map,
    plot_steps_and_rewards,
    qtable_directions_map,
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
params = Params(epsilon=0.1, n_runs=3, numEpisodes=100)
params

# %% [markdown]
# ## Load the environment and the agent algorithms

# %%
# Load the environment
env = Environment(params)

# %%
# State space
np.reshape(list(env.state_space), (env.rows, env.cols))

# %%
# Load the agent algorithms
learner = QLearningFuncApprox(
    learning_rate=params.alpha,
    gamma=params.gamma,
    state_size=env.numStates,
    action_size=env.numActions,
    jointRep=False,
)
explorer = EpsilonGreedy(epsilon=params.epsilon)

# %% [markdown]
# ## Main loop

# %%
rewards = np.zeros((params.numEpisodes, params.n_runs))
steps = np.zeros((params.numEpisodes, params.n_runs))
episodes = np.arange(params.numEpisodes)
qtables = np.zeros((params.n_runs, *learner.qtable.shape))

for run in range(params.n_runs):  # Run several times to account for stochasticity
    for episode in tqdm(
        episodes, desc=f"Run {run+1}/{params.n_runs} - Episodes", leave=False
    ):
        state = env.reset()  # Reset the environment
        step_count = 0
        done = False
        total_rewards = 0

        while not done:

            learner.qtable = learner.Q_hat(learner.weights, learner.features)

            action = explorer.choose_action(
                action_space=env.action_space, state=state, qtable=learner.qtable
            )

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
    qtables[run, :, :] = learner.qtable

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
qtable_directions_map(qtable, env.rows, env.cols)

# %% [markdown]
# ## Visualization

# %%
plot_features(learner.features)

# %%
plot_steps_and_rewards(res)

# %%
plot_q_values_map(qtable, env.rows, env.cols)

# %%
plot_rotated_q_values_map(qtable, env.rows, env.cols)
