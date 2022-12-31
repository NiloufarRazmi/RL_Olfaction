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

# %%
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# %load_ext lab_black
# %run utils.py

# %%
params = Params()
params

# %%
env = Environment(params)
env.transitionList

# %%
learner = Qlearning(
    learning_rate=params.alpha,
    gamma=params.gamma,
    state_size=params.numStates,
    action_size=params.numActions,
)
explorer = EpsilonGreedy(epsilon=params.epsilon)

# %%
rewards = np.zeros((params.numEpisodes, params.n_runs))
steps = np.zeros((params.numEpisodes, params.n_runs))
episodes = np.arange(params.numEpisodes)

for run in range(params.n_runs):  # Run several times to account for stochasticity
    for episode in tqdm(
        episodes, desc=f"Run {run}/{params.n_runs} - Episodes", leave=False
    ):
        state = env.reset()  # Reset the environment
        step = 0
        done = False
        total_rewards = 0

        while not done:

            action = explorer.choose_action(
                # action_space=env.action_space, state=state, qtable=learner.qtable
                action_space=params.actions,
                state=state,
                qtable=learner.qtable,
            )

            # Take the action (a) and observe the outcome state(s') and reward (r)
            new_state, reward = env.step(action, state)
            done = env.is_terminated(state, action)

            learner.qtable[state, action] = learner.update(
                state, action, reward, new_state
            )

            total_rewards += reward

            # Our new state is state
            state = new_state

        # explorer.epsilon = explorer.update_epsilon(episode)

        rewards[episode, run] = total_rewards
        steps[episode, run] = step
