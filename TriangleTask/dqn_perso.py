# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # DQN

# %% [markdown]
# ## Dependencies

import os

# %%
from pathlib import Path

import ipdb
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from imojify import imojify
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from torchinfo import summary
from tqdm import tqdm

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

from environment_tensor import CONTEXTS_LABELS, Actions, Cues, WrappedEnvironment

# %%
from utils import Params, random_choice

# %%
# Formatting & autoreload stuff
# %load_ext lab_black
# %load_ext autoreload
# %autoreload 2
# # %matplotlib ipympl

# %%
sns.set_theme(font_scale=1.5)
mpl.rcParams["font.family"] = ["sans-serif"]
mpl.rcParams["font.sans-serif"] = [
    "Fira Sans",
    "Computer Modern Sans Serif",
    "DejaVu Sans",
    "Verdana",
    "Arial",
    "Helvetica",
]
# plt.style.use("ggplot")

# %%
ROOT_PATH = Path("env").parent
PLOTS_PATH = ROOT_PATH / "plots"
print(f"Plots path: `{PLOTS_PATH.absolute()}`")


# %%
def check_plots():
    if not PLOTS_PATH.exists():
        os.mkdir(PLOTS_PATH)


# %% [markdown]
# ## Parameters

# %%
p = Params(
    seed=42,
    n_runs=1,
    total_episodes=200,
    epsilon=0.2,
    alpha=0.1,
    gamma=0.9,
    nHiddenUnits=5 * 5 + 2,
)
p

# %%
# # Set the seed
# p.rng = np.random.default_rng(p.seed)

# %% [markdown]
# ## The environment

# %%
# Load the environment
env = WrappedEnvironment(p)

# %%
# Get number of actions
# n_actions = env.action_space.n
p.n_actions = env.numActions

# Get the number of state observations
# state, info = env.reset()
state = env.reset()
p.n_observations = len(state)

print(f"Number of actions: {p.n_actions}")
print(f"Number of observations: {p.n_observations}")


# %% [markdown]
# ## Running the environment


# %%
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions, n_units=16):
        super(DQN, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_observations, n_units),
            nn.Linear(n_units, n_units),
            nn.Linear(n_units, n_units),
            # nn.ReLU(),
            nn.Linear(n_units, n_actions),
        )

    def forward(self, x):
        return self.mlp(x)


# %%
net = DQN(
    n_observations=p.n_observations, n_actions=p.n_actions, n_units=p.nHiddenUnits
).to(device)
net

# %%
# print("Model parameters:")
# print(list(net.parameters()))
print("\n\nParameters sizes summary:")
print([item.shape for item in net.parameters()])

# %%
summary(net, input_size=[state.shape], verbose=0)

# %%
optimizer = optim.AdamW(net.parameters(), lr=p.alpha, amsgrad=True)
optimizer


# %%
class EpsilonGreedy:
    def __init__(
        self,
        epsilon,
        rng=None,
    ):
        self.epsilon = epsilon
        # if rng:
        #     self.rng = rng

    def choose_action(self, action_space, state, state_action_values):
        """Choose an action a in the current world state (s)"""

        def sample(action_space):
            return random_choice(action_space)

        # # First we randomize a number
        # if hasattr(self, "rng"):
        #     explor_exploit_tradeoff = self.rng.uniform(0, 1)
        # else:
        #     explor_exploit_tradeoff = np.random.uniform(0, 1)
        explor_exploit_tradeoff = torch.rand(1)

        # Exploration
        if explor_exploit_tradeoff.item() < self.epsilon:
            # action = action_space.sample()
            action = sample(action_space)

        # Exploitation (taking the biggest Q-value for this state)
        else:
            # Break ties randomly
            # If all actions are the same for this state we choose a random one
            # (otherwise `argmax()` would always take the first one)
            if torch.all(state_action_values == state_action_values[0]):
                action = sample(action_space)
            else:
                action = torch.argmax(state_action_values)
        return action


# %%
explorer = EpsilonGreedy(epsilon=p.epsilon, rng=p.rng)

# %% [markdown]
# ### Main loop

# %%
rewards = torch.zeros((p.total_episodes, p.n_runs), device=device)
steps = torch.zeros((p.total_episodes, p.n_runs), device=device)
episodes = torch.arange(p.total_episodes, device=device)
all_states = []
all_actions = []
losses = [[] for _ in range(p.n_runs)]
episode_durations = []

for run in range(p.n_runs):  # Run several times to account for stochasticity
    # Reset model
    net = DQN(
        n_observations=p.n_observations, n_actions=p.n_actions, n_units=p.nHiddenUnits
    ).to(device)
    optimizer = optim.AdamW(net.parameters(), lr=p.alpha, amsgrad=True)

    for episode in tqdm(
        episodes, desc=f"Run {run+1}/{p.n_runs} - Episodes", leave=False
    ):
        state = env.reset()  # Reset the environment
        state = state.clone().float().detach().to(device)
        step_count = 0
        done = False
        total_rewards = 0

        while not done:
            state_action_values = net(state).to(device)
            action = explorer.choose_action(
                action_space=env.action_space,
                state=state,
                state_action_values=state_action_values,
            )

            # # Record states and actions
            all_states.append(state)
            all_actions.append(Actions(action.item()).name)

            observation, reward, done = env.step(
                action=action.item(), current_state=state
            )
            # if reward != 0:
            #     ipdb.set_trace()
            reward = torch.tensor([reward], device=device)

            # if terminated:
            if done:
                next_state = None
            else:
                next_state = observation.clone().float().detach()

            # Move to the next state
            state = next_state

            # # Compute V(s_{t+1}) for all next states.
            # # Expected values of actions are computed based
            # # on the "older" network, then selecting their best reward.
            # # Should be either the expected state value or 0 in case the state is final.
            # next_state_values = torch.zeros_like(state_action_values, device=device)
            # with torch.no_grad():
            #     next_state_values = net(state).max(1)[0]

            # Update only the weights related to action taken
            next_state_values = state_action_values
            if done:
                # next_state_values[action.item()] = 0
                next_state_values = torch.zeros(1, device=device)
            else:
                # next_state_values[action.item()] = net(state).max()
                next_state_values = net(state).max()

            # Compute the expected Q values
            expected_state_action_values = reward + p.gamma * next_state_values

            # Compute loss
            criterion = nn.MSELoss()
            # loss = criterion(state_action_values, expected_state_action_values)
            # if state_action_values[action.item()].shape != expected_state_action_values.shape:
            loss = criterion(
                state_action_values[action].unsqueeze(-1),
                expected_state_action_values,
            )

            # Optimize the model
            optimizer.zero_grad()
            loss.backward()

            # In-place gradient clipping
            torch.nn.utils.clip_grad_value_(net.parameters(), 100)
            optimizer.step()

            if done:
                episode_durations.append(step_count + 1)
                # plot_durations()

            total_rewards += reward
            step_count += 1
            losses[run].append(loss.item())

        rewards[episode, run] = total_rewards
        steps[episode, run] = step_count


# %%
# weights_metrics

# %%
# grads_metrics

# %%
# grads_metrics["avg_rolling"] = np.nan
# grads_metrics["std_rolling"] = np.nan
# for id in grads_metrics.id.unique():
#     grads_metrics.loc[grads_metrics[grads_metrics["id"] == id].index, "avg_rolling"] = (
#         grads_metrics.loc[grads_metrics[grads_metrics["id"] == id].index, "avg"]
#         .rolling(20)
#         .mean()
#     )
#     grads_metrics.loc[grads_metrics[grads_metrics["id"] == id].index, "std_rolling"] = (
#         grads_metrics.loc[grads_metrics[grads_metrics["id"] == id].index, "std"]
#         .rolling(20)
#         .mean()
#     )
# grads_metrics

# %% [markdown]
# ## Visualization

# %%
# def plot_weights(weights_metrics):
#     """Plot the weights."""
#     fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

#     sns.lineplot(x="steps_global", y="avg", hue="id", data=weights_metrics, ax=ax[0])
#     ax[0].set(ylabel="Weights (avg)")
#     ax[0].set(xlabel="Steps")

#     sns.lineplot(x="steps_global", y="std", hue="id", data=weights_metrics, ax=ax[1])
#     ax[1].set(ylabel="Weights (std)")
#     ax[1].set(xlabel="Steps")

#     fig.tight_layout()
#     plt.show()

# %%
# plot_weights(weights_metrics)

# %%
# def plot_gradients(grads_metrics):
#     """Plot the gradienta."""
#     fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

#     sns.lineplot(
#         x="steps_global",
#         y="avg_rolling",
#         hue="id",
#         data=grads_metrics,
#         ax=ax[0],
#         palette=sns.color_palette()[0 : len(grads_metrics.id.unique())],
#     )
#     ax[0].set(ylabel="Gradients (avg)")
#     ax[0].set(xlabel="Steps")

#     sns.lineplot(
#         x="steps_global",
#         y="std_rolling",
#         hue="id",
#         data=grads_metrics,
#         ax=ax[1],
#         palette=sns.color_palette()[0 : len(grads_metrics.id.unique())],
#     )
#     ax[1].set(ylabel="Gradients (std)")
#     ax[1].set(xlabel="Steps")

#     fig.tight_layout()
#     plt.show()

# %%
# plot_gradients(grads_metrics)


# %%
def postprocess(episodes, p, rewards, steps):
    """Convert the results of the simulation in dataframes."""
    res = pd.DataFrame(
        data={
            "Episodes": episodes.tile(p.n_runs),
            "Rewards": rewards.T.flatten(),
            "Steps": steps.T.flatten(),
        }
    )
    # res["cum_rewards"] = rewards.cumsum(axis=0).flatten(order="F")
    return res


# %%
res = postprocess(episodes, p, rewards, steps)
res


# %% [markdown]
# As a sanity check, we will plot the distributions of states and actions
# with the following function:


# %%
def plot_states_actions_distribution(states, actions):
    """Plot the distributions of states and actions."""
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(13, 5))
    # sns.histplot(data=states, ax=ax[0])
    ax[0].set_title("States")
    sns.histplot(data=actions, ax=ax[1])
    ax[1].set_xticks(
        [item.value for item in Actions], labels=[item.name for item in Actions]
    )
    ax[1].set_title("Actions")
    fig.tight_layout()
    plt.show()


# %%
plot_states_actions_distribution(all_states, all_actions)


# %%
def plot_steps_and_rewards(df):
    """Plot the steps and rewards from dataframes."""
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    sns.lineplot(data=df, x="Episodes", y="Rewards", ax=ax[0])
    ax[0].set(
        ylabel=f"Rewards\naveraged over {p.n_runs} runs"
        if p.n_runs > 1
        else "Steps number"
    )

    sns.lineplot(data=df, x="Episodes", y="Steps", ax=ax[1])
    ax[1].set(
        ylabel=f"Steps number\naveraged over {p.n_runs} runs"
        if p.n_runs > 1
        else "Steps number"
    )

    fig.tight_layout()
    plt.show()


# %%
plot_steps_and_rewards(res)

# %%
window_size = 10
for idx, loss in enumerate(losses):
    current_loss = torch.tensor(loss, device=device)
    losses_rolling_avg = nn.functional.avg_pool1d(
        current_loss.view(1, 1, -1), kernel_size=window_size
    ).squeeze()
    tmp_df = pd.DataFrame(
        data={
            "Run": idx * torch.ones(len(losses_rolling_avg), device=device).int(),
            "Steps": torch.arange(0, len(losses_rolling_avg), device=device),
            "Loss": losses_rolling_avg,
        }
    )
    if idx == 0:
        loss_df = tmp_df
    else:
        loss_df = pd.concat((loss_df, tmp_df))
loss_df

# %%
fig, ax = plt.subplots()
sns.lineplot(data=loss_df, x="Steps", y="Loss", ax=ax)
ax.set(
    ylabel=f"$Log_{{10}}(\\text{{Loss}})$\naveraged over {p.n_runs} runs"
    if p.n_runs > 1
    else "$Log_{10}(\\text{Loss})$"
)
ax.set(xlabel="Steps")
ax.set(yscale="log")
fig.tight_layout()
plt.show()

# %%
