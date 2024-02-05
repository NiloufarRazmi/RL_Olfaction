# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # DQN

# %% [markdown]
# ## Setup

# %% [markdown]
# ### Initialization

import os
from collections import namedtuple

# %%
from pathlib import Path

import ipdb
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from imojify import imojify
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from tqdm import tqdm

# from torchinfo import summary

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

import plotting
from agent_tensor import EpsilonGreedy
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
# ### Parameters

# %%
p = Params(
    seed=42,
    n_runs=1,
    total_episodes=1000,
    # epsilon=0.2,
    alpha=0.001,
    gamma=0.9,
    nHiddenUnits=(5 * 5 + 2) * 2,
    replay_buffer_max_size=1000,
    epsilon_min=0.1,
    epsilon_max=1.0,
    decay_rate=0.01,
    epsilon_warmup=300,
    batch_size=128,
)
p

# %%
# # Set the seed
# p.rng = np.random.default_rng(p.seed)

# %% [markdown]
# ### Environment definition

# %%
# Load the environment
env = WrappedEnvironment(one_hot_state=True)

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
# ### Network definition


# %%
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions, n_units=16):
        super(DQN, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_observations, n_units),
            nn.ReLU(),
            nn.Linear(n_units, n_units),
            nn.ReLU(),
            nn.Linear(n_units, n_units),
            nn.ReLU(),
            nn.Linear(n_units, n_actions),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.mlp(x)


# %%
def neural_network():
    # if env.one_hot_state:
    #     net = DQN(
    #         n_observations=p.n_observations,
    #         n_actions=p.n_actions,
    #         n_units=4 * p.n_observations,
    #     ).to(device)
    # else:
    #     net = DQN(
    #         n_observations=p.n_observations,
    #         n_actions=p.n_actions,
    #         n_units=p.nHiddenUnits,
    #     ).to(device)
    # net
    net = DQN(
        n_observations=p.n_observations,
        n_actions=p.n_actions,
        n_units=p.nHiddenUnits,
    ).to(device)
    return net


# %%
net = neural_network()
net

# %%
# print("Model parameters:")
# print(list(net.parameters()))
print("\n\nParameters sizes summary:")
print([item.shape for item in net.parameters()])

# %%
# summary(net, input_size=[state.shape], verbose=0)

# %% [markdown]
# ### Optimizer

# %%
optimizer = optim.AdamW(net.parameters(), lr=p.alpha, amsgrad=True)
optimizer

# %% [markdown]
# ### Explorer

# %%
explorer = EpsilonGreedy(
    epsilon=p.epsilon_max,
    epsilon_min=p.epsilon_min,
    epsilon_max=p.epsilon_max,
    decay_rate=p.decay_rate,
    epsilon_warmup=p.epsilon_warmup,
)
episodes = torch.arange(p.total_episodes, device=device)
epsilons = torch.empty_like(episodes) * torch.nan
for eps_i, epsi in enumerate(epsilons):
    epsilons[eps_i] = explorer.epsilon
    explorer.epsilon = explorer.update_epsilon(episodes[eps_i])

# %%
fig, ax = plt.subplots()
sns.lineplot(epsilons)
ax.set(ylabel="Epsilon")
ax.set(xlabel="Episodes")
fig.tight_layout()
plt.show()

# %%
explorer = EpsilonGreedy(
    epsilon=p.epsilon_max,
    epsilon_min=p.epsilon_min,
    epsilon_max=p.epsilon_max,
    decay_rate=p.decay_rate,
    epsilon_warmup=p.epsilon_warmup,
)


# %%
def collect_weights_biases(net):
    biases = {"val": [], "grad": []}
    weights = {"val": [], "grad": []}
    for layer in net.mlp.children():
        layer_params = layer.parameters()
        for idx, subparams in enumerate(layer_params):
            if idx > 2:
                raise ValueError(
                    "There should be max 2 sets of parameters: weights and biases"
                )
            if len(subparams.shape) > 2:
                raise ValueError("The weights have more dimensions than expected")

            if len(subparams.shape) == 1:
                biases["val"].append(subparams)
                biases["grad"].append(subparams.grad)
            elif len(subparams.shape) == 2:
                weights["val"].append(subparams)
                weights["grad"].append(subparams.grad)
    return weights, biases


# %%
def params_df_stats(weights, key, current_df=None):
    if not current_df is None:
        last_idx = current_df.index[-1] + 1
        df = current_df
    else:
        last_idx = 0
        df = None

    for idx, val in enumerate(weights[key]):
        tmp_df = pd.DataFrame(
            data={
                "Std": val.detach().cpu().std().item(),
                "Avg": val.detach().cpu().mean().item(),
                "Layer": idx,
                "Index": [last_idx + idx],
            },
            index=[last_idx + idx],
        )

        if df is None:
            df = tmp_df
        else:
            df = pd.concat((df, tmp_df))
    return df


# %% [markdown]
# ## Training loop

# %%
Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)

# %%
rewards = torch.zeros((p.total_episodes, p.n_runs), device=device)
steps = torch.zeros((p.total_episodes, p.n_runs), device=device)
episodes = torch.arange(p.total_episodes, device=device)
all_states = []
all_actions = []
losses = [[] for _ in range(p.n_runs)]

for run in range(p.n_runs):  # Run several times to account for stochasticity

    # Reset weights
    net = neural_network()
    optimizer = optim.AdamW(net.parameters(), lr=p.alpha, amsgrad=True)
    weights_val_stats = None
    biases_val_stats = None
    weights_grad_stats = None
    biases_grad_stats = None
    replay_buffer = []

    for episode in tqdm(
        episodes, desc=f"Run {run+1}/{p.n_runs} - Episodes", leave=False
    ):
        state = env.reset()  # Reset the environment
        state = state.clone().float().detach().to(device)
        step_count = 0
        done = False
        total_rewards = 0

        while not done:
            state_action_values = net(state).to(device)  # Q(s_t)
            action = explorer.choose_action(
                action_space=env.action_space,
                state=state,
                state_action_values=state_action_values,
            ).item()

            # Record states and actions
            all_states.append(state)
            # all_actions.append(Actions(action.item()).name)
            all_actions.append(Actions(action).name)

            next_state, reward, done = env.step(action=action, current_state=state)

            # Store transition in replay buffer
            if len(replay_buffer) >= p.replay_buffer_max_size:
                replay_buffer = replay_buffer[1:]
            # [current_state (2 or 28 x1), action (1x1), next_state (2 or 28 x1), reward (1x1), done (1x1 bool)]
            done = torch.tensor(done, device=device).unsqueeze(-1)
            replay_buffer.append(
                Transition(
                    state,
                    action,
                    reward,
                    next_state,
                    done,
                )
            )
            if len(replay_buffer) >= p.batch_size:
                transitions = random_choice(
                    replay_buffer,
                    length=len(replay_buffer),
                    num_samples=p.batch_size,
                )
                if p.batch_size > 1:
                    batch = Transition(*zip(*transitions, strict=True))
                else:
                    batch = Transition(*transitions)
                if p.batch_size > 1:
                    state_batch = torch.stack(batch.state)
                    # action_batch = torch.cat(batch.action)
                    action_batch = torch.tensor(batch.action, device=device)
                    reward_batch = torch.cat(batch.reward)
                    next_state_batch = torch.stack(batch.next_state)
                    done_batch = torch.cat(batch.done)
                else:
                    state_batch = batch.state
                    action_batch = batch.action
                    reward_batch = batch.reward
                    next_state_batch = batch.next_state
                    done_batch = batch.done
                # (
                #     state_sampled,
                #     action_sampled,
                #     reward_batch,
                #     next_state_batch,
                #     done_sampled,
                # ) = random_choice(replay_buffer, length=len(replay_buffer), num_samples=p.batch_size)

                # See DQN paper for equations: https://doi.org/10.1038/nature14236
                state_action_values_sampled = net(state_batch).to(device)  # Q(s_t)
                if p.batch_size > 1:
                    state_action_value = torch.gather(
                        input=state_action_values_sampled,
                        dim=1,
                        index=action_batch.unsqueeze(-1),
                    ).squeeze()  # Q(s_t, a)
                else:
                    state_action_value = state_action_values_sampled[
                        action_batch
                    ].unsqueeze(
                        -1
                    )  # Q(s_t, a)

                # if done_batch:
                #     expected_state_action_value = reward_batch
                # else:
                #     with torch.no_grad():
                #         next_state_values = (
                #             net(next_state_batch).to(device).max().unsqueeze(-1)
                #         )  # Q(s_t+1, a)
                #     expected_state_action_value = (
                #         reward_batch + p.gamma * next_state_values
                #     )  # y_j (Bellman optimality equation)

                done_false = torch.argwhere(done_batch == False).squeeze()
                done_true = torch.argwhere(done_batch == True).squeeze()
                expected_state_action_value = torch.empty_like(done_batch) * torch.nan
                with torch.no_grad():
                    if done_true.numel() > 0:
                        expected_state_action_value[done_true] = reward_batch[done_true]
                    if done_false.numel() > 0:
                        if len(next_state_batch.shape) > 1:
                            next_state_values = (
                                net(next_state_batch[done_false]).to(device).max(1)
                            )  # Q(s_t+1, a)
                            expected_state_action_value[done_false] = (
                                reward_batch[done_false]
                                + p.gamma * next_state_values.values
                            )  # y_j (Bellman optimality equation)
                        else:
                            next_state_values = (
                                net(next_state_batch).to(device).max()
                            )  # Q(s_t+1, a)
                            expected_state_action_value[done_false] = (
                                reward_batch[done_false] + p.gamma * next_state_values
                            )  # y_j (Bellman optimality equation)

                # Compute loss
                criterion = nn.MSELoss()
                loss = criterion(
                    input=state_action_value, target=expected_state_action_value
                )  # TD update

                # Optimize the model
                optimizer.zero_grad()
                loss.backward()

                # In-place gradient clipping
                torch.nn.utils.clip_grad_value_(net.parameters(), 100)
                optimizer.step()

                losses[run].append(loss.item())

                weights, biases = collect_weights_biases(net=net)
                weights_val_stats = params_df_stats(
                    weights, key="val", current_df=weights_grad_stats
                )
                biases_val_stats = params_df_stats(
                    biases, key="val", current_df=biases_val_stats
                )
                biases_grad_stats = params_df_stats(
                    biases, key="grad", current_df=biases_grad_stats
                )
                weights_grad_stats = params_df_stats(
                    weights, key="grad", current_df=weights_val_stats
                )

            total_rewards += reward
            step_count += 1

            # Move to the next state
            state = next_state

        explorer.epsilon = explorer.update_epsilon(episode)

        rewards[episode, run] = total_rewards
        steps[episode, run] = step_count
    weights_val_stats.set_index("Index", inplace=True)
    biases_val_stats.set_index("Index", inplace=True)
    biases_grad_stats.set_index("Index", inplace=True)
    weights_grad_stats.set_index("Index", inplace=True)


# %% [markdown]
# ## Visualization

# %% [markdown]
# ### States & actions distributions


# %%
def postprocess(episodes, p, rewards, steps):
    """Convert the results of the simulation in dataframes."""
    res = pd.DataFrame(
        data={
            "Episodes": episodes.tile(p.n_runs).cpu(),
            "Rewards": rewards.T.flatten().cpu(),
            "Steps": steps.T.flatten().cpu(),
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


# %% [markdown]
# ### Steps & rewards


# %%
def plot_steps_and_rewards(df):
    """Plot the steps and rewards from dataframes."""
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    sns.lineplot(data=df, x="Episodes", y="Rewards", ax=ax[0])
    ax[0].set(
        ylabel=f"Rewards\naveraged over {p.n_runs} runs" if p.n_runs > 1 else "Rewards"
    )

    sns.lineplot(data=df, x="Episodes", y="Steps", ax=ax[1])
    ax[1].set(
        ylabel=(
            f"Steps number\naveraged over {p.n_runs} runs"
            if p.n_runs > 1
            else "Steps number"
        )
    )

    fig.tight_layout()
    plt.show()


# %%
plot_steps_and_rewards(res)


# %%
def plot_steps_and_rewards_dist(df):
    """Plot the steps and rewards distributions from dataframes."""
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    sns.histplot(data=df, x="Rewards", ax=ax[0])
    sns.histplot(data=df, x="Steps", ax=ax[1])
    fig.tight_layout()
    plt.show()


# %%
plot_steps_and_rewards_dist(res)

# %% [markdown]
# ### Loss

# %%
window_size = 10
for idx, loss in enumerate(losses):
    current_loss = torch.tensor(loss, device=device)
    losses_rolling_avg = nn.functional.avg_pool1d(
        current_loss.view(1, 1, -1), kernel_size=window_size
    ).squeeze()
    tmp_df = pd.DataFrame(
        data={
            "Run": idx * torch.ones(len(losses_rolling_avg), device=device).int().cpu(),
            "Steps": torch.arange(0, len(losses_rolling_avg), device=device).cpu(),
            "Loss": losses_rolling_avg.cpu(),
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
    ylabel=(
        f"$Log_{{10}}(\\text{{Loss}})$\naveraged over {p.n_runs} runs"
        if p.n_runs > 1
        else "$Log_{10}(\\text{Loss})$"
    )
)
ax.set(xlabel="Steps")
# ax.set(yscale="log")
fig.tight_layout()
plt.show()

# %%
loss_df.iloc[-1].Loss

# %% [markdown]
# ### Policy learned

# %%
with torch.no_grad():
    q_values = torch.nan * torch.empty(
        (len(env.tiles_locations), len(Cues), p.n_actions), device=device
    )
    for tile_i, tile_v in enumerate(env.tiles_locations):
        for cue_i, cue_v in enumerate(Cues):
            state = torch.tensor([tile_v, cue_v.value], device=device).float()
            if env.one_hot_state:
                state = env.to_one_hot(state).float()
            q_values[tile_i, cue_i, :] = net(state).to(device)
q_values.shape


# %%
def qtable_directions_map(qtable, rows, cols):
    """Get the best learned action & map it to arrows."""
    qtable_val_max = qtable.max(axis=1).values.reshape(rows, cols)
    qtable_best_action = qtable.argmax(axis=1).reshape(rows, cols)
    directions = {
        Actions.UP: "↑",
        Actions.DOWN: "↓",
        Actions.LEFT: "←",
        Actions.RIGHT: "→",
    }
    qtable_directions = np.empty(qtable_best_action.flatten().shape, dtype=str)
    eps = torch.finfo(torch.float64).eps  # Minimum float number on the machine
    for idx, val in enumerate(qtable_best_action.flatten()):
        if qtable_val_max.flatten()[idx] > eps:
            # Assign an arrow only if a minimal Q-value has been learned as best action
            # otherwise since 0 is a direction, it also gets mapped on the tiles where
            # it didn't actually learn anything
            qtable_directions[idx] = directions[Actions(val.item())]
    qtable_directions = qtable_directions.reshape(rows, cols)
    return qtable_val_max, qtable_directions


# %%
def plot_policies(q_values, labels):
    """Plot the heatmap of the Q-values.

    Also plot the best action's direction with arrows."""

    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for idx, cue in enumerate(labels):
        qtable_val_max, qtable_directions = qtable_directions_map(
            qtable=q_values[:, idx, :], rows=env.rows, cols=env.cols
        )
        sns.heatmap(
            qtable_val_max.cpu(),
            annot=qtable_directions,
            fmt="",
            ax=ax.flatten()[idx],
            cmap=sns.color_palette("Blues", as_cmap=True),
            linewidths=0.7,
            linecolor="black",
            xticklabels=[],
            yticklabels=[],
            annot_kws={"fontsize": "xx-large"},
            cbar_kws={"label": "Q-value"},
        ).set(title=labels[cue])
        for _, spine in ax.flatten()[idx].spines.items():
            spine.set_visible(True)
            spine.set_linewidth(0.7)
            spine.set_color("black")

        # Annotate the ports names
        bbox = {
            "facecolor": "black",
            "edgecolor": "none",
            "boxstyle": "round",
            "alpha": 0.1,
        }
        ax.flatten()[idx].text(
            x=4.7,
            y=0.3,
            s="N",
            bbox=bbox,
            color="white",
        )
        ax.flatten()[idx].text(
            x=0.05,
            y=4.9,
            s="S",
            bbox=bbox,
            color="white",
        )
        ax.flatten()[idx].text(
            x=4.7,
            y=4.9,
            s="E",
            bbox=bbox,
            color="white",
        )
        ax.flatten()[idx].text(
            x=0.05,
            y=0.3,
            s="W",
            bbox=bbox,
            color="white",
        )

    # Make background transparent
    fig.patch.set_alpha(0)
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    plt.show()


# %%
plot_policies(q_values=q_values, labels=CONTEXTS_LABELS)

# %% [markdown]
# ### Weights & gradients metrics

# %%
weights, biases = collect_weights_biases(net=net)


# %%
def params_df_flat(weights):
    for idx, val in enumerate(weights):
        tmp_df = pd.DataFrame(
            data={
                "Val": val.detach().cpu().flatten(),
                "Layer": idx,
            }
        )
        if idx == 0:
            df = tmp_df
        else:
            df = pd.concat((df, tmp_df))
    return df


# %%
weights_val_df = params_df_flat(weights["val"])
weights_val_df

# %%
weights_val_df.describe()

# %%
biases_val_df = params_df_flat(biases["val"])
biases_val_df

# %%
biases_val_df.describe()

# %%
weights_grad_df = params_df_flat(weights["grad"])
weights_grad_df

# %%
weights_grad_df.describe()

# %%
biases_grad_df = params_df_flat(biases["grad"])
biases_grad_df

# %%
biases_grad_df.describe()


# %%
def check_grad_stats(grad_df):
    grad_stats = torch.tensor(
        [
            grad_df.Val.mean(),
            grad_df.Val.std(),
            grad_df.Val.min(),
            grad_df.Val.max(),
        ],
        device=device,
    )
    assert not torch.equal(
        torch.zeros_like(grad_stats),
        grad_stats,
    ), "Gradients are zero"


# %%
plotting.plot_weights_biases_distributions(
    weights_val_df, biases_val_df, label="Values"
)

# %%
check_grad_stats(weights_grad_df)

# %%
check_grad_stats(biases_grad_df)

# %%
plotting.plot_weights_biases_distributions(
    weights_grad_df, biases_grad_df, label="Gradients"
)

# %%
weights_val_stats

# %%
biases_val_stats

# %%
weights_grad_stats

# %%
biases_grad_stats


# %%
def plot_weights_biases_stats(weights_stats, biases_stats, label=None):
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(13, 8))

    if label:
        ax[0, 0].set_title("Weights " + label)
    else:
        ax[0, 0].set_title("Weights")
    ax[0, 0].set_xlabel("Episodes")
    palette = sns.color_palette()[0 : len(weights_stats.Layer.unique())]
    sns.lineplot(
        data=weights_stats,
        x="Index",
        y="Std",
        hue="Layer",
        palette=palette,
        ax=ax[0, 0],
    )
    ax[0, 0].set(yscale="log")

    if label:
        ax[0, 1].set_title("Weights " + label)
    else:
        ax[0, 1].set_title("Weights")
    ax[0, 1].set_xlabel("Episodes")
    palette = sns.color_palette()[0 : len(weights_stats.Layer.unique())]
    sns.lineplot(
        data=weights_stats,
        x="Index",
        y="Avg",
        hue="Layer",
        palette=palette,
        ax=ax[0, 1],
    )
    ax[0, 1].set(yscale="log")

    if label:
        ax[1, 0].set_title("Biases " + label)
    else:
        ax[1, 0].set_title("Biases")
    ax[1, 0].set_xlabel("Episodes")
    palette = sns.color_palette()[0 : len(biases_stats.Layer.unique())]
    sns.lineplot(
        data=biases_stats,
        x="Index",
        y="Std",
        hue="Layer",
        palette=palette,
        ax=ax[1, 0],
    )
    ax[1, 0].set(yscale="log")

    if label:
        ax[1, 1].set_title("Biases " + label)
    else:
        ax[1, 1].set_title("Biases")
    ax[1, 1].set_xlabel("Episodes")
    palette = sns.color_palette()[0 : len(biases_stats.Layer.unique())]
    sns.lineplot(
        data=biases_stats,
        x="Index",
        y="Avg",
        hue="Layer",
        palette=palette,
        ax=ax[1, 1],
    )
    ax[1, 1].set(yscale="log")

    fig.tight_layout()
    plt.show()


# %%
plot_weights_biases_stats(weights_val_stats, biases_val_stats, label="values")

# %%
plot_weights_biases_stats(weights_grad_stats, biases_grad_stats, label="gradients")
