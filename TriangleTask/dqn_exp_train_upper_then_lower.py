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

# %%
from pathlib import Path
import os
import datetime
import logging
import shutil

import ipdb

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import pandas as pd
import seaborn as sns
from imojify import imojify
from collections import namedtuple, deque

# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# from torchinfo import summary

# if GPU is to be used
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE

# %%
from utils import Params, random_choice, make_deterministic

from env_tensor_exp_train_upper_only_then_lower import (
    WrappedEnvironment,
    Actions,
    CONTEXTS_LABELS,
    Cues,
    TriangleState
)
from agent_tensor import EpsilonGreedy
import plotting

# %%
# Formatting & autoreload stuff
# # %load_ext lab_black
# %load_ext autoreload
# %autoreload 2
# # %matplotlib ipympl

# %%
sns.set_theme(font_scale=1.5)
# plt.style.use("ggplot")
print(shutil.which("latex"))
USETEX = True if shutil.which("latex") else False
mpl.rcParams["text.usetex"] = USETEX
if USETEX:
    mpl.rcParams["font.family"] = ["serif"]
else:
    mpl.rcParams["font.family"] = ["sans-serif"]
    mpl.rcParams["font.sans-serif"] = [
        "Fira Sans",
        "Computer Modern Sans Serif",
        "DejaVu Sans",
        "Verdana",
        "Arial",
        "Helvetica",
    ]

# %% [markdown]
# ### Save directory

# %%
now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
now

# %%
ROOT_PATH = Path("env").parent
SAVE_PATH = ROOT_PATH / "save"
CURRENT_PATH = SAVE_PATH / now
CURRENT_PATH.mkdir(parents=True, exist_ok=True)  # Create the tree of directories
print(f"Save path: `{CURRENT_PATH.absolute()}`")

# %%
# Configure logging
logfile = CURRENT_PATH / "training.log"
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler(logfile)
formatter = logging.Formatter(
    "%(asctime)s : %(name)s  : %(funcName)s : %(levelname)s : %(message)s"
)
handler.setFormatter(formatter)
logger.addHandler(handler)

# %% [markdown]
# ### Parameters

# %%
p = Params(
    seed=42,
    # seed=123,
    n_runs=20,
    total_episodes=600,
    epsilon=0.5,
    alpha=1e-4,
    gamma=0.99,
    # nHiddenUnits=(5 * 5 + 3) * 5,
    nHiddenUnits=128,
    replay_buffer_max_size=5000,
    epsilon_min=0.2,
    epsilon_max=1.0,
    decay_rate=0.01,
    epsilon_warmup=100,
    batch_size=32,
    # target_net_update=200,
    tau=0.005,
)
p

# %%
if p.batch_size < 2:
    raise ValueError("The batch size needs to be more that one data point")

# %%
# Set the seed
GENERATOR = make_deterministic(seed=p.seed)

# %% [markdown]
# ### Environment definition

# %%
# Load the environment
env = WrappedEnvironment(one_hot_state=True, seed=p.seed)

# %%
# Get number of actions
# n_actions = env.action_space.n
p.n_actions = env.numActions

# Get the number of state observations
# state, info = env.reset()
state = env.reset(triangle_state=TriangleState.lower)
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
            # nn.ReLU(),
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
    #     ).to(DEVICE)
    # else:
    #     net = DQN(
    #         n_observations=p.n_observations,
    #         n_actions=p.n_actions,
    #         n_units=p.nHiddenUnits,
    #     ).to(DEVICE)
    # net

    net = DQN(
        n_observations=p.n_observations,
        n_actions=p.n_actions,
        n_units=p.nHiddenUnits,
    ).to(DEVICE)

    target_net = DQN(
        n_observations=p.n_observations,
        n_actions=p.n_actions,
        n_units=p.nHiddenUnits,
    ).to(DEVICE)

    target_net.load_state_dict(net.state_dict())

    return net, target_net


# %%
net, target_net = neural_network()
net, target_net

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
    seed=p.seed,
)
episodes = torch.arange(p.total_episodes, device=DEVICE)
epsilons = torch.empty_like(episodes, device=DEVICE) * torch.nan
for eps_i, epsi in enumerate(epsilons):
    epsilons[eps_i] = explorer.epsilon
    explorer.epsilon = explorer.update_epsilon(episodes[eps_i])

# %%
fig, ax = plt.subplots()
sns.lineplot(epsilons.cpu())
ax.set(ylabel="Epsilon")
ax.set(xlabel="Episodes")
fig.tight_layout()
plt.show()


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
# ## Training

# %%
Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)


# %%
def train(
    triangle_state,
):

    state = env.reset(triangle_state=triangle_state)  # Reset the environment
    state = state.clone().float().detach().to(DEVICE)
    step_count = 0
    done = False
    total_rewards = 0
    loss = torch.ones(1, device=DEVICE) * torch.nan

    while not done:
        state_action_values = net(state).to(DEVICE)  # Q(s_t)
        action = explorer.choose_action(
            action_space=env.action_space,
            state=state,
            state_action_values=state_action_values,
        ).item()

        # Record states and actions
        # all_states.append(state)
        # all_actions.append(Actions(action.item()).name)
        all_actions.append(Actions(action).name)

        next_state, reward, done = env.step(action=action, current_state=state)

        # Store transition in replay buffer
        # [current_state (2 or 28 x1), action (1x1), next_state (2 or 28 x1), reward (1x1), done (1x1 bool)]
        done = torch.tensor(done, device=DEVICE).unsqueeze(-1)
        replay_buffer.append(
            Transition(
                state,
                action,
                reward,
                next_state,
                done,
            )
        )
        if len(replay_buffer) == p.replay_buffer_max_size:
            transitions = random_choice(
                replay_buffer,
                length=len(replay_buffer),
                num_samples=p.batch_size,
                generator=GENERATOR,
            )
            batch = Transition(*zip(*transitions, strict=True))
            state_batch = torch.stack(batch.state)
            action_batch = torch.tensor(batch.action, device=DEVICE)
            reward_batch = torch.cat(batch.reward)
            next_state_batch = torch.stack(batch.next_state)
            done_batch = torch.cat(batch.done)

            # See DQN paper for equations: https://doi.org/10.1038/nature14236
            state_action_values_sampled = net(state_batch).to(DEVICE)  # Q(s_t)
            state_action_values = torch.gather(
                input=state_action_values_sampled,
                dim=1,
                index=action_batch.unsqueeze(-1),
            ).squeeze()  # Q(s_t, a)

            # Compute a mask of non-final states and concatenate the batch elements
            # (a final state would've been the one after which simulation ended)
            non_final_mask = torch.tensor(
                tuple(map(lambda s: s == False, batch.done)),
                device=DEVICE,
                dtype=torch.bool,
            )
            non_final_next_states = torch.stack(
                [s[1] for s in zip(batch.done, batch.next_state) if s[0] == False]
            )

            # Compute V(s_{t+1}) for all next states.
            # Expected values of actions for non_final_next_states are computed based
            # on the "older" target_net; selecting their best reward with max(1).values
            # This is merged based on the mask, such that we'll have either the expected
            # state value or 0 in case the state was final.
            next_state_values = torch.zeros(p.batch_size, device=DEVICE)
            if non_final_next_states.numel() > 0 and non_final_mask.numel() > 0:
                with torch.no_grad():
                    next_state_values[non_final_mask] = (
                        target_net(non_final_next_states).max(1).values
                    )
            # Compute the expected Q values
            expected_state_action_values = reward_batch + (next_state_values * p.gamma)

            # Compute loss
            # criterion = nn.MSELoss()
            criterion = nn.SmoothL1Loss()
            loss = criterion(
                input=state_action_values,  # prediction
                target=expected_state_action_values,  # target/"truth" value
            )  # TD update

            # Optimize the model
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(
                net.parameters(), 100
            )  # In-place gradient clipping
            optimizer.step()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            net_state_dict = net.state_dict()
            for key in net_state_dict:
                target_net_state_dict[key] = net_state_dict[
                    key
                ] * p.tau + target_net_state_dict[key] * (1 - p.tau)
            target_net.load_state_dict(target_net_state_dict)

            losses[run].append(loss.item())

        total_rewards += reward
        step_count += 1

        # Move to the next state
        state = next_state

        if episode > p.total_episodes:
            explorer.epsilon = explorer.update_epsilon(episode - p.total_episodes)
        else:
            explorer.epsilon = explorer.update_epsilon(episode)
        epsilons.append(explorer.epsilon)

    return (
        total_rewards,
        step_count,
        loss,
    )


# %% [markdown]
# ### Main loop

# %%
rewards = torch.zeros((2 * p.total_episodes, p.n_runs), device=DEVICE)
steps = torch.zeros((2 * p.total_episodes, p.n_runs), device=DEVICE)
episodes = torch.arange(2 * p.total_episodes, device=DEVICE)
# all_states = []
all_actions = []
losses = [[] for _ in range(p.n_runs)]

for run in range(p.n_runs):  # Run several times to account for stochasticity

    # Reset everything
    net, target_net = neural_network()  # reset weights
    optimizer = optim.AdamW(net.parameters(), lr=p.alpha, amsgrad=True)
    explorer = EpsilonGreedy(
        epsilon=p.epsilon_max,
        epsilon_min=p.epsilon_min,
        epsilon_max=p.epsilon_max,
        decay_rate=p.decay_rate,
        epsilon_warmup=p.epsilon_warmup,
    )
    replay_buffer = deque([], maxlen=p.replay_buffer_max_size)
    epsilons = []

    # Train in lower triangle only
    for episode in tqdm(
        episodes[0 : p.total_episodes],
        desc=f"Run {run+1}/{p.n_runs} - Episodes",
        leave=False,
    ):

        (
            total_rewards,
            step_count,
            loss,
        ) = train(
            triangle_state=TriangleState.upper,
        )

        rewards[episode, run] = total_rewards
        steps[episode, run] = step_count
        logger.info(
            f"Run: {run+1}/{p.n_runs} - Episode: {episode+1}/{p.total_episodes} - Steps: {step_count} - Loss: {loss.item()}"
        )


    # Then switch to lower triangle
    for episode in tqdm(
        episodes[p.total_episodes :],
        desc=f"Run {run+1}/{p.n_runs} - Episodes",
        leave=False,
    ):

        (
            total_rewards,
            step_count,
            loss,
        ) = train(
            triangle_state=TriangleState.lower,
        )

        rewards[episode, run] = total_rewards
        steps[episode, run] = step_count
        logger.info(
            f"Run: {run+1}/{p.n_runs} - Episode: {episode+1}/{p.total_episodes} - Steps: {step_count} - Loss: {loss.item()}"
        )


# %% [markdown]
# ### Save data to disk

# %%
# data_path = CURRENT_PATH / "data.npz"
# with open(data_path, "wb") as f:
#     np.savez(
#         f,
#         rewards=rewards.cpu(),
#         steps=steps.cpu(),
#         episodes=episodes.cpu(),
#         all_actions=all_actions,
#         # losses=losses,
#         p=p,
#     )

# %% [markdown]
# ## Visualization

# %% [markdown]
# ### Load data from disk

# %%
# with open(data_path, "rb") as f:
#     # Load the arrays from the .npz file
#     data = np.load(f, allow_pickle=True)

#     # Access individual arrays by their names
#     rewards = data["rewards"]
#     steps = data["steps"]
#     episodes = data["episodes"]
#     all_actions = data["all_actions"]
#     losses = data["losses"]
#     p = data["p"][()]

# %% [markdown]
# ### Exploration rate

# %%
def plot_exploration_rate(epsilons, figpath=None):
    fig, ax = plt.subplots()
    sns.lineplot(epsilons)
    ax.set(ylabel="Epsilon")
    ax.set(xlabel="Steps")
    fig.tight_layout()
    fig.patch.set_alpha(0)
    fig.patch.set_facecolor("white")
    if figpath:
        fig.savefig(figpath / "exploration-rate.png", bbox_inches="tight")
    plt.show()


# %%
plot_exploration_rate(epsilons, figpath=CURRENT_PATH)


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
def plot_actions_distribution(actions, figpath=None):
    """Plot the distributions of states and actions."""
    fig, ax = plt.subplots()
    sns.histplot(data=actions, ax=ax)
    ax.set_xticks(
        [item.value for item in Actions], labels=[item.name for item in Actions]
    )
    ax.set_title("Actions")
    fig.tight_layout()
    fig.patch.set_alpha(0)
    fig.patch.set_facecolor("white")
    if figpath:
        fig.savefig(figpath / "actions-distribution.png", bbox_inches="tight")
    plt.show()


# %%
plot_actions_distribution(all_actions, figpath=CURRENT_PATH)


# %% [markdown]
# ### Steps & rewards

# %%
def plot_steps_and_rewards(df, figpath=None):
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
    fig.patch.set_alpha(0)
    fig.patch.set_facecolor("white")
    if figpath:
        fig.savefig(figpath / "steps-and-rewards.png", bbox_inches="tight")
    plt.show()


# %%
plot_steps_and_rewards(res, figpath=CURRENT_PATH)


# %%
def plot_steps_and_rewards_dist(df, figpath=None):
    """Plot the steps and rewards distributions from dataframes."""
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    sns.histplot(data=df, x="Rewards", ax=ax[0])
    sns.histplot(data=df, x="Steps", ax=ax[1])
    fig.tight_layout()
    fig.patch.set_alpha(0)
    fig.patch.set_facecolor("white")
    if figpath:
        fig.savefig(
            figpath / "steps-and-rewards-distrib.png",
            bbox_inches="tight",
        )
    plt.show()


# %%
plot_steps_and_rewards_dist(res, figpath=CURRENT_PATH)

# %% [markdown]
# ### Loss

# %%
window_size = 1
for idx, loss in enumerate(losses):
    current_loss = torch.tensor(loss, device=DEVICE)
    losses_rolling_avg = nn.functional.avg_pool1d(
        current_loss.view(1, 1, -1), kernel_size=window_size
    ).squeeze()
    tmp_df = pd.DataFrame(
        data={
            "Run": idx * torch.ones(len(losses_rolling_avg), device=DEVICE).int().cpu(),
            "Steps": torch.arange(0, len(losses_rolling_avg), device=DEVICE).cpu(),
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
if USETEX:
    ax.set(
        ylabel=(
            f"$Log_{{10}}(\mathrm{{Loss}})$\naveraged over {p.n_runs} runs"
            if p.n_runs > 1
            else "$Log_{10}(\mathrm{Loss})$"
        )
    )
else:
    ax.set(
        ylabel=(
            f"$Log_{{10}}(\\text{{Loss}})$\naveraged over {p.n_runs} runs"
            if p.n_runs > 1
            else "$Log_{10}(\\text{Loss})$"
        )
    )
ax.set(xlabel="Steps")
ax.set(yscale="log")
fig.tight_layout()
fig.patch.set_alpha(0)
fig.patch.set_facecolor("white")
fig.savefig(CURRENT_PATH / "loss.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ### Policy learned

# %%
with torch.no_grad():
    q_values = torch.nan * torch.empty(
        (len(env.tiles_locations), len(Cues), p.n_actions), device=DEVICE
    )
    for tile_i, tile_v in enumerate(env.tiles_locations):
        for cue_i, cue_v in enumerate(Cues):
            state = torch.tensor([tile_v, cue_v.value], device=DEVICE).float()
            if env.one_hot_state:
                state = env.to_one_hot(state).float()
            q_values[tile_i, cue_i, :] = net(state).to(DEVICE)
q_values.shape


# %%
# with torch.no_grad():
#     q_values = torch.nan * torch.empty(
#         (len(env.tiles_locations), len(OdorCues), len(LightCues), p.n_actions),
#         device=device,
#     )
#     for tile_i, tile_v in enumerate(env.tiles_locations):
#         for o_cue_i, o_cue_v in enumerate(OdorCues):
#             for l_cue_i, l_cue_v in enumerate(LightCues):
#                 state = torch.tensor(
#                     [tile_v, o_cue_v.value, l_cue_v.value], device=device
#                 ).float()
#                 if env.one_hot_state:
#                     state = env.to_one_hot(state).float()
#                 q_values[tile_i, cue_i, :] = net(state).to(device)
# q_values.shape

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
def plot_policies(q_values, labels, figpath=None):
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
    if figpath:
        fig.savefig(figpath / "policy.png", bbox_inches="tight")
    plt.show()


# %%
plot_policies(q_values=q_values, labels=CONTEXTS_LABELS, figpath=CURRENT_PATH)

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
        device=DEVICE,
    )
    assert not torch.equal(
        torch.zeros_like(grad_stats, device=DEVICE),
        grad_stats,
    ), "Gradients are zero"


# %%
plotting.plot_weights_biases_distributions(
    weights_val_df, biases_val_df, label="Values", figpath=CURRENT_PATH
)

# %%
check_grad_stats(weights_grad_df)

# %%
check_grad_stats(biases_grad_df)

# %%
plotting.plot_weights_biases_distributions(
    weights_grad_df, biases_grad_df, label="Gradients", figpath=CURRENT_PATH
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
def plot_weights_biases_stats(weights_stats, biases_stats, label=None, figpath=None):
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
    ax[1, 0].set_xlabel("Steps")
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
    ax[1, 1].set_xlabel("Steps")
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
    fig.patch.set_alpha(0)
    fig.patch.set_facecolor("white")
    if figpath:
        fig.savefig(
            figpath / f"weights-biases-stats-{label}.png",
            bbox_inches="tight",
        )
    plt.show()


# %%
plot_weights_biases_stats(
    weights_val_stats, biases_val_stats, label="values", figpath=CURRENT_PATH
)

# %%
plot_weights_biases_stats(
    weights_grad_stats, biases_grad_stats, label="gradients", figpath=CURRENT_PATH
)

# %%
# weights_val_stats.rolling(10, center=True).mean().dropna()

# %%
# rolling_win = 100
# plot_weights_biases_stats(
#     weights_val_stats.rolling(rolling_win, center=True).mean().dropna(),
#     biases_val_stats.rolling(rolling_win, center=True).mean().dropna(),
#     label="values",
# )

# %%
# rolling_win = 100
# plot_weights_biases_stats(
#     weights_grad_stats.rolling(rolling_win, center=True).mean().dropna(),
#     biases_grad_stats.rolling(rolling_win, center=True).mean().dropna(),
#     label="values",
# )
