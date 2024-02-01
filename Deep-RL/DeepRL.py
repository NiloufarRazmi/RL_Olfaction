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
# # Deep RL

# %% [markdown]
# 1. inputs need to reflect position in arena and odor (NOT CONJUNCTIONS)
# 2. outputs need to reflect action values
# 3. actions are selected via softmax on output neuron activity.
# 4. RPE requires knowing value of new state
#    -- so this will require a forward pass using "new state" inputs.

# %% [markdown]
# ## Dependencies

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

# %%
import numpy as np
import pandas as pd
import seaborn as sns
from agent import DQN, EpsilonGreedy
from deep_learning import Network
from env.RandomWalk1D import Actions, RandomWalk1D
from imojify import imojify
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

# from numpy.random import default_rng
from tqdm import tqdm

# %%
from utils import Params

# %%
# Formatting & autoreload stuff
# %load_ext lab_black
# %load_ext autoreload
# %autoreload 2
# # %matplotlib ipympl

# %%
sns.set_theme(font_scale=1.5)
mpl.rcParams["font.family"] = ["Fira Sans", "sans-serif"]

# %% [markdown]
# ## Parameters

# %%
p = Params(
    seed=42,
    n_runs=1,
    total_episodes=1000,
    epsilon=0.1,
    alpha=0.3,
    gamma=0.95,
    learning_rate=0.001,
    nLayers=3,
    nHiddenUnits=5,
)
p

# %%
# # Set the seed
# p.rng = np.random.default_rng(p.seed)

# %% [markdown]
# ## The environment

# %% [markdown]
# ![The task](https://juliareinforcementlearning.org/docs/assets/RandomWalk1D.png)

# %%
env = RandomWalk1D()

# %%
p.action_size = len(env.action_space)
p.state_size = len(env.observation_space)
print(f"Action size: {p.action_size}")
print(f"State size: {p.state_size}")

# %% [markdown]
# ## Running the environment

# %%
net = Network(
    nInputUnits=1,
    nLayers=p.nLayers,
    nOutputUnits=p.action_size,
    nHiddenUnits=p.nHiddenUnits,
)

# %%
[layer.shape for layer in net.wtMatrix]

# %%
learner = DQN(
    learning_rate=p.alpha,
    gamma=p.gamma,
    state_size=p.state_size,
    action_size=p.action_size,
)

# %%
explorer = EpsilonGreedy(epsilon=p.epsilon, rng=p.rng)

# %% [markdown]
# ### Main loop

# %%
rewards = np.zeros((p.total_episodes, p.n_runs))
steps = np.zeros((p.total_episodes, p.n_runs))
episodes = np.arange(p.total_episodes)
all_states = []
all_actions = []
steps_global = 0
X = np.array([])
y = np.array([])

for run in range(p.n_runs):  # Run several times to account for stochasticity
    learner.reset(
        action_size=env.numActions
    )  # Reset the Q-table and the weights between runs
    meanErrors = []
    weights_metrics = pd.DataFrame()
    grads_metrics = pd.DataFrame()

    for episode in tqdm(
        episodes, desc=f"Run {run+1}/{p.n_runs} - Episodes", leave=False
    ):
        state = env.reset()  # Reset the environment
        step = 0
        done = False
        total_rewards = 0

        while not done:
            # Obtain Q-values from network
            q_values = net.forward_pass(x_obs=[state])[-1]

            action = explorer.choose_action(
                action_space=env.action_space, state=state, q_values=q_values
            )

            # Take the action (a) and observe the outcome state(s') and reward (r)
            new_state, reward, done = env.step(action)

            # Obtain Q-value for selected action
            q_value = q_values[action]

            # Select next action with highest Q-value
            if new_state == done:
                new_q_value = 0  # No Q-value for terminal
            else:
                new_q_values = net.forward_pass(x_obs=[new_state])[
                    -1
                ]  # No gradient computation
                # TODO: Take a random action in case of
                new_action = np.argmax(new_q_values)
                new_q_value = new_q_values[new_action]

            # Compute observed Q-value
            q_update = reward + (learner.gamma * new_q_value)

            # # Compute loss value
            # loss = (q_update-q_value)**2

            # Update X and y for supervised learning
            if X.size == 0:
                X = np.array([state])[np.newaxis, :]
            else:
                X = np.append(X, np.array([state])[np.newaxis, :], axis=0)
            if y.size == 0:
                y = np.array([q_update])[np.newaxis, :]
            else:
                y = np.append(y, np.array([q_update])[np.newaxis, :])

            # Compute gradients and apply gradients to update network weights
            allError, y_hat, delta, activity = net.backprop(
                X=X, y=y, nLayers=p.nLayers, learning_rate=p.learning_rate
            )

            # Logging
            all_states.append(state)
            all_actions.append(action)
            meanErrors.append(np.nanmean(abs(allError)))

            for w_id, w_val in enumerate(net.wtMatrix):
                weights_metrics = pd.concat(
                    [
                        weights_metrics,
                        pd.DataFrame(
                            {
                                "avg": w_val.mean(),
                                "std": w_val.std(),
                                "id": w_id,
                                "step": step,
                                "steps_global": steps_global,
                            },
                            index=[steps_global],
                        ),
                    ],
                    ignore_index=True,
                )

            for d_id, d_val in enumerate(delta):
                grads_metrics = pd.concat(
                    [
                        grads_metrics,
                        pd.DataFrame(
                            {
                                "avg": d_val.mean(),
                                "std": d_val.std(),
                                "id": d_id,
                                "step": step,
                                "steps_global": steps_global,
                            },
                            index=[steps_global],
                        ),
                    ],
                    ignore_index=True,
                )

            total_rewards += reward
            step += 1
            steps_global += 1

            # Update the state
            state = new_state

        # Log all rewards and steps
        rewards[episode, run] = total_rewards
        steps[episode, run] = step
# weights_metrics.set_index("steps_global", inplace=True)
# grads_metrics.set_index("steps_global", inplace=True)
# grads_metrics["avg_rolling"] = grads_metrics.avg.rolling(10).mean()
# grads_metrics["std_rolling"] = grads_metrics["std"].rolling(10).mean()

# %%
weights_metrics

# %%
grads_metrics

# %%
grads_metrics["avg_rolling"] = np.nan
grads_metrics["std_rolling"] = np.nan
for id in grads_metrics.id.unique():
    grads_metrics.loc[grads_metrics[grads_metrics["id"] == id].index, "avg_rolling"] = (
        grads_metrics.loc[grads_metrics[grads_metrics["id"] == id].index, "avg"]
        .rolling(20)
        .mean()
    )
    grads_metrics.loc[grads_metrics[grads_metrics["id"] == id].index, "std_rolling"] = (
        grads_metrics.loc[grads_metrics[grads_metrics["id"] == id].index, "std"]
        .rolling(20)
        .mean()
    )
grads_metrics

# %% [markdown]
# ## Visualization

# %%
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(meanErrors)
ax.set_ylabel("Loss")
ax.set_xlabel("Steps")
plt.show()


# %%
def plot_weights(weights_metrics):
    """Plot the weights."""
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

    sns.lineplot(x="steps_global", y="avg", hue="id", data=weights_metrics, ax=ax[0])
    ax[0].set(ylabel="Weights (avg)")
    ax[0].set(xlabel="Steps")

    sns.lineplot(x="steps_global", y="std", hue="id", data=weights_metrics, ax=ax[1])
    ax[1].set(ylabel="Weights (std)")
    ax[1].set(xlabel="Steps")

    fig.tight_layout()
    plt.show()


# %%
plot_weights(weights_metrics)


# %%
def plot_gradients(grads_metrics):
    """Plot the gradienta."""
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

    sns.lineplot(
        x="steps_global",
        y="avg_rolling",
        hue="id",
        data=grads_metrics,
        ax=ax[0],
        palette=sns.color_palette()[0 : len(grads_metrics.id.unique())],
    )
    ax[0].set(ylabel="Gradients (avg)")
    ax[0].set(xlabel="Steps")

    sns.lineplot(
        x="steps_global",
        y="std_rolling",
        hue="id",
        data=grads_metrics,
        ax=ax[1],
        palette=sns.color_palette()[0 : len(grads_metrics.id.unique())],
    )
    ax[1].set(ylabel="Gradients (std)")
    ax[1].set(xlabel="Steps")

    fig.tight_layout()
    plt.show()


# %%
plot_gradients(grads_metrics)

# %%
grads_metrics.duplicated()


# %%
def postprocess(episodes, p, rewards, steps):
    """Convert the results of the simulation in dataframes."""
    res = pd.DataFrame(
        data={
            "Episodes": np.tile(episodes, reps=p.n_runs),
            "Rewards": rewards.flatten(order="F"),
            "Steps": steps.flatten(order="F"),
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
    sns.histplot(data=states, ax=ax[0])
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
        ylabel=(
            f"Rewards\naveraged over {p.n_runs} runs"
            if p.n_runs > 1
            else "Steps number"
        )
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
q_values = np.nan * np.empty((p.state_size, p.action_size))
for state_i, state_v in enumerate(np.arange(p.state_size)):
    if state_i in [0, len(q_values) - 1]:
        # q_values[state_i,] = 0
        continue
    q_values[state_i,] = net.forward_pass(x_obs=[state_v])[-1]
q_values

# %%
q_values.flatten()[np.newaxis, :]


# %%
def plot_q_values(q_values):
    fig, ax = plt.subplots(figsize=(15, 1.5))
    cmap = sns.color_palette("vlag", as_cmap=True)
    chart = sns.heatmap(
        q_values.flatten()[np.newaxis, :],
        annot=True,
        ax=ax,
        cmap=cmap,
        yticklabels=False,  # linewidth=0.5
        center=0,
    )
    states_nodes = np.arange(1, 14, 2)
    chart.set_xticks(states_nodes)
    chart.set_xticklabels([str(item) for item in np.arange(1, 8, 1)])
    chart.set_title("Q values")
    ax.tick_params(bottom=True)

    # Add actions arrows
    for node in states_nodes:
        y_height = 1.7
        if node in [1, 13]:
            continue
        arrows_left = {
            "x_tail": node,
            "y_tail": y_height,
            "x_head": node - 1,
            "y_head": y_height,
        }
        arrow = mpatches.FancyArrowPatch(
            (arrows_left["x_tail"], arrows_left["y_tail"]),
            (arrows_left["x_head"], arrows_left["y_head"]),
            mutation_scale=10,
            clip_on=False,
            color="k",
        )
        ax.add_patch(arrow)
        arrows_right = {
            "x_tail": node,
            "y_tail": y_height,
            "x_head": node + 1,
            "y_head": y_height,
        }
        arrow = mpatches.FancyArrowPatch(
            (arrows_right["x_tail"], arrows_right["y_tail"]),
            (arrows_right["x_head"], arrows_right["y_head"]),
            mutation_scale=10,
            clip_on=False,
            color="k",
        )
        ax.add_patch(arrow)

        # Add rectangle to separate each state pair
        rect = mpatches.Rectangle(
            (node - 1, 0),
            2,
            1,
            linewidth=2,
            edgecolor="k",
            facecolor="none",
            clip_on=False,
        )
        ax.add_patch(rect)

    def add_emoji(coords, emoji, ax):
        """Add emoji as image at absolute coordinates."""
        img = plt.imread(imojify.get_img_path(emoji))
        im = OffsetImage(img, zoom=0.08)
        im.image.axes = ax
        ab = AnnotationBbox(
            im, (coords[0], coords[1]), frameon=False, pad=0, annotation_clip=False
        )
        ax.add_artist(ab)

    emoji = [
        {"emoji": "🪨", "coords": [1, y_height]},
        {"emoji": "💎", "coords": [13, y_height]},
    ]
    for _, emo in enumerate(emoji):
        add_emoji(emo["coords"], emo["emoji"], ax)

    plt.show()


# %%
plot_q_values(q_values)

# %%
