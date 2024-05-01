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
import datetime
import logging
import shutil
from collections import deque, namedtuple, OrderedDict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
import ipdb

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# from torchinfo import summary

# if GPU is to be used
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE

# %%
import plotting as viz

# from environment_lights_tensor import (
#     WrappedEnvironment,
#     Actions,
#     CONTEXTS_LABELS,
#     OdorCues,
#     LightCues,
# )
from agent_tensor import EpsilonGreedy
from environment_tensor import (
    CONTEXTS_LABELS,
    Actions,
    Cues,
    WrappedEnvironment,
    TriangleState,
)
import utils

# %%
# Formatting & autoreload stuff
# %load_ext lab_black
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
EXP_TAG = "scratch"

# %%
ROOT_PATH = Path("env").parent
SAVE_PATH = ROOT_PATH / "save"
folder = f"{now}_{EXP_TAG}" if EXP_TAG else now
CURRENT_PATH = SAVE_PATH / folder
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
    # seed=42,
    # seed=123,
    n_runs=1,
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
    experiment_tag="scratch",
)
p

# %%
if p.batch_size < 2:
    raise ValueError("The batch size needs to be more that one data point")

# %% [markdown]
# ### Utilies

# %%
CURRENT_PATH = utils.create_save_path(p.experiment_tag)

# %%
LOGGER = utils.get_logger(current_path=CURRENT_PATH)

# %%
# Set the seed
GENERATOR = utils.make_deterministic(seed=p.seed)

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
state = env.reset()
p.n_observations = len(state)

print(f"Number of actions: {p.n_actions}")
print(f"Number of observations: {p.n_observations}")

# %% [markdown]
# ### Network definition

# %%
ENCODER_NEURONS_NUM = 5


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
weights_untrained = [layer.detach() for layer in net.parameters()]

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
viz.plot_exploration_rate(epsilons, xlabel="Episodes")

# %% [markdown]
# ## Training

# %%
Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)

# %% [markdown]
# ### Main loop

# %%
rewards = torch.zeros((p.total_episodes, p.n_runs), device=DEVICE)
steps = torch.zeros((p.total_episodes, p.n_runs), device=DEVICE)
episodes = torch.arange(p.total_episodes, device=DEVICE)
# all_states = []
all_actions = []
losses = [[] for _ in range(p.n_runs)]

for run in range(p.n_runs):  # Run several times to account for stochasticity
    # Reset everything
    net, target_net = neural_network()  # Reset weights
    optimizer = optim.AdamW(net.parameters(), lr=p.alpha, amsgrad=True)
    explorer = EpsilonGreedy(
        epsilon=p.epsilon_max,
        epsilon_min=p.epsilon_min,
        epsilon_max=p.epsilon_max,
        decay_rate=p.decay_rate,
        epsilon_warmup=p.epsilon_warmup,
    )
    weights_val_stats = None
    biases_val_stats = None
    weights_grad_stats = None
    biases_grad_stats = None
    replay_buffer = deque([], maxlen=p.replay_buffer_max_size)
    epsilons = []

    for episode in tqdm(
        episodes, desc=f"Run {run+1}/{p.n_runs} - Episodes", leave=False
    ):
        state = env.reset()  # Reset the environment
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

            # Start training when `replay_buffer` is full
            if len(replay_buffer) == p.replay_buffer_max_size:
                transitions = utils.random_choice(
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

                # done_false = torch.argwhere(done_batch == False).squeeze()
                # done_true = torch.argwhere(done_batch == True).squeeze()
                # expected_state_action_values = (
                #     torch.zeros_like(done_batch, device=DEVICE)
                # ).float()
                # with torch.no_grad():
                #     if done_true.numel() > 0:
                #         expected_state_action_values[done_true] = reward_batch[
                #             done_true
                #         ]
                #     if done_false.numel() > 0:
                #         next_state_values = (
                #             target_net(next_state_batch[done_false]).to(DEVICE).max(1)
                #         )  # Q(s_t+1, a)
                #         expected_state_action_values[done_false] = (
                #             reward_batch[done_false]
                #             + p.gamma * next_state_values.values
                #         )  # y_j (Bellman optimality equation)

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
                expected_state_action_values = reward_batch + (
                    next_state_values * p.gamma
                )

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

                # # Reset the target network
                # if step_count % p.target_net_update == 0:
                #     target_net.load_state_dict(net.state_dict())

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

                weights, biases = utils.collect_weights_biases(net=net)
                weights_val_stats = utils.params_df_stats(
                    weights, key="val", current_df=weights_grad_stats
                )
                biases_val_stats = utils.params_df_stats(
                    biases, key="val", current_df=biases_val_stats
                )
                biases_grad_stats = utils.params_df_stats(
                    biases, key="grad", current_df=biases_grad_stats
                )
                weights_grad_stats = utils.params_df_stats(
                    weights, key="grad", current_df=weights_val_stats
                )

            total_rewards += reward
            step_count += 1

            # Move to the next state
            state = next_state

            explorer.epsilon = explorer.update_epsilon(episode)
            epsilons.append(explorer.epsilon)

        rewards[episode, run] = total_rewards
        steps[episode, run] = step_count
        LOGGER.info(
            f"Run: {run+1}/{p.n_runs} - Episode: {episode+1}/{p.total_episodes} - Steps: {step_count} - Loss: {loss.item()}"
        )
    weights_val_stats.set_index("Index", inplace=True)
    biases_val_stats.set_index("Index", inplace=True)
    biases_grad_stats.set_index("Index", inplace=True)
    weights_grad_stats.set_index("Index", inplace=True)

# %% [markdown]
# ### Save data to disk

# %%
# data_path = CURRENT_PATH / "data.npz"
# with open(data_path, "wb") as fhd:
#     np.savez(
#         fhd,
#         rewards=rewards.cpu(),
#         steps=steps.cpu(),
#         episodes=episodes.cpu(),
#         all_actions=all_actions,
#         losses=losses,
#         p=p,
#     )

# %%
data_dict = {
    "rewards": rewards.cpu(),
    "steps": steps.cpu(),
    "episodes": episodes.cpu(),
    "all_actions": all_actions,
    "losses": losses,
    "p": p,
}

# %%
data_path = utils.save_data(data_dict=data_dict, current_path=CURRENT_PATH)

# %% [markdown]
# ## Visualization

# %% [markdown]
# ### Exploration rate

# %%
viz.plot_exploration_rate(epsilons, xlabel="Steps", figpath=CURRENT_PATH)

# %% [markdown]
# ### States & actions distributions

# %%
rew_steps_df = utils.postprocess_rewards_steps(
    episodes=episodes, n_runs=p.n_runs, rewards=rewards, steps=steps
)
rew_steps_df

# %%
viz.plot_actions_distribution(all_actions, figpath=CURRENT_PATH)

# %% [markdown]
# ### Steps & rewards

# %%
viz.plot_steps_and_rewards_dist(rew_steps_df, figpath=CURRENT_PATH)

# %%
viz.plot_steps_and_rewards(rew_steps_df, n_runs=p.n_runs, figpath=CURRENT_PATH)

# %% [markdown]
# ### Loss

# %%
loss_df = utils.postprocess_loss(losses=losses, window_size=1)
loss_df

# %%
viz.plot_loss(loss_df, n_runs=p.n_runs, figpath=CURRENT_PATH)

# %% [markdown]
# ### Policy learned

# %%
q_values = utils.get_q_values_by_states(
    env=env, cues=Cues, n_actions=p.n_actions, net=net
)
q_values.shape

# %%
viz.plot_policies(
    q_values=q_values,
    labels=CONTEXTS_LABELS,
    n_rows=env.rows,
    n_cols=env.cols,
    figpath=CURRENT_PATH,
)

# %% [markdown]
# ### Weights matrix

# %%
viz.plot_weights_matrices(
    weights_untrained=weights_untrained,
    weights_trained=[layer for layer in net.parameters()],
    figpath=CURRENT_PATH,
)

# %% [markdown]
# ### Activations learned

# %%
# [item for item in net.mlp.named_children()]

# %%
# # Hook to capture the activations
# activations = {}


# def get_activation(name):
#     def hook(module, args, output):
#         activations[name] = output.detach()

#     return hook

# %%
# # Register the hooks for all layers
# for name, layer in net.mlp.named_children():
#     layer.register_forward_hook(get_activation(name))

# %%
# x = torch.randn(28)
# output = net(x)

# %%
# [val.shape for key, val in activations.items()]

# %%
# # Construct input dictionnary to be fed to the network
# input_cond = OrderedDict({})
# for cue_obj, cue_txt in CONTEXTS_LABELS.items():
#     for loc in env.state_space["location"]:
#         current_state = torch.tensor([loc, cue_obj.value], device=DEVICE)
#         if env.one_hot_state:
#             current_state = env.to_one_hot(current_state)
#         input_cond[f"{loc}-{cue_txt}"] = current_state.float()

# %%
# layer = list(net.mlp.children())[6]
# parameters = list(layer.named_parameters())
# weights = [params[1] for params in parameters if params[0] == "weight"][0]
# neurons_num = weights.shape[1]
# neurons_num

# %%
# # Get the activations from the network
# layer_inspected = 6 - 1
# activations_layer = (
#     torch.ones((len(input_cond), ENCODER_NEURONS_NUM), device=DEVICE) * torch.nan
# )
# for idx, (cond, input_val) in enumerate(input_cond.items()):
#     net(input_val)
#     activations_layer[idx, :] = activations[str(layer_inspected)]

# %%
# # cols = pd.MultiIndex.from_tuples(
# #     [("neuron", str(item)) for item in range(1, ENCODER_NEURONS_NUM + 1)]
# # )
# activations_layer_df = pd.DataFrame(activations_layer)  # , columns=cols)
# activations_layer_df["Input"] = list(input_cond.keys())
# activations_layer_df.set_index("Input", inplace=True)
# activations_layer_df

# %%
input_cond, activations_layer_df = utils.get_activations_learned(
    net=net, env=env, layer_inspected=p.layer_inspected, contexts_labels=CONTEXTS_LABELS
)

# %%
# input_cond

# %%
activations_layer_df

# %%
viz.plot_activations(
    activations_layer_df=activations_layer_df,
    input_cond=input_cond,
    labels=CONTEXTS_LABELS,
    layer_inspected=p.layer_inspected,
    figpath=CURRENT_PATH,
)

# %% [markdown]
# ### Weights & gradients metrics

# %%
weights, biases = utils.collect_weights_biases(net=net)

# %%
weights_val_df = utils.postprocess_weights(weights["val"])
weights_val_df

# %%
weights_val_df.describe()

# %%
biases_val_df = utils.postprocess_weights(biases["val"])
biases_val_df

# %%
biases_val_df.describe()

# %%
weights_grad_df = utils.postprocess_weights(weights["grad"])
weights_grad_df

# %%
weights_grad_df.describe()

# %%
biases_grad_df = utils.postprocess_weights(biases["grad"])
biases_grad_df

# %%
biases_grad_df.describe()

# %%
viz.plot_weights_biases_distributions(
    weights_val_df, biases_val_df, label="Values", figpath=CURRENT_PATH
)

# %%
assert utils.check_grad_stats(weights_grad_df), "Gradients are zero"

# %%
assert utils.check_grad_stats(biases_grad_df), "Gradients are zero"

# %%
viz.plot_weights_biases_distributions(
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
viz.plot_weights_biases_stats(
    weights_val_stats, biases_val_stats, label="values", figpath=CURRENT_PATH
)

# %%
viz.plot_weights_biases_stats(
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
