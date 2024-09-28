"""Utilities functions."""

import configparser
import datetime
import logging
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tensordict.tensordict import TensorDict

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Params:
    """Container class to keep track of all hyperparameters."""

    # General
    seed: Optional[int] = None

    # Experiment
    n_runs: int = 10
    total_episodes: int = 100  # Set up the task

    # epsilon-greedy
    epsilon: float = 0.2  # Action-selection parameters
    epsilon_min: float = 0.1
    epsilon_max: float = 1.0
    decay_rate: float = 0.05
    epsilon_warmup: int = 100

    # Learning parameters
    gamma: float = 0.8
    alpha: float = 0.1

    # Deep network
    nLayers: int = 5
    n_hidden_units: int = 2**7

    # Environment
    # action_size: Optional[int] = None
    # state_size: Optional[int] = None
    n_observations: Optional[int] = None
    n_actions: Optional[int] = None

    replay_buffer_max_size: int = 1000
    batch_size: int = 2**5
    target_net_update: int = 100
    tau: float = 0.005

    experiment_tag: str = ""
    taskid: str = ""
    layer_inspected: int = None


def random_choice(choices_array, length=None, num_samples=1, generator=None):
    """
    PyTorch version of `numpy.random.choice`.

    Generates a random sample from a given 1-D array
    """
    if length:
        weights = torch.ones(length, device=DEVICE)
    else:
        weights = torch.ones_like(choices_array, dtype=float, device=DEVICE)
    idx = torch.multinomial(
        input=weights, num_samples=num_samples, replacement=False, generator=generator
    )
    # idx = torch.distributions.categorical.Categorical(logits=logits).sample()
    if num_samples == 1:
        random_res = choices_array[idx]
    elif num_samples > 1:
        random_res = [choices_array[idj] for idj in idx]
    else:
        raise ValueError(
            "The number of samples has to be positive and greater than zero"
        )
    return random_res


def make_deterministic(seed=None):
    """Make everything deterministic in a single call."""
    generator = None

    if seed:
        # PyTorch
        generator = torch.Generator(device=DEVICE)
        generator.manual_seed(seed)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # Numpy
        np.random.seed(seed)

        # Built-in Python
        # random.seed(seed)
        # https://docs.python.org/3/using/cmdline.html#envvar-PYTHONHASHSEED
        os.environ["PYTHONHASHSEED"] = str(seed)

    return generator


def check_grad_stats(grad_df):
    """Check that gradients are not zero."""
    grad_stats = torch.tensor(
        [
            grad_df.Val.mean(),
            grad_df.Val.std(),
            grad_df.Val.min(),
            grad_df.Val.max(),
        ],
        device=DEVICE,
    )
    return not torch.equal(
        torch.zeros_like(grad_stats, device=DEVICE),
        grad_stats,
    )


def get_logger(current_path):
    """Configure logging."""
    logfile = current_path / "training.log"
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(logfile)
    formatter = logging.Formatter(
        "%(asctime)s : %(name)s  : %(funcName)s : %(levelname)s : %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def create_save_path(task, experiment_tag):
    """Make saving directory for the experiment results."""
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    root_path = Path("env").parent
    save_path = root_path / "save"
    folder = f"{now}_{task}_{experiment_tag}" if experiment_tag else now
    current_path = save_path / folder
    current_path.mkdir(parents=True, exist_ok=True)  # Create the tree of directories
    print(f"Current path: {current_path.absolute()}")
    return current_path


def collect_weights_biases(net):
    """Collect weights & baisis in a dataframe."""
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


def params_df_stats(weights, key, current_df=None):
    """Collect weights stats in a dataframe."""
    if current_df is not None:
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

        df = tmp_df if df is None else pd.concat((df, tmp_df))
    return df


def save_data(data_dict, current_path):
    """Save variables to disk."""
    data_path = current_path / "data.npz"
    with open(data_path, "wb") as fhd:
        np.savez(fhd, **data_dict)
    return data_path


def postprocess_rewards_steps(episodes, n_runs, rewards, steps):
    """Convert the results of the simulation in dataframes."""
    res = pd.DataFrame(
        data={
            "Episodes": np.tile(episodes, n_runs),
            "Rewards": rewards.T.flatten(),
            "Steps": steps.T.flatten(),
        }
    )
    # res["cum_rewards"] = rewards.cumsum(axis=0).flatten(order="F")
    return res


def postprocess_loss(losses, window_size=1):
    """Convert losses to dataframe."""
    for idx, loss in enumerate(losses):
        current_loss = torch.tensor(loss, device=DEVICE)
        losses_rolling_avg = nn.functional.avg_pool1d(
            current_loss.view(1, 1, -1), kernel_size=window_size
        ).squeeze()
        tmp_df = pd.DataFrame(
            data={
                "Run": idx
                * torch.ones(len(losses_rolling_avg), device=DEVICE).int().cpu(),
                "Steps": torch.arange(0, len(losses_rolling_avg), device=DEVICE).cpu(),
                "Loss": losses_rolling_avg.cpu(),
            }
        )
        if idx == 0:  # noqa SIM108
            loss_df = tmp_df
        else:
            loss_df = pd.concat((loss_df, tmp_df))
    return loss_df


def get_q_values_by_states(env, cues, net):
    """Run the forward pass for all states."""
    with torch.no_grad():
        q_values = torch.nan * torch.empty(
            (
                len(cues),
                len(env.tiles_locations["x"]),
                len(env.tiles_locations["y"]),
                len(env.head_angle_space),
                len(env.action_space),
            ),
            device=DEVICE,
        )
        for cue_i, cue_v in enumerate(cues):
            for x_i, x_v in enumerate(env.tiles_locations["x"]):
                for y_i, y_v in enumerate(env.tiles_locations["y"]):
                    for direction_i, direction_v in enumerate(env.head_angle_space):
                        state = env.conv_dict_to_flat_duplicated_coords(
                            TensorDict(
                                {
                                    "cue": torch.tensor([cue_v.value], device=DEVICE),
                                    "x": torch.tensor([x_v], device=DEVICE),
                                    "y": torch.tensor([y_v], device=DEVICE),
                                    "direction": torch.tensor(
                                        [direction_v], device=DEVICE
                                    ),
                                },
                                batch_size=[1],
                                device=DEVICE,
                            )
                        )
                        q_values[cue_i, x_i, y_i, direction_i, :] = net(state).to(
                            DEVICE
                        )
    return q_values


def postprocess_weights(weights):
    """Convert weights to dataframe."""
    for idx, val in enumerate(weights):
        tmp_df = pd.DataFrame(
            data={
                "Val": val.detach().cpu().flatten(),
                "Layer": idx,
            }
        )
        if idx == 0:  # noqa SIM108
            df = tmp_df
        else:
            df = pd.concat((df, tmp_df))
    return df


def get_activations_learned(net, env, layer_inspected, contexts_labels):
    """
    Extract activations learned from the network.

    The `layer_inspected` should be the index of one of the sequential layers, e.g.:
    ```
    DQN(
      (mlp): Sequential(
        (0): Linear(in_features=10, out_features=512, bias=True)
        (1): ReLU()
        (2): Linear(in_features=512, out_features=512, bias=True)
        (3): ReLU()
        (4): Linear(in_features=512, out_features=512, bias=True)
        (5): ReLU()
        (6): Linear(in_features=512, out_features=512, bias=True)
        (7): ReLU()
        (8): Linear(in_features=512, out_features=3, bias=True)
      )
    )
    ```
    """
    activations = {}

    def get_activations_hook(name):
        """Provide hook to capture the activations."""

        def hook(module, args, output):
            activations[name] = output.detach()

        return hook

    # Register the hooks for all layers
    for name, layer in net.mlp.named_children():
        layer.register_forward_hook(get_activations_hook(name))

    # Construct input dictionnary to be fed to the network
    input_cond = OrderedDict({})
    for cue_obj, cue_txt in contexts_labels.items():
        for _, direction_v in enumerate(env.head_angle_space):
            for _, x_v in enumerate(env.tiles_locations["x"]):
                for _, y_v in enumerate(env.tiles_locations["y"]):
                    current_state = env.conv_dict_to_flat_duplicated_coords(
                        TensorDict(
                            {
                                "cue": torch.tensor([cue_obj.value], device=DEVICE),
                                "x": torch.tensor([x_v], device=DEVICE),
                                "y": torch.tensor([y_v], device=DEVICE),
                                "direction": torch.tensor([direction_v], device=DEVICE),
                            },
                            batch_size=[1],
                            device=DEVICE,
                        )
                    )
                    input_cond[
                        f"{cue_txt} | {(x_v.item(), y_v.item())} | {direction_v}°"
                    ] = current_state.float()

    # Get the number of neurons in the layer inspected
    layer = list(net.mlp.children())[layer_inspected]
    parameters = list(layer.named_parameters())
    weights = [params[1] for params in parameters if params[0] == "weight"][0]
    neurons_num = weights.shape[1]

    # Get the activations from the network
    activations_layer = (
        torch.ones((len(input_cond), neurons_num), device=DEVICE) * torch.nan
    )
    for idx, (_, input_val) in enumerate(input_cond.items()):
        net(input_val)
        activations_layer[idx, :] = activations[str(layer_inspected)]

    activations_layer_df = pd.DataFrame(activations_layer.cpu())  # , columns=cols)
    activations_layer_df["Input"] = list(input_cond.keys())
    activations_layer_df.set_index("Input", inplace=True)
    return input_cond, activations_layer_df


def get_exp_params_from_config(config_path):
    """Extract parameters from config file."""
    print(f"Experiments parameters path: {config_path.absolute()}")
    config = configparser.ConfigParser()
    config.read(config_path)
    params = {}
    for key in config["experiment"]:
        if key in [
            "seed",
            "n_runs",
            "total_episodes",
            "n_layers",
            "n_observations",
            "n_actions",
            "replay_buffer_max_size",
            "target_net_update",
            "layer_inspected",
            "epsilon_warmup",
        ]:
            params[key] = int(config["experiment"][key])
        elif key in [
            "epsilon",
            "epsilon_min",
            "epsilon_max",
            "decay_rate",
            "gamma",
            "alpha",
            "tau",
        ]:
            params[key] = float(config["experiment"][key])
        elif key in ["n_hidden_units", "batch_size"]:
            # Strings that need to be evaluated first
            params[key] = int(eval(config["experiment"][key]))
        else:
            params[key] = config["experiment"][key]
    p = Params(**params)
    return p
