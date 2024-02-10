from dataclasses import dataclass
from typing import Optional

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Params:
    """Container class to keep track of all hyperparameters."""

    # General
    seed: Optional[int] = None
    rng: Optional[int] = None

    # Experiment
    n_runs: int = 10
    total_episodes: int = 100  # Set up the task

    # epsilon-greedy
    epsilon: float = 0.2  # Action-selection parameters
    epsilon_min: float = 0.1
    epsilon_max: float = 1.0
    decay_rate: float = 0.05
    epsilon_warmup: float = 100

    # Learning parameters
    gamma: float = 0.8
    alpha: float = 0.1

    # Deep network
    nLayers: int = 5
    nHiddenUnits: int = 20

    # Environment
    # action_size: Optional[int] = None
    # state_size: Optional[int] = None
    n_observations: Optional[int] = None
    n_actions: Optional[int] = None

    replay_buffer_max_size: int = 1000
    batch_size: int = 32
    target_net_update: int = 100


def random_choice(choices_array, length=None, num_samples=1):
    """
    PyTorch version of `numpy.random.choice`.

    Generates a random sample from a given 1-D array
    """
    if length:
        weights = torch.ones(length, device=DEVICE)
    else:
        weights = torch.ones_like(choices_array, dtype=float, device=DEVICE)
    idx = torch.multinomial(input=weights, num_samples=num_samples, replacement=False)
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
