from dataclasses import dataclass
from typing import Optional

import torch


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


def random_choice(choices_array):
    logits = torch.ones_like(choices_array)
    idx = torch.distributions.categorical.Categorical(logits=logits).sample()
    random_choice = choices_array[idx]
    return random_choice
