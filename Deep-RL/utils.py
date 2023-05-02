from dataclasses import dataclass
from typing import Optional

import numpy as np


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
    epsilon: float = 0.1  # Action-selection parameters

    # QLearning parameters
    gamma: float = 0.8
    alpha: float = 0.05
    jointRep: bool = True

    # Deep network
    learning_rate: float = 0.001
    nLayers: int = 5
    nHiddenUnits: int = 20

    # Environment
    action_size: Optional[int] = None
    state_size: Optional[int] = None


class Sigmoid:
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def gradient(self, x):
        return self(x) * (1 - self(x))
