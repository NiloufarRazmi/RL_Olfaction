from dataclasses import dataclass

import numpy as np


@dataclass
class Params:
    """Container class to keep track of all hyperparameters."""

    #     epsilon: float = 0.1  # Action-selection parameters

    #     # QLearning parameters
    #     gamma: float = 0.8
    #     alpha: float = 0.05
    #     jointRep: bool = True

    #     n_runs: int = 5
    #     numEpisodes: int = 100  # Set up the task

    # Deep network
    learning_rate: float = 0.001
    nLayers: int = 5
    nHiddenUnits: int = 20


class Sigmoid:
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def gradient(self, x):
        return self(x) * (1 - self(x))
