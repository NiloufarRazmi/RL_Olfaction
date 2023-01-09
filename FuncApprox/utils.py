from dataclasses import dataclass


@dataclass
class Params:
    """Container class to keep track of all hyperparameters."""

    epsilon: float = 0.1  # Action-selection parameters

    # QLearning parameters
    gamma: float = 0.8
    alpha: float = 0.05
    jointRep: bool = True

    n_runs: int = 5
    numEpisodes: int = 100  # Set up the task
