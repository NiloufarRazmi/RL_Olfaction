from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import minmax_scale


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


def get_location_count(
    all_state_composite, tiles_locations, cols, rows, cue=None, scale=True
):
    """Count the occurences for each tile location.

    Optionally filter by `cue`"""
    location_count = np.zeros(len(tiles_locations))
    for tile in tiles_locations:
        if cue:  # Select based on chosen cue
            location_count[tile] = len(
                all_state_composite[
                    (all_state_composite.location == tile)
                    & (all_state_composite.cue == cue)
                ]
            )
        else:  # Select
            location_count[tile] = len(
                all_state_composite[all_state_composite.location == tile]
            )

    if scale:
        # Scale to [0, 1]
        locations_scaled = minmax_scale(location_count.flatten())
        loc_count = locations_scaled.reshape((rows, cols))
    else:
        loc_count = location_count.reshape((rows, cols))

    return loc_count
