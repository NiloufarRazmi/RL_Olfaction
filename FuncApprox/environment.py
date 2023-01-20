from enum import Enum

import numpy as np


class OdorCondition(Enum):
    pre = 1
    post = 2


class Ports(Enum):
    North = 4
    South = 20
    West = 0
    East = 24


class Cues(Enum):
    North = Ports.North.value
    South = Ports.South.value


class Actions(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


CONTEXTS_LABELS = [
    "Pre odor - North port",
    "Pre odor - South port",
    "Post odor - North port",
    "Post odor - South port",
]


class Environment:
    """Environment logic."""

    def __init__(self, params, rng=None):
        if rng:
            self.rng = rng
        self.rows = 5
        self.cols = 5
        self.tiles_locations = set(np.arange(self.rows * self.cols))

        self.action_space = set([item.value for item in Actions])
        self.numActions = len(self.action_space)

        self.state_space = {"location": self.tiles_locations, "cue": Cues}
        self.numStates = tuple(len(item) for item in self.state_space.values())
        self.reset()

    def reset(self):
        """Reset the environment."""
        start_state = {
            "location": np.random.randint(
                low=min(self.tiles_locations),
                high=max(self.tiles_locations) + 1,
            ),
            # "location": 12,
            "cue": np.random.choice(Cues).value,
            # "cue": Cues.South.value,
            # "cue": Cues.North.value,
        }
        self.odor_condition = OdorCondition.pre
        # self.odor_condition = OdorCondition.post
        return start_state

    def is_terminated(self, state):
        """Returns if the episode is terminated or not."""
        if self.odor_condition == OdorCondition.post and (
            state["location"] == Ports.West.value
            or state["location"] == Ports.East.value
        ):
            return True
        else:
            return False

    def reward(self, state):
        """Observe the reward."""
        reward = 0
        if self.odor_condition == OdorCondition.post:
            if (
                state["cue"] == Cues.North.value
                and state["location"] == Ports.West.value
            ):
                reward = 10
            elif (
                state["cue"] == Cues.South.value
                and state["location"] == Ports.East.value
            ):
                reward = 10
        return reward

    def step(self, action, current_state):
        """Take an action, observe reward and the next state."""
        new_state = {}
        new_state["cue"] = current_state["cue"]
        row, col = self.to_row_col(current_state)
        newrow, newcol = self.move(row, col, action)
        new_state["location"] = self.to_state_location(newrow, newcol)

        # Update odor condition
        if new_state["location"] == new_state["cue"]:
            self.odor_condition = OdorCondition.post

        reward = self.reward(new_state)
        done = self.is_terminated(new_state)
        return new_state, reward, done

    def to_state_location(self, row, col):
        """Convenience function to convert row and column to state number."""
        return row * self.cols + col

    def to_row_col(self, state):
        """Convenience function to convert state to row and column."""
        states_locations = np.reshape(
            list(self.tiles_locations), (self.rows, self.cols)
        )
        (row, col) = np.argwhere(states_locations == state["location"]).flatten()
        return (row, col)

    def move(self, row, col, a):
        """Where the agent ends up on the map."""
        if a == Actions.LEFT.value:
            col = max(col - 1, 0)
        elif a == Actions.DOWN.value:
            row = min(row + 1, self.rows - 1)
        elif a == Actions.RIGHT.value:
            col = min(col + 1, self.cols - 1)
        elif a == Actions.UP.value:
            row = max(row - 1, 0)
        return (row, col)


class WrappedEnvironment(Environment):
    """Wrap the base Environment class.

    Results in numerical only and flattened state space"""

    def __init__(self, params, rng=None):
        # Initialize the base class to get the base properties
        super().__init__(params, rng=None)

        self.state_space = set(
            np.arange(self.rows * self.cols * len(Cues) * len(OdorCondition))
        )
        self.numStates = len(self.state_space)
        self.reset()

    def convert_composite_to_flat_state(self, state):
        """Convert composite state dictionary to a flat single number."""
        conv_state = None
        tiles_num = len(self.tiles_locations)

        if self.odor_condition == OdorCondition.pre:  # Is that cheating??
            if state["cue"] == Cues.North.value:
                conv_state = state["location"]
            elif state["cue"] == Cues.South.value:
                conv_state = state["location"] + tiles_num
        elif self.odor_condition == OdorCondition.post:  # Is that cheating??
            if state["cue"] == Cues.North.value:
                conv_state = state["location"] + 2 * tiles_num
            elif state["cue"] == Cues.South.value:
                conv_state = state["location"] + 3 * tiles_num

        if conv_state is None:
            raise ValueError("Impossible value for composite state")

        return conv_state

    def convert_flat_state_to_composite(self, state):
        """Convert back flattened state to original composite state."""
        tiles_num = len(self.tiles_locations)
        if state >= 3 * tiles_num and state < 4 * tiles_num:
            conv_state = {"location": state - 3 * tiles_num, "cue": Cues.South.value}
        elif state >= 2 * tiles_num and state < 3 * tiles_num:
            conv_state = {"location": state - 2 * tiles_num, "cue": Cues.North.value}
        elif state >= tiles_num and state < 2 * tiles_num:
            conv_state = {"location": state - tiles_num, "cue": Cues.South.value}
        elif state < tiles_num:
            conv_state = {"location": state, "cue": Cues.North.value}
        else:
            raise ValueError("Impossible number for flat state")
        return conv_state

    # def is_terminated(self, state):
    #     """Wrapper around the base method."""
    #     conv_state = self.convert_flat_state_to_composite(state)
    #     return super().is_terminated(conv_state)

    # def reward(self, state):
    #     """Wrapper around the base method."""
    #     conv_state = self.convert_flat_state_to_composite(state)
    #     return super().reward(conv_state)

    def step(self, action, current_state):
        """Wrapper around the base method."""
        current_conv_state = self.convert_flat_state_to_composite(current_state)
        new_state, reward, done = super().step(action, current_conv_state)
        new_state_conv = self.convert_composite_to_flat_state(new_state)
        return new_state_conv, reward, done

    # def to_row_col(self, state):
    #     """Wrapper around the base method."""
    #     conv_state = self.convert_flat_state_to_composite(state)
    #     return super().to_row_col(conv_state)

    def reset(self):
        """Wrapper around the base method."""
        state = super().reset()
        conv_state = self.convert_composite_to_flat_state(state)
        return conv_state
