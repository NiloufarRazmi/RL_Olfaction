from collections import OrderedDict
from enum import Enum

import numpy as np


class OdorCondition(Enum):
    pre = 1
    post = 2


# class OdorID(Enum):
#     A = 1
#     B = 2


class Cues(Enum):
    NoOdor = 0
    OdorA = 1
    OdorB = 2


class Ports(Enum):
    North = 4
    South = 20
    West = 0
    East = 24


# class LightCues(Enum):
#     North = Ports.North.value
#     South = Ports.South.value


class OdorPorts(Enum):
    North = Ports.North.value
    South = Ports.South.value


class Actions(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class TriangleState(Enum):
    upper = 1
    lower = 2


CONTEXTS_LABELS = OrderedDict(
    [
        # (LightCues.North, "Pre odor - North light"),
        # (LightCues.South, "Pre odor - South light"),
        # (OdorID.A, "Post odor - Odor A"),
        # (OdorID.B, "Post odor - Odor B"),
        (Cues.NoOdor, "Pre odor"),
        (Cues.OdorA, "Odor A"),
        (Cues.OdorB, "Odor B"),
    ]
)


class ActionSpace:
    def __init__(self):
        self.action_space = set([item.value for item in Actions])

    def __call__(self):
        return self.action_space

    def sample(self):
        return np.random.choice(list(self.action_space))


class Environment:
    """Environment logic."""

    def __init__(self, params, rng=None):
        if rng:
            self.rng = rng
        self.rows = 5
        self.cols = 5
        self.tiles_locations = set(np.arange(self.rows * self.cols))
        # self.cues = [*LightCues, *OdorID]

        # self.action_space = set([item.value for item in Actions])
        self.action_space = ActionSpace()
        self.numActions = len(self.action_space())

        self.state_space = {
            "location": self.tiles_locations,
            # "cue": set(OdorID).union(LightCues),
            "cue": set(Cues),
        }
        self.numStates = tuple(len(item) for item in self.state_space.values())
        self.reset()

    def reset(self):
        """Reset the environment."""
        self.TriangleState = np.random.choice(TriangleState)
        start_state = {
            "location": np.random.choice(self.get_allowed_tiles()),
            # "cue": np.random.choice(LightCues),
            "cue": Cues.NoOdor,
        }
        self.odor_condition = OdorCondition.pre
        self.odor_ID = Cues(
            np.random.choice([item.value for item in Cues if item.name != "NoOdor"])
        )
        return start_state

    def get_allowed_tiles(self):
        """Get allowed tiles based on `TriangleState`"""
        squared_tiles_array = np.array(list(self.tiles_locations)).reshape(
            (self.rows, self.cols)
        )
        if self.TriangleState == TriangleState.lower:
            allowed_tiles = np.tri(self.rows, self.cols, 0).astype(bool)
        elif self.TriangleState == TriangleState.upper:
            allowed_tiles = np.tri(self.rows, self.cols, 0).astype(bool).T
        else:
            raise ValueError("Impossible value, must be `upper` or `lower`")
        allowed_tiles = squared_tiles_array[allowed_tiles == True]
        return allowed_tiles

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
                state["cue"] == Cues.OdorA
                and state["location"] == Ports.West.value
                or state["cue"] == Cues.OdorB
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

        # Update internal states
        # if new_state["location"] == new_state["cue"].value:
        if new_state["location"] in {item.value for item in OdorPorts}:
            self.odor_condition = OdorCondition.post
            new_state["cue"] = self.odor_ID

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
        # if a == Actions.LEFT.value:
        #     col = max(col - 1, 0)
        # elif a == Actions.DOWN.value:
        #     row = min(row + 1, self.rows - 1)
        # elif a == Actions.RIGHT.value:
        #     col = min(col + 1, self.cols - 1)
        # elif a == Actions.UP.value:
        #     row = max(row - 1, 0)
        row_max, col_max = self.wall(row, col, a)
        if a == Actions.LEFT.value:
            col = max(col - 1, col_max)
        elif a == Actions.DOWN.value:
            row = min(row + 1, row_max)
        elif a == Actions.RIGHT.value:
            col = min(col + 1, col_max)
        elif a == Actions.UP.value:
            row = max(row - 1, row_max)
        return (row, col)

    def wall(self, row, col, a):
        """Get wall according to current location and next action."""
        if self.TriangleState == self.TriangleState.upper:
            if a == Actions.LEFT.value:
                col = row
            elif a == Actions.DOWN.value:
                row = col
            elif a == Actions.RIGHT.value:
                col = self.cols - 1
            elif a == Actions.UP.value:
                row = 0

        elif self.TriangleState == self.TriangleState.lower:
            if a == Actions.LEFT.value:
                col = 0
            elif a == Actions.DOWN.value:
                row = self.rows - 1
            elif a == Actions.RIGHT.value:
                col = row
            elif a == Actions.UP.value:
                row = col

        else:
            raise ValueError("Impossible value, must be `upper` or `lower`")
        return (row, col)


class WrappedEnvironment(Environment):
    """
    Wrap the base Environment class.

    Results in numerical only and flattened state space
    """

    def __init__(self, params, rng=None):
        # Initialize the base class to get the base properties
        super().__init__(params, rng=None)

        self.state_space = set(
            # np.arange(self.rows * self.cols * len(LightCues) * len(OdorCondition))
            np.arange(self.rows * self.cols * len(Cues))
        )
        self.numStates = len(self.state_space)
        self.reset()

    def convert_composite_to_flat_state(self, state):
        """Convert composite state dictionary to a flat single number."""
        conv_state = None
        tiles_num = len(self.tiles_locations)

        # if self.odor_condition == OdorCondition.pre:
        #     if state["cue"] == LightCues.North:
        #         conv_state = state["location"]
        #     elif state["cue"] == LightCues.South:
        #         conv_state = state["location"] + tiles_num
        # elif self.odor_condition == OdorCondition.post:
        #     if state["cue"] == OdorID.A:
        #         conv_state = state["location"] + 2 * tiles_num
        #     elif state["cue"] == OdorID.B:
        #         conv_state = state["location"] + 3 * tiles_num
        if state["cue"] == Cues.NoOdor:
            conv_state = state["location"] + 0 * tiles_num
        elif state["cue"] == Cues.OdorA:
            conv_state = state["location"] + 1 * tiles_num
        elif state["cue"] == Cues.OdorB:
            conv_state = state["location"] + 2 * tiles_num

        if conv_state is None:
            raise ValueError("Impossible value for composite state")

        return conv_state

    def convert_flat_state_to_composite(self, state):
        """Convert back flattened state to original composite state."""
        tiles_num = len(self.tiles_locations)
        # if state >= 3 * tiles_num and state < 4 * tiles_num:
        #     conv_state = {
        #         "location": state - 3 * tiles_num,
        #         "cue": OdorID.B,
        #     }
        # elif state >= 2 * tiles_num and state < 3 * tiles_num:
        #     conv_state = {
        #         "location": state - 2 * tiles_num,
        #         "cue": OdorID.A,
        #     }
        # elif state >= tiles_num and state < 2 * tiles_num:
        #     conv_state = {"location": state - tiles_num, "cue": LightCues.South}
        # elif state >= 0 and state < tiles_num:
        #     conv_state = {"location": state, "cue": LightCues.North}
        if state >= 2 * tiles_num and state < 3 * tiles_num:
            conv_state = {
                "location": state - 2 * tiles_num,
                "cue": Cues.OdorB,
            }
        elif state >= 1 * tiles_num and state < 2 * tiles_num:
            conv_state = {
                "location": state - 1 * tiles_num,
                "cue": Cues.OdorA,
            }
        elif state >= 0 and state < 1 * tiles_num:
            conv_state = {"location": state, "cue": Cues.NoOdor}
        else:
            raise ValueError("Impossible number for flat state")
        return conv_state

    def step(self, action, current_state):
        """Wrapper around the base method."""
        current_conv_state = self.convert_flat_state_to_composite(current_state)
        new_state, reward, done = super().step(action, current_conv_state)
        new_state_conv = self.convert_composite_to_flat_state(new_state)
        return new_state_conv, reward, done

    def reset(self):
        """Wrapper around the base method."""
        state = super().reset()
        conv_state = self.convert_composite_to_flat_state(state)
        return conv_state
