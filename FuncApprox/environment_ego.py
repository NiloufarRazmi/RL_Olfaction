from collections import OrderedDict
from enum import Enum

import numpy as np


class OdorCondition(Enum):
    pre = 1
    post = 2


class OdorID(Enum):
    A = 1
    B = 2


class Ports(Enum):
    North = 4
    South = 20
    West = 0
    East = 24


class LightCues(Enum):
    North = Ports.North.value
    South = Ports.South.value


class Actions(Enum):
    FORWARD = 0
    RIGHT = 1
    LEFT = 2


CONTEXTS_LABELS = OrderedDict(
    [
        (LightCues.North, "Pre odor - North light"),
        (LightCues.South, "Pre odor - South light"),
        (OdorID.A, "Post odor - Odor A"),
        (OdorID.B, "Post odor - Odor B"),
    ]
)


class Environment:
    """Environment logic."""

    def __init__(self, params, rng=None):
        if rng:
            self.rng = rng
        self.rows = 5
        self.cols = 5
        self.tiles_locations = set(np.arange(self.rows * self.cols))
        self.head_angle_space = [0, 90, 180, 270]  # In degrees, 0° being north
        self.cues = [*LightCues, *OdorID]

        self.action_space = set([item.value for item in Actions])
        self.numActions = len(self.action_space)

        self.state_space = {
            "location": self.tiles_locations,
            "direction": self.head_angle_space,
            "cue": self.cues,
        }
        self.numStates = tuple(len(item) for item in self.state_space.values())
        self.reset()

    def reset(self):
        """Reset the environment."""
        start_state = {
            "location": np.random.randint(
                low=min(self.tiles_locations),
                high=max(self.tiles_locations) + 1,
            ),
            "direction": np.random.choice(self.head_angle_space),
            "cue": np.random.choice(LightCues),
        }
        self.odor_condition = OdorCondition.pre
        self.odor_ID = np.random.choice(OdorID)
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
            if state["cue"] == OdorID.A and state["location"] == Ports.West.value:
                reward = 10
            elif state["cue"] == OdorID.B and state["location"] == Ports.East.value:
                reward = 10
        return reward

    def step(self, action, current_state):
        """Take an action, observe reward and the next state."""
        new_state = {}
        new_state["cue"] = current_state["cue"]
        row, col = self.to_row_col(current_state)
        newrow, newcol, new_state["direction"] = self.move(
            row, col, action, current_state["direction"]
        )
        new_state["location"] = self.to_state_location(newrow, newcol)

        # Update internal states
        if new_state["location"] == new_state["cue"].value:
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

    def move(self, row, col, action, head_direction):
        """Where the agent ends up on the map."""

        def LEFT(col):
            "Moving left in the allocentric sense."
            col = max(col - 1, 0)
            angle = 270
            return angle, col

        def DOWN(row):
            "Moving down in the allocentric sense."
            row = min(row + 1, self.rows - 1)
            angle = 180
            return angle, row

        def RIGHT(col):
            "Moving right in the allocentric sense."
            col = min(col + 1, self.cols - 1)
            angle = 90
            return angle, col

        def UP(row):
            "Moving up in the allocentric sense."
            row = max(row - 1, 0)
            angle = 0
            return angle, row

        if head_direction == 0:  # Facing north
            if action == Actions.LEFT.value:
                # head_direction, col = LEFT(col)
                head_direction = 270
            elif action == Actions.RIGHT.value:
                # head_direction, col = RIGHT(col)
                head_direction = 90
            elif action == Actions.FORWARD.value:
                head_direction, row = UP(row)

        elif head_direction == 90:  # Facing east
            if action == Actions.LEFT.value:
                # head_direction, row = UP(row)
                head_direction = 0
            elif action == Actions.RIGHT.value:
                # head_direction, row = DOWN(row)
                head_direction = 180
            elif action == Actions.FORWARD.value:
                head_direction, col = RIGHT(col)

        elif head_direction == 180:  # Facing south
            if action == Actions.LEFT.value:
                # head_direction, col = RIGHT(col)
                head_direction = 90
            elif action == Actions.RIGHT.value:
                # head_direction, col = LEFT(col)
                head_direction = 270
            elif action == Actions.FORWARD.value:
                head_direction, row = DOWN(row)

        elif head_direction == 270:  # Facing west
            if action == Actions.LEFT.value:
                # head_direction, row = DOWN(row)
                head_direction = 180
            elif action == Actions.RIGHT.value:
                # head_direction, row = UP(row)
                head_direction = 0
            elif action == Actions.FORWARD.value:
                head_direction, col = LEFT(col)

        return (row, col, head_direction)


class WrappedEnvironment(Environment):
    """Wrap the base Environment class.

    Results in numerical only and flattened state space"""

    def __init__(self, params, rng=None):
        # Initialize the base class to get the base properties
        super().__init__(params, rng=None)

        self.state_space = set(
            np.arange(
                self.rows
                * self.cols
                * len(self.head_angle_space)
                * len(LightCues)
                * len(OdorCondition)
            )
        )
        self.numStates = len(self.state_space)
        self.reset()

    def convert_composite_to_flat_state(self, state):
        """Convert composite state dictionary to a flat single number."""
        conv_state = None
        tiles_num = len(self.tiles_locations)

        if self.odor_condition == OdorCondition.pre:
            if state["cue"] == LightCues.North:
                if state["direction"] == 0:
                    conv_state = state["location"] + 0 * tiles_num
                elif state["direction"] == 90:
                    conv_state = state["location"] + 1 * tiles_num
                elif state["direction"] == 180:
                    conv_state = state["location"] + 2 * tiles_num
                elif state["direction"] == 270:
                    conv_state = state["location"] + 3 * tiles_num
            elif state["cue"] == LightCues.South:
                if state["direction"] == 0:
                    conv_state = state["location"] + 4 * tiles_num
                elif state["direction"] == 90:
                    conv_state = state["location"] + 5 * tiles_num
                elif state["direction"] == 180:
                    conv_state = state["location"] + 6 * tiles_num
                elif state["direction"] == 270:
                    conv_state = state["location"] + 7 * tiles_num
        elif self.odor_condition == OdorCondition.post:
            if state["cue"] == OdorID.A:
                if state["direction"] == 0:
                    conv_state = state["location"] + 8 * tiles_num
                elif state["direction"] == 90:
                    conv_state = state["location"] + 9 * tiles_num
                elif state["direction"] == 180:
                    conv_state = state["location"] + 10 * tiles_num
                elif state["direction"] == 270:
                    conv_state = state["location"] + 11 * tiles_num
            elif state["cue"] == OdorID.B:
                if state["direction"] == 0:
                    conv_state = state["location"] + 12 * tiles_num
                elif state["direction"] == 90:
                    conv_state = state["location"] + 13 * tiles_num
                elif state["direction"] == 180:
                    conv_state = state["location"] + 14 * tiles_num
                elif state["direction"] == 270:
                    conv_state = state["location"] + 15 * tiles_num

        if conv_state is None:
            raise ValueError("Impossible value for composite state")

        return conv_state

    def get_states_structure(self):
        """Returns a human readable states dictionnary."""
        states = OrderedDict()
        count = 0
        for idx, cue in enumerate(self.cues):
            states[cue] = OrderedDict()
            for jdx, angle in enumerate(self.head_angle_space):
                states[cue][angle] = np.arange(
                    (idx + jdx + count) * self.rows * self.cols,
                    (idx + jdx + count + 1) * self.rows * self.cols,
                )
            count += jdx
        return states

    def convert_flat_state_to_composite(self, state):
        """Convert back flattened state to original composite state."""
        states_struct = self.get_states_structure()
        conv_state = None
        for cue in states_struct.keys():
            for angle in states_struct[cue].keys():
                if np.isin(states_struct[cue][angle], state).any():
                    conv_state = {
                        "location": np.argwhere(
                            states_struct[cue][angle] == state
                        ).flatten()[0],
                        "direction": angle,
                        "cue": cue,
                    }
                    break

        if conv_state is None:
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
