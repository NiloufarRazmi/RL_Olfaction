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


class Actions(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class Environment:
    """Environment logic."""

    def __init__(self, params, rng=None):
        if rng:
            self.rng = rng
        # self.transitionList = buildGrid(params, self.start)
        self.rows = 5
        self.cols = 5
        self.numStates = self.rows * self.cols
        self.state_space = set(np.arange(self.numStates))
        self.action_space = set([item.value for item in Actions])
        self.numActions = len(self.action_space)
        # wallsLoc = [1, 2, 4, 5, 6, 16, 21, 22, 24, 25, 20, 10]
        self.reset()

    def reset(self):
        """Reset the environment."""
        self.start = np.random.randint(
            low=min(self.state_space), high=max(self.state_space)
        )
        self.odor_condition = OdorCondition.pre
        self.cue_port = np.random.choice([Ports.North, Ports.South])
        return self.start

    def is_terminated(self, state):
        """Returns if the episode is terminated or not."""
        if self.odor_condition == OdorCondition.post and (
            state == Ports.West.value or state == Ports.East.value
        ):
            return True
        else:
            return False

    def reward(self, state):
        """Observe the reward."""
        reward = 0
        if self.odor_condition == OdorCondition.post:
            if self.cue_port == Ports.North and state == Ports.West.value:
                reward = 10
            elif self.cue_port == Ports.South and state == Ports.East.value:
                reward = 10
        return reward

    def step(self, action, current_state):
        """Take an action, observe reward and the next state."""
        # new_state = self.transitionList[prevState, action]
        row, col = self.to_row_col(current_state)
        newrow, newcol = self.move(row, col, action)
        new_state = self.to_state(newrow, newcol)

        # Update odor condition
        if new_state == self.cue_port.value:
            self.odor_condition = OdorCondition.post

        reward = self.reward(new_state)
        done = self.is_terminated(new_state)
        return new_state, reward, done

    def to_state(self, row, col):
        """Convenience function to convert row and column to state number."""
        return row * self.cols + col

    def to_row_col(self, state):
        """Convenience function to convert state to row and column."""
        states = np.reshape(list(self.state_space), (self.rows, self.cols))
        (row, col) = np.argwhere(states == state).flatten()
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
