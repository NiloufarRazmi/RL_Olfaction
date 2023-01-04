from dataclasses import dataclass
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


@dataclass
class Params:
    """Container class to keep track of all hyperparameters."""

    epsilon = 0.1  # Action-selection parameters

    # QLearning parameters
    gamma = 0.8
    alpha = 0.05

    n_runs = 5
    numEpisodes = 100  # Set up the task


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


class Qlearning:
    def __init__(self, learning_rate, gamma, state_size, action_size):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.qtable = np.zeros((state_size, action_size))

    def update(self, state, action, reward, new_state):
        """Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]"""
        q_updated = self.qtable[state, action] + self.learning_rate * (
            reward
            + self.gamma * np.max(self.qtable[new_state, :])
            - self.qtable[state, action]
        )
        return q_updated


class EpsilonGreedy:
    def __init__(
        self,
        epsilon,
        epsilon_min=0.1,
        epsilon_max=1.0,
        decay_rate=0.05,
        epsilon_warmup=25,
        rng=None,
    ):
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.decay_rate = decay_rate
        self.epsilon_warmup = epsilon_warmup
        if rng:
            self.rng = rng

    def choose_action(self, action_space, state, qtable):
        """Choose an action a in the current world state (s)"""

        def sample(action_space):
            return np.random.choice(list(action_space))

        # First we randomize a number
        if hasattr(self, "rng"):
            explor_exploit_tradeoff = self.rng.uniform(0, 1)
        else:
            explor_exploit_tradeoff = np.random.uniform(0, 1)

        # Exploration
        if explor_exploit_tradeoff < self.epsilon:
            # action = action_space.sample()
            action = sample(action_space)

        # Exploitation (taking the biggest Q-value for this state)
        else:
            action = np.argmax(qtable[state, :])
        return action

    def update_epsilon(self, ep):
        if ep > self.epsilon_warmup:
            """Reduce epsilon (because we need less and less exploration)"""
            epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * np.exp(
                -self.decay_rate * (ep - self.epsilon_warmup)
            )
        else:
            epsilon = self.epsilon
        return epsilon


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
        if self.odor_condition == OdorCondition.post and state == Ports.West.value:
            reward = 10
        else:
            reward = 0
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


def plot_steps_and_rewards(df1, df2):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    # sns.lineplot(data=df1, x="Episodes", y="cum_rewards", hue="map_size", ax=ax[0])
    # ax[0].set(ylabel="Cummulated rewards")
    sns.lineplot(data=df1, x="Episodes", y="cum_rewards", hue="map_size", ax=ax[0])
    ax[0].set(ylabel="Cumulated rewards")

    sns.lineplot(data=df2, x="Episodes", y="Steps", hue="map_size", ax=ax[1])
    ax[1].set(ylabel="Averaged steps number")

    for axi in ax:
        axi.legend(title="map size")
    plt.show()
