import numpy as np
from dataclasses import dataclass


@dataclass
class Params:
    """Container class to keep track of all hyperparameters."""

    epsilon = 0.1  # Action-selection parameters

    # QLearning parameters
    gamma = 0.8
    alpha = 0.05

    n_runs = 30
    numEpisodes = 200  # Set up the task

    # TODO: Move in the environment class
    rows = 5
    cols = 5
    numStates = rows * cols
    states = np.arange(numStates)
    actions = ["U", "D", "L", "R"]
    numActions = len(actions)
    wallsLoc = [1, 2, 4, 5, 6, 16, 21, 22, 24, 25, 20, 10]


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
        # First we randomize a number
        if hasattr(self, "rng"):
            explor_exploit_tradeoff = self.rng.uniform(0, 1)
        else:
            explor_exploit_tradeoff = np.random.uniform(0, 1)

        # Exploration
        if explor_exploit_tradeoff < self.epsilon:
            action = action_space.sample()

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


def createTransitionList(params):
    """CREATETRANSITIONLIST takes in params, a matrix of states, and whether
    there are walls (true or false). The output is a state action state
    transition list. Each row is a state and the columns are the resulting
    states of taking the four actions (0 = 'U', 1 = 'D', 2 = 'R', 3 = 'L').
    """

    transitionList = np.zeros((params.numStates, params.numActions))
    states = np.reshape(params.states, (params.rows, params.cols))
    for state in states.flatten():
        (row, col) = np.argwhere(states == state).flatten()

        # Up column
        if row == 0:
            transitionList[state, 0] = state
        else:
            transitionList[state, 0] = states[row - 1, col]

        # Down column
        if row == params.rows:
            transitionList[state, 1] = state
        else:
            transitionList[state, 1] = states[row, col]

        # Right column
        if col == params.cols:
            transitionList[state, 2] = state
        else:
            transitionList[state, 2] = states[row, col]

        # Left column
        if col == 0 or col == params.cols + 1:
            transitionList[state, 3] = state
        else:
            transitionList[state, 3] = states[row, col - 1]

    return transitionList


def buildGrid(params, start):
    """BUILDGRID builds the grid for gridworld using the params and the goal.
    It outputs a gridworld, matrix of Q values, a matrix of states,
    the goal state, and a transitionList.
    """

    # Create transition list
    transitionList = createTransitionList(params)
    tmp = transitionList

    # TODO: replace hard coded numbers by variables
    transitionList = np.vstack((transitionList, tmp + 25, tmp + 50, tmp + 75))

    # If odor is on top:
    if start < 26:  # TODO: replace hard coded number by a variable
        if np.random.rand() < 0.5:
            # Go to the context with rewarding state on the right
            transitionList[11, 1] = 51
        else:
            # Go to the context with rewarding state on the left
            transitionList[11, 1] = 76
    else:
        if np.random.rand() < 0.5:
            transitionList[40, 2] = 51
        else:
            transitionList[40, 2] = 76

    return transitionList


class Environment:
    """Environment logic."""

    def __init__(self, params, rng=None):
        if rng:
            self.rng = rng
        self.reset()
        self.transitionList = buildGrid(params, self.start)

    def reset(self):
        """Reset the environment."""
        self.start = np.random.randint(
            low=1, high=50
        )  # TODO: replace hard coded number by a variable
        return self.start

    def is_terminated(self, prevState, action):
        """Returns if the episode is terminated or not."""

        # TODO: replace hard coded numbers by variables
        if prevState == 73 and action == 3:
            return True
        elif prevState == 78 and action == 4:
            return True
        else:
            return False

    def reward(self, prevState, action):
        """Update the reward."""
        # Observe reward -- curently there are two rewarding actions
        # based on the odor location (but they are not simulatneously present)
        # TODO: replace hard coded numbers by variables
        if prevState == 73 and action == 3:
            reward = 10
        elif prevState == 78 and action == 4:
            reward = 10
        else:
            reward = 0
        return reward

    def step(self, action, prevState):
        """Take an action, observe reward and the next state."""
        reward = self.reward(prevState, action)
        new_state = self.transitionList[prevState, action]
        return new_state, reward
