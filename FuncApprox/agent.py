import numpy as np
import numpy.matlib


class Qlearning:
    def __init__(self, learning_rate, gamma, state_size, action_size):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.qtable = np.zeros((state_size, action_size))

    def update(self, state, action, reward, new_state):
        """Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]"""
        delta = (
            reward
            + self.gamma * np.max(self.qtable[new_state, :])
            - self.qtable[state, action]
        )
        q_update = self.qtable[state, action] + self.learning_rate * delta
        return q_update


class QLearningFuncApprox:
    def __init__(self, learning_rate, gamma, state_size, action_size, jointRep):
        self.learning_rate = learning_rate
        self.gamma = gamma
        if jointRep:
            self.features = np.eye(state_size * action_size, state_size * action_size)
        else:
            tmp = np.eye(state_size, state_size)
            self.features = np.matlib.repmat(tmp, action_size, action_size)
        self.weights = np.zeros((self.features.shape[0], action_size))
        self.Q_hat_table = np.zeros((self.weights.shape))

    def Q_hat(self, weights, features):
        """Compute the approximated Q-value."""
        Q_hat = (weights.T @ features).T
        return Q_hat

    def update_weights(self, state, action, reward, new_state):
        """Update the weights."""
        delta = (
            reward
            + self.gamma * np.max(self.Q_hat_table[new_state, :])
            - self.Q_hat_table[state, action]
        )
        weights_update = (
            self.weights[:, action].T
            + self.learning_rate * delta * self.features[:, state]
        )
        return weights_update


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
