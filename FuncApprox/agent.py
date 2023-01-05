import numpy as np


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
