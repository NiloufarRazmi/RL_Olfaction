import torch
import torch.nn as nn
from utils import make_deterministic, random_choice


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EpsilonGreedy:
    def __init__(
        self,
        epsilon,
        epsilon_min=0.1,
        epsilon_max=1.0,
        decay_rate=0.05,
        epsilon_warmup=25,
        seed=None,
    ):
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.decay_rate = decay_rate
        self.epsilon_warmup = epsilon_warmup
        if seed:
            self.generator = make_deterministic(seed=seed)
        else:
            self.generator = None

    def choose_action(self, action_space, state, state_action_values):
        """Choose an action a in the current world state (s)"""

        def sample(action_space, generator=None):
            return random_choice(action_space, generator=self.generator)

        # # First we randomize a number
        explor_exploit_tradeoff = torch.rand(1, generator=self.generator)

        # Exploration
        if explor_exploit_tradeoff.item() < self.epsilon:
            # action = action_space.sample()
            action = sample(action_space)

        # Exploitation (taking the biggest Q-value for this state)
        else:
            # Break ties randomly
            # If all actions are the same for this state we choose a random one
            # (otherwise `argmax()` would always take the first one)
            if torch.all(state_action_values == state_action_values[0]):
                action = sample(action_space)
            else:
                action = torch.argmax(state_action_values)
        return action

    def update_epsilon(self, ep):
        if ep > self.epsilon_warmup:
            """Reduce epsilon (because we need less and less exploration)"""
            epsilon = (
                self.epsilon_min
                + (self.epsilon_max - self.epsilon_min)
                * torch.exp(-self.decay_rate * (ep - self.epsilon_warmup)).item()
            )
        else:
            epsilon = self.epsilon
        return epsilon


ENCODER_NEURONS_NUM = 5


class DQN(nn.Module):
    """Define network."""

    def __init__(self, n_observations, n_actions, n_units=16):
        super(DQN, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_observations, n_units),
            nn.ReLU(),
            nn.Linear(n_units, n_units),
            nn.ReLU(),
            nn.Linear(n_units, ENCODER_NEURONS_NUM),
            nn.ReLU(),
            nn.Linear(ENCODER_NEURONS_NUM, n_units),
            nn.ReLU(),
            nn.Linear(n_units, n_units),
            nn.ReLU(),
            nn.Linear(n_units, n_actions),
            # nn.ReLU(),
        )

    def forward(self, x):
        return self.mlp(x)


def neural_network(n_observations, n_actions, nHiddenUnits):
    """Define policy and target networks."""
    # if env.one_hot_state:
    #     net = DQN(
    #         n_observations=n_observations,
    #         n_actions=n_actions,
    #         n_units=4 * n_observations,
    #     ).to(DEVICE)
    # else:
    #     net = DQN(
    #         n_observations=n_observations,
    #         n_actions=n_actions,
    #         n_units=nHiddenUnits,
    #     ).to(DEVICE)
    # net

    policy_net = DQN(
        n_observations=n_observations,
        n_actions=n_actions,
        n_units=nHiddenUnits,
    ).to(DEVICE)

    target_net = DQN(
        n_observations=n_observations,
        n_actions=n_actions,
        n_units=nHiddenUnits,
    ).to(DEVICE)

    target_net.load_state_dict(policy_net.state_dict())

    return policy_net, target_net
