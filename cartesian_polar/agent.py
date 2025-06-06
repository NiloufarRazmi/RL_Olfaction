"""Agent related routines."""

import torch
import torch.nn as nn

from .utils import make_deterministic, random_choice

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EpsilonGreedy:
    """Define the Epsilon-greedy algorithm to chose an action."""

    """
    - The algorithm balances exploration and exploitation when choosing actions
    
    - 1. Generate random number
    - 2. If number is less than epsilon, choose a random action (explore)
    - 3. Otherwise, choose best action (highest Q-value)

    - We decay epsilon over time, encouraging early exploration and late exploitation
    """

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

    """
    action_space : space of possible actions
    state_action_values : Q-values associated with each action in the action space
    """
    def choose_action(self, action_space, state, state_action_values):
        """Choose an action a in the current world state (s)."""

        def sample(action_space, generator=None):
            return random_choice(action_space, generator=self.generator)

        # # First we randomize a number
        explor_exploit_tradeoff = torch.rand(1, generator=self.generator)

        # Exploration
        if explor_exploit_tradeoff.item() < self.epsilon:
            # action = action_space.sample()
            action = sample(action_space) # Random action from action space

        # Exploitation (taking the biggest Q-value for this state)
        else:
            # Break ties randomly
            # If all actions have the same Q-values for this state we choose a random one
            # (otherwise `argmax()` would always take the first one)
            if torch.all(state_action_values == state_action_values[0]):
                action = sample(action_space)
            else:
                action = torch.argmax(state_action_values) # Take action with highest Q-value
        return action

    """
    Function to decay epsilon over time (episodes)

    epsilon_warmup : number of episodes to wait before beginning decay
    epsilon_min : a floor to prevent the agent becoming fully greedy and getting stuck in local optima

    Decays according to classic exponential decay, a smooth reduction

    !!! POINT OF EXPERIMENTATION !!! Play around with epsilon values and decay? What makes most sense for the mouse?
    """
    def update_epsilon(self, ep):
        """Reduce epsilon after `ep` episodes."""
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


"""Reinforcement Learning (DQN) Model Architecture"""
class DQN(nn.Module):
    """Define network."""

    """
    n_observations -- sensory input, in this case Cue, Cartesian, and Polar coordinates
    n_actions -- actions the agent can take, in this case Forward, Left, Right
    n_units -- number of hidden neurons
    """
    def __init__(self, n_observations, n_actions, n_units=16):
        super().__init__()
        """
        Network Architecture: 6 layers of neurons, 4 hidden layers
        """
        self.mlp = nn.Sequential(
            nn.Linear(n_observations, n_units), 
            nn.Linear(n_units, n_units),
            nn.ReLU(),
            nn.Linear(n_units, n_units),
            nn.ReLU(),
            nn.Linear(n_units, n_units),
            nn.ReLU(),
            nn.Linear(n_units, n_actions),
        )

    def forward(self, x):
        """Define the forward pass."""
        return self.mlp(x)


def neural_network(n_observations, n_actions, nHiddenUnits):
    """Define policy and target networks."""

    """
    Policy Network : network agent uses to interact with environment and learn
    Target Network : copy of policy, but updated less frequently
                     - used to compute target Q-values, i.e. a stable snapshot I can use to estimate future rewards
                     - helps stabilize training
    """
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
