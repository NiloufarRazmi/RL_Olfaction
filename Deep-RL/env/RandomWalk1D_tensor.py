from enum import Enum

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Actions(Enum):
    Left = 0
    Right = 1


class RandomWalk1D:
    """`RandomWalk1D` to test the Q-learning algorithm.

    The agent (A) starts in state 3.
    The actions it can take are going left or right.
    The episode ends when it reaches state 0 or 6.
    When it reaches state 0, it gets a reward of -1,
    when it reaches state 6, it gets a reward of +1.
    At any other state it gets a reward of zero.

    Rewards:  -1   <-A->  +1
    States:  <-0-1-2-3-4-5-6->
    """

    def __init__(self):
        self.observation_space = torch.arange(0, 7, device=DEVICE)
        self.action_space = torch.tensor(
            [item.value for item in list(Actions)], device=DEVICE
        )
        self.right_boundary = 6
        self.left_boundary = 0
        self.numActions = self.action_space.shape[0]
        self.reset()

    def reset(self):
        self.current_state = 3
        return torch.tensor([self.current_state], device=DEVICE)

    def step(self, action):
        if action == Actions.Left.value:
            new_state = (
                torch.tensor(
                    [self.left_boundary, self.current_state - 1], device=DEVICE
                )
                .max()
                .unsqueeze(-1)
            )
        elif action == Actions.Right.value:
            new_state = (
                torch.tensor(
                    [self.right_boundary, self.current_state + 1], device=DEVICE
                )
                .min()
                .unsqueeze(-1)
            )
        else:
            raise ValueError("Impossible action type")
        self.current_state = new_state
        reward = self.reward(self.current_state)
        is_terminated = self.is_terminated(self.current_state)
        return new_state, reward, is_terminated

    def reward(self, observation):
        reward = 0
        if observation.item() == self.right_boundary:
            reward = 1
        elif observation.item() == self.left_boundary:
            reward = -1
        return reward

    def is_terminated(self, observation):
        is_terminated = False
        if observation == self.right_boundary or observation == self.left_boundary:
            is_terminated = True
        return is_terminated
