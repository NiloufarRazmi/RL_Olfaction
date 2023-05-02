from enum import Enum

import numpy as np


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
        self.observation_space = np.arange(0, 7)
        self.action_space = [item.value for item in list(Actions)]
        self.right_boundary = 6
        self.left_boundary = 0
        self.numActions = len(Actions)
        self.reset()

    def reset(self):
        self.current_state = 3
        return self.current_state

    def step(self, action):
        if action == Actions.Left.value:
            new_state = np.max([self.left_boundary, self.current_state - 1])
        elif action == Actions.Right.value:
            new_state = np.min([self.right_boundary, self.current_state + 1])
        else:
            raise ValueError("Impossible action type")
        self.current_state = new_state
        reward = self.reward(self.current_state)
        is_terminated = self.is_terminated(self.current_state)
        return new_state, reward, is_terminated

    def reward(self, observation):
        reward = 0
        if observation == self.right_boundary:
            reward = 1
        elif observation == self.left_boundary:
            reward = -1
        return reward

    def is_terminated(self, observation):
        is_terminated = False
        if observation == self.right_boundary or observation == self.left_boundary:
            is_terminated = True
        return is_terminated
