"""Tests on the environment."""

import pytest
import torch
from tensordict.tensordict import TensorDict

from .agent import EpsilonGreedy
from .cartesian_polar_env import (
    Actions,
    Cues,
    DuplicatedCoordsEnv,
    Environment,
    OdorCondition,
    TaskID,
    TriangleState,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test_env_general_props():
    """Test general properties of the environment."""
    env = Environment()
    assert env
    state = env.reset()
    assert env.numActions == 4
    # assert len(state) == 9
    assert isinstance(state, TensorDict)
    assert state.shape == torch.Size([1])
    assert state["cue"].shape == torch.Size([1])
    assert state["x"].shape == torch.Size([1])
    assert state["y"].shape == torch.Size([1])
    assert state["direction"].shape == torch.Size([1])
    assert env.odor_condition == OdorCondition.pre


def test_starting_point_coords():
    """Test that the starting points are in the allwed boundaries of the arena."""
    env = Environment()
    for _ in range(100):
        agent_coords = env.sample_coord_position()
        assert agent_coords[0] >= env.rangeX["min"]
        assert agent_coords[0] <= env.rangeX["max"]
        assert agent_coords[1] >= env.rangeX["min"]
        assert agent_coords[0] <= env.rangeX["max"]

        # Check tha there is no decimals
        for coord in agent_coords:
            split_nb = str(float(coord)).split(".")
            if len(split_nb) > 1:  # if there are decimals
                assert split_nb[1] == str(0)  # can only be zero


@pytest.mark.parametrize(
    "x_orig, y_orig, direction_orig, TriangleState, action, "
    "x_expect, y_expect, direction_expect",
    [
        (
            0,
            0,
            180,
            TriangleState.upper,
            Actions.right.value,
            0,
            0,
            270,
        ),
        (
            0,
            0,
            270,
            TriangleState.upper,
            Actions.left.value,
            0,
            0,
            180,
        ),
        (
            2,
            1,
            0,
            TriangleState.upper,
            Actions.right.value,
            2,
            1,
            90,
        ),
        (
            0,
            2,
            0,
            TriangleState.upper,
            Actions.forward.value,
            0,
            2,
            0,
        ),
        (
            0,
            0,
            0,
            TriangleState.lower,
            Actions.right.value,
            0,
            0,
            90,
        ),
        (
            0,
            0,
            90,
            TriangleState.lower,
            Actions.left.value,
            0,
            0,
            0,
        ),
        (
            -2,
            -1,
            0,
            TriangleState.lower,
            Actions.left.value,
            -2,
            -1,
            270,
        ),
        (
            0,
            -2,
            90,
            TriangleState.lower,
            Actions.right.value,
            0,
            -2,
            180,
        ),
    ],
    ids=[
        "move left from diagonal - upper triangle",
        "move down from diagonal - upper triangle",
        "move to right wall - upper triangle",
        "move to upper wall - upper triangle",
        "move right from diagonal - lower triangle",
        "move up from diagonal - lower triangle",
        "move to left wall - lower triangle",
        "move to lower wall - lower triangle",
    ],
)
def test_walls(
    x_orig,
    y_orig,
    direction_orig,
    TriangleState,
    action,
    x_expect,
    y_expect,
    direction_expect,
):
    """Test moves against walls."""
    env = Environment()
    env.TriangleState = TriangleState
    state_orig = TensorDict(
        {
            "cue": torch.tensor(Cues.NoOdor.value, device=DEVICE).unsqueeze(-1),
            "x": torch.tensor([x_orig], device=DEVICE),
            "y": torch.tensor([y_orig], device=DEVICE),
            "direction": torch.tensor([direction_orig], device=DEVICE),
        },
        batch_size=[1],
    )
    new_state, _, _ = env.step(action=action, current_state=state_orig)
    state_expected = TensorDict(
        {
            "cue": torch.tensor(Cues.NoOdor.value, device=DEVICE).unsqueeze(-1),
            "x": torch.tensor([x_expect], device=DEVICE),
            "y": torch.tensor([y_expect], device=DEVICE),
            "direction": torch.tensor([direction_expect], device=DEVICE),
        },
        batch_size=[1],
    )
    torch.testing.assert_close(
        actual=new_state,
        expected=state_expected,
    )


@pytest.mark.parametrize(
    "x_orig, y_orig, direction_orig, TriangleState, action, "
    "x_expect, y_expect, direction_expect",
    [
        (
            1,
            1,
            0,
            TriangleState.upper,
            Actions.forward.value,
            1,
            2,
            0,
        ),
        (
            1,
            1,
            0,
            TriangleState.upper,
            Actions.right.value,
            2,
            1,
            90,
        ),
        (
            1,
            1,
            0,
            TriangleState.upper,
            Actions.backward.value,
            1,
            0,
            0,
        ),
        (
            1,
            1,
            0,
            TriangleState.upper,
            Actions.left.value,
            0,
            1,
            270,
        ),
        (
            -1,
            -1,
            180,
            TriangleState.lower,
            Actions.forward.value,
            -1,
            -2,
            180,
        ),
        (
            -1,
            -1,
            180,
            TriangleState.lower,
            Actions.right.value,
            -2,
            -1,
            270,
        ),
        (
            -1,
            -1,
            180,
            TriangleState.lower,
            Actions.backward.value,
            -1,
            -0,
            180,
        ),
        (
            -1,
            -1,
            180,
            TriangleState.lower,
            Actions.left.value,
            -0,
            -1,
            90,
        ),
    ],
)
def test_basic_moves(
    x_orig,
    y_orig,
    direction_orig,
    TriangleState,
    action,
    x_expect,
    y_expect,
    direction_expect,
):
    """Test basic moves."""
    env = Environment()
    env.TriangleState = TriangleState
    state_orig = TensorDict(
        {
            "cue": torch.tensor(Cues.NoOdor.value, device=DEVICE).unsqueeze(-1),
            "x": torch.tensor([x_orig], device=DEVICE),
            "y": torch.tensor([y_orig], device=DEVICE),
            "direction": torch.tensor([direction_orig], device=DEVICE),
        },
        batch_size=[1],
    )
    new_state, _, _ = env.step(action=action, current_state=state_orig)
    state_expected = TensorDict(
        {
            "cue": torch.tensor(Cues.NoOdor.value, device=DEVICE).unsqueeze(-1),
            "x": torch.tensor([x_expect], device=DEVICE),
            "y": torch.tensor([y_expect], device=DEVICE),
            "direction": torch.tensor([direction_expect], device=DEVICE),
        },
        batch_size=[1],
    )
    torch.testing.assert_close(
        actual=new_state,
        expected=state_expected,
    )


@pytest.mark.parametrize(
    "task, triangle, expected_reward",
    [
        (
            TaskID.EastWest,
            TriangleState.upper,
            1,
        ),
        (
            TaskID.EastWest,
            TriangleState.upper,
            0,
        ),
        (
            TaskID.EastWest,
            TriangleState.lower,
            1,
        ),
        (
            TaskID.EastWest,
            TriangleState.lower,
            0,
        ),
        (
            TaskID.LeftRight,
            TriangleState.upper,
            1,
        ),
        (
            TaskID.LeftRight,
            TriangleState.upper,
            0,
        ),
        (
            TaskID.LeftRight,
            TriangleState.lower,
            1,
        ),
        (
            TaskID.LeftRight,
            TriangleState.lower,
            0,
        ),
    ],
    ids=[
        "east/west - upper - solved",
        "east/west - upper - not solved",
        "east/west - lower - solved",
        "east/west - lower - not solved",
        "left/right - upper - solved",
        "left/right - upper - not solved",
        "left/right - lower - solved",
        "left/right - lower - not solved",
    ],
)
def test_env_logic(task, triangle, expected_reward):
    """Test environment logic."""
    env = Environment()
    env.reset()
    env.TaskID = task
    env.TriangleState = triangle
    assert env.odor_condition == OdorCondition.pre

    if triangle == TriangleState.upper:
        x_orig = 2
        y_orig = 1
        direction_orig = 0
    elif triangle == TriangleState.lower:
        x_orig = -2
        y_orig = -1
        direction_orig = 180

    state = TensorDict(
        {
            "cue": torch.tensor(Cues.NoOdor.value, device=DEVICE).unsqueeze(-1),
            "x": torch.tensor([x_orig], device=DEVICE),
            "y": torch.tensor([y_orig], device=DEVICE),
            "direction": torch.tensor([direction_orig], device=DEVICE),
        },
        batch_size=[1],
    )
    state, reward, done = env.step(action=Actions.forward.value, current_state=state)
    assert done is False
    assert env.odor_condition == OdorCondition.post
    assert state["cue"] == Cues.OdorA.value or Cues.OdorB.value

    # Post odor presentation
    if task == TaskID.EastWest:
        if triangle == TriangleState.upper:
            if state["cue"] == Cues.OdorA.value:
                if expected_reward == 1:
                    action = Actions.left.value
                    state, reward, done = env.step(action=action, current_state=state)
                else:
                    action = Actions.right.value
                    state, reward, done = env.step(action=action, current_state=state)
                    state, reward, done = env.step(action=action, current_state=state)
            elif state["cue"] == Cues.OdorB.value:
                if expected_reward == 1:
                    action = Actions.right.value
                    state, reward, done = env.step(action=action, current_state=state)
                    state, reward, done = env.step(action=action, current_state=state)
                else:
                    action = Actions.left.value
                    state, reward, done = env.step(action=action, current_state=state)
        elif triangle == TriangleState.lower:
            if state["cue"] == Cues.OdorA.value:
                if expected_reward == 1:
                    action = Actions.right.value
                    state, reward, done = env.step(action=action, current_state=state)
                    state, reward, done = env.step(action=action, current_state=state)
                else:
                    action = Actions.left.value
                    state, reward, done = env.step(action=action, current_state=state)
            elif state["cue"] == Cues.OdorB.value:
                if expected_reward == 1:
                    action = Actions.left.value
                    state, reward, done = env.step(action=action, current_state=state)
                else:
                    action = Actions.right.value
                    state, reward, done = env.step(action=action, current_state=state)
                    state, reward, done = env.step(action=action, current_state=state)
    elif task == TaskID.LeftRight:
        if triangle == TriangleState.upper:
            if state["cue"] == Cues.OdorA.value:
                if expected_reward == 1:
                    action = Actions.left.value
                    state, reward, done = env.step(action=action, current_state=state)
                else:
                    action = Actions.right.value
                    state, reward, done = env.step(action=action, current_state=state)
                    state, reward, done = env.step(action=action, current_state=state)
            elif state["cue"] == Cues.OdorB.value:
                if expected_reward == 1:
                    action = Actions.right.value
                    state, reward, done = env.step(action=action, current_state=state)
                    state, reward, done = env.step(action=action, current_state=state)
                else:
                    action = Actions.left.value
                    state, reward, done = env.step(action=action, current_state=state)
        elif triangle == TriangleState.lower:
            if state["cue"] == Cues.OdorB.value:
                if expected_reward == 1:
                    action = Actions.right.value
                    state, reward, done = env.step(action=action, current_state=state)
                    state, reward, done = env.step(action=action, current_state=state)
                else:
                    action = Actions.left.value
                    state, reward, done = env.step(action=action, current_state=state)
            elif state["cue"] == Cues.OdorA.value:
                if expected_reward == 1:
                    action = Actions.left.value
                    state, reward, done = env.step(action=action, current_state=state)
                else:
                    action = Actions.right.value
                    state, reward, done = env.step(action=action, current_state=state)
                    state, reward, done = env.step(action=action, current_state=state)

    action = Actions.forward.value
    for _ in range(3):
        state, reward, done = env.step(action=action, current_state=state)

    assert done is True
    assert reward == expected_reward


@pytest.mark.parametrize(
    "coords_orig, coords_north_cart, coords_south_cart, "
    "coords_north_polar, coords_south_polar",
    [
        (
            torch.tensor([0, 0]),
            torch.tensor([2, 2]),
            torch.tensor([2, 2]),
            torch.tensor([2.8, 45]),
            torch.tensor([2.8, 45]),
        ),
        (
            torch.tensor([2, 2]),
            torch.tensor([0, 0]),
            torch.tensor([4, 4]),
            torch.tensor([0, 0]).to(torch.float32),
            torch.tensor([5.7, 45]),
        ),
        (
            torch.tensor([-2, -2]),
            torch.tensor([4, 4]),
            torch.tensor([0, 0]),
            torch.tensor([5.7, 45]),
            torch.tensor([0, 0]).to(torch.float32),
        ),
        (
            torch.tensor([2, -2]),
            torch.tensor([0, 4]),
            torch.tensor([4, 0]),
            torch.tensor([4, 90]).to(torch.float32),
            torch.tensor([4, 0]).to(torch.float32),
        ),
        (
            torch.tensor([-2, 2]),
            torch.tensor([4, 0]),
            torch.tensor([0, 4]),
            torch.tensor([4, 0]).to(torch.float32),
            torch.tensor([4, 90]).to(torch.float32),
        ),
        (
            torch.tensor([3, 3]),
            torch.tensor([-1, -1]),
            torch.tensor([5, 5]),
            torch.tensor([1.4, -135]),
            torch.tensor([7.1, 45]),
        ),
        (
            torch.tensor([-3, -3]),
            torch.tensor([5, 5]),
            torch.tensor([-1, -1]),
            torch.tensor([7.1, 45]),
            torch.tensor([1.4, -135]),
        ),
        (
            torch.tensor([1, -1]),
            torch.tensor([1, 3]),
            torch.tensor([3, 1]),
            torch.tensor([3.2, 72]),
            torch.tensor([3.2, 18]),
        ),
        (
            torch.tensor([-1, 1]),
            torch.tensor([3, 1]),
            torch.tensor([1, 3]),
            torch.tensor([3.2, 18]),
            torch.tensor([3.2, 72]),
        ),
    ],
)
def test_coords_convertions(
    coords_orig,
    coords_north_cart,
    coords_south_cart,
    coords_north_polar,
    coords_south_polar,
):
    """Test the conversion routines."""
    env = DuplicatedCoordsEnv()
    assert torch.equal(
        input=env.conv2north_cartesian(coords_orig), other=coords_north_cart
    )
    assert torch.equal(
        input=env.conv2south_cartesian(coords_orig), other=coords_south_cart
    )
    torch.testing.assert_close(
        actual=env.conv2north_polar(coords_orig),
        expected=coords_north_polar,
        # rtol=1e-5,
        # atol=1e-4,
    )
    torch.testing.assert_close(
        actual=env.conv2south_polar(coords_orig),
        expected=coords_south_polar,
        # rtol=1e-5,
        # atol=1e-4,
    )

    # Test round trip conversion
    assert torch.equal(
        input=env.conv_north_cartesian2orig(env.conv2north_cartesian(coords_orig)),
        other=coords_orig,
    )


@pytest.mark.parametrize(
    "task, triangle, expected_reward",
    [
        (
            TaskID.EastWest,
            TriangleState.upper,
            1,
        ),
        (
            TaskID.EastWest,
            TriangleState.upper,
            0,
        ),
        (
            TaskID.EastWest,
            TriangleState.lower,
            1,
        ),
        (
            TaskID.EastWest,
            TriangleState.lower,
            0,
        ),
        (
            TaskID.LeftRight,
            TriangleState.upper,
            1,
        ),
        (
            TaskID.LeftRight,
            TriangleState.upper,
            0,
        ),
        (
            TaskID.LeftRight,
            TriangleState.lower,
            1,
        ),
        (
            TaskID.LeftRight,
            TriangleState.lower,
            0,
        ),
    ],
    ids=[
        "east/west - upper - solved",
        "east/west - upper - not solved",
        "east/west - lower - solved",
        "east/west - lower - not solved",
        "left/right - upper - solved",
        "left/right - upper - not solved",
        "left/right - lower - solved",
        "left/right - lower - not solved",
    ],
)
def test_DuplicatedCoordsEnv_logic(task, triangle, expected_reward):
    """Test environment logic."""
    env = DuplicatedCoordsEnv()
    env.reset()
    env.TaskID = task
    env.TriangleState = triangle
    assert env.odor_condition == OdorCondition.pre

    if triangle == TriangleState.upper:
        x_orig = 2
        y_orig = 1
        direction_orig = 0
    elif triangle == TriangleState.lower:
        x_orig = -2
        y_orig = -1
        direction_orig = 180

    coords_orig = torch.tensor([x_orig, y_orig])
    state = torch.cat(
        (
            torch.tensor(Cues.NoOdor.value).unsqueeze(-1),
            torch.tensor(direction_orig).unsqueeze(-1),
            env.conv2north_cartesian(coords_orig),
            env.conv2south_cartesian(coords_orig),
            env.conv2north_polar(coords_orig),
            env.conv2south_polar(coords_orig),
        )
    )
    state, reward, done = env.step(action=Actions.forward.value, current_state=state)
    assert done is False
    assert env.odor_condition == OdorCondition.post
    assert state[0] == Cues.OdorA.value or Cues.OdorB.value

    # Post odor presentation
    if task == TaskID.EastWest:
        if triangle == TriangleState.upper:
            if state[0] == Cues.OdorA.value:
                if expected_reward == 1:
                    action = Actions.left.value
                    state, reward, done = env.step(action=action, current_state=state)
                else:
                    action = Actions.right.value
                    state, reward, done = env.step(action=action, current_state=state)
                    state, reward, done = env.step(action=action, current_state=state)
            elif state[0] == Cues.OdorB.value:
                if expected_reward == 1:
                    action = Actions.right.value
                    state, reward, done = env.step(action=action, current_state=state)
                    state, reward, done = env.step(action=action, current_state=state)
                else:
                    action = Actions.left.value
                    state, reward, done = env.step(action=action, current_state=state)
        elif triangle == TriangleState.lower:
            if state[0] == Cues.OdorA.value:
                if expected_reward == 1:
                    action = Actions.right.value
                    state, reward, done = env.step(action=action, current_state=state)
                    state, reward, done = env.step(action=action, current_state=state)
                else:
                    action = Actions.left.value
                    state, reward, done = env.step(action=action, current_state=state)
            elif state[0] == Cues.OdorB.value:
                if expected_reward == 1:
                    action = Actions.left.value
                    state, reward, done = env.step(action=action, current_state=state)
                else:
                    action = Actions.right.value
                    state, reward, done = env.step(action=action, current_state=state)
                    state, reward, done = env.step(action=action, current_state=state)
    elif task == TaskID.LeftRight:
        if triangle == TriangleState.upper:
            if state[0] == Cues.OdorA.value:
                if expected_reward == 1:
                    action = Actions.left.value
                    state, reward, done = env.step(action=action, current_state=state)
                else:
                    action = Actions.right.value
                    state, reward, done = env.step(action=action, current_state=state)
                    state, reward, done = env.step(action=action, current_state=state)
            elif state[0] == Cues.OdorB.value:
                if expected_reward == 1:
                    action = Actions.right.value
                    state, reward, done = env.step(action=action, current_state=state)
                    state, reward, done = env.step(action=action, current_state=state)
                else:
                    action = Actions.left.value
                    state, reward, done = env.step(action=action, current_state=state)
        elif triangle == TriangleState.lower:
            if state[0] == Cues.OdorB.value:
                if expected_reward == 1:
                    action = Actions.right.value
                    state, reward, done = env.step(action=action, current_state=state)
                    state, reward, done = env.step(action=action, current_state=state)
                else:
                    action = Actions.left.value
                    state, reward, done = env.step(action=action, current_state=state)
            elif state[0] == Cues.OdorA.value:
                if expected_reward == 1:
                    action = Actions.left.value
                    state, reward, done = env.step(action=action, current_state=state)
                else:
                    action = Actions.right.value
                    state, reward, done = env.step(action=action, current_state=state)
                    state, reward, done = env.step(action=action, current_state=state)

    action = Actions.forward.value
    for _ in range(3):
        state, reward, done = env.step(action=action, current_state=state)

    assert done is True
    assert reward == expected_reward


def test_random_policy():
    """Test that using a random policy ends up in 50%-50% solved/unsolved task."""
    total_ep = 100
    env = DuplicatedCoordsEnv()
    episodes = torch.arange(total_ep, device=DEVICE)
    explorer = EpsilonGreedy(epsilon=1)
    rewards = torch.empty_like(episodes) * torch.nan

    for episode in episodes:
        state = env.reset()  # Reset the environment
        state = state.clone().float().detach().to(DEVICE)
        done = False
        total_rewards = 0

        while not done:
            state_action_values = torch.rand(len(env.action_space))
            action = explorer.choose_action(
                action_space=env.action_space,
                state=state,
                state_action_values=state_action_values,
            )
            next_state, reward, done = env.step(
                action=action.item(), current_state=state
            )
            total_rewards += reward

            # Move to the next state
            state = next_state

        rewards[episode] = total_rewards
    ep_counts = torch.histc(rewards, bins=len(rewards.unique()))
    torch.testing.assert_close(
        actual=ep_counts,
        expected=torch.tensor(len(episodes) / len(ep_counts)).repeat(len(ep_counts)),
        atol=total_ep * 15 / 100,  # allow 15% random error
        rtol=1e-2,
    )