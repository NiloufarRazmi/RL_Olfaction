"""Tests on the environment."""

import pytest
import torch
from tensordict.tensordict import TensorDict

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
            Actions.right,
            0,
            0,
            270,
        ),
        (
            0,
            0,
            270,
            TriangleState.upper,
            Actions.left,
            0,
            0,
            180,
        ),
        (
            2,
            1,
            0,
            TriangleState.upper,
            Actions.right,
            2,
            1,
            90,
        ),
        (
            0,
            2,
            0,
            TriangleState.upper,
            Actions.forward,
            0,
            2,
            0,
        ),
        (
            0,
            0,
            0,
            TriangleState.lower,
            Actions.right,
            0,
            0,
            90,
        ),
        (
            0,
            0,
            90,
            TriangleState.lower,
            Actions.left,
            0,
            0,
            0,
        ),
        (
            -2,
            -1,
            0,
            TriangleState.lower,
            Actions.left,
            -2,
            -1,
            270,
        ),
        (
            0,
            -2,
            90,
            TriangleState.lower,
            Actions.right,
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
            Actions.forward,
            1,
            2,
            0,
        ),
        (
            1,
            1,
            0,
            TriangleState.upper,
            Actions.right,
            2,
            1,
            90,
        ),
        (
            1,
            1,
            0,
            TriangleState.upper,
            Actions.backward,
            1,
            0,
            0,
        ),
        (
            1,
            1,
            0,
            TriangleState.upper,
            Actions.left,
            0,
            1,
            270,
        ),
        (
            -1,
            -1,
            180,
            TriangleState.lower,
            Actions.forward,
            -1,
            -2,
            180,
        ),
        (
            -1,
            -1,
            180,
            TriangleState.lower,
            Actions.right,
            -2,
            -1,
            270,
        ),
        (
            -1,
            -1,
            180,
            TriangleState.lower,
            Actions.backward,
            -1,
            -0,
            180,
        ),
        (
            -1,
            -1,
            180,
            TriangleState.lower,
            Actions.left,
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
    state, reward, done = env.step(action=Actions.forward, current_state=state)
    assert env.odor_condition == OdorCondition.post
    assert state["cue"] == Cues.OdorA or Cues.OdorB

    # Post odor presentation
    if task == TaskID.EastWest:
        if triangle == TriangleState.upper:
            if state["cue"] == Cues.OdorA:
                if expected_reward == 1:
                    action = Actions.left
                    state, reward, done = env.step(action=action, current_state=state)
                else:
                    action = Actions.right
                    state, reward, done = env.step(action=action, current_state=state)
                    state, reward, done = env.step(action=action, current_state=state)
            elif state["cue"] == Cues.OdorB:
                if expected_reward == 1:
                    action = Actions.right
                    state, reward, done = env.step(action=action, current_state=state)
                    state, reward, done = env.step(action=action, current_state=state)
                else:
                    action = Actions.left
                    state, reward, done = env.step(action=action, current_state=state)
        elif triangle == TriangleState.lower:
            if state["cue"] == Cues.OdorA:
                if expected_reward == 1:
                    action = Actions.right
                    state, reward, done = env.step(action=action, current_state=state)
                    state, reward, done = env.step(action=action, current_state=state)
                else:
                    action = Actions.left
                    state, reward, done = env.step(action=action, current_state=state)
            elif state["cue"] == Cues.OdorB:
                if expected_reward == 1:
                    action = Actions.left
                    state, reward, done = env.step(action=action, current_state=state)
                else:
                    action = Actions.right
                    state, reward, done = env.step(action=action, current_state=state)
                    state, reward, done = env.step(action=action, current_state=state)
    elif task == TaskID.LeftRight:
        if triangle == TriangleState.upper:
            if state["cue"] == Cues.OdorA:
                if expected_reward == 1:
                    action = Actions.left
                    state, reward, done = env.step(action=action, current_state=state)
                else:
                    action = Actions.right
                    state, reward, done = env.step(action=action, current_state=state)
                    state, reward, done = env.step(action=action, current_state=state)
            elif state["cue"] == Cues.OdorB:
                if expected_reward == 1:
                    action = Actions.right
                    state, reward, done = env.step(action=action, current_state=state)
                    state, reward, done = env.step(action=action, current_state=state)
                else:
                    action = Actions.left
                    state, reward, done = env.step(action=action, current_state=state)
        elif triangle == TriangleState.lower:
            if state["cue"] == Cues.OdorB:
                if expected_reward == 1:
                    action = Actions.right
                    state, reward, done = env.step(action=action, current_state=state)
                    state, reward, done = env.step(action=action, current_state=state)
                else:
                    action = Actions.left
                    state, reward, done = env.step(action=action, current_state=state)
            elif state["cue"] == Cues.OdorA:
                if expected_reward == 1:
                    action = Actions.left
                    state, reward, done = env.step(action=action, current_state=state)
                else:
                    action = Actions.right
                    state, reward, done = env.step(action=action, current_state=state)
                    state, reward, done = env.step(action=action, current_state=state)

    action = Actions.forward
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
            torch.tensor([2.8284, 45]),
            torch.tensor([2.8284, 45]),
        ),
        (
            torch.tensor([2, 2]),
            torch.tensor([0, 0]),
            torch.tensor([4, 4]),
            torch.tensor([0, 0]).to(torch.float32),
            torch.tensor([5.6568, 45]),
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
    assert torch.equal(env.conv2north_cartesian(coords_orig), coords_north_cart)
    assert torch.equal(env.conv2south_cartesian(coords_orig), coords_south_cart)
    torch.testing.assert_close(
        actual=env.conv2north_polar(coords_orig),
        expected=coords_north_polar,
        rtol=1e-5,
        atol=1e-4,
    )
    torch.testing.assert_close(
        actual=env.conv2south_polar(coords_orig),
        expected=coords_south_polar,
        rtol=1e-5,
        atol=1e-4,
    )
