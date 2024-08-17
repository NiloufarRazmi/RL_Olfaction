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
    "cue_orig, x_orig, y_orig, direction_orig, TriangleState, action, "
    "cue_expect, x_expect, y_expect, direction_expect",
    [
        (
            Cues.NoOdor,
            0,
            0,
            270,
            TriangleState.upper,
            Actions.left,
            Cues.NoOdor,
            0,
            0,
            180,
        ),
    ],
)
def test_moves(
    cue_orig,
    x_orig,
    y_orig,
    direction_orig,
    TriangleState,
    action,
    cue_expect,
    x_expect,
    y_expect,
    direction_expect,
):
    """Test basic moves."""
    env = Environment()
    env.TriangleState = TriangleState
    state_orig = TensorDict(
        {
            "cue": torch.tensor(cue_orig.value, device=DEVICE).unsqueeze(-1),
            "x": torch.tensor([x_orig], device=DEVICE),
            "y": torch.tensor([y_orig], device=DEVICE),
            "direction": torch.tensor([direction_orig], device=DEVICE),
        },
        batch_size=[1],
    )
    new_state, _, _ = env.step(action=action, current_state=state_orig)
    state_expected = TensorDict(
        {
            "cue": torch.tensor(cue_expect.value, device=DEVICE).unsqueeze(-1),
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
