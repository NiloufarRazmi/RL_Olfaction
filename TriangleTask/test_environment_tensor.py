import pytest
import torch
from environment_tensor import (
    Actions,
    Cues,
    OdorCondition,
    TriangleState,
    WrappedEnvironment,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test_TriangleTask_general_props():
    env = WrappedEnvironment()
    assert env
    assert env.numActions == 4
    state = env.reset()
    assert len(state) == 2
    assert env.odor_condition == OdorCondition.pre


@pytest.mark.parametrize(
    "tile_num, cue_val, action_val, trianglestate, exp_tile_num, exp_cue_val, exp_reward, exp_done, odor_cond",
    [
        (
            2,
            Cues.NoOdor.value,
            Actions.UP.value,
            TriangleState.upper,
            2,
            Cues.NoOdor.value,
            0,
            False,
            OdorCondition.pre,
        ),
        (
            0,
            Cues.NoOdor.value,
            Actions.LEFT.value,
            TriangleState.upper,
            0,
            Cues.NoOdor.value,
            0,
            False,
            OdorCondition.pre,
        ),
        (
            9,
            Cues.NoOdor.value,
            Actions.RIGHT.value,
            TriangleState.upper,
            9,
            Cues.NoOdor.value,
            0,
            False,
            OdorCondition.pre,
        ),
        (
            12,
            Cues.NoOdor.value,
            Actions.DOWN.value,
            TriangleState.upper,
            12,
            Cues.NoOdor.value,
            0,
            False,
            OdorCondition.pre,
        ),
        (
            12,
            Cues.NoOdor.value,
            Actions.LEFT.value,
            TriangleState.upper,
            12,
            Cues.NoOdor.value,
            0,
            False,
            OdorCondition.pre,
        ),
        (
            15,
            Cues.NoOdor.value,
            Actions.LEFT.value,
            TriangleState.lower,
            15,
            Cues.NoOdor.value,
            0,
            False,
            OdorCondition.pre,
        ),
        (
            22,
            Cues.NoOdor.value,
            Actions.DOWN.value,
            TriangleState.lower,
            22,
            Cues.NoOdor.value,
            0,
            False,
            OdorCondition.pre,
        ),
        (
            12,
            Cues.NoOdor.value,
            Actions.UP.value,
            TriangleState.lower,
            12,
            Cues.NoOdor.value,
            0,
            False,
            OdorCondition.pre,
        ),
        (
            12,
            Cues.NoOdor.value,
            Actions.RIGHT.value,
            TriangleState.lower,
            12,
            Cues.NoOdor.value,
            0,
            False,
            OdorCondition.pre,
        ),
        (
            8,
            Cues.NoOdor.value,
            Actions.UP.value,
            TriangleState.upper,
            3,
            Cues.NoOdor.value,
            0,
            False,
            OdorCondition.pre,
        ),
        (
            8,
            Cues.NoOdor.value,
            Actions.DOWN.value,
            TriangleState.upper,
            13,
            Cues.NoOdor.value,
            0,
            False,
            OdorCondition.pre,
        ),
        (
            8,
            Cues.NoOdor.value,
            Actions.LEFT.value,
            TriangleState.upper,
            7,
            Cues.NoOdor.value,
            0,
            False,
            OdorCondition.pre,
        ),
        (
            8,
            Cues.NoOdor.value,
            Actions.RIGHT.value,
            TriangleState.upper,
            9,
            Cues.NoOdor.value,
            0,
            False,
            OdorCondition.pre,
        ),
    ],
    ids=[
        "No change in position when moving UP from top border",
        "No change in position when moving LEFT from upper left border",
        "No change in position when moving RIGHT from right border",
        "No change in position when moving DOWN from upper triangle border",
        "No change in position when moving LEFT from upper triangle border",
        "No change in position when moving LEFT from lower left border",
        "No change in position when moving DOWN from bottom border",
        "No change in position when moving UP from lower triangle border",
        "No change in position when moving RIGHT from lower triangle border",
        "Moving UP",
        "Moving DOWN",
        "Moving LEFT",
        "Moving RIGHT",
    ],
)
def test_TriangleTask_moves(
    tile_num,
    cue_val,
    action_val,
    trianglestate,
    exp_tile_num,
    exp_cue_val,
    exp_reward,
    exp_done,
    odor_cond,
):
    """
    env.tiles_locations.reshape((env.rows, env.cols))

    tensor([[ 0,  1,  2,  3,  4],
            [ 5,  6,  7,  8,  9],
            [10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19],
            [20, 21, 22, 23, 24]])
    """
    env = WrappedEnvironment()
    state = torch.tensor([tile_num, cue_val], device=DEVICE)
    action = torch.tensor(action_val, device=DEVICE)
    env.TriangleState = trianglestate
    observation, reward, done = env.step(action=action, current_state=state)
    assert torch.equal(
        observation, torch.tensor([exp_tile_num, exp_cue_val], device=DEVICE)
    )
    assert reward == exp_reward
    assert done == exp_reward
    assert env.TriangleState == trianglestate
    assert env.odor_condition == odor_cond


# def test_TriangleTask_logic():
#     """
#     env.tiles_locations.reshape((env.rows, env.cols))

#     tensor([[ 0,  1,  2,  3,  4],
#             [ 5,  6,  7,  8,  9],
#             [10, 11, 12, 13, 14],
#             [15, 16, 17, 18, 19],
#             [20, 21, 22, 23, 24]])
#     """
#     env = WrappedEnvironment()
#     state = torch.tensor([tile_num, cue_val], device=DEVICE)
#     action = torch.tensor(action_val, device=DEVICE)
#     env.TriangleState = trianglestate
#     observation, reward, done = env.step(action=action, current_state=state)
