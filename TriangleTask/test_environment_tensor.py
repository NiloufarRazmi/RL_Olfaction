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


@pytest.mark.parametrize(
    "trianglestate, cue_val, correct_reward_port",
    [
        (
            TriangleState.upper,
            Cues.OdorA,
            True,
        ),
        (
            TriangleState.upper,
            Cues.OdorB,
            True,
        ),
        (
            TriangleState.lower,
            Cues.OdorA,
            True,
        ),
        (
            TriangleState.lower,
            Cues.OdorB,
            True,
        ),
        (
            TriangleState.upper,
            Cues.OdorA,
            False,
        ),
        (
            TriangleState.upper,
            Cues.OdorB,
            False,
        ),
        (
            TriangleState.lower,
            Cues.OdorA,
            False,
        ),
        (
            TriangleState.lower,
            Cues.OdorB,
            False,
        ),
    ],
)
def test_TriangleTask_logic(trianglestate, cue_val, correct_reward_port):
    """
    env.tiles_locations.reshape((env.rows, env.cols))

    tensor([[ 0,  1,  2,  3,  4],
            [ 5,  6,  7,  8,  9],
            [10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19],
            [20, 21, 22, 23, 24]])
    """

    # Go to the odor port
    if trianglestate == TriangleState.upper:
        tile_num = 3
        action_val = Actions.RIGHT.value
    elif trianglestate == TriangleState.lower:
        tile_num = 21
        action_val = Actions.LEFT.value
    env = WrappedEnvironment()
    assert env.odor_condition == OdorCondition.pre
    env.odor_ID = Cues(cue_val)
    state = torch.tensor([tile_num, 0], device=DEVICE)
    action = torch.tensor(action_val, device=DEVICE)
    env.TriangleState = trianglestate
    observation, reward, done = env.step(action=action, current_state=state)
    assert observation[1] == cue_val.value
    assert reward == 0
    assert done == False
    assert env.TriangleState == trianglestate
    assert env.odor_condition == OdorCondition.post

    # Get the reward
    if trianglestate == TriangleState.upper:
        if correct_reward_port:
            if cue_val == Cues.OdorA:
                tile_num = 1
                action_val = Actions.LEFT.value
            elif cue_val == Cues.OdorB:
                tile_num = 19
                action_val = Actions.DOWN.value
        else:
            if cue_val == Cues.OdorA:
                tile_num = 19
                action_val = Actions.DOWN.value
            elif cue_val == Cues.OdorB:
                tile_num = 1
                action_val = Actions.LEFT.value
    elif trianglestate == TriangleState.lower:
        if correct_reward_port:
            if cue_val == Cues.OdorA:
                tile_num = 5
                action_val = Actions.UP.value
            elif cue_val == Cues.OdorB:
                tile_num = 23
                action_val = Actions.RIGHT.value
        else:
            if cue_val == Cues.OdorA:
                tile_num = 23
                action_val = Actions.RIGHT.value
            elif cue_val == Cues.OdorB:
                tile_num = 5
                action_val = Actions.UP.value
    state = torch.tensor([tile_num, cue_val.value], device=DEVICE)
    action = torch.tensor(action_val, device=DEVICE)
    observation, reward, done = env.step(action=action, current_state=state)
    if correct_reward_port:
        assert reward > 0
        assert done == True
    else:
        assert reward <= 0
        assert done == True