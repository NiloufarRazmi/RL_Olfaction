import pytest
import torch
from .agent_tensor import EpsilonGreedy
from .environment_tensor import (
    Actions,
    Cues,
    OdorCondition,
    TriangleState,
    WrappedEnvironment,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.mark.parametrize(
    "one_hot_state, state_len",
    [(False, 2), (True, 28)],
    ids=["normal state", "state one hot encoded"],
)
def test_TriangleTask_general_props(one_hot_state, state_len):
    env = WrappedEnvironment(one_hot_state=one_hot_state)
    assert env
    assert env.numActions == 4
    state = env.reset()
    assert len(state) == state_len
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
    envs = [
        WrappedEnvironment(one_hot_state=False),
        WrappedEnvironment(one_hot_state=True),
    ]
    for env in envs:
        if env.one_hot_state:
            state = env.to_one_hot(torch.tensor([tile_num, cue_val], device=DEVICE))
        else:
            state = torch.tensor([tile_num, cue_val], device=DEVICE)
        action = torch.tensor(action_val, device=DEVICE)
        env.TriangleState = trianglestate
        observation, reward, done = env.step(action=action, current_state=state)
        if env.one_hot_state:
            assert torch.equal(
                observation,
                env.to_one_hot(
                    torch.tensor([exp_tile_num, exp_cue_val], device=DEVICE)
                ),
            )
        else:
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
    envs = [
        WrappedEnvironment(one_hot_state=False),
        WrappedEnvironment(one_hot_state=True),
    ]
    for env in envs:
        # Go to the odor port
        if trianglestate == TriangleState.upper:
            tile_num = 3
            action_val = Actions.RIGHT.value
        elif trianglestate == TriangleState.lower:
            tile_num = 21
            action_val = Actions.LEFT.value
        assert env.odor_condition == OdorCondition.pre
        env.odor_ID = Cues(cue_val)
        if env.one_hot_state:
            state = env.to_one_hot(torch.tensor([tile_num, 0], device=DEVICE))
        else:
            state = torch.tensor([tile_num, 0], device=DEVICE)

        action = torch.tensor(action_val, device=DEVICE)
        env.TriangleState = trianglestate
        observation, reward, done = env.step(action=action, current_state=state)
        if env.one_hot_state:
            assert env.from_one_hot(observation)[1] == cue_val.value
        else:
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
        if env.one_hot_state:
            state = env.to_one_hot(
                torch.tensor([tile_num, cue_val.value], device=DEVICE)
            )
        else:
            state = torch.tensor([tile_num, cue_val.value], device=DEVICE)
        action = torch.tensor(action_val, device=DEVICE)
        observation, reward, done = env.step(action=action, current_state=state)
        if correct_reward_port:
            assert reward > 0
            assert done == True
        else:
            assert reward <= 0
            assert done == True


@pytest.mark.parametrize(
    "one_hot_state",
    [(False), (True)],
)
def test_random_actions(one_hot_state):
    total_ep = 100
    env = WrappedEnvironment(one_hot_state=one_hot_state)
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
