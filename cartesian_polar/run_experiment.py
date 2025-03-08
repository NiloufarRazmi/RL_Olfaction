"""Run experiment from CLI."""

import shutil
from collections import deque, namedtuple
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from . import plotting as viz
from . import utils
from .agent import EpsilonGreedy, neural_network
from .environment import CONTEXTS_LABELS, Actions, Cues, DuplicatedCoordsEnv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def training_loop(p, current_path, logger, generator=None):
    """Run the main training loop."""
    Transition = namedtuple(
        "Transition", ("state", "action", "reward", "next_state", "done")
    )

    rewards = torch.zeros((p.total_episodes, p.n_runs), device=DEVICE)
    steps = torch.zeros((p.total_episodes, p.n_runs), device=DEVICE)
    episodes = torch.arange(p.total_episodes, device=DEVICE)
    all_states = [[[] for _ in range(p.total_episodes)] for _ in range(p.n_runs)]
    all_actions = [[[] for _ in range(p.total_episodes)] for _ in range(p.n_runs)]
    losses = [[] for _ in range(p.n_runs)]

    for run in range(p.n_runs):  # Run several times to account for stochasticity
        # Get the number of states and actions from the environment
        env = DuplicatedCoordsEnv(taskid=p.taskid, seed=p.seed)
        state = env.reset()
        p.n_actions = env.numActions
        p.n_observations = len(state)
        logger.info(f"Number of actions: {p.n_actions}")
        logger.info(f"Number of observations: {p.n_observations}")
        logger.info(f"Number of hidden units in the network: {p.n_hidden_units}")

        # Reset everything
        net, target_net = neural_network(
            n_observations=p.n_observations,
            n_actions=p.n_actions,
            nHiddenUnits=p.n_hidden_units,
        )  # Reset weights
        logger.info(f"Network architecture:\n{net}")
        optimizer = optim.AdamW(net.parameters(), lr=p.alpha, amsgrad=True)
        explorer = EpsilonGreedy(
            epsilon=p.epsilon_max,
            epsilon_min=p.epsilon_min,
            epsilon_max=p.epsilon_max,
            decay_rate=p.decay_rate,
            epsilon_warmup=p.epsilon_warmup,
        )
        weights = None
        biases = None
        weights_val_stats = None
        biases_val_stats = None
        weights_grad_stats = None
        biases_grad_stats = None
        replay_buffer = deque([], maxlen=p.replay_buffer_max_size)
        epsilons = []

        for episode in tqdm(
            episodes, desc=f"Run {run + 1}/{p.n_runs} - Episodes", leave=False
        ):
            state = env.reset()  # Reset the environment
            state = state.clone().float().detach().to(DEVICE)
            step_count = 0
            done = False
            total_rewards = 0
            loss = torch.ones(1, device=DEVICE) * torch.nan

            while not done:
                state_action_values = net(state).to(DEVICE)  # Q(s_t)
                action = explorer.choose_action(
                    action_space=env.action_space,
                    state=state,
                    state_action_values=state_action_values,
                ).item()

                # Record states and actions
                all_states[run][episode].append(state.cpu())
                all_actions[run][episode].append(Actions(action).name)

                next_state, reward, done = env.step(action=action, current_state=state)

                # Store transition in replay buffer
                # [current_state (2 or 28 x1), action (1x1), next_state (2 or 28 x1),
                # reward (1x1), done (1x1 bool)]
                done = torch.tensor(done, device=DEVICE).unsqueeze(-1)
                replay_buffer.append(
                    Transition(
                        state,
                        action,
                        reward,
                        next_state,
                        done,
                    )
                )

                # Start training when `replay_buffer` is full
                if len(replay_buffer) == p.replay_buffer_max_size:
                    transitions = utils.random_choice(
                        replay_buffer,
                        length=len(replay_buffer),
                        num_samples=p.batch_size,
                        generator=generator,
                    )
                    batch = Transition(*zip(*transitions, strict=True))
                    state_batch = torch.stack(batch.state)
                    action_batch = torch.tensor(batch.action, device=DEVICE)
                    reward_batch = torch.cat(batch.reward)
                    # next_state_batch = torch.stack(batch.next_state)
                    # done_batch = torch.cat(batch.done)

                    # See DQN paper for equations: https://doi.org/10.1038/nature14236
                    state_action_values_sampled = net(state_batch).to(DEVICE)  # Q(s_t)
                    state_action_values = torch.gather(
                        input=state_action_values_sampled,
                        dim=1,
                        index=action_batch.unsqueeze(-1),
                    ).squeeze()  # Q(s_t, a)

                    # Compute a mask of non-final states and concatenate
                    # the batch elements
                    # (a final state would've been the one after which simulation ended)
                    non_final_mask = torch.tensor(
                        tuple(map(lambda s: not s, batch.done)),
                        device=DEVICE,
                        dtype=torch.bool,
                    )
                    non_final_next_states = torch.stack(
                        [s[1] for s in zip(batch.done, batch.next_state) if not s[0]]
                    )

                    # Compute V(s_{t+1}) for all next states.
                    # Expected values of actions for non_final_next_states are computed
                    # based on the "older" target_net;
                    # selecting their best reward with max(1).values
                    # This is merged based on the mask,
                    # such that we'll have either the expected
                    # state value or 0 in case the state was final.
                    next_state_values = torch.zeros(p.batch_size, device=DEVICE)
                    if non_final_next_states.numel() > 0 and non_final_mask.numel() > 0:
                        with torch.no_grad():
                            next_state_values[non_final_mask] = (
                                target_net(non_final_next_states).max(1).values
                            )
                    # Compute the expected Q values
                    expected_state_action_values = reward_batch + (
                        next_state_values * p.gamma
                    )

                    # Compute loss
                    # criterion = nn.MSELoss()
                    criterion = nn.SmoothL1Loss()
                    loss = criterion(
                        input=state_action_values,  # prediction
                        target=expected_state_action_values,  # target/"truth" value
                    )  # TD update

                    # Optimize the model
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_value_(
                        net.parameters(), 100
                    )  # In-place gradient clipping
                    optimizer.step()

                    # # Reset the target network
                    # if step_count % p.target_net_update == 0:
                    #     target_net.load_state_dict(net.state_dict())

                    # Soft update of the target network's weights
                    # θ′ ← τ θ + (1 −τ )θ′
                    target_net_state_dict = target_net.state_dict()
                    net_state_dict = net.state_dict()
                    for key in net_state_dict:
                        target_net_state_dict[key] = net_state_dict[
                            key
                        ] * p.tau + target_net_state_dict[key] * (1 - p.tau)
                    target_net.load_state_dict(target_net_state_dict)

                    losses[run].append(loss.item())

                    weights, biases = utils.collect_weights_biases(net=net)
                    weights_val_stats = utils.params_df_stats(
                        weights, key="val", current_df=weights_grad_stats
                    )
                    biases_val_stats = utils.params_df_stats(
                        biases, key="val", current_df=biases_val_stats
                    )
                    biases_grad_stats = utils.params_df_stats(
                        biases, key="grad", current_df=biases_grad_stats
                    )
                    weights_grad_stats = utils.params_df_stats(
                        weights, key="grad", current_df=weights_val_stats
                    )

                total_rewards += reward
                step_count += 1

                # Move to the next state
                state = next_state

                explorer.epsilon = explorer.update_epsilon(episode)
                epsilons.append(explorer.epsilon)

            all_states[run][episode].append(state.cpu())
            rewards[episode, run] = total_rewards
            steps[episode, run] = step_count
            logger.info(
                f"Run: {run + 1}/{p.n_runs} - Episode: {episode + 1}/{p.total_episodes}"
                f" - Steps: {step_count} - Loss: {loss.item()}"
                f" - epsilon: {explorer.epsilon}"
            )
        if weights_val_stats is not None and weights_val_stats.get("Index") is not None:
            weights_val_stats.set_index("Index", inplace=True)
        if biases_val_stats is not None and biases_val_stats.get("Index") is not None:
            biases_val_stats.set_index("Index", inplace=True)
        if biases_grad_stats is not None and biases_grad_stats.get("Index") is not None:
            biases_grad_stats.set_index("Index", inplace=True)
        if (
            weights_grad_stats is not None
            and weights_grad_stats.get("Index") is not None
        ):
            weights_grad_stats.set_index("Index", inplace=True)

        # Save agent trained state
        model_path = current_path / f"trained-agent-state-{run}.pt"
        torch.save(net.state_dict(), model_path)

    # Save data to disk
    data_dict = {
        "rewards": rewards.cpu(),
        "steps": steps.cpu(),
        "episodes": episodes.cpu(),
        "all_states": np.array(all_states, dtype=object),
        "all_actions": np.array(all_actions, dtype=object),
        "losses": np.array(losses, dtype=object),
        "p": p,
        "epsilons": epsilons,
        # "weights": weights,
        # "biases": biases,
        "weights_val_stats": weights_val_stats,
        "biases_val_stats": biases_val_stats,
        "weights_grad_stats": weights_grad_stats,
        "biases_grad_stats": biases_grad_stats,
        "net": net,
        "env": env,
    }
    if weights is not None and biases is not None:
        data_dict["weights"] = weights
        data_dict["biases"] = biases
    data_path = utils.save_data(data_dict=data_dict, current_path=current_path)
    return data_path


def visualization_plots(data_path, p, current_path, logger):
    """Make all the plots."""
    # Load data from disk
    data_dict = torch.load(data_path, weights_only=False, map_location=DEVICE)
    # with open(data_path, "rb") as fhd:
    #     # Load the arrays from the .npz file
    #     data_dict = np.load(fhd, allow_pickle=True)

    # Access individual arrays by their names
    rewards = data_dict["rewards"]
    steps = data_dict["steps"]
    episodes = data_dict["episodes"]
    all_actions = data_dict["all_actions"]
    # all_states = data_dict["all_states"]
    losses = data_dict["losses"]
    p = data_dict["p"]
    epsilons = data_dict["epsilons"]
    weights = data_dict["weights"]
    biases = data_dict["biases"]
    weights_val_stats = pd.DataFrame(
        data_dict["weights_val_stats"], columns=["Std", "Avg", "Layer"]
    ).index.rename("Index", inplace=True)
    biases_val_stats = pd.DataFrame(
        data_dict["biases_val_stats"], columns=["Std", "Avg", "Layer"]
    ).index.rename("Index", inplace=True)
    weights_grad_stats = pd.DataFrame(
        data_dict["weights_grad_stats"], columns=["Std", "Avg", "Layer"]
    ).index.rename("Index", inplace=True)
    biases_grad_stats = pd.DataFrame(
        data_dict["biases_grad_stats"], columns=["Std", "Avg", "Layer"]
    ).index.rename("Index", inplace=True)
    env = data_dict["env"]
    net = data_dict["net"]

    untrained_net, _ = neural_network(
        n_observations=p.n_observations,
        n_actions=p.n_actions,
        nHiddenUnits=p.n_hidden_units,
    )
    weights_untrained = [layer.detach() for layer in untrained_net.parameters()]

    # Postprocessing
    rew_steps_df = utils.postprocess_rewards_steps(
        episodes=episodes, n_runs=p.n_runs, rewards=rewards, steps=steps
    )
    loss_df = utils.postprocess_loss(losses=losses, window_size=1)
    q_values = utils.get_q_values_by_states(env=env, cues=Cues, net=net)
    weights_val_df = utils.postprocess_weights(weights["val"])
    biases_val_df = utils.postprocess_weights(biases["val"])
    weights_grad_df = utils.postprocess_weights(weights["grad"])
    biases_grad_df = utils.postprocess_weights(biases["grad"])
    input_cond, activations_layer_df = utils.get_activations_learned(
        net=net,
        env=env,
        layer_inspected=p.layer_inspected,
        contexts_labels=CONTEXTS_LABELS,
    )

    # Plots
    viz.plot_exploration_rate(
        epsilons=epsilons,
        steps=steps,
        figpath=current_path,
        logger=logger,
    )
    viz.plot_actions_distribution(all_actions, figpath=current_path, logger=logger)
    viz.plot_steps_and_rewards_dist(rew_steps_df, figpath=current_path, logger=logger)
    viz.plot_steps_and_rewards(
        rew_steps_df, n_runs=p.n_runs, figpath=current_path, logger=logger
    )
    viz.plot_loss(loss_df, n_runs=p.n_runs, figpath=current_path, logger=logger)
    viz.plot_policies_by_head_directions(
        q_values=q_values,
        env=env,
        cues=Cues,
        labels=CONTEXTS_LABELS,
        figpath=current_path,
        logger=logger,
    )
    viz.plot_weights_matrices(
        weights_untrained=weights_untrained,
        weights_trained=[layer for layer in net.parameters()],
        figpath=current_path,
        logger=logger,
    )
    viz.plot_activations(
        activations_layer_df=activations_layer_df,
        input_cond=input_cond,
        labels=CONTEXTS_LABELS,
        layer_inspected=p.layer_inspected,
        figpath=current_path,
        logger=logger,
    )
    viz.plot_weights_biases_distributions(
        weights_val_df,
        biases_val_df,
        label="Values",
        figpath=current_path,
        logger=logger,
    )
    if utils.check_grad_stats(weights_grad_df) or utils.check_grad_stats(
        biases_grad_df
    ):
        viz.plot_weights_biases_distributions(
            weights_grad_df,
            biases_grad_df,
            label="Gradients",
            figpath=current_path,
            logger=logger,
        )
    else:
        msg = "Gradients are zero"
        print(msg)
        logger.warning(msg)
    if weights_val_stats and biases_val_stats:
        viz.plot_weights_biases_stats(
            weights_val_stats,
            biases_val_stats,
            label="values",
            figpath=current_path,
            logger=logger,
        )
    if weights_grad_stats and biases_grad_stats:
        viz.plot_weights_biases_stats(
            weights_grad_stats,
            biases_grad_stats,
            label="gradients",
            figpath=current_path,
            logger=logger,
        )


@click.command()
@click.argument(
    "paramsfile",
    type=click.Path(exists=True),
    required=True,
)
def cli(paramsfile):
    """Run the main command-line function."""
    paramsfile = Path(paramsfile)
    p = utils.get_exp_params_from_config(config_path=paramsfile)
    current_path = utils.create_save_path(
        task=p.taskid, experiment_tag=p.experiment_tag
    )
    shutil.copyfile(src=paramsfile, dst=current_path / paramsfile.name)
    logger = utils.get_logger(current_path=current_path)
    generator = utils.make_deterministic(seed=p.seed)

    data_path = training_loop(
        p=p, current_path=current_path, logger=logger, generator=generator
    )
    visualization_plots(
        data_path=data_path, p=p, current_path=current_path, logger=logger
    )


if __name__ == "__main__":
    cli()
