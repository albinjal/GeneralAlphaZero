from collections import Counter
from matplotlib import pyplot as plt
import numpy as np
import torch as th
import gymnasium as gym
import wandb

from environments.environment import (
    investigate_model,
    obs_to_tensor,
    plot_policy_network,
    plot_value_network,
)


def add_training_metrics_wandb(
    value_losses,
    policy_losses,
    value_sims,
    regularization_loss,
    total_losses,
    buffer_size,
    learning_rate,
    step,
):
    wandb.log(
        {
            "Training/Value_loss": np.mean(value_losses),
            "Training/Policy_loss": np.mean(policy_losses),
            "Training/Value_Simularities": np.mean(value_sims),
            "Training/Value_and_Policy_loss": np.mean(total_losses),
            "Training/Regularization_loss": regularization_loss,
            "Training/Replay_Buffer_Size": buffer_size,
        },
        step=step,
    )
    if learning_rate is not None:
        wandb.log({"Training/Learning_Rate": learning_rate[0]}, step=step)


def log_model_wandb(model: th.nn.Module, env: gym.Env):
    # Log model parameters and gradients (if any)
    wandb.watch(model)


def add_self_play_metrics_wandb(
    rewards,
    time_steps,
    entropies,
    tot_tim,
    cumulative_reward,
    ema_reward,
    global_step,
):
    mean_reward = np.mean(rewards)
    wandb.log(
        {
            "Self_Play/Rewards": wandb.Histogram(rewards),
            "Self_Play/Mean_Reward": mean_reward,
            "Self_Play/Reward_STD": np.sqrt(np.var(rewards, ddof=1)),
            "Self_Play/Max_Reward": np.max(rewards),
            "Self_Play/Min_Reward": np.min(rewards),
            "Self_Play/Timesteps": wandb.Histogram(time_steps),
            "Self_Play/Min_Timesteps": np.min(time_steps),
            "Self_Play/Mean_Timesteps": np.mean(time_steps),
            "Self_Play/Timesteps_STD": np.sqrt(np.var(time_steps, ddof=1)),
            "Self_Play/Total_Timesteps": np.sum(time_steps),
            "Self_Play/Runtime_per_Timestep": tot_tim.microseconds / np.sum(time_steps),
            "Self_Play/Entropies": wandb.Histogram(entropies),
            "Self_Play/Mean_Entropy": np.mean(entropies),
            "Self_Play/Total_Runtime": tot_tim.microseconds,
            "Self_Play/EMA_Reward": ema_reward,
            "Self_Play/Cumulative_Reward": cumulative_reward,
            "Self_Play/Total_Average_Reward": cumulative_reward / (global_step+1),
        },
        step=global_step,
    )


def show_model_in_wandb(
    observation_space: gym.spaces.Discrete, model: th.nn.Module, step
):
    rows, cols = 6, 12
    outputs = investigate_model(observation_space, model)
    value_fig = plot_value_network(outputs, nrows=rows, ncols=cols)
    policy_fig = plot_policy_network(outputs, nrows=rows, ncols=cols)

    wandb.log(
        {
            "value_network": wandb.Image(value_fig),
            "policy_network": wandb.Image(policy_fig),
        },
        step=step,
    )
    plt.close(value_fig)
    plt.close(policy_fig)


def plot_visits_to_wandb_with_counter(
    visit_counts: Counter,
    observation_space,
    nrows,
    ncols,
    step,
    title="State Visit Counts",
):
    grid = np.zeros((nrows, ncols))

    for obs in range(observation_space.n):
        obs_tensor = tuple(
            obs_to_tensor(observation_space, obs, dtype=th.float32).tolist()
        )
        count = visit_counts.get(obs_tensor, 0)
        row, col = divmod(obs, ncols)
        grid[row, col] = count

    fig, ax = plt.subplots()
    ax.imshow(grid, cmap="viridis", interpolation="nearest")
    ax.set_title(title)
    for obs in range(observation_space.n):
        row, col = divmod(obs, ncols)
        ax.text(
            col, row, f"{grid[row, col]:.0f}", ha="center", va="center", color="white"
        )
    plt.tight_layout()

    wandb.log({"visit_counts": wandb.Image(fig)}, step=step)
    plt.close(fig)
