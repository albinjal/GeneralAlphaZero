from collections import Counter
from matplotlib import pyplot as plt
import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter
import torch as th
import gymnasium as gym
from az.model import AlphaZeroModel
from logs.investigate_model import investigate_model, plot_policy_network, plot_value_network, plot_visits_with_counter

from environments.observation_embeddings import CoordinateEmbedding, ObservationEmbedding

def show_model_in_tensorboard(model: AlphaZeroModel, writer, step):
    assert isinstance(model.observation_embedding, CoordinateEmbedding)
    outputs = investigate_model(model)
    value_fig = plot_value_network(outputs, nrows=model.observation_embedding.nrows, ncols=model.observation_embedding.ncols)
    policy_fig = plot_policy_network(outputs, nrows=model.observation_embedding.nrows, ncols=model.observation_embedding.ncols)
    writer.add_figure("value_network", value_fig, global_step=step)
    writer.add_figure("policy_network", policy_fig, global_step=step)
    plt.close(value_fig)
    plt.close(policy_fig)


def plot_visits_with_counter_tensorboard(visit_counts: Counter, observation_embedding: CoordinateEmbedding, writer, step, title="State Visit Counts"):
    fig = plot_visits_with_counter(visit_counts, observation_embedding, step, title)
    # Log the figure to Tensorboard
    writer.add_figure("visit_counts", fig, global_step=step)
    plt.close(fig)



def add_training_metrics(
    writer: SummaryWriter,
    value_losses,
    policy_losses,
    value_sims,
    regularization_loss,
    total_losses,
    buffer_size,
    learning_rate,
    i,
):
    writer.add_scalar("Training/Value_loss", np.mean(value_losses), i)
    writer.add_scalar("Training/Policy_loss", np.mean(policy_losses), i)
    writer.add_scalar("Training/Value_Simularities", np.mean(value_sims), i)
    writer.add_scalar("Training/Value_and_Policy_loss", np.mean(total_losses), i)
    writer.add_scalar("Training/Regularization_loss", regularization_loss, i)
    writer.add_scalar("Training/Replay_Buffer_Size", buffer_size, i)
    if learning_rate is not None:
        writer.add_scalar("Training/Learning_Rate", learning_rate[0], i)


def add_self_play_metrics(
    writer: SummaryWriter,
    mean_reward,
    reward_variance,
    time_steps,
    entropies,
    tot_tim,
    global_step,
):
    # Log the statistics
    writer.add_scalar("Self_Play/Mean_Reward", mean_reward, global_step)
    writer.add_scalar("Self_Play/Reward_STD", np.sqrt(reward_variance), global_step)
    writer.add_scalar("Self_Play/Mean_Timesteps", np.mean(time_steps), global_step)
    writer.add_scalar(
        "Self_Play/Timesteps_STD", np.sqrt(np.var(time_steps, ddof=1)), global_step
    )
    writer.add_scalar(
        "Self_Play/Runtime_per_Timestep",
        tot_tim.microseconds / np.sum(time_steps),
        global_step,
    )

    writer.add_scalar("Self_Play/Mean_Entropy", np.mean(entropies), global_step)

    # count the total number of timesteps
    writer.add_scalar("Self_Play/Total_Timesteps", np.sum(time_steps), global_step)


def log_model(writer: SummaryWriter, observation_embedding: ObservationEmbedding, model: th.nn.Module, env: gym.Env):
    if model.device == th.device("cpu"):
        obs = env.reset()[0]
        writer.add_graph(
            model,
            observation_embedding.obs_to_tensor(obs, dtype=th.float32),
        )
