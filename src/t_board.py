
import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter
import torch as th
import gymnasium as gym

from environment import obs_to_tensor

def add_training_metrics(writer: SummaryWriter, value_losses, policy_losses, value_sims,regularization_loss, total_losses,buffer_size, learning_rate, i):
    writer.add_scalar("Training/Value_loss", np.mean(value_losses), i)
    writer.add_scalar("Training/Policy_loss", np.mean(policy_losses), i)
    writer.add_scalar("Training/Value_Simularities", np.mean(value_sims), i)
    writer.add_scalar("Training/Value_and_Policy_loss", np.mean(total_losses), i)
    writer.add_scalar("Training/Regularization_loss", regularization_loss, i)
    writer.add_scalar("Training/Replay_Buffer_Size", buffer_size, i)
    if learning_rate is not None:
        writer.add_scalar("Training/Learning_Rate", learning_rate[0], i)

def add_self_play_metrics(writer: SummaryWriter, mean_reward, reward_variance, time_steps, entropies, tot_tim, global_step):
                                  # Log the statistics
    writer.add_scalar("Self_Play/Mean_Reward", mean_reward, global_step)
    writer.add_scalar(
        "Self_Play/Reward_STD", np.sqrt(reward_variance), global_step
    )
    writer.add_scalar(
        "Self_Play/Mean_Timesteps", np.mean(time_steps), global_step
    )
    writer.add_scalar(
        "Self_Play/Timesteps_STD", np.sqrt(np.var(time_steps, ddof=1)), global_step
    )
    writer.add_scalar(
        "Self_Play/Runtime_per_Timestep",
        tot_tim.microseconds / np.sum(time_steps),
        global_step,
    )

    writer.add_scalar(
        "Self_Play/Mean_Entropy", np.mean(entropies), global_step
    )

    # count the total number of timesteps
    writer.add_scalar(
        "Self_Play/Total_Timesteps", np.sum(time_steps), global_step
    )

def log_model(writer: SummaryWriter, model: th.nn.Module, env: gym.Env):
    if model.device == th.device("cpu"):
        obs = env.reset()[0]
        writer.add_graph(
            model,
            obs_to_tensor(env.observation_space, obs, dtype=th.float32),
        )
