import torch as th
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from torchvision.transforms import ToTensor
from PIL import Image
import io



def obs_to_tensor_coordinates(observation_space: gym.Space, observation, ncols=12, *args, **kwargs):
    """
    Returns a tensor of shape (2,) with the coordinates of the observation
    """
    cords = divmod(observation, ncols)
    return th.tensor(cords, *args, **kwargs)


def obs_dim_coordinates(observation_space: gym.Space):
    return 2


def obs_to_tensor_flat(observation_space: gym.Space, observation, *args, **kwargs):
    return th.tensor(
        gym.spaces.flatten(observation_space, observation),
        *args,
        **kwargs,
    )

def obs_dim_flat(observation_space: gym.Space):
    return gym.spaces.flatdim(observation_space)

obs_to_tensor = obs_to_tensor_coordinates
obs_dim = obs_dim_coordinates

@th.no_grad()
def investigate_model(observation_space: gym.spaces.Discrete, model: th.nn.Module):
    """
    returns a dict of {obs: value} for each obs in the observation space
    """
    output = {}
    model.eval()
    for obs in range(observation_space.n):
        output[obs] = model(obs_to_tensor(observation_space, obs, dtype=th.float32))
    return output

def create_figure_and_axes():
    fig, ax = plt.subplots()
    ax.grid(False)
    ax.axis("off")
    return fig, ax

def plot_image(fig, ax, image, title):
    ax.imshow(image, interpolation="nearest")
    ax.set_title(title)
    plt.tight_layout(pad=0)
    if fig is not None:
        plt.close(fig)

def plot_value_network(outputs, nrows=4, ncols=12):
    plt.ioff()
    grid = np.zeros((nrows, ncols))
    for state, value in outputs.items():
        row, col = divmod(state, ncols)
        grid[row, col] = value[0]
    fig, ax = create_figure_and_axes()
    for i in range(nrows):
        for j in range(ncols):
            ax.text(j, i, f"{grid[i, j]:.0f}", ha="center", va="center", color="blue")
    plot_image(fig, ax, grid, "Cliff Walking Value Network")
    return fig

def plot_policy_network(
    outputs, nrows=4, ncols=12, title="Cliff Walking Policy Network"
):
    plt.ioff()
    action_arrows = {0: "↑", 1: "→", 2: "↓", 3: "←"}
    preffered_actions = np.zeros((nrows, ncols), dtype="<U2")
    entropy = np.zeros((nrows, ncols))
    for state, action in outputs.items():
        row, col = divmod(state, ncols)
        preffered_actions[row, col] = action_arrows[np.argmax(action[1]).item()]
        entropy[row, col] = th.distributions.Categorical(
            probs=action[1]
        ).entropy().item() / np.log(len(action_arrows))
    fig, ax = create_figure_and_axes()
    for i in range(nrows):
        for j in range(ncols):
            ax.text(j, i, f"{entropy[i, j]:.2f}", ha="center", va="top", color="black")
            ax.text(
                j,
                i,
                f"{preffered_actions[i, j]}",
                ha="center",
                va="bottom",
                color="red",
                fontsize=16,
            )
    plot_image(fig, ax, entropy, title)
    return fig

def show_model_in_tensorboard(
    observation_space: gym.spaces.Discrete, model: th.nn.Module, writer, step
):
    rows, cols = 6, 12
    outputs = investigate_model(observation_space, model)
    value_fig = plot_value_network(outputs, nrows=rows, ncols=cols)
    policy_fig = plot_policy_network(outputs, nrows=rows, ncols=cols)
    writer.add_figure("value_network", value_fig, global_step=step)
    writer.add_figure("policy_network", policy_fig, global_step=step)
