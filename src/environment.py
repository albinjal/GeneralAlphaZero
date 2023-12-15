import torch as th
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from torchvision.transforms import ToTensor
from PIL import Image
import io




from model import AlphaZeroModel

def obs_to_tensor(observation_space: gym.Space, observation, *args, **kwargs):
    return th.tensor(
        gym.spaces.flatten(observation_space, observation),
        *args,
        **kwargs,
    )


@th.no_grad()
def investigate_model(observation_space: gym.spaces.Discrete, model: AlphaZeroModel):
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
    ax.axis('off')
    return fig, ax

def plot_image(fig, ax, image, title):
    ax.imshow(image, interpolation='nearest')
    ax.set_title(title)
    plt.tight_layout(pad=0)
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
            ax.text(j, i, f'{grid[i, j]:.0f}', ha='center', va='center', color='blue')
    plot_image(fig, ax, grid, 'Cliff Walking Value Network')
    return fig

def plot_policy_network(outputs, nrows=4, ncols=12, title='Cliff Walking Policy Network'):
    plt.ioff()
    action_arrows = {0: '↑', 1: '→', 2: '↓', 3: '←'}
    preffered_actions = np.zeros((nrows, ncols), dtype='<U2')
    entropy = np.zeros((nrows, ncols))
    for state, action in outputs.items():
        row, col = divmod(state, ncols)
        preffered_actions[row, col] = action_arrows[np.argmax(action[1]).item()]
        entropy[row, col] = th.distributions.Categorical(probs=action[1]).entropy().item() / np.log(len(action_arrows))
    fig, ax = create_figure_and_axes()
    for i in range(nrows):
        for j in range(ncols):
            ax.text(j, i, f'{entropy[i, j]:.2f}', ha='center', va='top', color='black')
            ax.text(j, i, f'{preffered_actions[i, j]}', ha='center', va='bottom', color='red', fontsize=16)
    plot_image(fig, ax, entropy, title)
    return fig

def plot_to_tensor(fig):
    """Convert a Matplotlib figure to a 4D tensor for TensorBoard."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf).convert('RGB')  # Convert image to RGB mode
    tensor = ToTensor()(img).unsqueeze(0)
    return tensor

def show_model_in_tensorboard(observation_space: gym.spaces.Discrete, model: AlphaZeroModel, writer, step):
    outputs = investigate_model(observation_space, model)
    value_fig = plot_value_network(outputs)
    policy_fig = plot_policy_network(outputs)
    value_tensor = plot_to_tensor(value_fig)
    policy_tensor = plot_to_tensor(policy_fig)
    writer.add_image('value_network', value_tensor, step, dataformats='NCHW')
    writer.add_image('policy_network', policy_tensor, step, dataformats='NCHW')
