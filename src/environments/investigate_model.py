from collections import Counter
import torch as th
import numpy as np
import matplotlib.pyplot as plt
from az.model import AlphaZeroModel

from environments.observation_embeddings import CoordinateEmbedding, ObservationEmbedding




@th.no_grad()
def investigate_model(model: AlphaZeroModel):
    """
    returns a dict of {obs: value} for each obs in the observation space
    """
    tensor_observations = []
    assert isinstance(model.observation_embedding, CoordinateEmbedding)

    # Convert each observation into a tensor and add it to the list
    for obs in range(model.observation_embedding.observation_space.n):
        tensor_obs = model.observation_embedding.obs_to_tensor(obs, dtype=th.float32)
        tensor_observations.append(tensor_obs)

    # Stack all tensor observations into a single batch
    batch = th.stack(tensor_observations)

    # Pass the batch through the model
    values, policies = model(batch)


    return {obs: (value, policy) for obs, value, policy in zip(range(model.observation_embedding.observation_space.n), values, policies)}

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

def plot_value_network(outputs, nrows=4, ncols=12, title = "Cliff Walking Value Network"):
    plt.ioff()
    grid = np.zeros((nrows, ncols))
    for state, value in outputs.items():
        row, col = divmod(state, ncols)
        grid[row, col] = value[0]
    fig, ax = create_figure_and_axes()
    for i in range(nrows):
        for j in range(ncols):
            ax.text(j, i, f"{grid[i, j]:.1f}", ha="center", va="center", color="blue")
    plot_image(fig, ax, grid, title)
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



def plot_visits_with_counter(
    visit_counts: Counter,
    observation_embedding: CoordinateEmbedding,
    step,
    title="State Visit Counts",
):
    grid = np.zeros((observation_embedding.nrows, observation_embedding.ncols))

    for obs in range(observation_embedding.observation_space.n):
        obs_tensor = tuple(
            observation_embedding.obs_to_tensor(obs, dtype=th.float32).tolist()
        )
        count = visit_counts.get(obs_tensor, 0)
        row, col = divmod(obs, observation_embedding.ncols)
        grid[row, col] = count

    fig, ax = plt.subplots()
    ax.imshow(grid, cmap="viridis", interpolation="nearest")
    ax.set_title(f"{title}, Step: {step}")
    for obs in range(observation_embedding.observation_space.n):
        row, col = divmod(obs, observation_embedding.ncols)
        ax.text(
            col, row, f"{grid[row, col]:.0f}", ha="center", va="center", color="white"
        )
    plt.tight_layout()
    return fig
