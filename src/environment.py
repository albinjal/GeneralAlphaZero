import torch as th
import gymnasium as gym

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
