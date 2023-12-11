import torch as th
import gymnasium as gym

def obs_to_tensor(observation_space, observation, *args, **kwargs):
    return th.tensor(
        gym.spaces.flatten(observation_space, observation),
        *args,
        **kwargs,
    )


