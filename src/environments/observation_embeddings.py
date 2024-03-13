
from abc import ABC, abstractmethod
import gymnasium as gym
import numpy as np
import torch as th




class ObservationEmbedding(ABC):
    observation_space: gym.Space

    def __init__(self, observation_space: gym.Space) -> None:
        self.observation_space = observation_space

    @abstractmethod
    def obs_to_tensor(observation) -> th.Tensor:
        pass

    @abstractmethod
    def obs_dim() -> int:
        pass



class DefaultEmbedding(ObservationEmbedding):
    def obs_to_tensor(self, observation, *args, **kwargs):
        return th.tensor(
            gym.spaces.flatten(self.observation_space, observation),
            *args,
            **kwargs,
        )

    def obs_dim(self):
        return gym.spaces.flatdim(self.observation_space)


class CoordinateEmbedding(ObservationEmbedding):
    ncols: int
    nrows: int
    observation_space: gym.spaces.Discrete

    def __init__(self, observation_space: gym.spaces.Discrete, *args, ncols=8, **kwargs) -> None:
        super().__init__(observation_space, *args, **kwargs)
        self.ncols = ncols
        self.nrows = observation_space.n // ncols
        print(f"nrows: {self.nrows}, ncols: {self.ncols}")


    def obs_to_tensor(self, observation, *args, **kwargs):
        """
        Returns a tensor of shape (2,) with the coordinates of the observation
        """
        cords = divmod(observation, self.ncols)
        # make cords between -1 and 1
        # cols between 0 and ncols-1, rows between 0 and nrows-1
        cords = (np.array(cords) / np.array([self.nrows-1, self.ncols-1])) * 2 - 1
        return th.tensor(cords, *args, **kwargs)


    def obs_dim(self):
        return 2


embedding_dict = {
    "default": DefaultEmbedding,
    "coordinate": CoordinateEmbedding,
}
