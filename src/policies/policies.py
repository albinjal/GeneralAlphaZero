from abc import ABC, abstractmethod
import torch as th
import numpy as np
from core.node import Node



class Policy(ABC):
    def __call__(self, node: Node) -> np.int64:
        return self.sample(node)

    @abstractmethod
    def sample(self, node: Node) -> np.int64:
        """Take a node and return an action"""


class PolicyDistribution(Policy):
    """Also lets us view the full distribution of the policy, not only sample from it.
    When we have the distribution, we can choose how to sample from it.
    We can either sample stochasticly from distribution or deterministically choose the action with the highest probability.
    We can also apply softmax with temperature to the distribution.
    """

    def sample(self, node: Node) -> th.Tensor:
        """
        Returns a random action from the distribution
        """
        return self.distribution(node).sample()

    @abstractmethod
    def distribution(self, node: Node, include_self=False) -> th.distributions.Categorical:
        """The distribution of the policy. Must sum to 1 and be all positive."""
        pass


class OptionalPolicy(ABC):
    def __call__(self, node: Node) -> np.int64 | None:
        return self.sample(node)

    @abstractmethod
    def sample(self, node: Node) -> np.int64 | None:
        """Take a node and return an action or None if no action is chosen"""
        pass



class RandomPolicy(Policy):
    def sample(self, node: Node) -> np.int64:
        return node.action_space.sample()
