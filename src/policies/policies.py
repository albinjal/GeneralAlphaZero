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
    temperature: float
    def __init__(self, temperature: float = None) -> None:
        super().__init__()
        self.temperature = temperature


    def sample(self, node: Node) -> th.Tensor:
        """
        Returns a random action from the distribution
        """
        return self.softmaxed_distribution(node).sample()

    @abstractmethod
    def _distribution(self, node: Node, include_self=False) -> th.distributions.Categorical:
        """The distribution of the policy. Must sum to 1 and be all positive."""
        pass

    def softmaxed_distribution(self, node: Node, *args, **kwargs) -> th.distributions.Categorical:
        """Returns the softmaxed distribution of the policy"""
        if self.temperature is None:
            return self._distribution(node, *args, **kwargs)
        elif self.temperature == 0:
            # return a uniform distribution over the actions with the highest probability
            logits = self._distribution(node, *args, **kwargs).logits
            max_logits = th.max(logits)
            return th.distributions.Categorical(logits == max_logits)
        else:
            return th.distributions.Categorical(
                logits=self._distribution(node, *args, **kwargs).logits / self.temperature
            )


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
