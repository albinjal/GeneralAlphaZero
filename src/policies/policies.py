from abc import ABC, abstractmethod
import torch as th
import numpy as np
from core.node import Node


def add_self_to_probs(node: Node, probs: th.Tensor) -> th.Tensor:
    """
    Takes the current policy and adds one extra value to it, which is the probability of selecting the node itself.
    Should return a tensor with one extra value at the end
    The default choice is to set it to 1/visits
    Note that policy is not yet normalized, so we can't just add 1/visits to the last value
    """
    self_prob = probs.sum() / (node.visits - 1)
    return th.cat([probs, th.tensor([self_prob])])


class Policy(ABC):
    def __call__(self, node: Node) -> int:
        return self.sample(node)

    @abstractmethod
    def sample(self, node: Node) -> int:
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


    def sample(self, node: Node) -> int:
        """
        Returns a random action from the distribution
        """
        return int(self.softmaxed_distribution(node).sample().item())

    @abstractmethod
    def _probs(self, node: Node) -> th.Tensor:
        """
        Returns the relative probabilities of the actions
        """
        pass

    def cleaned_probs(self, node: Node, include_self: bool) -> th.Tensor:
        """
        Relative probabilities with self handling
        """
        # policy for leaf nodes
        if include_self and len(node.children) == 0:
            probs = th.zeros(int(node.action_space.n) + include_self, dtype=th.float32)
            probs[-1] = 1.0
            return probs

        probs = self._probs(node)
        if include_self:
            probs = add_self_to_probs(node, probs)

        assert probs.shape[-1] == node.action_space.n + include_self, f"Probs shape: {probs.shape}, action space: {node.action_space.n}, include_self: {include_self}"
        return probs

    def distribution(self, node: Node, include_self) -> th.distributions.Categorical:
        return th.distributions.Categorical(probs=self.cleaned_probs(node, include_self))

    def softmaxed_distribution(self, node: Node, include_self=False, **kwargs) -> th.distributions.Categorical:
        """Returns the softmaxed distribution of the policy"""

        relative_probs = self.cleaned_probs(node, include_self)

        if self.temperature is None:
            return th.distributions.Categorical(probs=relative_probs)
        elif self.temperature == 0.0 :
            # return a uniform distribution over the actions with the highest probability
            max_logits = th.max(relative_probs)
            return th.distributions.Categorical(probs=relative_probs == max_logits)
        else:
            return th.distributions.Categorical(
                logits=relative_probs / self.temperature
            )

class OptionalPolicy(ABC):
    def __call__(self, node: Node) -> int | None:
        return self.sample(node)

    @abstractmethod
    def sample(self, node: Node) -> int | None:
        """Take a node and return an action or None if no action is chosen"""
        pass



class RandomPolicy(Policy):
    def sample(self, node: Node) -> int:
        return node.action_space.sample()
