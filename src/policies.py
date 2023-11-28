from abc import ABC, abstractmethod
from typing import Generic, TypeVar
import torch as th
import numpy as np
from node import Node

ObservationType = TypeVar("ObservationType")

class Policy(ABC, Generic[ObservationType]):
    """Take a node and return an action"""
    @abstractmethod
    def __call__(self, node: Node[ObservationType]) -> np.int64:
        pass

class PolicyDistribution(Policy[ObservationType]):
    """Also lets us view the full distribution of the policy, not only sample from it.
    When we have the distribution, we can choose how to sample from it.
    We can either sample stochasticly from distribution or deterministically choose the action with the highest probability.
    We can also apply softmax with temperature to the distribution.
    """

    def __call__(self, node: Node[ObservationType]) -> np.int64:
        """
        Returns a random action from the distribution
        """
        return np.random.choice(node.action_space.n, p=self.distribution(node))

    @abstractmethod
    def distribution(self, node: Node[ObservationType]) -> th.Tensor:
        """The distribution of the policy. Must sum to 1 and be all positive."""
        pass



class SelectionPolicy(ABC, Generic[ObservationType]):
    @abstractmethod
    def __call__(self, node: Node[ObservationType]) -> np.int64 | None:
        pass


class UCB(SelectionPolicy[ObservationType]):
    def __init__(self, c: float):
        self.c = c

    def __call__(self, node: Node[ObservationType]) -> np.int64 | None:
        # if not fully expanded, return None
        if not node.is_fully_expanded():
            return None

        # if fully expanded, return the action with the highest UCB value
        # Idea: potentially mess around with making this stochastic
        return max(node.children, key=lambda action: self.ucb(node, action))

    def ucb(self, node: Node[ObservationType], action: np.int64) -> float:
        child = node.children[action]
        return child.default_value() + self.c * (node.visits / child.visits) ** 0.5


class RandomPolicy(Policy[ObservationType]):
    def __call__(self, node: Node[ObservationType]) -> np.int64:
        return node.action_space.sample()


class DefaultExpansionPolicy(Policy[ObservationType]):
    def __call__(self, node: Node[ObservationType]) -> np.int64:
        # returns a uniformly random unexpanded action
        return node.sample_unexplored_action()


class DefaultTreeEvaluator(Policy[ObservationType]):
    # the default tree evaluator selects the action with the most visits
    def __call__(self, node: Node[ObservationType]) -> np.int64:
        return max(node.children, key=lambda action: node.children[action].visits)
