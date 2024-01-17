from abc import ABC, abstractmethod
from typing import Generic, TypeVar
import torch as th
import numpy as np
from node import Node

ObservationType = TypeVar("ObservationType")


class Policy(ABC, Generic[ObservationType]):
    def __call__(self, node: Node[ObservationType]) -> np.int64:
        return self.sample(node)

    @abstractmethod
    def sample(self, node: Node[ObservationType]) -> np.int64:
        """Take a node and return an action"""


class PolicyDistribution(Policy[ObservationType]):
    """Also lets us view the full distribution of the policy, not only sample from it.
    When we have the distribution, we can choose how to sample from it.
    We can either sample stochasticly from distribution or deterministically choose the action with the highest probability.
    We can also apply softmax with temperature to the distribution.
    """

    def sample(self, node: Node[ObservationType]) -> th.Tensor:
        """
        Returns a random action from the distribution
        """
        return self.distribution(node).sample()

    @abstractmethod
    def distribution(self, node: Node[ObservationType], include_self=False) -> th.distributions.Categorical:
        """The distribution of the policy. Must sum to 1 and be all positive."""
        pass


class OptionalPolicy(ABC, Generic[ObservationType]):
    def __call__(self, node: Node[ObservationType]) -> np.int64 | None:
        return self.sample(node)

    @abstractmethod
    def sample(self, node: Node[ObservationType]) -> np.int64 | None:
        """Take a node and return an action or None if no action is chosen"""
        pass


class UCT(OptionalPolicy[ObservationType]):
    def __init__(self, c: float):
        self.c = c

    def sample(self, node: Node[ObservationType]) -> np.int64 | None:
        # if not fully expanded, return None
        if not node.is_fully_expanded():
            return None

        # if fully expanded, return the action with the highest UCB value
        # Idea: potentially mess around with making this stochastic
        return max(node.children, key=lambda action: self.ucb(node, action))

    def ucb(self, node: Node[ObservationType], action: np.int64) -> float:
        child = node.children[action]
        return child.default_value() + self.c * (node.visits / child.visits) ** 0.5


class PUCT(OptionalPolicy[ObservationType]):
    def __init__(self, c: float, dir_alpha: float = 0.0):
        self.c = c
        self.dir_alpha = dir_alpha


    def sample(self, node: Node) -> np.int64 | None:
        # if not fully expanded, return None
        if not node.is_fully_expanded():
            return None

        if self.dir_alpha != 0.0:
            # sample from the dirichlet distribution
            dirichlet = th.distributions.dirichlet.Dirichlet(
                th.ones(int(node.action_space.n)) * self.dir_alpha
            ).sample()

            # if fully expanded, return the action with the highest UCB value
            # Idea: potentially mess around with making this stochastic
            return max(node.children, key=lambda action: self.puct(node, action, dirichlet))

        else:
            return max(node.children, key=lambda action: self.puct(node, action, None))

    # TODO: this can def be sped up (calculate the denominator once)
    def puct(self, node: Node, action: np.int64, dirichlet: th.Tensor | None) -> float:
        child = node.children[action]
        if dirichlet is None:
            prior = node.prior_policy[action]
        else:
            prior = node.prior_policy[action] * (1.0 - self.dir_alpha) + dirichlet[action] * self.dir_alpha
        return child.default_value() + self.c * prior * (node.visits**0.5) / (
            child.visits + 1
        )


class RandomPolicy(Policy[ObservationType]):
    def sample(self, node: Node[ObservationType]) -> np.int64:
        return node.action_space.sample()


class DefaultExpansionPolicy(Policy[ObservationType]):
    def sample(self, node: Node[ObservationType]) -> np.int64:
        # returns a uniformly random unexpanded action
        return node.sample_unexplored_action()

class ExpandFromPriorPolicy(Policy[ObservationType]):
    def sample(self, node: Node[ObservationType]):
        prior = node.prior_policy
        # return the action with the highest prior that has not been expanded yet
        for action in reversed(np.argsort(prior)):
            action = np.int64(action)
            if action not in node.children:
                return action


class DefaultTreeEvaluator(PolicyDistribution[ObservationType]):
    # the default tree evaluator selects the action with the most visits
    def distribution(self, node: Node[ObservationType], include_self = False) -> th.distributions.Categorical:
        visits = th.zeros(int(node.action_space.n) + include_self)
        for action, child in node.children.items():
            visits[action] = child.visits

        if include_self:
            visits[-1] = 1

        return th.distributions.Categorical(visits)


class SoftmaxDefaultTreeEvaluator(PolicyDistribution[ObservationType]):
    """
    Same as DefaultTreeEvaluator but with softmax applied to the visits. temperature controls the softmax temperature.
    """

    def __init__(self, temperature: float):
        self.temperature = temperature

    # the default tree evaluator selects the action with the most visits
    def distribution(self, node: Node[ObservationType], include_self=False) -> th.distributions.Categorical:
        visits = th.zeros(int(node.action_space.n), dtype=th.float32)
        for action, child in node.children.items():
            visits[action] = child.visits

        return th.distributions.Categorical(
            th.softmax(visits / self.temperature, dim=-1)
        )

class InverseVarianceTreeEvaluator(PolicyDistribution[ObservationType]):
    """
    Selects the action with the highest inverse variance of the q value.
    Should return the same as the default tree evaluator
    """

    def distribution(self, node: Node[ObservationType], include_self=False) -> th.distributions.Categorical:
        inverse_variances = th.zeros(int(node.action_space.n) + include_self, dtype=th.float32)

        for action, child in node.children.items():
            inverse_variances[action] = 1.0 / independent_policy_value_variance(child, self, 1.0)


        if include_self:
            # check if this is correct
            inverse_variances[-1] = value_evaluation_variance(node)

        return th.distributions.Categorical(
            inverse_variances
        )

# TODO: can improve this implementation
def policy_value(node: Node, policy: PolicyDistribution, discount_factor: float):
    # return the q value the node with the given policy
    # with the defualt tree evaluator, this should return the same as the default value
    pi = policy.distribution(node, include_self=True)

    probabilities = pi.probs
    own_propability = probabilities[-1] # type: ignore
    child_propabilities = probabilities[:-1] # type: ignore
    child_values = th.zeros_like(child_propabilities, dtype=th.float32)
    for action, child in node.children.items():
        child_values[action] = policy_value(child, policy, discount_factor)


    return node.reward + discount_factor * (
        own_propability * node.value_evaluation
        + (child_propabilities * child_values).sum()
    )


def reward_variance(node: Node):
    return 0.0

def value_evaluation_variance(node: Node):
    if node.terminal:
        return 1e-6
    else:
        return 1.0


def independent_policy_value_variance(node: Node, policy: PolicyDistribution, discount_factor: float):
    # return the variance of the q value the node with the given policy
    pi = policy.distribution(node, include_self=True)

    probabilities_squared = pi.probs ** 2 # type: ignore
    own_propability_squared = probabilities_squared[-1]
    child_propabilities_squared = probabilities_squared[:-1]
    child_variances = th.zeros_like(child_propabilities_squared, dtype=th.float32)
    for action, child in node.children.items():
        child_variances[action] = independent_policy_value_variance(child, policy, discount_factor)

    return reward_variance(node) + discount_factor ** 2 * (
        own_propability_squared * value_evaluation_variance(node)
        + (child_propabilities_squared * child_variances).sum()
    )
