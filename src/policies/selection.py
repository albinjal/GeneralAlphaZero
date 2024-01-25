import numpy as np
import torch as th

from core.node import Node
from policies.policies import OptionalPolicy, PolicyDistribution
from policies.utility_functions import policy_value


class UCT(OptionalPolicy):
    def __init__(self, c: float):
        self.c = c

    def sample(self, node: Node) -> np.int64 | None:
        # if not fully expanded, return None
        if not node.is_fully_expanded():
            return None

        # if fully expanded, return the action with the highest UCB value
        # Idea: potentially mess around with making this stochastic
        return max(node.children, key=lambda action: self.ucb(node, action))

    def ucb(self, node: Node, action: np.int64) -> float:
        child = node.children[action]
        return child.default_value() + self.c * (node.visits / child.visits) ** 0.5


class PUCT(OptionalPolicy):
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

class PolicyUCT(UCT):
    def __init__(self, c: float, policy: PolicyDistribution, discount_factor: float = 1.0):
        super().__init__(c)
        self.policy = policy
        self.discount_factor = discount_factor

    def ucb(self, node: Node, action: np.int64) -> float:
        child = node.children[action]
        # replace the default value with the policy value
        return policy_value(child, self.policy, self.discount_factor) + self.c * (node.visits / child.visits) ** 0.5



class PolicyPUCT(PUCT):
    def __init__(self, c: float, policy: PolicyDistribution, discount_factor: float = 1.0):
        super().__init__(c)
        self.policy = policy
        self.discount_factor = discount_factor

    def puct(self, node: Node, action: np.int64, dirichlet: th.Tensor | None) -> float:
        child = node.children[action]
        if dirichlet is None:
            prior = node.prior_policy[action]
        else:
            prior = node.prior_policy[action] * (1.0 - self.dir_alpha) + dirichlet[action] * self.dir_alpha
        val = policy_value(child, self.policy, self.discount_factor)

        return val + self.c * prior * (node.visits**0.5) / (
            child.visits + 1
        )
