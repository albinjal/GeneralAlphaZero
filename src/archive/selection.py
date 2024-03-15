import numpy as np
import torch as th

from core.node import Node
from policies.policies import OptionalPolicy, PolicyDistribution
from policies.utility_functions import policy_value


class UCB(OptionalPolicy):
    def Q(self, node: Node, action: int) -> float:
        raise NotImplementedError

    def U(self, node: Node, action: int) -> float:
        raise NotImplementedError

    def ucb_score(self, node: Node, action: int) -> float:
        return self.Q(node, action) + self.U(node, action)

    def sample(self, node: Node) -> int | None:
        # if not fully expanded, return None
        if not node.is_fully_expanded():
            return None

        return max(node.children, key=lambda action: self.ucb_score(node, action))


class UCT(UCB):
    def __init__(self, c: float,*args,  **kwargs):
        super().__init__(*args, **kwargs)
        self.c = c

    def Q(self, node: Node, action: int) -> float:
        child = node.children[action]
        return child.default_value()

    def U(self, node: Node, action: int) -> float:
        child = node.children[action]
        return self.c * (np.log(node.visits) / child.visits) ** 0.5


class PolicyUCT(UCT):
    def __init__(self, *args, policy: PolicyDistribution, discount_factor: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy = policy
        self.discount_factor = discount_factor

    def Q(self, node: Node, action: int) -> float:
        child = node.children[action]
        return policy_value(child, self.policy, self.discount_factor)


class PUCT(UCT):

    def U(self, node: Node, action: int) -> float:
        assert node.prior_policy is not None
        child = node.children[action]
        return self.c * node.prior_policy[action] * (node.visits**0.5) / (child.visits + 1)


class PolicyPUCT(PolicyUCT, PUCT):
    pass



selection_dict_fn = lambda c, policy, discount: {
    "UCT": UCT(c),
    "PUCT": PUCT(c),
    "PolicyUCT": PolicyUCT(c, policy=policy, discount_factor=discount),
    "PolicyPUCT": PolicyPUCT(c, policy=policy, discount_factor=discount),
}
