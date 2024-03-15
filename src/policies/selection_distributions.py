import numpy as np
import torch as th

from core.node import Node
from policies.policies import PolicyDistribution
from policies.utility_functions import get_children_policy_values, get_children_visits, get_transformed_default_values, policy_value

# use distributional selection policies instead of OptionalPolicy
class SelectionPolicy(PolicyDistribution):
    def __init__(self, temperature: float = 0) -> None:
        # by default, we use argmax in selection
        super().__init__(temperature)



class UCT(SelectionPolicy):
    def __init__(self, c: float,*args,  **kwargs):
        super().__init__(*args, **kwargs)
        self.c = c

    def Q(self, node: Node) -> th.Tensor:
        return get_transformed_default_values(node)

    def _probs(self, node: Node) -> th.Tensor:
        child_visits = get_children_visits(node)
        # if any child_visit is 0
        if th.any(child_visits == 0):
            # return 1 for all children with 0 visits
            return child_visits == 0

        return self.Q(node) + self.c * th.sqrt(th.log(th.tensor(node.visits)) / child_visits)



class PolicyUCT(UCT):
    def __init__(self, *args, policy: PolicyDistribution, discount_factor: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy = policy
        self.discount_factor = discount_factor

    def Q(self, node: Node) -> float:
        return get_children_policy_values(node, self.policy, self.discount_factor)


class PUCT(UCT):
    # def U(self, node: Node, action: int) -> float:
    #     assert node.prior_policy is not None
    #     child = node.children[action]
    #     return self.c * node.prior_policy[action] * (node.visits**0.5) / (child.visits + 1)

    def _probs(self, node: Node) -> th.Tensor:
        child_visits = get_children_visits(node)
        # if any child_visit is 0
        if th.any(child_visits == 0):
            # return 1 for all children with 0 visits
            return node.prior_policy

        return self.Q(node) + self.c * node.prior_policy * th.sqrt(th.tensor(node.visits)) / (child_visits + 1)


class PolicyPUCT(PolicyUCT, PUCT):
    pass



selection_dict_fn = lambda c, policy, discount: {
    "UCT": UCT(c, temperature=0.0),
    "PUCT": PUCT(c, temperature=0.0),
    "PolicyUCT": PolicyUCT(c, policy=policy, discount_factor=discount,temperature=0.0),
    "PolicyPUCT": PolicyPUCT(c, policy=policy, discount_factor=discount,temperature=0.0),
}
