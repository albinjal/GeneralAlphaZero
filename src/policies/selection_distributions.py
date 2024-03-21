import numpy as np
import torch as th

from core.node import Node
from policies.policies import PolicyDistribution
from policies.utility_functions import get_children_policy_values, get_children_visits, get_transformed_default_values, policy_value

# use distributional selection policies instead of OptionalPolicy
class SelectionPolicy(PolicyDistribution):
    def __init__(self, *args, temperature: float = 0, **kwargs) -> None:
        # by default, we use argmax in selection
        super().__init__(*args, temperature=temperature, **kwargs)


class UCT(SelectionPolicy):
    def __init__(self, c: float, *args,  **kwargs):
        super().__init__(*args, **kwargs)
        self.c = c

    def Q(self, node: Node) -> th.Tensor:
        return get_transformed_default_values(node, self.value_transform)

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
        return get_children_policy_values(node, self.policy, self.discount_factor, self.value_transform)


class PUCT(UCT):
    def _probs(self, node: Node) -> th.Tensor:
        child_visits = get_children_visits(node)
        # if any child_visit is 0
        unvisited = child_visits == 0
        if th.any(unvisited):
            return node.prior_policy * unvisited

        return self.Q(node) + self.c * node.prior_policy * th.sqrt(th.tensor(node.visits)) / (child_visits + 1)


class PolicyPUCT(PolicyUCT, PUCT):
    pass



selection_dict_fn = lambda c, policy, discount, value_transform: {
    "UCT": UCT(c, temperature=0.0, value_transform=value_transform),
    "PUCT": PUCT(c, temperature=0.0, value_transform=value_transform),
    "PolicyUCT": PolicyUCT(c, policy=policy, discount_factor=discount,temperature=0.0, value_transform=value_transform),
    "PolicyPUCT": PolicyPUCT(c, policy=policy, discount_factor=discount,temperature=0.0, value_transform=value_transform),
}
