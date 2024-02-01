import numpy as np
from core.node import Node
from policies.policies import Policy



class DefaultExpansionPolicy(Policy):
    def sample(self, node: Node) -> np.int64:
        # returns a uniformly random unexpanded action
        return node.sample_unexplored_action()

class ExpandFromPriorPolicy(Policy):
    def sample(self, node: Node):
        prior = node.prior_policy
        # return the action with the highest prior that has not been expanded yet
        for action in reversed(np.argsort(prior)):
            action = np.int64(action)
            if action not in node.children:
                return action


expansion_policy_dict = {
    "default": DefaultExpansionPolicy,
    "fromprior": ExpandFromPriorPolicy,
}
