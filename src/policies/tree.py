from math import nan
import torch as th

from core.node import Node
from policies.policies import PolicyDistribution
from policies.utility_functions import get_children_policy_values, get_children_policy_values_and_inverse_variance, get_children_inverse_variances, get_children_visits, puct_multiplier


class VistationPolicy(PolicyDistribution):
    # the default tree evaluator selects the action with the most visits
    def _probs(self, node: Node) -> th.Tensor:
        return get_children_visits(node)


class InverseVarianceTreeEvaluator(PolicyDistribution):
    """
    Selects the action with the highest inverse variance of the q value.
    Should return the same as the default tree evaluator
    """
    def __init__(self, discount_factor = 1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.discount_factor = discount_factor

    def _probs(self, node: Node) -> th.Tensor:
        return get_children_inverse_variances(node, self, self.discount_factor)


# minimal-variance constraint policy
class MinimalVarianceConstraintPolicy(PolicyDistribution):
    """
    Selects the action with the highest inverse variance of the q value.
    Should return the same as the default tree evaluator
    """
    def __init__(self, beta: float, discount_factor = 1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.discount_factor = discount_factor

    def get_beta(self, node: Node):
        return self.beta

    def _probs(self, node: Node) -> th.Tensor:

        beta = self.get_beta(node)

        normalized_vals, inv_vars = get_children_policy_values_and_inverse_variance(node, self, self.discount_factor)
        probs = th.zeros(int(node.action_space.n), dtype=th.float32)

        for action in node.children:
            probs[action] = th.exp(beta * (normalized_vals[action] - normalized_vals.max())) * inv_vars[action]

        return probs

class MVCP_Dynamic_Beta(MinimalVarianceConstraintPolicy):
    """
    From the mcts as policy optimization paper
    """
    def __init__(self, c: float, discount_factor=1, *args, **kwargs):
        super(MinimalVarianceConstraintPolicy, self).__init__(*args, **kwargs)
        self.c = c
        self.discount_factor = discount_factor

    def get_beta(self, node: Node):
        return 1 / puct_multiplier(self.c, node)

class ReversedRegPolicy(PolicyDistribution):
    """
    From the mcts as policy optimization paper
    """
    def __init__(self, c: float, discount_factor=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c = c
        self.discount_factor = discount_factor


    def _probs(self, node: Node) -> th.Tensor:
        vals = get_children_policy_values(node, self, self.discount_factor)
        probs = th.zeros_like(vals, dtype=th.float32)
        mult = puct_multiplier(self.c, node)

        for action in node.children:
            probs[action] = node.prior_policy[action] * th.exp((vals[action] - vals.max()) / mult)

        probs


class MVTOPolicy(PolicyDistribution):
    """
    Solution to argmax mu + lambda * var
    """

    def __init__(self, lamb: float, discount_factor=1, *args, **kwargs):
        """
        Note that lambda > max_i sum_j (var_j^-1 ( x_j - x_i))/ 2
        """
        super().__init__(*args, **kwargs)
        self.lamb = lamb
        self.discount_factor = discount_factor


    def _probs(self, node: Node) -> th.Tensor:
        vals, inv_vars = get_children_policy_values_and_inverse_variance(node, self, self.discount_factor)
        inv_var_policy = inv_vars / inv_vars.sum()

        probs = th.zeros_like(vals, dtype=th.float32)

        piv_sum = (inv_var_policy * vals).nansum()
        a = .5 / self.lamb

        for action in node.children:
            probs[action] = inv_var_policy[action] + a * inv_vars[action] * (vals[action] - piv_sum)

        if probs.min() < 0:
            # seems like lambda is too small, follow the greedy policy instead
            # alternativly we could just set the negative values to 0
            print("lambda too small, using greedy policy")
            g =  ValuePolicy(self.discount_factor, temperature=0).softmaxed_distribution(node).probs
            return g

        return probs

class ValuePolicy(PolicyDistribution):
    def __init__(self, discount_factor = 1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.discount_factor = discount_factor

    """
    Determinstic policy that selects the action with the highest value
    """
    def _probs(self, node: Node) -> th.Tensor:
        return get_children_policy_values(node, self, self.discount_factor)

tree_dict = {
    "visit": VistationPolicy,
    "inverse_variance": InverseVarianceTreeEvaluator,
    "mvc": MinimalVarianceConstraintPolicy,
    'mvc_dynbeta': MVCP_Dynamic_Beta,
    'reversedregpolicy': ReversedRegPolicy,
    "mvto": MVTOPolicy,
}

tree_eval_dict = lambda param, discount, c=1.0, temperature=None: {
    "visit": VistationPolicy(temperature),
    "inverse_variance": InverseVarianceTreeEvaluator(discount_factor=discount, temperature=temperature),
    "mvc": MinimalVarianceConstraintPolicy(discount_factor=discount, beta=param, temperature=temperature),
    'mvc_dynbeta': MVCP_Dynamic_Beta(c=c, discount_factor=discount, temperature=temperature),
    'reversedregpolicy': ReversedRegPolicy(c=c, discount_factor=discount, temperature=temperature),
    "mvto": MVTOPolicy(lamb=param, discount_factor=discount, temperature=temperature),
}
