import torch as th

from core.node import Node
from policies.policies import PolicyDistribution
from policies.utility_functions import get_children_policy_values, get_children_policy_values_and_inverse_variance, get_children_inverse_variances, get_children_visits, get_transformed_default_values, puct_multiplier
from policies.value_transforms import IdentityValueTransform, ValueTransform


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

        normalized_vals, inv_vars = get_children_policy_values_and_inverse_variance(node, self, self.discount_factor, self.value_transform)
        # for action in node.children:
        #     probs[action] = th.exp(beta * (normalized_vals[action] - normalized_vals.max())) * inv_vars[action]
        logits = beta * th.nan_to_num(normalized_vals)
        probs = inv_vars * th.exp(logits - logits.max())
        return probs
class MinimalVarianceConstraintPolicyPrior(MinimalVarianceConstraintPolicy):
    def _probs(self, node: Node) -> th.Tensor:
        return super()._probs(node) * node.prior_policy


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
        vals = get_children_policy_values(node, self, self.discount_factor, self.value_transform)
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
        vals, inv_vars = get_children_policy_values_and_inverse_variance(node, self, self.discount_factor, self.value_transform)
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
            g =  ValuePolicy(self.discount_factor, temperature=0.0).softmaxed_distribution(node).probs
            return g

        return probs

class ValuePolicy(PolicyDistribution):
    def __init__(self, discount_factor = 1.0, **kwargs):
        assert "temperature" in kwargs and kwargs["temperature"] is not None, "temperature must be set"
        super().__init__(**kwargs)
        self.discount_factor = discount_factor

    """
    Determinstic policy that selects the action with the highest value
    """
    def _probs(self, node: Node) -> th.Tensor:
        vals = get_children_policy_values(node, self, self.discount_factor, self.value_transform)
        return vals

class PriorStdPolicy(PolicyDistribution):
    def Q(self, node: Node) -> th.Tensor:
        pass

    def inv_std(self, node: Node) -> th.Tensor:
        pass

    def Q_inv_std(self, node: Node):
        return self.Q(node), self.inv_std(node)

    def _probs(self, node: Node) -> th.Tensor:
        vals, inv_std = self.Q_inv_std(node)
        transformed_vals = th.nan_to_num(vals * inv_std)
        return node.prior_policy * th.exp(transformed_vals - transformed_vals.max())

class VistationPriorStdPolicy(PriorStdPolicy):
    def __init__(self, sigma: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigma = sigma

    def Q(self, node: Node) -> th.Tensor:
        return get_transformed_default_values(node, self.value_transform)

    def inv_std(self, node: Node) -> th.Tensor:
        return th.sqrt(get_children_visits(node)) / self.sigma


class BellmanPriorStdPolicy(PriorStdPolicy):
    def __init__(self, sigma: float, *args, discount_factor: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigma = sigma
        self.discount_factor = discount_factor

    def Q_inv_std(self, node: Node):
        q, var = get_children_policy_values_and_inverse_variance(node, self, self.discount_factor, self.value_transform)
        return q, th.sqrt(var) / self.sigma

tree_dict = {
    "visit": VistationPolicy,
    "inverse_variance": InverseVarianceTreeEvaluator,
    "mvc": MinimalVarianceConstraintPolicy,
    'mvc_dynbeta': MVCP_Dynamic_Beta,
    'reversedregpolicy': ReversedRegPolicy,
    "mvto": MVTOPolicy,
}

tree_eval_dict = lambda param, discount, c=1.0, temperature=None, value_transform=IdentityValueTransform: {
    "visit": VistationPolicy(temperature, value_transform=value_transform),
    "inverse_variance": InverseVarianceTreeEvaluator(discount_factor=discount, temperature=temperature, value_transform=value_transform),
    "mvc": MinimalVarianceConstraintPolicy(discount_factor=discount, beta=param, temperature=temperature, value_transform=value_transform),
    'mvc_dynbeta': MVCP_Dynamic_Beta(c=c, discount_factor=discount, temperature=temperature, value_transform=value_transform),
    'reversedregpolicy': ReversedRegPolicy(c=c, discount_factor=discount, temperature=temperature, value_transform=value_transform),
    "mvto": MVTOPolicy(lamb=param, discount_factor=discount, temperature=temperature, value_transform=value_transform),
    'visit_prior_std': VistationPriorStdPolicy(sigma=param, temperature=temperature, value_transform=value_transform),
    'bellman_prior_std': BellmanPriorStdPolicy(sigma=param, temperature=temperature, value_transform=value_transform),
    'mvcp': MinimalVarianceConstraintPolicyPrior(discount_factor=discount, beta=param, temperature=temperature, value_transform=value_transform),
}
