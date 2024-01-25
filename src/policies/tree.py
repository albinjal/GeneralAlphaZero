import torch as th

from core.node import Node
from policies.policies import PolicyDistribution
from policies.utility_functions import independent_policy_value_variance, policy_value, value_evaluation_variance


class DefaultTreeEvaluator(PolicyDistribution):
    # the default tree evaluator selects the action with the most visits
    def distribution(self, node: Node, include_self = False) -> th.distributions.Categorical:
        visits = th.zeros(int(node.action_space.n) + include_self)
        for action, child in node.children.items():
            visits[action] = child.visits

        if include_self:
            visits[-1] = 1

        return th.distributions.Categorical(visits)


class SoftmaxDefaultTreeEvaluator(PolicyDistribution):
    """
    Same as DefaultTreeEvaluator but with softmax applied to the visits. temperature controls the softmax temperature.
    """

    def __init__(self, temperature: float):
        self.temperature = temperature

    # the default tree evaluator selects the action with the most visits
    def distribution(self, node: Node, include_self=False) -> th.distributions.Categorical:
        visits = th.zeros(int(node.action_space.n), dtype=th.float32)
        for action, child in node.children.items():
            visits[action] = child.visits

        return th.distributions.Categorical(
            th.softmax(visits / self.temperature, dim=-1)
        )

class InverseVarianceTreeEvaluator(PolicyDistribution):
    """
    Selects the action with the highest inverse variance of the q value.
    Should return the same as the default tree evaluator
    """
    def __init__(self, discount_factor = 1.0):
        self.discount_factor = discount_factor

    def distribution(self, node: Node, include_self=False) -> th.distributions.Categorical:
        inverse_variances = th.zeros(int(node.action_space.n) + include_self, dtype=th.float32)

        for action, child in node.children.items():
            inverse_variances[action] = 1.0 / independent_policy_value_variance(child, self, self.discount_factor)


        if include_self:
            # check if this is correct, might need to add discount factor
            inverse_variances[-1] = 1.0 / (self.discount_factor ** 2 * value_evaluation_variance(node))

        return th.distributions.Categorical(
            inverse_variances
        )


# minimal-variance constraint policy
class MinimalVarianceConstraintPolicy(PolicyDistribution):
    """
    Selects the action with the highest inverse variance of the q value.
    Should return the same as the default tree evaluator
    """
    def __init__(self, beta: float, discount_factor = 1.0):
        self.beta = beta
        self.discount_factor = discount_factor

    def distribution(self, node: Node, include_self=False) -> th.distributions.Categorical:
        if len(node.children) == 0:
            d = th.zeros(int(node.action_space.n) + include_self, dtype=th.float32)
            d[-1] = 1.0
            return th.distributions.Categorical(d)

        vals = th.zeros(int(node.action_space.n) + include_self, dtype=th.float32)
        inv_vars = th.zeros_like(vals, dtype=th.float32)

        for action, child in node.children.items():
            pi = self.distribution(child, include_self=True)
            vals[action] = policy_value(child, pi, self.discount_factor)
            inv_vars[action] = 1/independent_policy_value_variance(child, pi, self.discount_factor)

        # if include_self:
        #     # TODO: this probably has to be updated
        #     vals[-1] = th.tensor(node.value_evaluation)
        #     inv_vars[-1] = 1.0 / (self.discount_factor ** 2 * value_evaluation_variance(node))

        # risk for numerical instability if vals are large/small
        # Solution: subtract the mean
        # This should be equivalent to muliplying by a constant which we can ignore
        policy = th.exp(self.beta * (vals - vals.max())) * inv_vars


        # make a numerical check. If the policy is all 0, then we should return a uniform distribution
        if policy.sum() == 0:
            policy[:-1] = th.ones_like(policy[:-1])


        # # if we have some infinities, we should return a uniform distribution over the infinities
        # if th.isinf(policy).any():
        #     return th.distributions.Categorical(
        #         th.isinf(policy)
        #     )

        if include_self:
            # set it to 1/visits
            policy[-1] = policy.sum() / (node.visits - 1)


        return th.distributions.Categorical(
            policy
        )
