import torch as th

from core.node import Node
from policies.policies import PolicyDistribution

# TODO: can improve this implementation
def policy_value(node: Node, policy: PolicyDistribution | th.distributions.Categorical, discount_factor: float):
    # return the q value the node with the given policy
    # with the defualt tree evaluator, this should return the same as the default value

    if node.terminal:
        return th.tensor(node.reward, dtype=th.float32)

    if node.policy_value:
        return node.policy_value

    if isinstance(policy, th.distributions.Categorical):
        pi = policy
    else:
        pi = policy.distribution(node, include_self=True)

    probabilities = pi.probs
    own_propability = probabilities[-1] # type: ignore
    child_propabilities = probabilities[:-1] # type: ignore
    child_values = th.zeros_like(child_propabilities, dtype=th.float32)
    for action, child in node.children.items():
        child_values[action] = policy_value(child, policy, discount_factor)


    val = node.reward + discount_factor * (
        own_propability * node.value_evaluation
        + (child_propabilities * child_values).sum()
    )
    node.policy_value = val
    return val


def reward_variance(node: Node):
    return 0.0

def value_evaluation_variance(node: Node):
    # if we want to duplicate the default tree evaluator, we can return 1 / visits
    # In reality, the variance should be lower for terminal nodes
    if node.terminal:
        return 1.0 / float(node.visits)
    else:
        return 1.0


def independent_policy_value_variance(node: Node, policy: PolicyDistribution | th.distributions.Categorical, discount_factor: float):
    if node.variance is not None:
        return node.variance
    # return the variance of the q value the node with the given policy
    if isinstance(policy, th.distributions.Categorical):
        pi = policy
    else:
        pi = policy.distribution(node, include_self=True)


    probabilities_squared = pi.probs ** 2 # type: ignore
    own_propability_squared = probabilities_squared[-1]
    child_propabilities_squared = probabilities_squared[:-1]
    child_variances = th.zeros_like(child_propabilities_squared, dtype=th.float32)
    for action, child in node.children.items():
        child_variances[action] = independent_policy_value_variance(child, policy, discount_factor)

    var = reward_variance(node) + discount_factor ** 2 * (
        own_propability_squared * value_evaluation_variance(node)
        + (child_propabilities_squared * child_variances).sum()
    )
    node.variance = var
    return var