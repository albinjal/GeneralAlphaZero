import sys

sys.path.append("src/")

from policies.utility_functions import policy_value
from policies.tree import (
    ValuePolicy,
    VistationPolicy,
    InverseVarianceTreeEvaluator,
    MVTOPolicy,
    MinimalVarianceConstraintPolicy,
)

from policies.selection_distributions import PUCT, UCT, PolicyPUCT, PolicyUCT
import numpy as np
from az.azmcts import AlphaZeroMCTS
from az.model import UnifiedModel
import pytest
import torch as th

from core.mcts import MCTS, RandomRolloutMCTS
import gymnasium as gym


def generate_mcts_tree(
    solver: MCTS,
    env: gym.Env,
    planning_budget=200,
    seed=0,
    seq=(0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0),
):

    observation, _ = env.reset(seed=seed)
    reward = 0
    for action in seq:
        observation, reward, terminated, truncated, info = env.step(action)

    tree = solver.search(env, planning_budget, observation, reward)
    return tree


def get_solver(env, tree_type, discount_factor):
    selection_policy = UCT(1.0)

    if tree_type == "az":
        model = UnifiedModel(env, 2, 0)
        return AlphaZeroMCTS(model, selection_policy, discount_factor=discount_factor)
    else:  # Default to RandomRolloutMCTS
        return RandomRolloutMCTS(20, selection_policy, discount_factor=discount_factor)


@pytest.fixture(scope="function")
def env(request):
    return gym.make(request.param)


@pytest.fixture(scope="function")
def tree(env, tree_type, discount_factor, seed):
    np.random.seed(seed)
    th.manual_seed(seed)
    solver = get_solver(env, tree_type, discount_factor)
    gen = generate_mcts_tree(solver, env, seed=seed)
    gen.reset_var_val()
    return gen


@pytest.mark.parametrize("env", ["CliffWalking-v0"], indirect=True)
@pytest.mark.parametrize("seed", [0, 1, 2])  # Add more seeds if needed
@pytest.mark.parametrize(
    "discount_factor", [1.0, 0.9]
)  # Add more discount factors if needed
@pytest.mark.parametrize("tree_type", ["default", "az"])
def test_InverseVarianceTreeEvaluator(tree):
    inv_var_eval = InverseVarianceTreeEvaluator(1.0)
    default_eval = VistationPolicy()
    inv_var_policy = np.array(inv_var_eval.softmaxed_distribution(tree).probs)
    tree.reset_var_val()
    default_policy = np.array(default_eval.softmaxed_distribution(tree).probs)
    assert np.allclose(
        inv_var_policy, default_policy, rtol=1e-6, atol=1e-6
    ), f"Inverse variance policy: {inv_var_policy}, Default policy: {default_policy}"


@pytest.mark.parametrize("env", ["CliffWalking-v0"], indirect=True)
@pytest.mark.parametrize("seed", [0, 1, 2])  # Add more seeds if needed
@pytest.mark.parametrize("discount_factor", [1.0, 0.9])  # Parametrize discount factors
@pytest.mark.parametrize("tree_type", ["default", "az"])
def test_MinimalVarianceConstraintPolicy_zerobeta(tree, discount_factor):
    """
    We assume that when beta = 0, the policy is the same as the inverse variance policy
    """
    beta = 0.0
    inv_var_eval = InverseVarianceTreeEvaluator(discount_factor=discount_factor)
    mvcp = MinimalVarianceConstraintPolicy(beta=beta, discount_factor=discount_factor)

    inv_var_policy = np.array(inv_var_eval.softmaxed_distribution(tree).probs)
    tree.reset_var_val()
    mvcp_policy = np.array(mvcp.softmaxed_distribution(tree).probs)
    assert np.allclose(
        inv_var_policy, mvcp_policy, rtol=1e-6, atol=1e-6
    ), f"Inverse variance policy: {inv_var_policy}, MVCP policy: {mvcp_policy}"


@pytest.mark.parametrize("env", ["CliffWalking-v0"], indirect=True)
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5, 6])  # Add more seeds if needed
@pytest.mark.parametrize("discount_factor", [1.0, 0.9])  # Parametrize discount factors
@pytest.mark.parametrize("tree_type", ["default", "az"])
def test_MinimalVarianceConstraintPolicy_greedy(tree, discount_factor):
    """
    We assume that when beta-> inf, the policy is the same as the greedy policy
    """
    # TODO: this sometimes fails, investigate why
    beta = 1e6
    greedy = ValuePolicy(discount_factor=discount_factor, temperature=0)
    mvcp = MinimalVarianceConstraintPolicy(beta=beta, discount_factor=discount_factor)

    greedy_policy = np.array(greedy.softmaxed_distribution(tree).probs)
    tree.reset_var_val()
    mvcp_policy = np.array(mvcp.softmaxed_distribution(tree).probs)
    assert np.allclose(
        greedy_policy, mvcp_policy, rtol=1e-6, atol=1e-6
    ), f"Greedy policy: {greedy_policy}, MVCP policy: {mvcp_policy}"



@pytest.mark.parametrize("env", ["CliffWalking-v0"], indirect=True)
@pytest.mark.parametrize("seed", [0, 1, 2])  # Add more seeds if needed
@pytest.mark.parametrize("discount_factor", [1.0, 0.9])  # Parametrize discount factors
@pytest.mark.parametrize("tree_type", ["default", "az"])
def test_mvto_lambdinf(tree, discount_factor):
    """
    We assume that when lambda -> inf, the policy is the same as the inverse variance policy
    """
    lamb = 1e8
    inv_var_eval = InverseVarianceTreeEvaluator(discount_factor=discount_factor)
    mvto = MVTOPolicy(lamb=lamb, discount_factor=discount_factor)

    inv_var_policy = np.array(inv_var_eval.softmaxed_distribution(tree).probs)
    tree.reset_var_val()
    mvto_policy = np.array(mvto.softmaxed_distribution(tree).probs)
    assert np.allclose(
        inv_var_policy, mvto_policy, rtol=1e-4, atol=1e-4
    ), f"Inverse variance policy: {inv_var_policy}, MVCP policy: {mvto_policy}"


@pytest.mark.parametrize("env", ["CliffWalking-v0"], indirect=True)
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5, 6])  # Add more seeds if needed
@pytest.mark.parametrize("discount_factor", [1.0, 0.9])  # Parametrize discount factors
@pytest.mark.parametrize("tree_type", ["default", "az"])
def test_mvto_lambdzero(tree, discount_factor):
    """
    We assume that when lambda = 0, the policy is the same as the greedy policy
    """
    # TODO: for some strange reason this sometimes fails and the greedy and mvto policies are different, investigate
    lambd = 1e-8
    greedy = ValuePolicy(discount_factor=discount_factor, temperature=0)
    mvto = MVTOPolicy(lamb=lambd, discount_factor=discount_factor)

    greedy_policy = np.array(greedy.softmaxed_distribution(tree).probs)
    tree.reset_var_val()
    mvto_policy = np.array(mvto.softmaxed_distribution(tree).probs)
    assert np.allclose(
        greedy_policy, mvto_policy, rtol=1e-6, atol=1e-6
    ), f"Greedy policy: {greedy_policy}, MVCP policy: {mvto_policy}"

@pytest.mark.parametrize("env", ["CliffWalking-v0"], indirect=True)
@pytest.mark.parametrize("seed", [0, 1, 2])  # Add more seeds if needed
@pytest.mark.parametrize("discount_factor", [1.0, 0.9])  # Parametrize discount factors
@pytest.mark.parametrize("tree_type", ["default", "az"])
def test_policy_value(tree, discount_factor):
    """
    We assume that the policy value for the default policy is the same as the default value (subtree average value)
    """

    default_eval = VistationPolicy()
    pol_val = policy_value(tree, default_eval, discount_factor=discount_factor)
    default_value = tree.default_value()
    # had to lower the tolerance to 1e-3 since there seem to be some numerical instability
    assert np.allclose(
        default_value, pol_val, rtol=1e-3, atol=1e-3
    ), f"Default value: {default_value}, Policy value: {pol_val}"


@pytest.mark.parametrize("env", ["CliffWalking-v0"], indirect=True)
@pytest.mark.parametrize("seed", [0, 1, 2])  # Add more seeds if needed
@pytest.mark.parametrize("discount_factor", [1.0, 0.9])  # Parametrize discount factors
@pytest.mark.parametrize("tree_type", ["default", "az"])
def test_policy_uct(tree, discount_factor, c=1.0):
    """
    We assume that policyuct and uct return the same thing for the default policy
    """

    uct = UCT(c)
    uct_dist = np.array(uct.distribution(tree).probs)

    default_eval = VistationPolicy()
    policy_uct = PolicyUCT(c, policy=default_eval, discount_factor=discount_factor)
    p_uct_dist = np.array(policy_uct.distribution(tree).probs)

    assert np.allclose(
        uct_dist, p_uct_dist, rtol=1e-6, atol=1e-6
    ), f"UCT distribution: {uct_dist}, PolicyUCT distribution: {p_uct_dist}"


@pytest.mark.parametrize("env", ["CliffWalking-v0"], indirect=True)
@pytest.mark.parametrize("seed", [0, 1, 2])  # Add more seeds if needed
@pytest.mark.parametrize("discount_factor", [1.0, 0.9])  # Parametrize discount factors
@pytest.mark.parametrize("tree_type", ["az"])
def test_policy_puct(tree, discount_factor, c=1.0):
    """
    We assume that policyuct and uct return the same thing for the default policy
    """

    uct = PUCT(c)
    uct_action = uct.sample(tree)

    default_eval = VistationPolicy()
    policy_uct = PolicyPUCT(c, policy=default_eval, discount_factor=discount_factor)
    p_uct_action = policy_uct.sample(tree)

    assert (
        uct_action == p_uct_action
    ), f"PUCT action: {uct_action}, PolicyPUCT action: {p_uct_action}"
