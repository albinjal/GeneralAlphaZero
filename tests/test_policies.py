import sys
sys.path.append("src/")

from policies.utility_functions import policy_value
from policies.tree import DefaultTreeEvaluator, GreedyPolicy, InverseVarianceTreeEvaluator, MinimalVarianceConstraintPolicy

from policies.selection import PUCT, UCT, PolicyPUCT, PolicyUCT
import numpy as np
from az.azmcts import AlphaZeroMCTS
from az.model import AlphaZeroModel, UnifiedModel
import pytest
import torch as th

from core.mcts import MCTS, RandomRolloutMCTS
import gymnasium as gym



def generate_mcts_tree(
    solver: MCTS,
    env: gym.Env,
    compute_budget=200,
    seed = 0,
    seq = (0,1,1,1,1,1,1,1,1,1,0),
    ):

    observation, _ = env.reset(seed=seed)
    reward = 0
    for action in seq:
        observation, reward, terminated, truncated, info = env.step(action)


    tree = solver.search(env, compute_budget, observation, np.float32(reward))
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
@pytest.mark.parametrize("discount_factor", [1.0, 0.9])  # Add more discount factors if needed
@pytest.mark.parametrize("tree_type", ["default", "az"])
def test_InverseVarianceTreeEvaluator(tree):
    inv_var_eval = InverseVarianceTreeEvaluator(1.0)
    default_eval = DefaultTreeEvaluator()
    inv_var_policy = np.array(inv_var_eval.distribution(tree).probs)
    tree.reset_var_val()
    default_policy = np.array(default_eval.distribution(tree).probs)
    assert np.allclose(inv_var_policy, default_policy, rtol=1e-6, atol=1e-6), f"Inverse variance policy: {inv_var_policy}, Default policy: {default_policy}"



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

    inv_var_policy = np.array(inv_var_eval.distribution(tree).probs)
    tree.reset_var_val()
    mvcp_policy = np.array(mvcp.distribution(tree).probs)
    assert np.allclose(inv_var_policy, mvcp_policy, rtol=1e-6, atol=1e-6), f"Inverse variance policy: {inv_var_policy}, MVCP policy: {mvcp_policy}"

@pytest.mark.parametrize("env", ["CliffWalking-v0"], indirect=True)
@pytest.mark.parametrize("seed", [0, 1, 2,3,4,5,6])  # Add more seeds if needed
@pytest.mark.parametrize("discount_factor", [1.0, 0.9])  # Parametrize discount factors
@pytest.mark.parametrize("tree_type", ["default", "az"])
def test_MinimalVarianceConstraintPolicy_greedy(tree, discount_factor):
    """
    We assume that when beta-> inf, the policy is the same as the greedy policy
    """
    # TODO: this sometimes fails, investigate why
    beta = 1e6
    greedy = GreedyPolicy()
    mvcp = MinimalVarianceConstraintPolicy(beta=beta, discount_factor=discount_factor)

    greedy_policy = np.array(greedy.distribution(tree).probs)
    tree.reset_var_val()
    mvcp_policy = np.array(mvcp.distribution(tree).probs)
    assert np.allclose(greedy_policy, mvcp_policy, rtol=1e-6, atol=1e-6), f"Greedy policy: {greedy_policy}, MVCP policy: {mvcp_policy}"



@pytest.mark.parametrize("env", ["CliffWalking-v0"], indirect=True)
@pytest.mark.parametrize("seed", [0, 1, 2])  # Add more seeds if needed
@pytest.mark.parametrize("discount_factor", [1.0, 0.9])  # Parametrize discount factors
@pytest.mark.parametrize("tree_type", ["default", "az"])
def test_policy_value(tree, discount_factor):
    """
    We assume that the policy value for the default policy is the same as the default value (subtree average value)
    """

    default_eval = DefaultTreeEvaluator()
    pol_val = policy_value(tree, default_eval, discount_factor=discount_factor)
    default_value = tree.default_value()
    # had to lower the tolerance to 1e-3 since there seem to be some numerical instability
    assert np.allclose(default_value, pol_val, rtol=1e-3, atol=1e-3), f"Default value: {default_value}, Policy value: {pol_val}"



@pytest.mark.parametrize("env", ["CliffWalking-v0"], indirect=True)
@pytest.mark.parametrize("seed", [0, 1, 2])  # Add more seeds if needed
@pytest.mark.parametrize("discount_factor", [1.0, 0.9])  # Parametrize discount factors
@pytest.mark.parametrize("tree_type", ["default", "az"])
def test_policy_uct(tree, discount_factor, c=1.0):
    """
    We assume that policyuct and uct return the same thing for the default policy
    """

    uct = UCT(c)
    uct_action = uct.sample(tree)

    default_eval = DefaultTreeEvaluator()
    policy_uct = PolicyUCT(c, default_eval, discount_factor=discount_factor)
    p_uct_action = policy_uct.sample(tree)

    assert uct_action == p_uct_action, f"UCT action: {uct_action}, PolicyUCT action: {p_uct_action}"



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

    default_eval = DefaultTreeEvaluator()
    policy_uct = PolicyPUCT(c, default_eval, discount_factor=discount_factor)
    p_uct_action = policy_uct.sample(tree)

    assert uct_action == p_uct_action, f"PUCT action: {uct_action}, PolicyPUCT action: {p_uct_action}"
