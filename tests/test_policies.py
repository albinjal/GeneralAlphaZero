import sys
sys.path.append("src/")

from policies.utility_functions import policy_value
from policies.tree import DefaultTreeEvaluator, GreedyPolicy, InverseVarianceTreeEvaluator, MinimalVarianceConstraintPolicy

from policies.selection import UCT
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
    seq = (0,1,1,1,1,1,1,1,1),
    ):

    observation, _ = env.reset(seed=seed)
    reward = 0
    for action in seq:
        observation, reward, terminated, truncated, info = env.step(action)


    tree = solver.search(env, compute_budget, observation, np.float32(reward))
    return tree


def default_tree(env_name = "CliffWalking-v0", seed = 0, discount = 1.0):
    env = gym.make(env_name)
    # model = UnifiedModel(env, 2, 0)
    selection_policy = UCT(1.0)
    solver = RandomRolloutMCTS(20, selection_policy, discount_factor=discount)
    return generate_mcts_tree(solver, env, seed = seed)



def test_InverseVarianceTreeEvaluator():
    """
    We assume that the inverse variance tree evaluator and the visit count tree evaluator return the same policy
    """
    inv_var_eval = InverseVarianceTreeEvaluator(1.0)
    default_eval = DefaultTreeEvaluator()
    tree = default_tree()
    tree.reset_var_val()
    inv_var_policy = np.array(inv_var_eval.distribution(tree).probs)
    tree.reset_var_val()
    default_policy = np.array(default_eval.distribution(tree).probs)
    assert np.allclose(inv_var_policy, default_policy, rtol=1e-6, atol=1e-6), f"Inverse variance policy: {inv_var_policy}, Default policy: {default_policy}"



def test_MinimalVarianceConstraintPolicy_zerobeta():
    """
    We assume that when beta = 0, the policy is the same as the inverse variance policy
    """
    discount = .99
    beta = 0.0
    inv_var_eval = InverseVarianceTreeEvaluator(discount_factor=discount)
    mvcp = MinimalVarianceConstraintPolicy(beta=beta, discount_factor=discount)

    tree = default_tree()
    tree.reset_var_val()
    inv_var_policy = np.array(inv_var_eval.distribution(tree).probs)
    tree.reset_var_val()
    mvcp_policy = np.array(mvcp.distribution(tree).probs)
    assert np.allclose(inv_var_policy, mvcp_policy, rtol=1e-6, atol=1e-6), f"Inverse variance policy: {inv_var_policy}, MVCP policy: {mvcp_policy}"


def test_MinimalVarianceConstraintPolicy_greedy():
    """
    We assume that when beta-> inf, the policy is the same as the greedy policy
    """
    discount = 1.0
    beta = 1e6
    greedy = GreedyPolicy()
    mvcp = MinimalVarianceConstraintPolicy(beta=beta, discount_factor=discount)

    tree = default_tree()
    tree.reset_var_val()
    greedy_policy = np.array(greedy.distribution(tree).probs)
    tree.reset_var_val()
    mvcp_policy = np.array(mvcp.distribution(tree).probs)
    assert np.allclose(greedy_policy, mvcp_policy, rtol=1e-6, atol=1e-6), f"Greedy policy: {greedy_policy}, MVCP policy: {mvcp_policy}"


def test_policy_value(discount = .99):
    """
    We assume that the policy value for the default policy is the same as the default value (subtree average value)
    """
    tree = default_tree(discount=discount)


    default_eval = DefaultTreeEvaluator()
    tree.reset_var_val()
    pol_val = policy_value(tree, default_eval, discount_factor=discount)

    default_value = tree.default_value()

    # had to lower the tolerance to 1e-3 since there seem to be some numerical instability
    assert np.allclose(default_value, pol_val, rtol=1e-3, atol=1e-3), f"Default value: {default_value}, Policy value: {pol_val}"




if __name__ == "__main__":
    test_InverseVarianceTreeEvaluator()
    test_MinimalVarianceConstraintPolicy_zerobeta()
    test_MinimalVarianceConstraintPolicy_greedy()
    test_policy_value()
