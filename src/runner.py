

import copy
from typing import Any, List, Tuple
import gymnasium as gym
import numpy as np
from mcts import MCTS, RandomRolloutMCTS
from policies import UCB, DefaultTreeEvaluator, Policy, PolicyDistribution


def run_episode(
    solver: MCTS,
    env: gym.Env,
    tree_evaluation_policy: PolicyDistribution,
    compute_budget=1000,
    max_steps=1000,
    verbose=False,
    goal_obs=None,
    render=False,
    seed=None,
) -> List[Tuple[Any, np.ndarray, float]]:
    assert isinstance(env.action_space, gym.spaces.Discrete)
    observation, info = env.reset(seed=seed)
    reward = .0
    action = None
    terminal = False
    total_reward = 0.0
    trajectory = []
    for step in range(max_steps):
        tree = solver.search(env, compute_budget, observation, float(reward))

        policy_dist = tree_evaluation_policy.distribution(tree)
        v_target = tree.default_value()
        trajectory.append((observation, policy_dist, v_target))

        action = np.random.choice(env.action_space.n, p=policy_dist)
        observation, reward, terminated, truncated, _ = env.step(action)
        terminal = terminated or truncated
        total_reward += float(reward)

        if verbose:
            if goal_obs is not None:
                vis_counter = tree.state_visitation_counts()
                print(f"Visits to goal state: {vis_counter[goal_obs]}")
            norm_entropy = -float(np.sum(policy_dist * np.log(policy_dist))) / np.log(env.action_space.n)
            print(f"Policy: {policy_dist}, Norm Entropy: {norm_entropy: .2f}")
            print(
                f"{step}.O: {observation}, A: {action}, R: {reward}, T: {terminal}, total_reward: {total_reward}"
            )
        if terminal:
            break


    return trajectory



def vis_tree(solver: MCTS, env: gym.Env, compute_budget=100, max_depth=None):
    observation, _ = env.reset()
    tree = solver.search(env, compute_budget, observation, .0)
    return tree.visualize(max_depth=max_depth)


if __name__ == "__main__":
    seed = 0
    actType = np.int64
    env_id = "CliffWalking-v0"
    # env_id = "FrozenLake-v1"
    # env_id = "Taxi-v3"
    env: gym.Env[Any, actType] = gym.make(env_id, render_mode="ansi")

    selection_policy = UCB(c=40)
    tree_evaluation_policy = DefaultTreeEvaluator()

    mcts = RandomRolloutMCTS(selection_policy=selection_policy, rollout_budget=20)
    # vis_tree(mcts, env, compute_budget=100, max_depth=None)
    total_reward = run_episode(
        mcts,
        env,
        tree_evaluation_policy,
        compute_budget=1000,
        verbose=True,
        goal_obs=47,
        seed=seed,
        render=True,
    )
    env.close()
    print(f"Total reward: {total_reward}")
