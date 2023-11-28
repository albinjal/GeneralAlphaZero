

from typing import Any
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
    render_env=None,
    verbose=False,
    goal_obs=None,
):
    total_reward = 0.0
    assert isinstance(env.action_space, gym.spaces.Discrete)

    for step in range(max_steps):
        tree = solver.search(env, compute_budget)
        policy_dist = tree_evaluation_policy.distribution(tree)
        action = np.random.choice(env.action_space.n, p=policy_dist)


        observation, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        if render_env is not None:
            render_env.step(action)
        if verbose:
            if goal_obs is not None:
                vis_counter = tree.state_visitation_counts()
                print(f"Visits to goal state: {vis_counter[goal_obs]}")
            norm_entropy = -float(np.sum(policy_dist * np.log(policy_dist))) / np.log(env.action_space.n)
            print(f"Policy: {policy_dist}, Norm Entropy: {norm_entropy: .2f}")
            print(
                f"{step}.O: {observation}, A: {action}, R: {reward}, T: {terminated}, Tr: {truncated}, total_reward: {total_reward}"
            )
        if terminated or truncated:
            break

    return total_reward



def vis_tree(solver: MCTS, env: gym.Env, compute_budget=100, max_depth=None):
    tree = solver.search(env, compute_budget)
    return tree.visualize(max_depth=max_depth)


if __name__ == "__main__":
    seed = 0
    actType = np.int64
    env_id = "CliffWalking-v0"
    # env_id = "FrozenLake-v1"
    # env_id = "Taxi-v3"
    args: dict = {"id": env_id}
    env: gym.Env[Any, actType] = gym.make(**args)
    env.reset(seed=seed)
    render_env: gym.Env[Any, actType] = gym.make(**args, render_mode="human")
    render_env.reset(seed=seed)
    selection_policy = UCB[Any](c=50)
    tree_evaluation_policy = DefaultTreeEvaluator[Any]()

    mcts = RandomRolloutMCTS(selection_policy=selection_policy, rollout_budget=20)
    # vis_tree(mcts, env, compute_budget=100, max_depth=None)
    total_reward = run_episode(
        mcts,
        env,
        tree_evaluation_policy,
        compute_budget=1000,
        render_env=render_env,
        verbose=True,
        goal_obs=47
    )
    print(f"Total reward: {total_reward}")
    env.close()
    render_env.close()
