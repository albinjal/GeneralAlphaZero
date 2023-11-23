

from typing import Any
import gymnasium as gym
import numpy as np
from mcts import MCTS, RandomRolloutMCTS
from policies import UCB, DefaultTreeEvaluator, Policy


def run_episode(
    solver: MCTS,
    env: gym.Env,
    tree_evaluation_policy: Policy,
    compute_budget=1000,
    max_steps=1000,
    render_env=None,
    verbose=False,
):
    total_reward = 0.0

    for step in range(max_steps):
        tree = solver.search(env, compute_budget)
        action = tree_evaluation_policy(tree)
        print(f"Found {tree.children[action].default_value()}")
        observation, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        if render_env is not None:
            render_env.step(action)
        if verbose:
            print(
                f"{step}. A: {action}, R: {reward}, T: {terminated}, Tr: {truncated}, total_reward: {total_reward}"
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
    # env_id = "CliffWalking-v0"
    env_id = "FrozenLake-v1"
    # env_id = "Taxi-v3"
    args = {"id": env_id, "map_name": "8x8", "is_slippery":False}
    env: gym.Env[Any, actType] = gym.make(**args)
    env.reset(seed=seed)
    render_env: gym.Env[Any, actType] = gym.make(**args, render_mode="human")
    render_env.reset(seed=seed)
    selection_policy = UCB[Any, actType](c=0.5)
    tree_evaluation_policy = DefaultTreeEvaluator[Any]()

    mcts = RandomRolloutMCTS(selection_policy=selection_policy)
    # vis_tree(mcts, env, compute_budget=10000, max_depth=None)
    total_reward = run_episode(
        mcts,
        env,
        tree_evaluation_policy,
        compute_budget=1000,
        render_env=render_env,
        verbose=True,
    )
    print(f"Total reward: {total_reward}")
    env.close()
    render_env.close()
