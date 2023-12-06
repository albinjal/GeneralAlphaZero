
from tensordict import TensorDict
import torch as th
import copy
from typing import Any, List, Tuple
import gymnasium as gym
import numpy as np
from mcts import MCTS, RandomRolloutMCTS
from policies import PUCT, UCT, DefaultTreeEvaluator, Policy, PolicyDistribution



def run_episode(
    solver: MCTS,
    env: gym.Env,
    tree_evaluation_policy: PolicyDistribution,
    compute_budget=1000,
    max_steps=1000,
    verbose=False,
    goal_obs=None,
    seed=None,
    ):
    """Runs an episode using the given solver and environment.
    For each timestep, the trajectory contains the observation, the policy distribution, the action taken and the reward received.
    """
    assert isinstance(env.action_space, gym.spaces.Discrete)
    observation, info = env.reset(seed=seed)
    trajectory = []
    for step in range(max_steps):

        tree = solver.search(env, compute_budget, observation, np.float32(.0))
        policy_dist = tree_evaluation_policy.distribution(tree)
        action = policy_dist.sample()
        # res will now contain the obersevation, policy distribution, action, as well as the reward and terminal we got from executing the action
        next_observation, reward, terminated, truncated, _ = env.step(action.item())
        terminal = terminated or truncated

        trajectory.append((observation, policy_dist.probs, action, reward, terminal))

        if verbose:
            if goal_obs is not None:
                vis_counter = tree.state_visitation_counts()
                print(f"Visits to goal state: {vis_counter[goal_obs]}")
            norm_entropy = policy_dist.entropy() / np.log(env.action_space.n)
            print(f"Policy: {policy_dist.probs}, Norm Entropy: {norm_entropy: .2f}")
            print(
                f"{step}. O: {observation}, A: {action}, R: {reward}, T: {terminal}"
            )
        observation = next_observation

        if terminal:
            break

    # if we terminated early, we need to add the final observation to the trajectory as well for value estimation
    # trajectory.append((observation, None, None, None, None))


    return trajectory



def vis_tree(solver: MCTS, env: gym.Env, compute_budget=100, max_depth=None):
    observation, _ = env.reset()
    tree = solver.search(env, compute_budget, observation, np.float32(0.0))
    return tree.visualize(max_depth=max_depth)


if __name__ == "__main__":
    seed = 0
    actType = np.int64
    env_id = "CliffWalking-v0"
    # env_id = "FrozenLake-v1"
    # env_id = "Taxi-v3"
    env: gym.Env[Any, actType] = gym.make(env_id, render_mode="ansi")

    selection_policy = UCT(c=40)
    tree_evaluation_policy = DefaultTreeEvaluator()

    mcts = RandomRolloutMCTS(selection_policy=selection_policy, rollout_budget=20)
    # vis_tree(mcts, env, compute_budget=100, max_depth=None)
    trajectory = run_episode(
        mcts,
        env,
        tree_evaluation_policy,
        compute_budget=1000,
        verbose=True,
        goal_obs=47,
        seed=seed,
        max_steps=5
    )
    env.close()
    print(trajectory)
