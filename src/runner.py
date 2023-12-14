import time
from tensordict import TensorDict, tensorclass
import torch as th
import copy
from typing import Any, List, Tuple
import gymnasium as gym
import numpy as np
from environment import obs_to_tensor
from mcts import MCTS, RandomRolloutMCTS
from node import Node
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
    render=False,
):
    """Runs an episode using the given solver and environment.
    For each timestep, the trajectory contains the observation, the policy distribution, the action taken and the reward received.
    """
    assert isinstance(env.action_space, gym.spaces.Discrete)
    n = env.action_space.n

    observation, info = env.reset(seed=seed)
    terminal = False

    observation_tensor = obs_to_tensor(env.observation_space, observation, dtype=th.float32)
    trajectory = TensorDict(
        source={
            "observations": th.zeros(max_steps, gym.spaces.flatdim(env.observation_space), dtype=observation_tensor.dtype),
            "rewards": th.zeros(max_steps, dtype=th.float32),
            "policy_distributions": th.zeros(max_steps, int(n), dtype=th.float32),
            "actions": th.zeros(max_steps, dtype=th.int64),
            "mask": th.zeros(max_steps, dtype=th.bool),
            "terminals": th.zeros(max_steps, dtype=th.bool),
        },
        batch_size=[max_steps],
    )

    root_node = Node(
        env=copy.deepcopy(env),
        parent=None, reward=np.float32(.0), action_space=env.action_space, observation=observation
    )
    value = solver.value_function(root_node)
    root_node.value_evaluation = value
    for step in range(max_steps):
        tree = solver.build_tree(root_node, compute_budget)
        policy_dist = tree_evaluation_policy.distribution(tree)
        action = policy_dist.sample()
        # res will now contain the obersevation, policy distribution, action, as well as the reward and terminal we got from executing the action
        new_obs, reward, terminated, truncated, _ = env.step(action.item())
        root_node = tree.step(np.int64(action.item()))
        root_node.parent = None

        new_observation_tensor = obs_to_tensor(env.observation_space, new_obs)
        # TODO: check the difference between terminated and truncated
        next_terminal = terminated or truncated
        trajectory["observations"][step] = observation_tensor
        trajectory["rewards"][step] = reward
        trajectory["policy_distributions"][step] = policy_dist.probs
        trajectory["actions"][step] = action
        trajectory["mask"][step] = True
        trajectory["terminals"][step] = terminal

        if verbose:
            if goal_obs is not None:
                vis_counter = tree.state_visitation_counts()
                print(f"Visits to goal state: {vis_counter[goal_obs]}")
            norm_entropy = policy_dist.entropy() / np.log(n)
            print(f"Policy: {policy_dist.probs}, Norm Entropy: {norm_entropy: .2f}")
            print(f"{step}. O: {observation}, A: {action}, R: {reward}, T: {terminal}")

        observation_tensor = new_observation_tensor
        terminal = next_terminal

        if terminal:
            break

    # if we terminated early, we need to add the final observation to the trajectory as well for value estimation
    # trajectory.append((observation, None, None, None, None))
    # observations.append(observation)
    # convert render to tensor


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
    env: gym.Env[Any, actType] = gym.make(env_id, render_mode="rgb_array")

    selection_policy = UCT(c=1)
    tree_evaluation_policy = DefaultTreeEvaluator()

    mcts = RandomRolloutMCTS(selection_policy=selection_policy, rollout_budget=20)
    # vis_tree(mcts, env, compute_budget=100, max_depth=None)
    trajectory = run_episode(
        mcts,
        env,
        tree_evaluation_policy,
        compute_budget=100,
        verbose=True,
        goal_obs=47,
        seed=seed,
        max_steps=200,
    )
    env.close()
    print(trajectory)
