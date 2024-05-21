import multiprocessing
from tensordict import TensorDict
import torch as th
import gymnasium as gym
import numpy as np
from core.mcts import MCTS
from environments.observation_embeddings import ObservationEmbedding
from policies.policies import PolicyDistribution, custom_softmax

def run_episode_process(args):
    """Wrapper function for multiprocessing that unpacks arguments and runs a single episode."""
    return run_episode(*args)

def collect_trajectories(tasks, workers=1):
    if workers > 1:
        with multiprocessing.Pool(workers) as pool:
            # Run the tasks using map
            results = pool.map(run_episode_process, tasks)
    else:
        results = [run_episode_process(task) for task in tasks]
    res_tensor =  th.stack(results)
    return res_tensor

@th.no_grad()
def run_episode(
    solver: MCTS,
    env: gym.Env,
    tree_evaluation_policy: PolicyDistribution,
    observation_embedding: ObservationEmbedding,
    planning_budget=1000,
    max_steps=1000,
    seed=None,
    temperature=None,
    return_trees=False,
    ):
    """Runs an episode using the given solver and environment.
    For each timestep, the trajectory contains the observation, the policy distribution, the action taken and the reward received.
    """
    assert isinstance(env.action_space, gym.spaces.Discrete)
    n = int(env.action_space.n)
    if seed is not None:
        th.manual_seed(seed)
        np.random.seed(seed)
    observation, info = env.reset(seed=seed)

    observation_tensor: th.Tensor = observation_embedding.obs_to_tensor(observation, dtype=th.float32)
    trajectory = TensorDict(
        source={
            "observations": th.zeros(
                max_steps,
                observation_embedding.obs_dim(),
                dtype=observation_tensor.dtype,
            ),
            "rewards": th.zeros(max_steps, dtype=th.float32),
            "policy_distributions": th.zeros(max_steps, n, dtype=th.float32),
            "actions": th.zeros(max_steps, dtype=th.int64),
            "mask": th.zeros(max_steps, dtype=th.bool),
            "terminals": th.zeros(max_steps, dtype=th.bool),
            "root_values": th.zeros(max_steps, dtype=th.float32),
        },
        batch_size=[max_steps],
    )
    if return_trees:
        trees = []
    tree = solver.search(env, planning_budget, observation, 0.0)
    for step in range(max_steps):
        root_value = tree.value_evaluation

        tree.reset_var_val()
        policy_dist = tree_evaluation_policy.softmaxed_distribution(tree)
        if return_trees:
            trees.append(tree)
        # apply extra softmax
        action = th.distributions.Categorical(probs=custom_softmax(policy_dist.probs, temperature, None)).sample().item()
        # res will now contain the obersevation, policy distribution, action, as well as the reward and terminal we got from executing the action
        new_obs, reward, terminated, truncated, _ = env.step(action)
        assert not truncated

        next_terminal = terminated
        trajectory["observations"][step] = observation_tensor
        trajectory["rewards"][step] = reward
        trajectory["policy_distributions"][step] = policy_dist.probs
        trajectory["actions"][step] = action
        trajectory["mask"][step] = True
        trajectory["terminals"][step] = next_terminal
        trajectory["root_values"][step] = th.tensor(root_value, dtype=th.float32)
        if next_terminal or truncated:
            break

        tree = solver.search(env, planning_budget, new_obs, reward)

        new_observation_tensor = observation_embedding.obs_to_tensor(new_obs, dtype=th.float32)
        observation_tensor = new_observation_tensor

    # if we terminated early, we need to add the final observation to the trajectory as well for value estimation
    # trajectory.append((observation, None, None, None, None))
    # observations.append(observation)
    # convert render to tensor

    if return_trees:
        return trajectory, trees

    return trajectory
