from tensordict import TensorDict
import torch as th
import gymnasium as gym
import numpy as np
from core.mcts import MCTS
from environments.observation_embeddings import ObservationEmbedding
from policies.policies import PolicyDistribution

@th.no_grad()
def run_episode(
    solver: MCTS,
    env: gym.Env,
    tree_evaluation_policy: PolicyDistribution,
    observation_embedding: ObservationEmbedding,
    planning_budget=1000,
    max_steps=1000,
    verbose=False,
    seed=None,
    eval=False,
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
    tree = solver.search(env, planning_budget, observation, 0.0)
    for step in range(max_steps):
        root_value = tree.value_evaluation

        tree.reset_var_val()
        policy_dist = tree_evaluation_policy.softmaxed_distribution(tree)
        if eval:
            action = policy_dist.probs.argmax().item()
        else:
            action = policy_dist.sample().item()
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

        if verbose:
            norm_entropy = policy_dist.entropy() / np.log(n)
            print(f"Policy: {policy_dist.probs}, Norm Entropy: {norm_entropy: .2f}")
            print(
                f"{step}. O: {observation}, A: {action}, R: {reward}, T: {next_terminal}"
            )
        if next_terminal or truncated:
            break

        tree = solver.search(env, planning_budget, observation, reward)

        new_observation_tensor = observation_embedding.obs_to_tensor(new_obs, dtype=th.float32)
        observation_tensor = new_observation_tensor

    # if we terminated early, we need to add the final observation to the trajectory as well for value estimation
    # trajectory.append((observation, None, None, None, None))
    # observations.append(observation)
    # convert render to tensor

    return trajectory
