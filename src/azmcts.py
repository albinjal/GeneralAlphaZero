import torch as th
import gymnasium as gym
from environment import obs_to_tensor
import copy
import numpy as np

from mcts import MCTS
from model import AlphaZeroModel
from node import Node


class AlphaZeroMCTS(MCTS):
    model: AlphaZeroModel

    def __init__(self, model: AlphaZeroModel, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model

    @th.no_grad()
    def value_function(
        self,
        node: Node,
        env: gym.Env,
    ):
        observation = node.observation
        # flatten the observation
        assert observation is not None
        # run the model
        # convert observation from int to tensor float 1x1 tensor
        tensor_obs = obs_to_tensor(
            env.observation_space, observation, device=self.model.device, dtype=th.float32
        )
        value, policy = self.model.forward(tensor_obs)
        # store the policy
        node.prior_policy = policy
        # return float 32 value
        return value.item()

    def handle_all(self, node: Node, env: gym.Env):
        """
            should do the same as
        def handle_single(
            self,
            node: Node[ObservationType],
            env: gym.Env[ObservationType, np.int64],
            action: np.int64,
        ):
            eval_node = self.expand(node, env, action)
            # evaluate the node
            value = self.value_function(eval_node, env)
            # backupagate the value
            eval_node.value_evaluation = value
            eval_node.backup(value)

        def handle_all(
            self, node: Node[ObservationType], env: gym.Env[ObservationType, np.int64]
        ):
            for action in range(node.action_space.n):
                self.handle_single(node, copy.deepcopy(env), np.int64(action))

        """
        observations = []
        all_actions = np.arange(node.action_space.n)
        for i, action in enumerate(all_actions):
            if i == len(all_actions) - 1:
                # we can mess up the env on the last action
                new_node = self.expand(node, env, action)
            else:
                new_env = copy.deepcopy(env)
                new_node = self.expand(node, new_env, action)
            observations.append(
                obs_to_tensor(
                    env.observation_space,
                    new_node.observation,
                    device=self.model.device,
                    dtype=th.float32,
                )
            )

        tensor_obs = th.stack(observations)  # actions x obs_dim tensor
        values, policies = self.model.forward(tensor_obs)

        value_to_backup = np.float32(0.0)
        for action, value, policy in zip(all_actions, values, policies):
            new_node = node.children[action]
            new_node.prior_policy = policy
            new_node.visits = 1
            new_node.value_evaluation = np.float32(value.item())
            new_node.subtree_sum = new_node.reward + new_node.value_evaluation
            value_to_backup += new_node.subtree_sum

        # backup
        # the value to backup from the parent should be the sum of the value and the reward for all children
        node.backup(value_to_backup, len(all_actions))

    # def backup_all_children(self, parent: Node, values: th.Tensor):
