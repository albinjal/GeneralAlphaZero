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
    ):
        observation = node.observation
        # flatten the observation
        assert observation is not None
        # run the model
        # convert observation from int to tensor float 1x1 tensor
        assert node.env is not None
        tensor_obs = obs_to_tensor(
            node.env.observation_space, observation, device=self.model.device, dtype=th.float32
        )
        value, policy = self.model.forward(tensor_obs)
        # store the policy
        node.prior_policy = policy
        # return float 32 value
        return value.item()

    def handle_all(self, node: Node):
        observations = []
        all_actions = np.arange(node.action_space.n)
        assert node.env is not None

        for i, action in enumerate(all_actions):
            obs_space = node.env.observation_space
            new_node = self.expand(node, action)
            observations.append(
                obs_to_tensor(
                    obs_space,
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
