from typing import List
import torch as th
import gymnasium as gym
from environments.environment import obs_to_tensor
import copy
import numpy as np

from core.mcts import MCTS
from az.model import AlphaZeroModel
from core.node import Node


class AlphaZeroMCTS(MCTS):
    model: AlphaZeroModel

    def __init__(self, model: AlphaZeroModel, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model

    @th.no_grad()
    def value_function(
        self,
        node: Node,
    ) -> np.float32:
        if node.is_terminal():
            return np.float32(0.0)
        observation = node.observation
        # flatten the observation
        assert observation is not None
        # run the model
        # convert observation from int to tensor float 1x1 tensor
        assert node.env is not None
        tensor_obs = obs_to_tensor(
            node.env.observation_space,
            observation,
            device=self.model.device,
            dtype=th.float32,
        )
        value, policy = self.model.forward(tensor_obs.unsqueeze(0))
        # store the policy
        node.prior_policy = policy.squeeze(0)
        # return float 32 value
        return np.float32(value.item())

    @th.no_grad()
    def value_funciton_multiple(self, nodes: List[Node]):
        pass

    # @th.no_grad()
    # def handle_all(self, node: Node):
    #     all_actions = np.arange(node.action_space.n)
    #     assert node.env is not None
    #     obs_space = node.env.observation_space

    #     terminal_included = False
    #     for action in all_actions:
    #         new_node = self.expand(node, action)
    #         if new_node.is_terminal():
    #             terminal_included = True

    #     children = node.children

    #     # if there is a terminal node along the new nodes, simply do the regular backup
    #     if terminal_included:
    #         for child in children.values():
    #             value = self.value_function(child)
    #             # backupagate the value
    #             child.value_evaluation = value
    #             child.backup(value)

    #         return

    #     tensor_obs = th.stack([obs_to_tensor(obs_space, children[action].observation, device=self.model.device, dtype=th.float32)
    #                            for action in all_actions])  # actions x obs_dim tensor
    #     values, policies = self.model.forward(tensor_obs)

    #     value_to_backup = np.float32(0.0)
    #     for action, value, policy in zip(all_actions, values, policies):
    #         new_node = node.children[action]
    #         new_node.prior_policy = policy
    #         new_node.visits = 1
    #         new_node.value_evaluation = np.float32(value.item())
    #         new_node.subtree_sum = new_node.reward + new_node.value_evaluation
    #         value_to_backup += new_node.subtree_sum

    #     # backup
    #     # the value to backup from the parent should be the sum of the value and the reward for all children
    #     node.backup(value_to_backup, len(all_actions))

    # def backup_all_children(self, parent: Node, values: th.Tensor):
