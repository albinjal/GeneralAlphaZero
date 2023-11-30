import copy
import gymnasium as gym
from typing import Generic, TypeVar, Tuple
import numpy as np
from node import Node

from policies import DefaultExpansionPolicy, OptionalPolicy, Policy

ObservationType = TypeVar("ObservationType")


class MCTS(Generic[ObservationType]):
    """
    This class contains the basic MCTS algorithm without assumtions on the value function.
    """

    env: gym.Env[ObservationType, np.int64]
    selection_policy: OptionalPolicy[ObservationType]
    expansion_policy: Policy[
        ObservationType
    ]  # the expansion policy is usually "pick uniform non explored action"

    def __init__(
        self,
        selection_policy: OptionalPolicy[ObservationType],
        expansion_policy: Policy[ObservationType] = DefaultExpansionPolicy[
            ObservationType
        ](),
    ):
        self.selection_policy = selection_policy  # the selection policy should return None if the input node should be expanded
        self.expansion_policy = expansion_policy

    def search(self, env: gym.Env[ObservationType, np.int64], iterations: int, obs: ObservationType, reward: float) -> Node[ObservationType]:
        # the env should be in the state we want to search from
        self.env = env
        # assert that the type of the action space is discrete
        assert isinstance(env.action_space, gym.spaces.Discrete)
        # root_node = Node[ObservationType](
        #     parent=None, reward=reward, action_space=env.action_space, observaton=obs
        # )
        # # evaluate the root node
        # value = self.value_function(root_node, copy.deepcopy(self.env))
        # # backpropagate the value (just updates value est)
        # root_node.backprop(value)
        # return self.build_tree(root_node, iterations - 1)
        root_node = Node[ObservationType](
            parent=None, reward=reward, action_space=env.action_space, observaton=obs
        )
        value = self.value_function(root_node, copy.deepcopy(self.env))
        root_node.value_evaluation = value
        return self.build_tree(root_node, iterations)

    def build_tree(self, from_node: Node[ObservationType], iterations: int) -> Node[ObservationType]:
        for _ in range(iterations):
            # traverse the tree and select the node to expand
            selected_node_for_expansion, env = self.select_node_to_expand(from_node)
            # check if the node is terminal
            if selected_node_for_expansion.is_terminal():
                # if the node is terminal, we can not expand it
                # the value (sum of future reward) of the node is 0
                # the backprop will still propagate the visit and reward
                selected_node_for_expansion.backprop(0)
            else:
                # expand the node
                eval_node = self.expand(selected_node_for_expansion, env)
                # evaluate the node
                value = self.value_function(eval_node, env)
                # backpropagate the value
                eval_node.backprop(value)
        return from_node

    def value_function(
        self,
        node: Node[ObservationType],
        env: gym.Env[ObservationType, np.int64],
    ) -> float:
        """The point of the value function is to estimate the value of the node.
        The value is defined as the expected future reward if we get to the node given some policy.
        """
        return 0.0

    def select_node_to_expand(
        self, from_node: Node[ObservationType]
    ) -> Tuple[Node[ObservationType], gym.Env[ObservationType, np.int64]]:
        """
        Returns the node to be expanded next.
        Returns None if the node is terminal.
        The selection policy returns None if the input node should be expanded.
        """

        node = from_node
        # the reason we copy the env is because we want to keep the original env in the root state
        # Question: note that all envs will have the same seed, this might needs to be dealt with for stochastic envs
        env = copy.deepcopy(self.env)
        while not node.is_terminal():
            # select which node to step into
            action = self.selection_policy(node)
            # if the selection policy returns None, this indicates that the current node should be expanded
            if action is None:
                return node, env
            # step into the node
            node = node.step(action)
            # also step the environment
            # Question: right now we do not save the observation or reward from the env since we already have them saved
            # This might be worth considering though if we use stochastic envs since the rewards/states could vary each time we execute an action sequence
            env.step(action)

        return node, env

    def expand(
        self,
        node: Node[ObservationType],
        env: gym.Env[ObservationType, np.int64],
    ) -> Node[ObservationType]:
        """
        Expands the node and returns the expanded node.
        """
        # in the default case we sample a random unexplored action
        action = self.expansion_policy(node)
        # step the environment
        observation, reward, terminated, truncated, _ = env.step(action)
        terminal = terminated or truncated
        # create the node
        new_child = Node[ObservationType](
            parent=node,
            reward=float(reward),
            action_space=node.action_space,
            terminal=terminal,
            observaton=observation,
        )
        node.children[action] = new_child
        return new_child


class RandomRolloutMCTS(MCTS):
    def __init__(self, rollout_budget=40, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rollout_budget = rollout_budget

    def value_function(
        self,
        node: Node[ObservationType],
        env: gym.Env[ObservationType, np.int64],
    ) -> float:
        """
        The standard value function for MCTS is the the sum of the future reward when acting with uniformly random policy.
        """
        # if the node is terminal, return 0
        if node.is_terminal():
            return 0.0

        # if the node is not terminal, simulate the enviroment with random actions and return the accumulated reward until termination
        accumulated_reward = 0.0
        for _ in range(self.rollout_budget):
            _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            accumulated_reward += float(reward)
            if terminated or truncated:
                break

        return accumulated_reward
