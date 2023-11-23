import copy
import gymnasium as gym

from typing import Generic, TypeVar, Tuple
from node import Node

from policies import DefaultExpansionPolicy, Policy, SelectionPolicy

ActionType = TypeVar("ActionType")
ObservationType = TypeVar("ObservationType")

class MCTS(Generic[ObservationType, ActionType]):
    env: gym.Env[ObservationType, ActionType]
    selection_policy: SelectionPolicy[ObservationType, ActionType]
    expansion_policy: Policy[
        ObservationType, ActionType
    ]  # the expansion policy is usually "pick uniform non explored action"

    def __init__(
        self,
        selection_policy: SelectionPolicy[ObservationType, ActionType],
        expansion_policy: Policy[ObservationType, ActionType] = DefaultExpansionPolicy[
            ObservationType
        ](),
    ):
        self.selection_policy = selection_policy  # the selection policy should return None if the input node should be expanded
        self.expansion_policy = expansion_policy

    def search(self, env: gym.Env[ObservationType, ActionType], iterations: int):
        # the env should be in the state we want to search from
        self.env = env
        # assert that the type of the action space is discrete
        assert isinstance(env.action_space, gym.spaces.Discrete)
        root_node = Node[ObservationType, ActionType](
            parent=None, reward=0.0, action_space=env.action_space
        )
        return self.build_tree(root_node, iterations)

    def build_tree(self, from_node: Node[ObservationType, ActionType], iterations: int):
        for _ in range(iterations):
            selected_node_for_expansion, env = self.select_node_to_expand(from_node)
            # check if the node is terminal
            if selected_node_for_expansion.is_terminal():
                eval_node = selected_node_for_expansion
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
        node: Node[ObservationType, ActionType],
        env: gym.Env[ObservationType, ActionType],
    ) -> float:
        return node.reward

    def select_node_to_expand(
        self, from_node: Node[ObservationType, ActionType]
    ) -> Tuple[Node[ObservationType, ActionType], gym.Env[ObservationType, ActionType]]:
        """
        Returns the node to be expanded next.
        Returns None if the node is terminal.
        The selection policy returns None if the input node should be expanded.
        """

        node = from_node

        env = copy.deepcopy(self.env)
        while not node.is_terminal():
            action = self.selection_policy(node)
            if action is None:
                return node, env
            node = node.step(action)
            # also step the environment
            env.step(action)

        return node, env

    def expand(
        self,
        node: Node[ObservationType, ActionType],
        env: gym.Env[ObservationType, ActionType],
    ) -> Node[ObservationType, ActionType]:
        """
        Expands the node and returns the expanded node.
        """
        action = self.expansion_policy(node)
        # step the environment
        observation, reward, terminated, truncated, _ = env.step(action)
        terminal = terminated or truncated
        # create the node
        new_child = Node[ObservationType, ActionType](
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
        node: Node[ObservationType, ActionType],
        env: gym.Env[ObservationType, ActionType],
    ) -> float:
        # if the node is terminal, return the reward
        if node.is_terminal():
            return node.reward

        # if the node is not terminal, simulate the enviroment with random actions and return the accumulated reward until termination
        accumulated_reward = 0.0
        for _ in range(self.rollout_budget):
            _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            accumulated_reward += float(reward)
            if terminated or truncated:
                break

        return accumulated_reward
