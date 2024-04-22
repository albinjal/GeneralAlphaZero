import copy
from typing import Tuple
import gymnasium as gym
import numpy as np
from core.node import Node
import torch as th
from environments.observation_embeddings import CoordinateEmbedding, ObservationEmbedding
from policies.policies import Policy




class MCTS:
    """
    This class contains the basic MCTS algorithm without assumtions on the value function.
    """
    root_selection_policy: Policy
    selection_policy: Policy

    def __init__(
        self,
        selection_policy: Policy,
        discount_factor: float = 1.0,
        root_selection_policy: Policy | None = None,
    ):
        if root_selection_policy is None:
            root_selection_policy = selection_policy
        self.root_selection_policy = root_selection_policy
        self.selection_policy = selection_policy  # the selection policy should return None if the input node should be expanded
        self.discount_factor = discount_factor

    def search(
        self,
        env: gym.Env,
        iterations: int,
        obs,
        reward: float,
    ) -> Node:
        # the env should be in the state we want to search from
        # assert that the type of the action space is discrete
        assert isinstance(env.action_space, gym.spaces.Discrete)
        # root_node = Node(
        #     parent=None, reward=reward, action_space=env.action_space, observation=obs
        # )
        # # evaluate the root node
        # value = self.value_function(root_node, copy.deepcopy(self.env))
        # # backupagate the value (just updates value est)
        # root_node.backup(value)
        # return self.build_tree(root_node, iterations - 1)
        root_node = Node(
            env=copy.deepcopy(env),
            parent=None,
            reward=reward,
            action_space=env.action_space,
            observation=obs,
        )
        root_node.value_evaluation = self.value_function(root_node)
        self.backup(root_node, root_node.value_evaluation)
        return self.build_tree(root_node, iterations)

    def build_tree(self, from_node: Node, iterations: int) -> Node:
        while from_node.visits < iterations:
            # traverse the tree and select the node to expand
            selected_node_for_expansion, selected_action = self.select_node_to_expand(from_node)
            # check if the node is terminal
            if selected_node_for_expansion.is_terminal():
                # if the node is terminal, we can not expand it
                # the value (sum of future reward) of the node is 0
                # the backup will still propagate the visit and reward
                selected_node_for_expansion.value_evaluation = 0.0
                self.backup(selected_node_for_expansion, 0)
            else:
                self.handle_single(selected_node_for_expansion, selected_action)

        return from_node

    def handle_single(
        self,
        node: Node,
        action: int,
    ):
        eval_node = self.expand(node, action)
        # evaluate the node
        value = self.value_function(eval_node)
        # backupagate the value
        eval_node.value_evaluation = value
        self.backup(eval_node, value)

    # def handle_all(
    #     self, node: Node,
    # ):
    #     for action in range(node.action_space.n):
    #         self.handle_single(node, int(action))

    def value_function(
        self,
        node: Node,
    ) -> float:
        """The point of the value function is to estimate the value of the node.
        The value is defined as the expected future reward if we get to the node given some policy.
        """
        return .0

    def select_node_to_expand(
        self, from_node: Node
    ) -> Tuple[Node, int]:
        """
        Returns the node and action to be expanded next.
        Returns None if the node is terminal.
        The selection policy returns None if the input node should be expanded.
        """

        node = from_node

        # select which node to step into
        action = self.root_selection_policy.sample(node)
        # if the selection policy returns None, this indicates that the current node should be expanded
        if action not in node.children:
            return node, action
        # step into the node
        node = node.step(action)
        # the reason we copy the env is because we want to keep the original env in the root state
        # Question: note that all envs will have the same seed, this might needs to be dealt with for stochastic envs
        while not node.is_terminal():
            # select which node to step into
            action = self.selection_policy.sample(node)
            # if the selection policy returns None, this indicates that the current node should be expanded
            if action not in node.children:
                break
            # step into the node
            node = node.step(action)
            # also step the environment
            # Question: right now we do not save the observation or reward from the env since we already have them saved
            # This might be worth considering though if we use stochastic envs since the rewards/states could vary each time we execute an action sequence

        return node, action

    def expand(
        self, node: Node, action: int
    ) -> Node:
        """
        Expands the node and returns the expanded node.
        Note that the function will modify the env and the input node
        """
        # if this is the last child to be expanded, we do not need to copy the env
        if len(node.children) == int(node.action_space.n) - 1:
            env = node.env
            node.env = None
        else:
            env = copy.deepcopy(node.env)

        assert env is not None

        # step the environment
        observation, reward, terminated, truncated, _ = env.step(action)
        terminal = terminated
        assert not truncated
        if terminated:
            observation = None

        node_class = type(node)
        # create the node
        new_child = node_class(
            env=env,
            parent=node,
            reward=reward,
            action_space=node.action_space,
            terminal=terminal,
            observation=observation,
        )
        node.children[action] = new_child
        return new_child

    def backup(self, start_node: Node, value: float, new_visits: int = 1) -> None:
        # add the value and the reward to all parent nodes
        # we weight the reward by visit count of node (from mathematically derived formula)
        # for example, the immidiate reward will have the highest weight
        node = start_node
        cum_reward = value
        while node is not None:
            cum_reward *= self.discount_factor
            cum_reward += node.reward
            node.subtree_sum += cum_reward
            node.visits += new_visits
            # parent is None if node is root
            # NEW: reset the prior policy and value evaluation (mark as needing update)
            node.variance = None
            node.policy_value = None
            node = node.parent

class RandomRolloutMCTS(MCTS):
    def __init__(self, rollout_budget=40, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rollout_budget = rollout_budget

    def value_function(
        self,
        node: Node,
    ) -> float:
        """
        The standard value function for MCTS is the the sum of the future reward when acting with uniformly random policy.
        """
        # if the node is terminal, return 0
        if node.is_terminal():
            return 0.0

        # if the node is not terminal, simulate the enviroment with random actions and return the accumulated reward until termination
        accumulated_reward = 0.0
        discount = 0.0
        env = copy.deepcopy(node.env)
        assert env is not None
        for _ in range(self.rollout_budget):
            _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            accumulated_reward += reward * discount
            assert not truncated
            if terminated:
                break
            discount *= self.discount_factor

        return accumulated_reward



class DistanceMCTS(MCTS):
    def __init__(self, embedding: CoordinateEmbedding, goal_state: int | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if goal_state is None:
            # the goal is max row max col
            goal_state = embedding.nrows * embedding.ncols - 1
        self.goal_state = goal_state
        self.embedding = embedding


    def value_function(
        self,
        node: Node,
    ) -> float:
        """
        The value function for MCTS is the distance to the goal state.
        """
        if node.is_terminal():
            return 0.0
        cols = self.embedding.ncols
        assert cols is not None
        rows = self.embedding.nrows
        assert rows is not None
        observation = node.observation
        assert observation is not None

        """
        There are (rows-1) x cols + 1 possible states. The player cannot be at the cliff, nor at the goal as the latter results in the end of the episode. What remains are all the positions of the first 3 rows plus the bottom-left cell.
        The observation is a value representing the player’s current position as current_row * nrows + current_col (where both the row and col start at 0).
        For example, the stating position can be calculated as follows: (rows-1) * cols + 0 = 36.
        """
        goal_row = self.goal_state // cols
        goal_col = self.goal_state % cols
        current_row = observation // cols
        current_col = observation % cols

        col_diff = abs(goal_col - current_col)
        row_diff = abs(goal_row - current_row)
        manhattan_distance = col_diff + row_diff
        # special case for cliffwalking env
        # if we are in the last row, we need to add two since we cannot go directly to the goal (cuz cliff)
        if current_row == rows - 1:
            manhattan_distance += 2

        return - float(manhattan_distance)
