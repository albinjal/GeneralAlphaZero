from typing import Callable, Generic, Optional, TypeVar
from abc import ABC, abstractmethod
import numpy as np
import gymnasium as gym
import copy


ActionType = TypeVar("ActionType")
ObservationType = TypeVar("ObservationType")


class Node(Generic[ObservationType, ActionType]):
    parent: Optional["Node[ObservationType, ActionType]"]
    # Since we have to use Discrete action space the ActionType is an integer so we could also use a list
    children: dict[ActionType, "Node[ObservationType, ActionType]"]
    visits: int = 0
    subtree_value: float = 0.0
    value_evaluation: float | None
    reward: float
    # Discrete action space
    action_space: gym.spaces.Discrete

    def __init__(
        self,
        parent: Optional["Node[ObservationType, ActionType]"],
        reward: float,
        action_space: gym.spaces.Discrete,
    ):
        # TODO: lazy init
        self.children = {}
        self.action_space = action_space

    def is_terminal(self) -> bool:
        # TODO: returns if its a terminal node or not
        return False

    def step(self, action: ActionType) -> "Node[ObservationType, ActionType]":
        # steps into the action and returns that node
        return self.children[action]

    def backprop(self, value: float) -> None:
        node: Node[ObservationType, ActionType] | None = self
        while node is not None:
            node.visits += 1
            node = node.parent
            self.subtree_value += value

    def default_value(self) -> float:
        return self.reward + self.subtree_value / self.visits

    def is_fully_expanded(self) -> bool:
        return len(self.children) == self.action_space.n

    def sample_unexplored_action(self) -> np.int64:
        """
        mask – An optional mask for if an action can be selected. Expected np.ndarray of shape (n,) and dtype np.int8 where 1 represents valid actions and 0 invalid / infeasible actions. If there are no possible actions (i.e. np.all(mask == 0)) then space.start will be returned.
        """
        mask = np.ones(self.action_space.n, dtype=np.int8)
        for action in self.children:
            mask[action] = 0
        return self.action_space.sample(mask=mask)



class Policy(ABC, Generic[ObservationType, ActionType]):
    @abstractmethod
    def __call__(self, node: Node[ObservationType, ActionType]) -> ActionType:
        pass


class SelectionPolicy(ABC, Generic[ObservationType, ActionType]):
    @abstractmethod
    def __call__(self, node: Node[ObservationType, ActionType]) -> ActionType | None:
        pass


class UCB(SelectionPolicy[ObservationType, ActionType]):
    def __init__(self, c: float):
        self.c = c

    def __call__(self, node: Node[ObservationType, ActionType]) -> ActionType | None:
        # if not fully expanded, return None
        if not node.is_fully_expanded():
            return None

        # if fully expanded, return the action with the highest UCB value
        # Idea: potentially mess around with making this stochastic
        return max(node.children, key=lambda action: self.ucb(node, action))

    def ucb(self, node: Node[ObservationType, ActionType], action: ActionType) -> float:
        child = node.children[action]
        return child.default_value() + self.c * (node.visits / child.visits) ** 0.5


class RandomPolicy(Policy[ObservationType, np.int64]):
    def __call__(self, node: Node[ObservationType, np.int64]) -> np.int64:
        return node.action_space.sample()

class DefaultExpansionPolicy(Policy[ObservationType, np.int64]):
    def __call__(self, node: Node[ObservationType, np.int64]) -> np.int64:
        # returns a uniformly random unexpanded action
        return node.sample_unexplored_action()

class MCTS(Generic[ObservationType, ActionType]):
    env: gym.Env[ObservationType, ActionType]
    tree_evaluation_policy: Policy[ObservationType, ActionType]
    selection_policy: SelectionPolicy[ObservationType, ActionType]
    expansion_policy: Policy[
        ObservationType, ActionType
    ]  # the expansion policy is usually "pick uniform non explored action"
    value_function: Callable[["Node[ObservationType, ActionType]"], float]

    def __init__(
        self,
        tree_evaluation_policy: Policy[ObservationType, ActionType],
        selection_policy: SelectionPolicy[ObservationType, ActionType],
        value_function: Callable[["Node[ObservationType, ActionType]"], float],
        expansion_policy: Policy[ObservationType, ActionType] = DefaultExpansionPolicy[ObservationType](),
    ):
        self.tree_evaluation_policy = tree_evaluation_policy
        self.selection_policy = selection_policy  # the selection policy should return None if the input node should be expanded
        self.value_function = value_function
        self.expansion_policy = expansion_policy


    def search(self, env: gym.Env[ObservationType, ActionType], iterations: int):
        # the env should be in the state we want to search from
        self.env = env
        # build the tree
        # assert that the type of the action space is discrete
        assert isinstance(env.action_space, gym.spaces.Discrete)
        root_node = Node[ObservationType, ActionType](parent=None, reward=0.0, action_space=env.action_space)
        return self.build_tree(root_node, iterations)


    def build_tree(self, from_node: Node[ObservationType, ActionType], iterations: int):
        for _ in range(iterations):
            selected_node_for_expansion = self.select_node_to_expand(from_node)
            # check if the node is terminal
            if not selected_node_for_expansion.is_terminal():
                # expand the node
                expanded_node = self.expand(selected_node_for_expansion)
                # evaluate the node
                value = self.value_function(expanded_node)
                # backpropagate the value
                expanded_node.backprop(value)

            else:
                # node is terminal
                pass

    def select_node_to_expand(
        self, from_node: Node[ObservationType, ActionType]
    ) -> Node[ObservationType, ActionType]:
        """
        Returns the node to be expanded next.
        Returns None if the node is terminal.
        The selection policy returns None if the input node should be expanded.
        """

        node = from_node
        while not node.is_terminal():
            action = self.selection_policy(node)
            if action is None:
                return node
            node = node.step(action)
            # also step the environment
            self.env.step(action)

        return node

    def expand(
        self, node: Node[ObservationType, ActionType]
    ) -> Node[ObservationType, ActionType]:
        """
        Expands the node and returns the expanded node.
        """
        action = self.expansion_policy(node)
        # assert that the


if __name__ == "__main__":
    obsType = np.int64
    actType = np.int64
    env: gym.Env[obsType, actType] = gym.make("CliffWalking-v0")
    env.reset()
    selection_policy = UCB[obsType, actType](c=1.0)
    tree_evaluation_policy = RandomPolicy[obsType]()
    value_function = None

    mcts = MCTS[obsType, actType](
        tree_evaluation_policy=tree_evaluation_policy,
        selection_policy=selection_policy,
        value_function=lambda node: node.default_value(),
        environment=env,
    )
