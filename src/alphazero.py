from typing import Callable, Generic, TypeVar
from abc import ABC, abstractmethod

import gymnasium as gym


ActionType = TypeVar('ActionType')
ObservationType = TypeVar('ObservationType')



class Node(Generic[ObservationType, ActionType]):
    parent: "Node[ObservationType, ActionType]" | None
    children: dict[ActionType, "Node[ObservationType, ActionType]"]
    visits: int = 0
    subtree_value: float = 0.0
    value_evaluation: float | None
    reward: float

    def __init__(self, parent: "Node[ObservationType, ActionType]" | None, reward: float):
        # TODO: lazy init
        self.children = {}


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





class Policy(ABC, Generic[ObservationType, ActionType]):
    @abstractmethod
    def __call__(self, node: Node[ObservationType, ActionType]) -> ActionType:
        pass

class SelectionPolicy(ABC, Generic[ObservationType, ActionType]):
    @abstractmethod
    def __call__(self, node: Node[ObservationType, ActionType]) -> ActionType | None:
        pass


class MCTS(Generic[ObservationType, ActionType]):
    env: gym.Env[ObservationType, ActionType]
    tree_evaluation_policy: Policy[ObservationType, ActionType]
    selection_policy: SelectionPolicy[ObservationType, ActionType]
    expansion_policy: Policy[ObservationType, ActionType] # the expansion policy is usually "pick uniform non explored action"
    value_function: Callable[["Node[ObservationType, ActionType]"], float]

    def __init__(self, tree_evaluation_policy: Policy[ObservationType, ActionType],
                 selection_policy: SelectionPolicy[ObservationType, ActionType],
                 value_function: Callable[["Node[ObservationType, ActionType]"], float],
                 expansion_policy: Policy[ObservationType, ActionType],
                 environment: gym.Env[ObservationType, ActionType]
                 ):
        self.tree_evaluation_policy = tree_evaluation_policy
        self.selection_policy = selection_policy # the selection policy should return None if the input node should be expanded
        self.value_function = value_function
        self.expansion_policy = expansion_policy
        self.env = environment



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




    def select_node_to_expand(self, from_node: Node[ObservationType, ActionType]) -> Node[ObservationType, ActionType]:
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

        return node

    def expand(self, node: Node[ObservationType, ActionType]) -> Node[ObservationType, ActionType]:
        """
        Expands the node and returns the expanded node.
        """
        action = self.expansion_policy(node)
        return node.step(action)










if __name__ == "__main__":
    pass
