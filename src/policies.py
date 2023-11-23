from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import numpy as np
from node import Node

ActionType = TypeVar("ActionType")
ObservationType = TypeVar("ObservationType")

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


class DefaultTreeEvaluator(Policy[ObservationType, np.int64]):
    # the default tree evaluator selects the action with the most visits
    def __call__(self, node: Node[ObservationType, np.int64]) -> np.int64:
        return max(node.children, key=lambda action: node.children[action].visits)
