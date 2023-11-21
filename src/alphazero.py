from typing import Callable


class Action:
    pass
class Node:
    parent: "Node" | None
    children: dict[Action, "Node"]
    visits: int = 0
    subtree_value: float = 0.0

    def __init__(self):
        # TODO: lazy init
        self.children = {}
        

    def is_terminal(self) -> bool:
        # TODO: returns if its a terminal node or not
        return False

    def step(self, action: Action) -> "Node":
        # steps into the action and returns that node
        # if the child exists, return it
        # if not, create it and return it
        if action in self.children:
            return self.children[action]
        else:
            self.children[action] = Node()
            return self.children[action]

    def backprop(self, value: float) -> None:
        node: Node | None = self
        while node is not None:
            node.visits += 1
            node = node.parent
            self.subtree_value += value


ValueFunction = Callable[[Node], float]


from abc import ABC, abstractmethod

class Policy(ABC):
    @abstractmethod
    def __call__(self, node: Node) -> Action:
        pass

class SelectionPolicy(ABC):
    @abstractmethod
    def __call__(self, node: Node) -> Action | None:
        pass

class Environment:
    pass

class MCTS:
    tree_evaluation_policy: Policy
    selection_policy: SelectionPolicy
    expansion_policy: Policy # the expansion policy is usually "pick uniform non explored action"
    value_function: ValueFunction

    def __init__(self, tree_evaluation_policy: Policy,
                 selection_policy: SelectionPolicy,
                 value_function: ValueFunction,
                 expansion_policy: Policy,
                 ):
        self.tree_evaluation_policy = tree_evaluation_policy
        self.selection_policy = selection_policy # the selection policy should return None if the input node should be expanded
        self.value_function = value_function
        self.expansion_policy = expansion_policy



    def build_tree(self, from_node: Node, iterations: int):

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




    def select_node_to_expand(self, from_node: Node) -> Node:
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

    def expand(self, node: Node) -> Node:
        """
        Expands the node and returns the expanded node.
        """
        action = self.expansion_policy(node)
        return node.step(action)










if __name__ == "__main__":
    pass
