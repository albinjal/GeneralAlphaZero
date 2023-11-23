from typing import Generic, TypeVar, Optional, Any, Callable, Tuple
import gymnasium as gym
import numpy as np
import graphviz

ActionType = TypeVar("ActionType")
ObservationType = TypeVar("ObservationType")


class Node(Generic[ObservationType, ActionType]):
    parent: Optional["Node[ObservationType, ActionType]"]
    # Since we have to use Discrete action space the ActionType is an integer so we could also use a list
    children: dict[ActionType, "Node[ObservationType, ActionType]"]
    visits: int = 0
    subtree_value: float = 0.0
    value_evaluation: float = 0.0
    reward: float
    # Discrete action space
    action_space: gym.spaces.Discrete
    observation: Optional[ObservationType]

    def __init__(
        self,
        parent: Optional["Node[ObservationType, ActionType]"],
        reward: float,
        action_space: gym.spaces.Discrete,
        observaton: Optional[ObservationType] = None,
        terminal: bool = False,
    ):
        # TODO: lazy init
        self.children = {}
        self.action_space = action_space
        self.reward = reward
        self.parent = parent
        self.terminal = terminal
        self.observation = observaton

    def is_terminal(self) -> bool:
        return self.terminal

    def step(self, action: ActionType) -> "Node[ObservationType, ActionType]":
        # steps into the action and returns that node
        child = self.children[action]
        return child

    def backprop(self, value: float) -> None:
        self.value_evaluation = value
        node: Node[ObservationType, ActionType] | None = self
        while node is not None:
            node.subtree_value += self.value_evaluation + self.reward
            node.visits += 1
            node = node.parent

    def default_value(self) -> float:
        return self.subtree_value / self.visits

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

    def visualize(
        self,
        var_fn: Optional[Callable[["Node[ObservationType, ActionType]"], Any]] = None,
        max_depth: Optional[int] = None,
    ) -> None:
        dot = graphviz.Digraph(comment="MCTS Tree")
        self._add_node_to_graph(dot, var_fn, max_depth=max_depth)
        dot.render(filename="mcts_tree.gv", view=True)

    def _add_node_to_graph(
        self,
        dot: graphviz.Digraph,
        var_fn: Optional[Callable[["Node[ObservationType, ActionType]"], Any]] = None,
        max_depth: Optional[int] = None,
    ) -> None:
        if max_depth is not None and max_depth == 0:
            return
        label = f"R: {self.reward}, MS: {self.default_value(): .2f}, V: {self.value_evaluation: .2f}\nVisit: {self.visits}, T: {int(self.terminal)}"
        if var_fn is not None:
            label += f", Var: {var_fn(self)}"
        dot.node(str(id(self)), label=label)
        for action, child in self.children.items():
            child._add_node_to_graph(
                dot, var_fn, max_depth=max_depth - 1 if max_depth is not None else None
            )

            dot.edge(str(id(self)), str(id(child)), label=f"Action: {action}")

    def __str__(self):
        return f"Visits: {self.visits}, ter: {int(self.terminal)}\nR: {self.reward}\nSub_sum: {self.subtree_value}\nRollout: {self.default_value()}"

    def __repr__(self):
        return self.__str__()
