from collections import Counter
from typing import Dict, Generic, List, TypeVar, Optional, Any, Callable, Tuple
import gymnasium as gym
import numpy as np
import torch as th


ObservationType = TypeVar("ObservationType")

NodeType = TypeVar("NodeType", bound="Node")



class Node(Generic[ObservationType]):
    parent: Optional["Node[ObservationType]"]
    children: Dict[int, "Node[ObservationType]"]
    visits: int = 0
    subtree_sum: float = 0.0  # sum of reward and value of all children
    value_evaluation: float  # expected future reward
    reward: float  # reward recived when stepping into this node
    # Discrete action space
    action_space: gym.spaces.Discrete  # the reference to the action space
    observation: Optional[ObservationType]
    prior_policy: th.Tensor
    env: Optional[gym.Env]
    variance: float | None = None
    policy_value: float | None = None

    def __init__(
        self,
        env: gym.Env,
        parent: Optional["Node[ObservationType]"],
        reward: float,
        action_space: gym.spaces.Discrete,
        observation: Optional[ObservationType],
        terminal: bool = False,
    ):
        # TODO: lazy init?
        self.children = {}
        self.action_space = action_space
        self.reward = reward
        self.parent = parent
        self.terminal = terminal
        self.observation = observation
        self.env = env

    def is_terminal(self) -> bool:
        return self.terminal

    def step(self, action: int) -> "Node[ObservationType]":
        # steps into the action and returns that node
        child = self.children[action]
        return child

    def default_value(self) -> float:
        """
        The default value estimate for taking this action is the average of the rewards + value estimates of all children
        """
        return self.subtree_sum / self.visits



    def is_fully_expanded(self) -> bool:
        return len(self.children) == self.action_space.n

    def sample_unexplored_action(self) -> int:
        """
        mask – An optional mask for if an action can be selected. Expected np.ndarray of shape (n,) and dtype np.int8 where 1 represents valid actions and 0 invalid / infeasible actions. If there are no possible actions (i.e. np.all(mask == 0)) then space.start will be returned.
        """
        mask = np.ones(self.action_space.n, dtype=np.int8)
        for action in self.children:
            mask[action] = 0
        return self.action_space.sample(mask=mask)

    def get_root(self) -> "Node[ObservationType]":
        node: Node[ObservationType] | None = self
        while node.parent is not None:
            node = node.parent
        return node


    def visualize(
        self,
        var_fn: Optional[Callable[["Node[ObservationType]"], Any]] = None,
        max_depth: Optional[int] = None,
    ) -> None:
        import graphviz

        dot = graphviz.Digraph(comment="MCTS Tree")
        self._add_node_to_graph(dot, var_fn, max_depth=max_depth)
        dot.render(filename="mcts_tree.gv", view=True)


    def _add_node_to_graph(
        self,
        dot,
        var_fn: Optional[Callable[["Node[ObservationType]"], Any]] = None,
        max_depth: Optional[int] = None,
    ) -> None:
        if max_depth is not None and max_depth == 0:
            return
        label = f"O: {self.observation}, R: {self.reward}, MS: {self.default_value(): .2f}, V: {self.value_evaluation: .2f}\nVisit: {self.visits}, T: {int(self.terminal)}"
        if var_fn is not None:
            label += f", VarFn: {var_fn(self)}"

        if self.policy_value is not None:
            label += f", PV: {self.policy_value: .2f}"
        if self.variance is not None:
            label += f", Var: {self.variance: .2f}"
        dot.node(str(id(self)), label=label)
        for action, child in self.children.items():
            child._add_node_to_graph(
                dot, var_fn, max_depth=max_depth - 1 if max_depth is not None else None
            )

            dot.edge(str(id(self)), str(id(child)), label=f"Action: {action}")

    def state_visitation_counts(self) -> Counter:
        """
        Returns a counter of the number of times each state has been visited
        """
        counter = Counter()
        # add the current node
        counter[self.observation] += self.visits
        # add all children
        for child in self.children.values():
            counter.update(child.state_visitation_counts())

        return counter

    def get_children(self):
        # return a list of all children
        l: List[Node | None] = [None] * self.action_space.n
        for key, child in self.children.items():
            l[key] = child
        return l

    def reset_policy_value(self):
        self.policy_value = None
        for child in self.children.values():
            child.reset_policy_value()

    def reset_variance(self):
        self.variance = None
        for child in self.children.values():
            child.reset_variance()

    def reset_var_val(self):
        self.variance = None
        self.policy_value = None
        for child in self.children.values():
            child.reset_var_val()

    def __str__(self):
        return f"Visits: {self.visits}, ter: {int(self.terminal)}\nR: {self.reward}\nSub_sum: {self.subtree_sum}\nRollout: {self.default_value()}"

    def __repr__(self):
        return self.__str__()
