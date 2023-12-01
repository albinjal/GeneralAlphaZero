import copy
import gymnasium as gym
from typing import Any, Generic, TypeVar, Tuple
import numpy as np
from node import Node

from policies import DefaultExpansionPolicy, OptionalPolicy, Policy

ObservationType = TypeVar("ObservationType")

NodeType = TypeVar("NodeType", bound="Node")

def try_to_set_state(env: gym.Env, state: Any):
    state_names = ["state", "_state", "s", "_s"]
    for state_name in state_names:
        if hasattr(env, state_name):
            setattr(env, state_name, state)
            return
    raise ValueError(f"Could not find state attribute in env. Tried: {state_names}")

def categorical_sample(prob_n, np_random: np.random.Generator):
    """Sample from categorical distribution where each row specifies class probabilities."""
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return np.argmax(csprob_n > np_random.random())

def step(env, a, state=None):
    if state is None:
        state = env.s
    transitions = env.P[state][a]
    i = categorical_sample([t[0] for t in transitions], env.np_random)
    p, s, r, t = transitions[i]
    env.s = s
    env.lastaction = a

    if env.render_mode == "human":
        env.render()
    return (int(s), r, t, False, {"prob": p})


class MCTS(Generic[ObservationType]):
    """
    This class contains the basic MCTS algorithm without assumtions on the value function.
    """

    env: gym.Env[ObservationType, np.int64]
    selection_policy: OptionalPolicy[ObservationType]
    expansion_policy: Policy[
        ObservationType
    ] | None  # the expansion policy is usually "pick uniform non explored action"

    def __init__(
        self,
        selection_policy: OptionalPolicy[ObservationType],
        expansion_policy: Policy[ObservationType] | None = None,
    ):
        self.selection_policy = selection_policy  # the selection policy should return None if the input node should be expanded
        self.expansion_policy = expansion_policy

    def search(
        self,
        env: gym.Env[ObservationType, np.int64],
        iterations: int,
        obs: ObservationType,
        reward: np.float32,
    ) -> Node[ObservationType]:
        # the env should be in the state we want to search from
        self.env = copy.deepcopy(env)
        # assert that the type of the action space is discrete
        assert isinstance(env.action_space, gym.spaces.Discrete)

        root_node = Node[ObservationType](
            parent=None, reward=reward, action_space=env.action_space, observation=obs
        )
        value = self.value_function(root_node)
        root_node.value_evaluation = value
        return self.build_tree(root_node, iterations)

    def build_tree(self, from_node: NodeType, iterations: int) -> NodeType:
        while from_node.visits < iterations:
            # traverse the tree and select the node to expand
            selected_node_for_expansion = self.select_node_to_expand(from_node)
            # check if the node is terminal
            if selected_node_for_expansion.is_terminal():
                # if the node is terminal, we can not expand it
                # the value (sum of future reward) of the node is 0
                # the backup will still propagate the visit and reward
                selected_node_for_expansion.value_evaluation = np.float32(0.0)
                selected_node_for_expansion.backup(np.float32(0))
            else:
                self.handle_selected_node(selected_node_for_expansion)

        return from_node

    def handle_selected_node(
        self, node: Node[ObservationType]):
        if self.expansion_policy is None:
            self.handle_all(node)
        else:
            action = self.expansion_policy(node)
            self.handle_single(node, action)

    def handle_single(
        self,
        node: Node[ObservationType],
        action: np.int64,
    ):
        eval_node = self.expand(node, action)
        # evaluate the node
        value = self.value_function(eval_node)
        # backupagate the value
        eval_node.value_evaluation = value
        eval_node.backup(value)

    def handle_all(
        self, node: Node[ObservationType]
    ):
        for action in range(node.action_space.n):
            self.handle_single(node, np.int64(action))

    def value_function(
        self,
        node: Node[ObservationType],
    ) -> np.float32:
        """The point of the value function is to estimate the value of the node.
        The value is defined as the expected future reward if we get to the node given some policy.
        """
        return np.float32(0.0)

    def select_node_to_expand(
        self, from_node: NodeType
    ) -> NodeType:
        """
        Returns the node to be expanded next.
        Returns None if the node is terminal.
        The selection policy returns None if the input node should be expanded.
        """

        node = from_node
        # the reason we copy the env is because we want to keep the original env in the root state
        # Question: note that all envs will have the same seed, this might needs to be dealt with for stochastic envs
        while not node.is_terminal():
            # select which node to step into
            action = self.selection_policy(node)
            # if the selection policy returns None, this indicates that the current node should be expanded
            if action is None:
                return node
            # step into the node
            node = node.step(action)
            # also step the environment
            # Question: right now we do not save the observation or reward from the env since we already have them saved
            # This might be worth considering though if we use stochastic envs since the rewards/states could vary each time we execute an action sequence

        return node

    def expand(
        self, node: NodeType, action: np.int64
    ) -> NodeType:
        """
        Expands the node and returns the expanded node.
        Note that the function will modify the env and the input node
        """

        observation, reward, terminated, truncated, _ = step(self.env, action, state=node.observation)
        terminal = terminated or truncated
        node_class = type(node)
        # create the node
        new_child = node_class(
            parent=node,
            reward=np.float32(reward),
            action_space=node.action_space,
            terminal=terminal,
            observation=observation,
        )
        node.children[action] = new_child
        return new_child


class RandomRolloutMCTS(MCTS):
    def __init__(self, rollout_budget=40, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rollout_budget = rollout_budget

    def value_function(
        self,
        node: Node,
    ) -> np.float32:
        """
        The standard value function for MCTS is the the sum of the future reward when acting with uniformly random policy.
        """
        # if the node is terminal, return 0
        if node.is_terminal():
            return np.float32(0.0)

        o = node.observation
        # if the node is not terminal, simulate the enviroment with random actions and return the accumulated reward until termination
        accumulated_reward = np.float32(0.0)
        for _ in range(self.rollout_budget):
            o, reward, terminated, truncated, _ = step(self.env, self.env.action_space.sample(), o)
            accumulated_reward += np.float32(reward)
            if terminated or truncated:
                break

        return accumulated_reward
