import pytest
import numpy as np
import gymnasium as gym
import sys
sys.path.append('src/')

from node import Node


@pytest.fixture
def node():
    parent_node = Node(None, np.float32(0.0), gym.spaces.Discrete(5), None)
    return Node(parent_node, np.float32(1.0), gym.spaces.Discrete(5), None)


def test_is_terminal(node):
    assert not node.is_terminal()


def test_step(node):
    action = np.int64(1)
    node.children[action] = Node(node, np.float32(2.0), gym.spaces.Discrete(5), None)
    child = node.step(action)
    assert child == node.children[action]


def test_backup(node):
    value = np.float32(3.0)
    node.backup(value)
    assert node.subtree_sum == value + node.reward
    assert node.visits == 1


def test_default_value(node):
    node.visits = 1
    node.subtree_sum = np.float32(3.0)
    assert node.default_value() == node.subtree_sum / node.visits


def test_is_fully_expanded(node):
    assert not node.is_fully_expanded()
    for i in range(5):
        node.children[i] = Node(node, np.float32(i), gym.spaces.Discrete(5), None)
    assert node.is_fully_expanded()


def test_sample_unexplored_action(node):
    for i in range(4):
        node.children[i] = Node(node, np.float32(i), gym.spaces.Discrete(5), None)
    unexplored_action = node.sample_unexplored_action()
    assert unexplored_action == 4


def test_get_root(node):
    root = node.get_root()
    assert root == node.parent


def test_state_visitation_counts(node):
    for i in range(5):
        node.children[i] = Node(node, np.float32(i), gym.spaces.Discrete(5), None)
        node.children[i].visits = i
    counts = node.state_visitation_counts()
    assert sum(counts.values()) == sum(range(5)) + node.visits
