import pytest
import gymnasium as gym
import numpy as np
import torch as th
from unittest.mock import Mock, MagicMock

import sys
sys.path.append('src/')

from policies import PolicyDistribution
from runner import run_episode
from mcts import MCTS

@pytest.fixture
def mock_solver():
    mock = Mock(spec=MCTS)
    mock.search.return_value = MagicMock()
    return mock

@pytest.fixture
def mock_env():
    mock = Mock(spec=gym.Env)
    mock.reset.return_value = (np.array([0.0]), {})
    mock.step.return_value = (np.array([0.0]), 0.0, False, False, {})
    mock.action_space = gym.spaces.Discrete(5)
    return mock

@pytest.fixture
def mock_policy():
    mock = Mock(spec=PolicyDistribution)
    mock.distribution.return_value = th.tensor([0.2, 0.2, 0.2, 0.2, 0.2])
    return mock

def test_run_episode(mock_solver, mock_env, mock_policy):
    trajectory, total_reward, total_entropy = run_episode(
        solver=mock_solver,
        env=mock_env,
        tree_evaluation_policy=mock_policy,
        compute_budget=1000,
        max_steps=1000,
        verbose=False,
        goal_obs=None,
        render=False,
        seed=None,
        device=None,
    )

    # Assert that the environment was reset
    mock_env.reset.assert_called_once()

    # Assert that the solver's search method was called
    mock_solver.search.assert_called()

    # Assert that the policy's distribution method was called
    mock_policy.distribution.assert_called()

    # Assert that the environment's step method was called
    mock_env.step.assert_called()

    # Assert that the trajectory is not empty
    assert len(trajectory) > 0

    # Assert that the total reward is a float
    assert isinstance(total_reward, float)

    # Assert that the total entropy is a float
    assert isinstance(total_entropy, float)
