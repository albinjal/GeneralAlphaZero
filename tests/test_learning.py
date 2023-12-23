import sys
import torch as th
import pytest

sys.path.append("src/")

from learning import n_step_value_targets, one_step_value_targets

# [Include the definitions of n_step_value_targets and one_step_value_targets here]


def test_value_targets_with_pytest():
    batch_size = 10
    n_steps = 20
    discount_factor = 0.99

    # Random tensors for rewards, values, and terminals
    rewards = th.randn(batch_size, n_steps)
    values = th.randn(batch_size, n_steps)
    terminals = th.randint(0, 2, (batch_size, n_steps)).bool()

    # Calculate targets
    n_step_targets = n_step_value_targets(
        rewards, values, terminals, discount_factor, 1
    )
    one_step_targets = one_step_value_targets(
        rewards, values, terminals, discount_factor
    )

    # Assert equality within a small tolerance
    assert th.allclose(n_step_targets, one_step_targets, rtol=1e-6, atol=1e-6)
