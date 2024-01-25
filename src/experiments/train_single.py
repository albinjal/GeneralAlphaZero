import datetime
import multiprocessing
import sys
sys.path.append("src/")

import gymnasium as gym
import numpy as np
import torch as th
from torchrl.data import (
    LazyTensorStorage,
    TensorDictReplayBuffer,
)
from torch.utils.tensorboard.writer import SummaryWriter
import numpy as np


from az.alphazero import AlphaZeroController
from policies.expansion import ExpandFromPriorPolicy
from policies.selection import PolicyUCT
from policies.tree import MinimalVarianceConstraintPolicy
from az.azmcts import AlphaZeroMCTS
from az.model import AlphaZeroModel



def train_alphazero():
    # set seed
    np.random.seed(0)
    # env_id = "CartPole-v1"
    env_id = "CliffWalking-v0"
    # env_id = "FrozenLake-v1"
    env = gym.make(env_id)

    iterations = 100
    discount_factor = .99

    tree_evaluation_policy = MinimalVarianceConstraintPolicy(1.0, discount_factor)
    selection_policy = PolicyUCT(c=1, policy=tree_evaluation_policy, discount_factor=discount_factor)
    expansion_policy = ExpandFromPriorPolicy()

    model = AlphaZeroModel(env, hidden_dim=2**5, layers=10, pref_gpu=False)
    agent = AlphaZeroMCTS(selection_policy=selection_policy, model=model, discount_factor=discount_factor, expansion_policy=expansion_policy)
    regularization_weight = 1e-9
    optimizer = th.optim.Adam(model.parameters(), lr=5e-4, weight_decay=regularization_weight)
    scheduler = th.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1, verbose=True)
    debug = False
    workers = 1 if debug else multiprocessing.cpu_count()

    self_play_games_per_iteration = workers
    replay_buffer_size = 10 * self_play_games_per_iteration
    sample_batch_size = replay_buffer_size // 5

    replay_buffer = TensorDictReplayBuffer(storage=LazyTensorStorage(replay_buffer_size), batch_size=sample_batch_size)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    assert env.spec is not None
    env_id = env.spec.id
    run_name = f"{env_id}_{current_time}"
    log_dir = f"./tensorboard_logs/{run_name}"
    writer = SummaryWriter(log_dir=log_dir)
    run_dir = f"./runs/{run_name}"


    controller = AlphaZeroController(
        env,
        agent,
        optimizer,
        replay_buffer = replay_buffer,
        max_episode_length=200,
        compute_budget=100,
        training_epochs=10,
        value_loss_weight=1.0,
        policy_loss_weight=1.0,
        writer=writer,
        run_dir=run_dir,
        self_play_iterations=self_play_games_per_iteration,
        tree_evaluation_policy=tree_evaluation_policy,
        self_play_workers=workers,
        scheduler=scheduler,
        discount_factor=discount_factor,
        n_steps_learning=10,
        use_visit_count=True,
    )
    controller.iterate(iterations)
    env.close()
    writer.close()

if __name__ == "__main__":
    train_alphazero()
