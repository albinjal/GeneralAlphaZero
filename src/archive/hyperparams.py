import datetime
import multiprocessing
import numpy as np
import gymnasium as gym
from torch.utils.tensorboard.writer import SummaryWriter
import torch as th
import random
import itertools
from torchrl.data import (
    LazyTensorStorage,
    TensorDictReplayBuffer,
)
import sys

from policies.selection_distributions import PUCT
from policies.tree_policies import VistationPolicy
sys.path.append("src/")

from az.alphazero import AlphaZeroController
from az.azmcts import AlphaZeroMCTS
from az.model import AlphaZeroModel

def tune_alphazero(hparams):
    env_id = hparams['env_id']
    env = gym.make(env_id)

    discount_factor = hparams['discount_factor']
    selection_policy = PUCT(c=hparams['puct_c'])
    tree_evaluation_policy = VistationPolicy()

    model = AlphaZeroModel(env, hidden_dim=hparams['hidden_dim'], layers=hparams['layers'])
    agent = AlphaZeroMCTS(selection_policy=selection_policy, model=model,
                          discount_factor=discount_factor)
    regularization_weight = hparams['regularization_weight']
    optimizer = th.optim.Adam(model.parameters(), lr=hparams['learning_rate'], weight_decay=regularization_weight)

    workers = multiprocessing.cpu_count()
    self_play_games_per_iteration = workers
    replay_buffer_size = hparams['replay_buffer_multiplier'] * self_play_games_per_iteration
    sample_batch_size = replay_buffer_size // hparams['sample_batch_ratio']

    replay_buffer = TensorDictReplayBuffer(storage=LazyTensorStorage(replay_buffer_size), batch_size=sample_batch_size)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    assert env.spec is not None
    env_id = env.spec.id
    run_name = f"{env_id}_{current_time}"
    log_dir = f"./tensorboard_logs/hyper/{run_name}"
    writer = SummaryWriter(log_dir=log_dir)
    run_dir = f"./runs/hyper/{run_name}"

    controller = AlphaZeroController(
        env,
        agent,
        optimizer,
        replay_buffer=replay_buffer,
        max_episode_length=hparams['max_episode_length'],
        planning_budget=hparams['planning_budget'],
        training_epochs=hparams['training_epochs'],
        value_loss_weight=hparams['value_loss_weight'],
        policy_loss_weight=hparams['policy_loss_weight'],
        writer=writer,
        run_dir=run_dir,
        self_play_iterations=self_play_games_per_iteration,
        tree_evaluation_policy=tree_evaluation_policy,
        self_play_workers=workers,
        scheduler=th.optim.lr_scheduler.ExponentialLR(optimizer, gamma=hparams['lr_gamma'], verbose=True),  # Optionally add scheduler here
        discount_factor=discount_factor,
        n_steps_learning=hparams['n_steps_learning'],
        checkpoint_interval=50,
        use_visit_count=hparams['use_visit_count']
    )

    metrics = controller.iterate(hparams['iterations'])
    env.close()

    writer.add_hparams(hparam_dict=hparams, metric_dict=metrics)
    writer.close()

    return metrics


def random_search(train_function, param_distributions, n_iter):
    for _ in range(n_iter):
        hparams = {key: random.choice(value) for key, value in param_distributions.items()}
        train_function(hparams)



def grid_search(train_function, param_grid):
    grid = list(itertools.product(*param_grid.values()))
    print(f"Grid search: {len(grid)} experiments")
    for params in grid:
        print(f"Running with params: {params}")
        hparams = dict(zip(param_grid.keys(), params))
        train_function(hparams)



def search_hyperparams():
    # Some are only a single value
    param_distributions = {
        'env_id': ['CliffWalking-v0'],
        'discount_factor': [1.0],
        'max_episode_length': [100],
        'iterations': [10],
        'planning_budget': [30],
        'training_epochs': [1, 10],
        'value_loss_weight': [1.0],
        'policy_loss_weight': [1.0],
        'lr_gamma': [1.0],
        'n_steps_learning': [1, 10],
        'puct_c': [1, 3],
        'hidden_dim': [128],
        'layers': [1],
        'regularization_weight': [1e-3, 1e-4],
        'learning_rate': [1e-4],
        'replay_buffer_multiplier': [10],
        'sample_batch_ratio': [5],
        'eval_param': [1.0],
        'use_visit_count': [True, False]
    }

    grid_search(tune_alphazero, param_distributions)





if __name__ == '__main__':
    search_hyperparams()
