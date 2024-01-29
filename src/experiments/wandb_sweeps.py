import sys
sys.path.append("src/")
import datetime
import multiprocessing
import numpy as np
import gymnasium as gym
from torch.utils.tensorboard.writer import SummaryWriter
import torch as th

from torchrl.data import (
    LazyTensorStorage,
    TensorDictReplayBuffer,
)
import wandb
from policies.expansion import DefaultExpansionPolicy, ExpandFromPriorPolicy
from policies.selection import PUCT
from policies.tree import DefaultTreeEvaluator

from az.alphazero import AlphaZeroController
from az.azmcts import AlphaZeroMCTS
from az.model import AlphaZeroModel
from experiments.sweep_configs import default_config
from policies.tree import expanded_tree_dict
from policies.selection import selection_dict_fn


def tune_alphazero_with_wandb(project_name="AlphaZero", entity = None, job_name = None):
    # Initialize Weights & Biases
    settings = wandb.Settings(job_name=job_name)
    run = wandb.init(project=project_name, entity=entity, settings=settings)
    assert run is not None
    hparams = wandb.config
    print(hparams)
    env = gym.make(hparams['env_id'])

    discount_factor = hparams['discount_factor']
    tree_evaluation_policy = expanded_tree_dict(discount_factor)[hparams['tree_evaluation_policy']]
    selection_policy = selection_dict_fn(hparams['puct_c'], tree_evaluation_policy, discount_factor)[hparams['selection_policy']]
    print(selection_policy)
    expansion_policy = ExpandFromPriorPolicy()

    model = AlphaZeroModel(env, hidden_dim=hparams['hidden_dim'], layers=hparams['layers'])
    agent = AlphaZeroMCTS(selection_policy=selection_policy, model=model,
                          discount_factor=discount_factor, expansion_policy=expansion_policy)

    optimizer = th.optim.Adam(model.parameters(), lr=hparams['learning_rate'], weight_decay=hparams['regularization_weight'])

    workers = multiprocessing.cpu_count()
    self_play_games_per_iteration = workers
    replay_buffer_size = hparams['replay_buffer_multiplier'] * self_play_games_per_iteration
    sample_batch_size = replay_buffer_size // hparams['sample_batch_ratio']

    replay_buffer = TensorDictReplayBuffer(storage=LazyTensorStorage(replay_buffer_size), batch_size=sample_batch_size)

    run_name = f"{hparams['env_id']}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    log_dir = f"./tensorboard_logs/hyper/{run_name}"
    writer = SummaryWriter(log_dir=log_dir)
    run_dir = f"./runs/hyper/{run_name}"

    controller = AlphaZeroController(
        env,
        agent,
        optimizer,
        replay_buffer=replay_buffer,
        max_episode_length=hparams['max_episode_length'],
        compute_budget=hparams['compute_budget'],
        training_epochs=hparams['training_epochs'],
        value_loss_weight=hparams['value_loss_weight'],
        policy_loss_weight=hparams['policy_loss_weight'],
        run_dir=run_dir,
        self_play_iterations=self_play_games_per_iteration,
        tree_evaluation_policy=tree_evaluation_policy,
        self_play_workers=workers,
        scheduler=th.optim.lr_scheduler.ExponentialLR(optimizer, gamma=hparams['lr_gamma'], verbose=True),
        discount_factor=discount_factor,
        n_steps_learning=hparams['n_steps_learning'],
        checkpoint_interval=-1,
        use_visit_count=hparams['use_visit_count'],
        writer=writer,
        save_plots=False,
    )

    metrics = controller.iterate(hparams['iterations'])

    env.close()
    run.log_code(root="./src")
    # Finish the WandB run
    run.finish()
    return metrics

def sweep_agent():
    tune_alphazero_with_wandb()




if __name__ == '__main__':

    sweep_id = wandb.sweep(sweep=default_config, project="AlphaZero")

    wandb.agent(sweep_id, function=sweep_agent)
