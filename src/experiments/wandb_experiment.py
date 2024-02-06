# import sys
# sys.path.append("src/")
# import datetime
# import multiprocessing
# import numpy as np
# import gymnasium as gym
# from torch.utils.tensorboard.writer import SummaryWriter
# import torch as th

# from torchrl.data import (
#     LazyTensorStorage,
#     TensorDictReplayBuffer,
# )
# import wandb
# from policies.expansion import DefaultExpansionPolicy
# from policies.selection import PUCT
# from policies.tree import DefaultTreeEvaluator

# from az.alphazero import AlphaZeroController
# from az.azmcts import AlphaZeroMCTS
# from az.model import AlphaZeroModel
# from experiments.hyperparams import grid_search


# def tune_alphazero_with_wandb(hparams, project_name="AlphaZero", entity = None, job_name = None):
#     # Initialize Weights & Biases
#     settings = wandb.Settings(job_name=job_name)
#     run = wandb.init(project=project_name, config=hparams, entity=entity, settings=settings)
#     assert run is not None

#     np.random.seed(0)
#     env = gym.make(hparams['env_id'])


#     discount_factor = hparams['discount_factor']
#     selection_policy = PUCT(c=hparams['puct_c'])
#     tree_evaluation_policy = DefaultTreeEvaluator()

#     model = AlphaZeroModel(env, hidden_dim=hparams['hidden_dim'], layers=hparams['layers'])
#     agent = AlphaZeroMCTS(selection_policy=selection_policy, model=model,
#                           discount_factor=discount_factor, expansion_policy=DefaultExpansionPolicy())

#     regularization_weight = hparams['regularization_weight']
#     optimizer = th.optim.Adam(model.parameters(), lr=hparams['learning_rate'], weight_decay=regularization_weight)

#     workers = multiprocessing.cpu_count()
#     self_play_games_per_iteration = workers
#     replay_buffer_size = hparams['replay_buffer_multiplier'] * self_play_games_per_iteration
#     sample_batch_size = replay_buffer_size // hparams['sample_batch_ratio']

#     replay_buffer = TensorDictReplayBuffer(storage=LazyTensorStorage(replay_buffer_size), batch_size=sample_batch_size)

#     run_name = f"{hparams['env_id']}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
#     log_dir = f"./tensorboard_logs/hyper/{run_name}"
#     writer = SummaryWriter(log_dir=log_dir)
#     run_dir = f"./runs/hyper/{run_name}"

#     controller = AlphaZeroController(
#         env,
#         agent,
#         optimizer,
#         replay_buffer=replay_buffer,
#         max_episode_length=hparams['max_episode_length'],
#         compute_budget=hparams['compute_budget'],
#         training_epochs=hparams['training_epochs'],
#         value_loss_weight=hparams['value_loss_weight'],
#         policy_loss_weight=hparams['policy_loss_weight'],
#         run_dir=run_dir,
#         self_play_iterations=self_play_games_per_iteration,
#         tree_evaluation_policy=tree_evaluation_policy,
#         self_play_workers=workers,
#         scheduler=th.optim.lr_scheduler.ExponentialLR(optimizer, gamma=hparams['lr_gamma'], verbose=True),
#         discount_factor=discount_factor,
#         n_steps_learning=hparams['n_steps_learning'],
#         checkpoint_interval=10,
#         use_visit_count=hparams['use_visit_count'],
#         writer=writer
#     )

#     metrics = controller.iterate(hparams['iterations'])

#     env.close()
#     # Finish the WandB run
#     run.finish()

#     return metrics


# def run():
#     # Some are only a single value
#     base_config = {
#         'env_id': ['CliffWalking-v0'],
#         'discount_factor': [1.0],
#         'max_episode_length': [200],
#         'iterations': [30],
#         'compute_budget': [50],
#         'training_epochs': [5],
#         'value_loss_weight': [1.0],
#         'policy_loss_weight': [1.0],
#         'lr_gamma': [1.0],
#         'n_steps_learning': [10],
#         'puct_c': [3],
#         'hidden_dim': [128],
#         'layers': [1],
#         'regularization_weight': [1e-4],
#         'learning_rate': [1e-4],
#         'replay_buffer_multiplier': [10],
#         'sample_batch_ratio': [5],
#         'eval_param': [1.0],
#         'use_visit_count': [False]
#     }

#     grid_search(tune_alphazero_with_wandb, base_config)



# if __name__ == '__main__':
#     run()
