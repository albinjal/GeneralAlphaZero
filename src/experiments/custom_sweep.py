import sys
import time
sys.path.append('./src/')
from tqdm import tqdm
import pandas as pd
import wandb
import asyncio
from experiments.wandb_sweeps import tune_alphazero_with_wandb
import experiments.sweep_configs as sweep_configs


if __name__ == '__main__':
    entity, project = "ajzero", "AlphaZero"
    nr_runs = 5

    config_modifications = {
    'iterations': 15,
    'tree_evaluation_policy': 'default',
    'selection_policy': 'PolicyPUCT',
    'eval_param': .1,
    'discount_factor': 1.0,
    'model_type': 'seperated',
    'max_episode_length': 200,
    'puct_c': 2.0,
    'compute_budget': 64,
    'use_visit_count': 1,
    'learning_rate': 1e-3
    }

    run_config = {**sweep_configs.base_parameters, **config_modifications}


    variable_configs = [{"learning_rate": i} for i in [1e-2, 1e-3, 1e-4]]
    series_configs = [{'use_visit_count': 1, 'tree_evaluation_policy': 'default', 'selection_policy': 'PUCT'},
                     {'use_visit_count': 1, 'tree_evaluation_policy': 'minimal_variance_constraint', 'selection_policy': 'PUCT'},
                     {'use_visit_count': 0, 'tree_evaluation_policy': 'default', 'selection_policy': 'PUCT'},
                     {'use_visit_count': 0, 'tree_evaluation_policy': 'minimal_variance_constraint', 'selection_policy': 'PUCT'},
                     ]
    # try every combination of variable_configs and series_config n times

    configs = [
        {**run_config, **variable_config, **series_config} for variable_config in variable_configs for series_config in series_configs
    ] * nr_runs

    time_name = time.strftime("%Y-%m-%d-%H-%M-%S")

    tags = ['custom_sweep', list(variable_configs[0])[0], time_name]
    for config in tqdm(configs):
        tune_alphazero_with_wandb(project, entity, config=config, tags=tags)
