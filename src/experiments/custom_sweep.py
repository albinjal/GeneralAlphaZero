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
    nr_runs = 1

    config_modifications = {
        'iterations': 15,
        'planning_budget': 32,
    }

    run_config = {**sweep_configs.base_parameters, **config_modifications}


    # variable_configs = [{"learning_rate": i} for i in [1e-2, 1e-3, 1e-4]]
    # series_configs = [{'use_visit_count': 1, 'tree_evaluation_policy': 'default', 'selection_policy': 'PUCT'},
    #                  {'use_visit_count': 1, 'tree_evaluation_policy': 'minimal_variance_constraint', 'selection_policy': 'PUCT'},
    #                  {'use_visit_count': 0, 'tree_evaluation_policy': 'default', 'selection_policy': 'PUCT'},
    #                  {'use_visit_count': 0, 'tree_evaluation_policy': 'minimal_variance_constraint', 'selection_policy': 'PUCT'},
    #                  ]
    # variable_configs = [{"planning_budget": 2**i} for i in range(4, 8)]
    variable_configs = [{
        "env_description": "CartPole-v1-300",
        "max_episode_length": 300,
        "env_params": dict(id='CartPole-v1', max_episode_steps=None),
    },{
        "env_description": "CliffWalking-v0-100",
        "max_episode_length": 100,
        "env_params": dict(id='CliffWalking-v0', max_episode_steps=None),
    },{
        "env_description": "FrozenLake-v1-4x4-100",
        "max_episode_length": 100,
        "env_params": dict(id='FrozenLake-v1-100', desc=None, map_name="4x4", is_slippery=False),
    },{
        "env_description": "FrozenLake-v1-8x8-100",
        "max_episode_length": 100,
        "env_params": dict(id='FrozenLake-v1', desc=None, map_name="8x8", is_slippery=False),
    },
                        ]
    series_configs = [{'tree_evaluation_policy': 'visit', 'selection_policy': 'PUCT'},
                     {'tree_evaluation_policy': 'mvc', 'selection_policy': 'PUCT'},
                     {'tree_evaluation_policy': 'mvc', 'selection_policy': 'PolicyPUCT'},
                     ]
    # try every combination of variable_configs and series_config n times

    configs = [
        {**run_config, **variable_config, **series_config} for variable_config in variable_configs for series_config in series_configs
    ] * nr_runs

    time_name = time.strftime("%Y-%m-%d-%H-%M-%S")

    tags = ['custom_sweep', list(variable_configs[0])[0], time_name]
    for config in tqdm(configs):
        tune_alphazero_with_wandb(project, entity, config=config, tags=tags)
