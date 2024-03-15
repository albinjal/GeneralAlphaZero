import functools
import multiprocessing
import sys
import time
sys.path.append('./src/')
import numpy as np
from tqdm import tqdm

from experiments.train_from_config import train_from_config
import experiments.parameters as parameters

def train_model(project, entity, config, tags):
    # Assuming train_from_config is your training function
    # wait 0-3 seconds (random)
    time.sleep(3 * np.random.random())
    print("Training with config:", config)
    train_from_config(project, entity, config=config, tags=tags, performance=True, debug=False)


if __name__ == '__main__':
    entity, project = "ajzero", "AlphaZero"
    nr_runs = 1

    config_modifications = {
        'planning_budget': 32,
        "workers": 1,
        "episodes_per_iteration": 6,
    }

    run_config = {**parameters.base_parameters, **config_modifications}


    # variable_configs = [{"learning_rate": i} for i in [1e-2, 1e-3, 1e-4]]
    # series_configs = [{'use_visit_count': 1, 'tree_evaluation_policy': 'default', 'selection_policy': 'PUCT'},
    #                  {'use_visit_count': 1, 'tree_evaluation_policy': 'minimal_variance_constraint', 'selection_policy': 'PUCT'},
    #                  {'use_visit_count': 0, 'tree_evaluation_policy': 'default', 'selection_policy': 'PUCT'},
    #                  {'use_visit_count': 0, 'tree_evaluation_policy': 'minimal_variance_constraint', 'selection_policy': 'PUCT'},
    #                  ]
    # variable_configs = [{"planning_budget": 2**i} for i in range(4, 8)]
    env_challenges = [{
        "env_description": "CartPole-v1-300-30",
        "max_episode_length": 300,
        "iterations": 30,
        "env_params": dict(id='CartPole-v1', max_episode_steps=None),
        "observation_embedding": "default",
        "ncols": None,
    },{
        "env_description": "CliffWalking-v0-100-15",
        "max_episode_length": 100,
        "iterations": 15,
        "env_params": dict(id='CliffWalking-v0', max_episode_steps=None),
        "observation_embedding": "coordinate",
        "ncols": 12,

    },{
        "env_description": "FrozenLake-v1-4x4-150-20",
        "max_episode_length": 150,
        "iterations": 20,
        "env_params": dict(id='FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, max_episode_steps=None),
        "observation_embedding": "coordinate",
        "ncols": 4,
    },{
        "env_description": "FrozenLake-v1-8x8-150-20",
        "max_episode_length": 150,
        "iterations": 20,
        "env_params": dict(id='FrozenLake-v1', desc=None, map_name="8x8", is_slippery=False, max_episode_steps=None),
        "observation_embedding": "coordinate",
        "ncols": 8,
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
    # for config in tqdm(configs):
    #     train_from_config(project, entity, config=config, tags=tags, performance=True, debug=False)

    partial_train_model = functools.partial(train_model, project, entity, tags=tags)
    with multiprocessing.Pool(6) as p:
        list(tqdm(p.imap_unordered(partial_train_model, configs), total=len(configs)))
