import functools
import multiprocessing
import sys
import time

sys.path.append('./src/')
import numpy as np
from tqdm import tqdm

from experiments.evaluate_from_config import eval_from_config
from experiments.train_from_config import train_from_config
import experiments.parameters as parameters

def train_model(project, entity, config, tags):
    # Assuming train_from_config is your training function
    # wait 0-3 seconds (random)
    time.sleep(3 * np.random.random())
    train_from_config(project, entity, config=config, tags=tags, performance=True)


def eval_agent(project, entity, config, tags):
    # Assuming train_from_config is your training function
    # wait 0-3 seconds (random)
    time.sleep(3 * np.random.random())
    eval_from_config(project, entity, config=config, tags=tags)

if __name__ == '__main__':
    entity, project = "ajzero", "AlphaZero"
    nr_runs = 1

    config_modifications = {
        "workers": 1,
        "eval_param": .1,
        "puct_c": 10.0,
        "runs": 10,
        "agent_type": "random_rollout",
        "rollout_budget": 20,
        # "eval_temp": 0.0,
        "planning_budget": 64,
    }

    run_config = {**parameters.base_parameters, **config_modifications}


    # variable_configs = [{"learning_rate": i} for i in [1e-2, 1e-3, 1e-4]]
    # series_configs = [{'use_visit_count': 1, 'tree_evaluation_policy': 'default', 'selection_policy': 'PUCT'},
    #                  {'use_visit_count': 1, 'tree_evaluation_policy': 'minimal_variance_constraint', 'selection_policy': 'PUCT'},
    #                  {'use_visit_count': 0, 'tree_evaluation_policy': 'default', 'selection_policy': 'PUCT'},
    #                  {'use_visit_count': 0, 'tree_evaluation_policy': 'minimal_variance_constraint', 'selection_policy': 'PUCT'},
    #                  ]
    # budget_configs = [{"planning_budget": 2**i} for i in range(4, 8)]
    # budget_configs = [{"planning_budget": i} for i in (16, 64, 256)]
    budget_configs = [{"eval_temp": x} for x in (.0, 1.0, 10.0)]

    series_configs = [
        {'tree_evaluation_policy': 'visit', 'selection_policy': 'UCT'},
                     {'tree_evaluation_policy': 'mvc', 'selection_policy': 'UCT'},
                     {'tree_evaluation_policy': 'mvc', 'selection_policy': 'PolicyUCT'},
        # {"selection_policy": 'PolicyPUCT',
        # "tree_evaluation_policy": "mvto",
        # "eval_param": 10.0,
        # "tree_value_transform": 'zero_one',},
        # {"selection_policy": 'PUCT',
        # "tree_evaluation_policy": "mvto",
        # "eval_param": 10.0,
        # "tree_value_transform": 'zero_one',},
                     ]
    # try every combination of variable_configs and series_config n times

    configs = [
        {**run_config, **env_config, **variable_config, **series_config} for env_config in parameters.env_challenges for variable_config in budget_configs for series_config in series_configs
    ] * nr_runs
    print(f"Number of runs: {len(configs)}")

    time_name = time.strftime("%Y-%m-%d-%H-%M-%S")

    tags = ['custom_sweep', time_name]
    # for config in tqdm(configs):
    #     train_from_config(project, entity, config=config, tags=tags, performance=True, debug=False)

    partial_train_model = functools.partial(eval_agent, project, entity, tags=tags)
    with multiprocessing.Pool(6) as p:
        list(tqdm(p.imap_unordered(partial_train_model, configs), total=len(configs)))
