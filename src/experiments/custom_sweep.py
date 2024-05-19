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

def train_model(project, entity, config, tags, sleep=3):
    # Assuming train_from_config is your training function
    # wait 0-3 seconds (random)
    time.sleep(sleep * np.random.random())
    train_from_config(project, entity, config=config, tags=tags, performance=True)


def eval_agent(project, entity, config, tags, sleep=3):
    # Assuming train_from_config is your training function
    # wait 0-3 seconds (random)
    time.sleep(sleep * np.random.random())
    eval_from_config(project, entity, config=config, tags=tags)

if __name__ == '__main__':
    entity, project = "ajzero", "AlphaZero"
    nr_runs = 2
    cpu_count = multiprocessing.cpu_count()

    if cpu_count >= 12:
        workers_per_run = 6
        total_workers = cpu_count
    else:
        workers_per_run = 1
        total_workers = min(6, cpu_count)
    sweep_workers = total_workers // workers_per_run
    config_modifications = {
        "workers": workers_per_run,
        # "runs": 6*10,
        # "agent_type": "random_rollout",
        # "rollout_budget": 50,
        # "eval_temp": 0.0,
    }

    if False:
        # env modificaitons
        mods = {
            "CartPole-v1": {
                "rollout_budget": 200,
                "puct_c": (c := 10.0),
                "eval_param": 1.0 / c,
                "max_episode_length": 1000,
                "planning_budget": 8,
            },
            "FrozenLake-v1-4x4": {
                "rollout_budget": 30,
                "planning_budget": 32,
                "puct_c": (c := 1.0),
                "eval_param": 1.0 / c,
            }
        }

        for env in parameters.env_challenges:
            if env["env_description"] in mods:
                env.update(mods[env["env_description"]])

    envs = parameters.env_challenges[3:4]


    run_config = {**parameters.base_parameters, **config_modifications}


    # variable_configs = [{"learning_rate": i} for i in [1e-2, 1e-3, 1e-4]]
    # series_configs = [{'use_visit_count': 1, 'tree_evaluation_policy': 'default', 'selection_policy': 'PUCT'},
    #                  {'use_visit_count': 1, 'tree_evaluation_policy': 'minimal_variance_constraint', 'selection_policy': 'PUCT'},
    #                  {'use_visit_count': 0, 'tree_evaluation_policy': 'default', 'selection_policy': 'PUCT'},
    #                  {'use_visit_count': 0, 'tree_evaluation_policy': 'minimal_variance_constraint', 'selection_policy': 'PUCT'},
    #                  ]
    budget_configs = [{"planning_budget": 2**i} for i in range(4, 8)]
    # budget_configs = [{"planning_budget": i} for i in (32)]
    # budget_configs = [{"use_visit_count": x} for x in [True]]
    # budget_configs = [{"eval_temp": x} for x in (.0,
    #                                              # 1.0, 10.0
    #                                              )]

    series_configs = [
        {'tree_evaluation_policy': 'visit', 'selection_policy': 'PUCT'},
                     {'tree_evaluation_policy': 'mvc', 'selection_policy': 'PUCT'},
                     {'tree_evaluation_policy': 'mvc', 'selection_policy': 'PolicyPUCT'},
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
        {**run_config, **env_config, **variable_config, **series_config} for env_config in envs for variable_config in budget_configs for series_config in series_configs
    ] * nr_runs
    print(f"Number of runs: {len(configs)}")

    time_name = time.strftime("%Y-%m-%d-%H-%M-%S")

    tags = ['custom_sweep', time_name]
    job = train_model
    if sweep_workers == 1:
        for config in tqdm(configs):
            job(project, entity, config, tags, sleep=0.0)
    else:
        partial_train_model = functools.partial(job, project, entity, tags=tags)
        with multiprocessing.Pool(sweep_workers) as p:
            list(tqdm(p.imap_unordered(partial_train_model, configs), total=len(configs)))
