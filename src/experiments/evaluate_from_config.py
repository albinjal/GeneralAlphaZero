import sys
sys.path.append("src/")

import numpy as np

from experiments.eval_agent import eval_agent

import multiprocessing


from core.mcts import RandomRolloutMCTS
import gymnasium as gym


import wandb

import experiments.parameters as parameters
from environments.observation_embeddings import ObservationEmbedding, embedding_dict
from az.azmcts import AlphaZeroMCTS
from az.model import (
    AlphaZeroModel,
)
from policies.tree_policies import tree_eval_dict
from policies.selection_distributions import selection_dict_fn
from policies.value_transforms import value_transform_dict
import torch as th
def agent_from_config(hparams: dict):
    env = gym.make(**hparams["env_params"])

    discount_factor = hparams["discount_factor"]
    if "tree_temperature" not in hparams:
        hparams["tree_temperature"] = None

    if "tree_value_transform" not in hparams or hparams["tree_value_transform"] is None:
        hparams["tree_value_transform"] = "identity"


    tree_evaluation_policy = tree_eval_dict(hparams["eval_param"], discount_factor, hparams["puct_c"], hparams["tree_temperature"], value_transform=value_transform_dict[hparams["tree_value_transform"]])[
        hparams["tree_evaluation_policy"]
    ]
    if "selection_value_transform" not in hparams or hparams["selection_value_transform"] is None:
        hparams["selection_value_transform"] = "identity"

    selection_policy = selection_dict_fn(
        hparams["puct_c"], tree_evaluation_policy, discount_factor, value_transform_dict[hparams["selection_value_transform"]]
    )[hparams["selection_policy"]]

    if "root_selection_policy" not in hparams or hparams["root_selection_policy"] is None:
        hparams["root_selection_policy"] = hparams["selection_policy"]

    root_selection_policy = selection_dict_fn(
        hparams["puct_c"], tree_evaluation_policy, discount_factor, value_transform_dict[hparams["selection_value_transform"]]
    )[hparams["root_selection_policy"]]

    observation_embedding: ObservationEmbedding = embedding_dict[hparams["observation_embedding"]](env.observation_space, hparams["ncols"] if "ncols" in hparams else None)
    if "observation_embedding" not in hparams:
        hparams["observation_embedding"] = "default"

    if hparams["agent_type"] == "random_rollout":
        if "rollout_budget" not in hparams:
            hparams["rollout_budget"] = 40
        agent = RandomRolloutMCTS(
            rollout_budget = hparams["rollout_budget"],
            root_selection_policy=root_selection_policy,
            selection_policy=selection_policy,
            discount_factor=discount_factor,
        )

    else:
        filename = hparams["model_file"]
        model: AlphaZeroModel = AlphaZeroModel.load_model(
            filename, env, False, hparams["hidden_dim"])

        model.eval()

        if "dir_epsilon" not in hparams:
            hparams["dir_epsilon"] = 0.0
            hparams["dir_alpha"] = None

        dir_epsilon = hparams["dir_epsilon"]
        dir_alpha = hparams["dir_alpha"]

        agent = AlphaZeroMCTS(
            root_selection_policy=root_selection_policy,
            selection_policy=selection_policy,
            model=model,
            dir_epsilon=dir_epsilon,
            dir_alpha=dir_alpha,
            discount_factor=discount_factor,
        )

    return agent, env, tree_evaluation_policy, observation_embedding, hparams["planning_budget"]


def eval_from_config(
    project_name="AlphaZero", entity=None, job_name=None, config=None, tags = None
):
    if tags is None:
        tags = []
    tags.append("evaluation")

    # Initialize Weights & Biases
    settings = wandb.Settings(job_name=job_name)
    run = wandb.init(
        project=project_name, entity=entity, settings=settings, config=config, tags=tags
    )
    assert run is not None
    hparams = wandb.config

    agent, env, tree_evaluation_policy, observation_embedding, planning_budget  = agent_from_config(hparams)
    if "workers" not in hparams or hparams["workers"] is None:
        hparams["workers"] = multiprocessing.cpu_count()
    workers = hparams["workers"]

    seeds = [None] * hparams["runs"]

    episode_returns, discounted_returns, time_steps, entropies = eval_agent(agent, env, tree_evaluation_policy, observation_embedding, planning_budget, hparams["max_episode_length"], seeds=seeds, temperature=hparams["eval_temp"], workers=workers)

    eval_res =  {
        "Evaluation/Returns": wandb.Histogram(np.array((episode_returns))),
        "Evaluation/Discounted_Returns": wandb.Histogram(np.array((discounted_returns))),
        "Evaluation/Timesteps": wandb.Histogram(np.array((time_steps))),
        "Evaluation/Entropies": wandb.Histogram(np.array(((th.sum(entropies, dim=-1) / time_steps)))),
        "Evaluation/Mean_Returns": episode_returns.mean().item(),
        "Evaluation/Mean_Discounted_Returns": discounted_returns.mean().item(),
        "Evaluation/Mean_Entropy": (th.sum(entropies, dim=-1) / time_steps).mean().item(),
    }
    run.log(data= eval_res)
    run.log_code(root="./src")
    # Finish the WandB run
    run.finish()



def eval_single():
    challenge = parameters.env_challenges[2]
    config_modifications = {
        "workers": 6,
        "tree_evaluation_policy": "visit",
        "eval_param": 1.0,
        "selection_policy": "UCT",
        "puct_c": 1.0,
        "runs": 200,
        "agent_type": "random_rollout",
        "rollout_budget": 40,
        "eval_temp": 0.0,
    }
    run_config = {**parameters.base_parameters, **challenge, **config_modifications}
    return eval_from_config(config=run_config)


if __name__ == "__main__":
    # sweep_id = wandb.sweep(sweep=coord_search, project="AlphaZero")

    # wandb.agent(sweep_id, function=sweep_agent)
    eval_single()