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

import experiments.parameters as parameters
from environments.observation_embeddings import ObservationEmbedding, embedding_dict
from az.alphazero import AlphaZeroController
from az.azmcts import AlphaZeroMCTS
from az.model import (
    AlphaZeroModel,
    activation_function_dict,
    norm_dict,
    models_dict,
)
from policies.tree_policies import tree_eval_dict
from policies.selection_distributions import selection_dict_fn
from policies.value_transforms import value_transform_dict


def train_from_config(
    project_name="AlphaZero", entity=None, job_name=None, config=None, performance=True, tags = None,
):
    if tags is None:
        tags = []
    tags.append("training")

    if performance:
        tags.append("performance")

    # Initialize Weights & Biases
    settings = wandb.Settings(job_name=job_name)
    run = wandb.init(
        project=project_name, entity=entity, settings=settings, config=config, tags=tags
    )
    assert run is not None
    hparams = wandb.config
    print(hparams)
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

    if "observation_embedding" not in hparams:
        hparams["observation_embedding"] = "default"
    observation_embedding: ObservationEmbedding = embedding_dict[hparams["observation_embedding"]](env.observation_space, hparams["ncols"] if "ncols" in hparams else None)
    model: AlphaZeroModel = models_dict[hparams["model_type"]](
        env,
        observation_embedding=observation_embedding,
        hidden_dim=hparams["hidden_dim"],
        nlayers=hparams["layers"],
        activation_fn=activation_function_dict[hparams["activation_fn"]],
        norm_layer=norm_dict[hparams["norm_layer"]],
    )

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

    optimizer = th.optim.Adam(
        model.parameters(),
        lr=hparams["learning_rate"],
        weight_decay=hparams["regularization_weight"],
    )

    if "workers" not in hparams or hparams["workers"] is None:
        hparams["workers"] = multiprocessing.cpu_count()
    workers = hparams["workers"]

    if "episodes_per_iteration" not in hparams or hparams["episodes_per_iteration"] is None:
        hparams["episodes_per_iteration"] = workers
    episodes_per_iteration = hparams["episodes_per_iteration"]

    replay_buffer_size = (
        hparams["replay_buffer_multiplier"] * episodes_per_iteration
    )
    sample_batch_size = replay_buffer_size // hparams["sample_batch_ratio"]

    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(replay_buffer_size)
    )

    run_name = (
        f"{env.spec.id}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    log_dir = f"./tensorboard_logs/hyper/{run_name}"
    writer = SummaryWriter(log_dir=log_dir)
    run_dir = f"./runs/hyper/{run_name}"

    controller = AlphaZeroController(
        env,
        agent,
        optimizer,
        replay_buffer=replay_buffer,
        max_episode_length=hparams["max_episode_length"],
        planning_budget=hparams["planning_budget"],
        training_epochs=hparams["training_epochs"],
        value_loss_weight=hparams["value_loss_weight"],
        policy_loss_weight=hparams["policy_loss_weight"],
        run_dir=run_dir,
        episodes_per_iteration=episodes_per_iteration,
        tree_evaluation_policy=tree_evaluation_policy,
        self_play_workers=workers,
        scheduler=th.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=hparams["lr_gamma"], verbose=True
        ),
        discount_factor=discount_factor,
        n_steps_learning=hparams["n_steps_learning"],
        checkpoint_interval=-1 if performance else 10,
        use_visit_count=bool(hparams["use_visit_count"]),
        writer=writer,
        save_plots=not performance,
        batch_size=sample_batch_size,
    )
    iterations = hparams["iterations"]
    # start_temp = 1.0
    # end_temp = 0.5
    # # Exponential decay from start_temp to end_temp
    # temp_schedule = np.exp(np.linspace(np.log(start_temp), np.log(end_temp), iterations))
    temp_schedule = [None] * iterations
    metrics = controller.iterate(temp_schedule=temp_schedule)

    env.close()
    run.log_code(root="./src")
    # Finish the WandB run
    run.finish()
    return metrics


def sweep_agent():
    train_from_config(performance=True)


def run_single():
    challenge = parameters.env_challenges[3]
    config_modifications = {
        "workers": 6,
        "tree_evaluation_policy": "mvc",
        "eval_param": 2.0,
        "planning_budget": 64,
        "selection_policy": "PolicyPUCT",
        "puct_c": 1.0,
        "n_steps_learning": 1,
        "training_epochs": 2,
        "learning_rate": 1e-3,
    }
    run_config = {**parameters.base_parameters, **challenge, **config_modifications}
    return train_from_config(config=run_config, performance=False)


if __name__ == "__main__":
    # sweep_id = wandb.sweep(sweep=coord_search, project="AlphaZero")

    # wandb.agent(sweep_id, function=sweep_agent)
    run_single()
