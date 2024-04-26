base_parameters = {
    "model_type": "seperated",
    "observation_embedding": "default",
    "activation_fn": "relu",
    "norm_layer": "none",
    "dir_epsilon": 0.4,
    "dir_alpha": 2.5,
    "selection_policy": "PUCT",
    "puct_c": 1.0,
    "selection_value_transform": "identity",
    "use_visit_count": False,
    "regularization_weight": 1e-6,
    "tree_evaluation_policy": "visit",
    "eval_param": 1.0,
    "tree_temperature": None,
    "tree_value_transform": "identity",
    "hidden_dim": 64,
    "learning_rate": 1e-3,
    "sample_batch_ratio": 4,
    "n_steps_learning": 3,
    "training_epochs": 4,
    "planning_budget": 32,
    "layers": 2,
    "replay_buffer_multiplier": 15,
    "discount_factor": 1.0,
    "lr_gamma": 1.0,
    "iterations": 40,
    "policy_loss_weight": 0.3,
    "value_loss_weight": 0.7,
    "max_episode_length": 200,
    "episodes_per_iteration": 6,
    "eval_temp": 0.0,
}

default_embedding_parameters = {
    "observation_embedding": "default",
    "use_visit_count": False,
    "sample_batch_ratio": 5,
    "n_steps_learning": 3,
    "training_epochs": 8,
    "puct_c": 3.0,
}

lake_discount_factor = 0.95
lake_config = {
    "max_episode_length": 100,
    "discount_factor": lake_discount_factor,
    "iterations": 30,
    "observation_embedding": "coordinate",
    "eval_param": 5.0,
    "training_epochs": 2,
    "puct_c": 1.0,
    "eval_param": 1.0,
}

env_challenges = [
    {
        "env_description": "CartPole-v1",
        "max_episode_length": 300,
        "iterations": 40,
        "env_params": dict(id="CartPole-v1", max_episode_steps=1000000000),
        "observation_embedding": "default",
        "ncols": None,
        "optimal_value": 300,
        "worst_value": 0.0,

    },
    {
        "env_description": "CliffWalking-v0",
        "max_episode_length": 100,
        "iterations": 30,
        "env_params": dict(id="CliffWalking-v0", max_episode_steps=1000000000),
        "ncols": 12,
        "optimal_value": -13,
        "worst_value": -200.0,
        "observation_embedding": "coordinate",
        "puct_c": 2.0,
        "eval_param": 1.0,
    },
    {
        **lake_config,
        "env_description": "FrozenLake-v1-4x4",
        "ncols": 4,
        "env_params": dict(
            id="FrozenLake-v1",
            desc=None,
            map_name="4x4",
            is_slippery=False,
            max_episode_steps=1000000000,
        ),
        "optimal_value": 1.0 * lake_discount_factor ** 5,
        "worst_value": -1.0,
    },
    {
        **lake_config,
        "env_description": "FrozenLake-v1-8x8",
        "ncols": 8,
        "env_params": dict(
            id="FrozenLake-v1",
            desc=None,
            map_name="8x8",
            is_slippery=False,
            max_episode_steps=1000000000,
        ),
        "optimal_value": 1.0 * lake_discount_factor ** 13,
        "worst_value": -1.0,

    },
    #     {
    #     "env_description": 'Acrobot-v1',
    #     "max_episode_length": 100,
    #     "iterations": 30,
    #     "env_params": dict(id='Acrobot-v1', max_episode_steps=1000000000),
    #     "observation_embedding": "default",
    #     "ncols": None,
    # }, # seems to work poorly cuz of exploration
    # {
    #     "env_description": "LunarLander-v2",
    #     "max_episode_length": 100,
    #     "iterations": 30,
    #     "env_params":dict(
    #         id="LunarLander-v2",
    #         max_episode_steps=1000000000,
    #         continuous=False,
    #         gravity=-10.0,
    #         enable_wind=False,
    #         wind_power=15.0,
    #         turbulence_power=1.5,
    #     ),
    #     "observation_embedding": "default",
    #     "ncols": None,
    # }, # https://github.com/openai/gym/issues/1292
]
