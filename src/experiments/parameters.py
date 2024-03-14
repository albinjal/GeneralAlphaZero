def selection_to_expansion(selection_policy):
    """
    There is a connection between selection policy and expansion policy.
    puct -> fromprior
    policy_puct -> fromprior
    default -> default
    policy_uct -> default
    """
    if selection_policy in ["UCT", "PolicyUCT"]:
        return "default"
    else:
        return "fromprior"



selection = "PUCT"
base_parameters = {
    "model_type": "seperated",
    "observation_embedding": "default",
    "expansion_policy": selection_to_expansion(selection),
    "activation_fn": "relu",
    "norm_layer": "none",
    "dir_epsilon": 0.1,
    "dir_alpha": 0.5,
    "selection_policy": selection,
    # "root_seleciton_policy": selection,
    "puct_c": 1.0,
    "use_visit_count": 0,
    "regularization_weight": 0,
    "tree_evaluation_policy": "visit",
    'eval_param': .1,
    "tree_temperature": None,
    "hidden_dim": 64,
    "learning_rate": 1e-3,
    "sample_batch_ratio": 1,
    "n_steps_learning": 1,
    "training_epochs": 2,
    "planning_budget": 32,
    "layers": 3,
    "replay_buffer_multiplier": 5,
    "discount_factor": .98,
    "lr_gamma": 1.0,
    "iterations": 40,
    "policy_loss_weight": .5,
    "value_loss_weight": .5,
    "max_episode_length": 200,
    "env_params": dict(id='CartPole-v1', max_episode_steps=None),
    "workers": 8,
    "episodes_per_iteration": 8,
}
