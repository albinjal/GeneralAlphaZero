from policies.tree import expanded_tree_dict
from policies.selection import selection_dict_fn
from az.model import activation_function_dict, norm_dict
"""
Sweep configuration options
A sweep configuration consists of nested key-value pairs. Use top-level keys within your sweep configuration to define qualities of your sweep search such as the parameters to search through (parameter key), the methodology to search the parameter space (method key), and more.

The proceeding table lists top-level sweep configuration keys and a brief description. See the respective sections for more information about each key.

Top-level keys	Description
program	(required) Training script to run.
entity	Specify the entity for this sweep.
project	Specify the project for this sweep.
description	Text description of the sweep.
name	The name of the sweep, displayed in the W&B UI.
method	(required) Specify the search strategy.
metric	Specify the metric to optimize (only used by certain search strategies and stopping criteria).
parameters	(required) Specify parameters bounds to search.
early_terminate	Specify any early stopping criteria.
command	Specify command structure for invoking and passing arguments to the training script.
run_cap	Specify a maximum number of runs in a sweep.
See the Sweep configuration structure for more information on how to structure your sweep configuration.

metric
Use the metric top-level sweep configuration key to specify the name, the goal, and the target metric to optimize.

Key	Description
name	Name of the metric to optimize.
goal	Either minimize or maximize (Default is minimize).
target	Goal value for the metric you are optimizing. The sweep does not create new runs when if or when a run reaches a target value that you specify. Active agents that have a run executing (when the run reaches the target) wait until the run completes before the agent stops creating new runs.
parameters
In your YAML file or Python script, specify parameters as a top level key. Within the parameters key, provide the name of a hyperparameter you want to optimize. Common hyperparameters include: learning rate, batch size, epochs, optimizers, and more. For each hyperparameter you define in your sweep configuration, specify one or more search constraints.

The proceeding table shows supported hyperparameter search constraints. Based on your hyperparameter and use case, use one of the search constraints below to tell your sweep agent where (in the case of a distribution) or what (value, values, and so forth) to search or use.

Search constraint	Description
values	Specifies all valid values for this hyperparameter. Compatible with grid.
value	Specifies the single valid value for this hyperparameter. Compatible with grid.
distribution	Specify a probability distribution. See the note following this table for information on default values.
probabilities	Specify the probability of selecting each element of values when using random.
min, max	(intor float) Maximum and minimum values. If int, for int_uniform -distributed hyperparameters. If float, for uniform -distributed hyperparameters.
mu	(float) Mean parameter for normal - or lognormal -distributed hyperparameters.
sigma	(float) Standard deviation parameter for normal - or lognormal -distributed hyperparameters.
q	(float) Quantization step size for quantized hyperparameters.
parameters	Nest other parameters inside a root level parameter.
INFO
W&B sets the following distributions based on the following conditions if a distribution is not specified:

categorical if you specify values
int_uniform if you specify max and min as integers
uniform if you specify max and min as floats
constant if you provide a set to value
method
Specify the hyperparameter search strategy with the method key. There are three hyperparameter search strategies to choose from: grid, random, and Bayesian search.

Grid search
Iterate over every combination of hyperparameter values. Grid search makes uninformed decisions on the set of hyperparameter values to use on each iteration. Grid search can be computationally costly.

Grid search executes forever if it is searching within in a continuous search space.

Random search
Choose a random, uninformed, set of hyperparameter values on each iteration based on a distribution. Random search runs forever unless you stop the process from the command line, within your python script, or the W&B App UI.

Specify the distribution space with the metric key if you choose random (method: random) search.

Bayesian search
In contrast to random and grid search, Bayesian models make informed decisions. Bayesian optimization uses a probabilistic model to decide which values to use through an iterative process of testing values on a surrogate function before evaluating the objective function. Bayesian search works well for small numbers of continuous parameters but scales poorly. For more information about Bayesian search, see the Bayesian Optimization Primer paper.

Bayesian search runs forever unless you stop the process from the command line, within your python script, or the W&B App UI.

Distribution options for random and Bayesian search
Within the parameter key, nest the name of the hyperparameter. Next, specify the distribution key and specify a distribution for the value.

The proceeding tables lists distributions W&B supports.

Value for distribution key	Description
constant	Constant distribution. Must specify the constant value (value) to use.
categorical	Categorical distribution. Must specify all valid values (values) for this hyperparameter.
int_uniform	Discrete uniform distribution on integers. Must specify max and min as integers.
uniform	Continuous uniform distribution. Must specify max and min as floats.
q_uniform	Quantized uniform distribution. Returns round(X / q) * q where X is uniform. q defaults to 1.
log_uniform	Log-uniform distribution. Returns a value X between exp(min) and exp(max)such that the natural logarithm is uniformly distributed between min and max.
log_uniform_values	Log-uniform distribution. Returns a value X between min and max such that log(X) is uniformly distributed between log(min) and log(max).
q_log_uniform	Quantized log uniform. Returns round(X / q) * q where X is log_uniform. q defaults to 1.
q_log_uniform_values	Quantized log uniform. Returns round(X / q) * q where X is log_uniform_values. q defaults to 1.
inv_log_uniform	Inverse log uniform distribution. Returns X, where log(1/X) is uniformly distributed between min and max.
inv_log_uniform_values	Inverse log uniform distribution. Returns X, where log(1/X) is uniformly distributed between log(1/max) and log(1/min).
normal	Normal distribution. Return value is normally distributed with mean mu (default 0) and standard deviation sigma (default 1).
q_normal	Quantized normal distribution. Returns round(X / q) * q where X is normal. Q defaults to 1.
log_normal	Log normal distribution. Returns a value X such that the natural logarithm log(X) is normally distributed with mean mu (default 0) and standard deviation sigma (default 1).
q_log_normal	Quantized log normal distribution. Returns round(X / q) * q where X is log_normal. q defaults to 1.
early_terminate
Use early termination (early_terminate) to stop poorly performing runs. If early termination occurs, W&B stops the current run before it creates a new run with a new set of hyperparameter values.

NOTE
You must specify a stopping algorithm if you use early_terminate. Nest the type key within early_terminate within your sweep configuration.

Stopping algorithm
INFO
W&B currently supports Hyperband stopping algorithm.

Hyperband hyperparameter optimization evaluates if a program should stop or if it should to continue at one or more pre-set iteration counts, called brackets.

When a W&B run reaches a bracket, the sweep compares that run's metric to all previously reported metric values. The sweep terminates the run if the run's metric value is too high (when the goal is minimization) or if the run's metric is too low (when the goal is maximization).

Brackets are based on the number of logged iterations. The number of brackets corresponds to the number of times you log the metric you are optimizing. The iterations can correspond to steps, epochs, or something in between. The numerical value of the step counter is not used in bracket calculations.

INFO
Specify either min_iter or max_iter to create a bracket schedule.

Key	Description
min_iter	Specify the iteration for the first bracket
max_iter	Specify the maximum number of iterations.
s	Specify the total number of brackets (required for max_iter)
eta	Specify the bracket multiplier schedule (default: 3).
strict	Enable 'strict' mode that prunes runs aggressively, more closely following the original Hyperband paper. Defaults to false.
"""


import datetime


default_config = lambda: {
    "name": f"AlphaZero_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "Self_Play/Cumulative_Reward"},
    "run_cap": 1000,
    "parameters": {
        "use_visit_count": {"values": [1, 0], "distribution": "categorical"},
        "regularization_weight": {
            "min": -10.0,
            "max": -1.0,
            "distribution": "log_uniform",
        },
        "selection_policy": {
            "values": list(selection_dict_fn(1.0, 1.0, None).keys()),
            "distribution": "categorical",
        },
        "tree_evaluation_policy": {
            "values": list(expanded_tree_dict(1.0).keys()),
            "distribution": "categorical",
        },
        "hidden_dim": {"min": 2, "max": 10, "distribution": "q_log_uniform"},
        "policy_loss_weight": {"min": 1, "max": 5, "distribution": "log_uniform"},
        "learning_rate": {"min": -15, "max": -3, "distribution": "log_uniform"},
        "sample_batch_ratio": {"min": 1, "max": 10, "distribution": "int_uniform"},
        "n_steps_learning": {"min": 1, "max": 50, "distribution": "int_uniform"},
        "training_epochs": {"min": 1, "max": 100, "distribution": "int_uniform"},
        "compute_budget": {"min": 10, "max": 50, "distribution": "int_uniform"},
        "puct_c": {"min": 0, "max": 10, "distribution": "int_uniform"},
        "eval_param": {"min": 0, "max": 10, "distribution": "int_uniform"},
        "layers": {"min": 1, "max": 3, "distribution": "int_uniform"},
        "replay_buffer_multiplier": {
            "min": 1,
            "max": 20,
            "distribution": "int_uniform",
        },
        "discount_factor": {"value": 0.99, "distribution": "constant"},
        "lr_gamma": {"value": 1.0, "distribution": "constant"},
        "iterations": {"value": 20, "distribution": "constant"},
        "env_id": {"value": "CliffWalking-v0", "distribution": "constant"},
        "value_loss_weight": {"value": 1.0, "distribution": "constant"},
        "max_episode_length": {"value": 100, "distribution": "constant"},
    },
}


beta_vs_c = lambda: {
    "name": f"AlphaZero_Beta_VS_C_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
    "method": "grid",
    "metric": {"goal": "maximize", "name": "Self_Play/Cumulative_Reward"},
    "run_cap": 100,
    "parameters": {
        "selection_policy": {
            "values": ["PUCT", "PolicyPUCT"],
            "distribution": "categorical",
        },
        "puct_c": {"min": 0, "max": 10, "distribution": "int_uniform"},
        "eval_param": {"min": 0, "max": 10, "distribution": "int_uniform"},
        "use_visit_count": {"value": 0, "distribution": "constant"},
        "regularization_weight": {"value": 1e-3, "distribution": "constant"},
        "tree_evaluation_policy": {
            "value": "minimal_variance_constraint",
            "distribution": "constant",
        },
        "hidden_dim": {"value": 128, "distribution": "constant"},
        "policy_loss_weight": {"value": 3, "distribution": "constant"},
        "learning_rate": {"value": 2e-4, "distribution": "constant"},
        "sample_batch_ratio": {"value": 8, "distribution": "constant"},
        "n_steps_learning": {"value": 5, "distribution": "constant"},
        "training_epochs": {"value": 30, "distribution": "constant"},
        "compute_budget": {"value": 40, "distribution": "constant"},
        "layers": {"value": 2, "distribution": "constant"},
        "replay_buffer_multiplier": {"value": 10, "distribution": "constant"},
        "discount_factor": {"value": 0.99, "distribution": "constant"},
        "lr_gamma": {"value": 1.0, "distribution": "constant"},
        "iterations": {"value": 30, "distribution": "constant"},
        "env_id": {"value": "CliffWalking-v0", "distribution": "constant"},
        "value_loss_weight": {"value": 1.0, "distribution": "constant"},
        "max_episode_length": {"value": 200, "distribution": "constant"},
    },
}


beta_vs_c_2 = lambda: {
    "name": f"AlphaZero_Beta_VS_C_2_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
    "method": "grid",
    "metric": {"goal": "maximize", "name": "Self_Play/Cumulative_Reward"},
    "run_cap": 100,
    "parameters": {
        "selection_policy": {
            "value": "PolicyPUCT",
            "distribution": "constant",
        },
        "puct_c": {"values": [0, 1, 5, 10]},
        "eval_param": {"values": [0, 1, 5, 10]},
        "use_visit_count": {"value": 0, "distribution": "constant"},
        "regularization_weight": {"value": 1e-3, "distribution": "constant"},
        "tree_evaluation_policy": {
            "value": "minimal_variance_constraint",
            "distribution": "constant",
        },
        "hidden_dim": {"value": 128, "distribution": "constant"},
        "policy_loss_weight": {"value": 3, "distribution": "constant"},
        "learning_rate": {"value": 2e-4, "distribution": "constant"},
        "sample_batch_ratio": {"value": 8, "distribution": "constant"},
        "n_steps_learning": {"value": 5, "distribution": "constant"},
        "training_epochs": {"value": 30, "distribution": "constant"},
        "compute_budget": {"value": 40, "distribution": "constant"},
        "layers": {"value": 2, "distribution": "constant"},
        "replay_buffer_multiplier": {"value": 10, "distribution": "constant"},
        "discount_factor": {"value": 0.99, "distribution": "constant"},
        "lr_gamma": {"value": 1.0, "distribution": "constant"},
        "iterations": {"value": 30, "distribution": "constant"},
        "env_id": {"value": "CliffWalking-v0", "distribution": "constant"},
        "value_loss_weight": {"value": 1.0, "distribution": "constant"},
        "max_episode_length": {"value": 200, "distribution": "constant"},
    },
}


coord_search = lambda: {
    "name": f"AlphaZero_Coord_architecture{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "Self_Play/EMA_Reward"},
    "run_cap": 1000,
    "parameters": {
        "activation_fn": activation_function_dict.keys(),
        "norm_layer": norm_dict.keys(),
        "use_visit_count": {"values": [1, 0], "distribution": "categorical"},
        "regularization_weight": {
            "min": -15.0,
            "max": -1.0,
            "distribution": "log_uniform",
        },
        "selection_policy": {
            "value": "PUCT",
            "distribution": "constant",
        },
        "tree_evaluation_policy": {
            "value": "default",
            "distribution": "constant",
        },
        "hidden_dim": {"min": 3, "max": 15, "distribution": "q_log_uniform"},
        "policy_loss_weight": {"min": 1, "max": 5, "distribution": "log_uniform"},
        "learning_rate": {"min": -15, "max": -3, "distribution": "log_uniform"},
        "sample_batch_ratio": {"min": 1, "max": 10, "distribution": "int_uniform"},
        "n_steps_learning": {"values": [1, 3, 10], "distribution": "categorical"},
        "training_epochs": {"min": 1, "max": 30, "distribution": "int_uniform"},
        "compute_budget": {"min": 20, "max": 50, "distribution": "int_uniform"},
        "puct_c": {"min": 0, "max": 10, "distribution": "int_uniform"},
        "eval_param": {"min": 0, "max": 10, "distribution": "int_uniform"},
        "layers": {"min": 1, "max": 10, "distribution": "int_uniform"},
        "replay_buffer_multiplier": {
            "min": 1,
            "max": 20,
            "distribution": "int_uniform",
        },
        "discount_factor": {"value": 0.99, "distribution": "constant"},
        "lr_gamma": {"value": 1.0, "distribution": "constant"},
        "iterations": {"value": 20, "distribution": "constant"},
        "env_id": {"value": "CliffWalking-v0", "distribution": "constant"},
        "value_loss_weight": {"value": 1.0, "distribution": "constant"},
        "max_episode_length": {"value": 100, "distribution": "constant"},
    },
}
