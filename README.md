# Pessimistic AlphaZero

This repository contains the code for the Pessimisc AlphaZero M.Sc thesis project.


## Structure
The src folder contains a modular MCTS + PyTorch AlphaZero implementation. The experimental data is recorded with both Tensorboard and Weights & Biases. The `experiments` folder contains the code for running the experiments, currently, I primarily use the `wandb_sweeps.py` file for running experiments and modify the hyperparameters in the `sweep_configs.py` file.

## Installation
TODO: needs to be updated, should install custom gym env with the 6 * 12 Cliffwalking env
