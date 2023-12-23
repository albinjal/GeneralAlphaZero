import copy
import datetime
import time
from typing import List, Tuple
import gymnasium as gym
import numpy as np
from tqdm import tqdm
from azmcts import AlphaZeroMCTS
from environment import obs_to_tensor, show_model_in_tensorboard
import torch as th
from torchrl.data import (
    ReplayBuffer,
    ListStorage,
    LazyTensorStorage,
    Storage,
    TensorDictReplayBuffer,
    TensorDictPrioritizedReplayBuffer
)
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np

import multiprocessing
from learning import n_step_value_targets, one_step_value_targets
from mcts import MCTS
from model import AlphaZeroModel
from node import Node
from policies import PUCT, UCT, DefaultExpansionPolicy, DefaultTreeEvaluator, Policy
from runner import run_episode





def run_episode_process(args):
    """Wrapper function for multiprocessing that unpacks arguments and runs a single episode."""
    agent, env, tree_evaluation_policy, compute_budget, max_episode_length = args
    return run_episode(
        agent, env, tree_evaluation_policy, compute_budget, max_episode_length
    )



class AlphaZeroController:
    """
    The Controller will be responsible for orchistrating the training of the model. With self play and training.
    """

    replay_buffer: ReplayBuffer
    training_epochs: int
    model: AlphaZeroModel

    def __init__(
        self,
        env: gym.Env,
        agent: AlphaZeroMCTS,
        optimizer: th.optim.Optimizer,
        replay_buffer = TensorDictReplayBuffer(),
        training_epochs=10,
        tree_evaluation_policy=DefaultTreeEvaluator(),
        compute_budget=100,
        max_episode_length=500,
        tensorboard_dir="./tensorboard_logs",
        runs_dir="./runs",
        checkpoint_interval=5,
        value_loss_weight=1.0,
        policy_loss_weight=1.0,
        self_play_iterations=10,
        self_play_workers = 1,
        secheduler = None,
        value_sim_loss = False,
        discount_factor = 1.0,
        n_steps_learning: int = 1,
    ) -> None:
        self.replay_buffer = replay_buffer
        self.training_epochs = training_epochs
        self.optimizer = optimizer
        self.agent = agent
        self.env = env
        self.tree_evaluation_policy = tree_evaluation_policy
        self.compute_budget = compute_budget
        self.max_episode_length = max_episode_length
        self.self_play_workers = self_play_workers
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        assert env.spec is not None
        env_id = env.spec.id
        self.run_name = f"{env_id}_{current_time}"
        log_dir = f"{tensorboard_dir}/{self.run_name}"
        self.writer = SummaryWriter(log_dir=log_dir)
        self.run_dir = f"{runs_dir}/{self.run_name}"
        self.value_sim_loss = value_sim_loss
        # create run dir if it does not exist
        os.makedirs(self.run_dir, exist_ok=True)

        self.checkpoint_interval = checkpoint_interval
        self.discount_factor = discount_factor
        self.n_steps_learning = n_steps_learning
        # Log the model
        if self.agent.model.device == th.device("cpu"):
            self.writer.add_graph(
                self.agent.model,
                th.tensor(
                    [
                        gym.spaces.flatten(
                            self.env.observation_space, self.env.reset()[0]
                        )
                    ],
                    dtype=th.float32,
                ),
            )

        self.value_loss_weight = value_loss_weight
        self.policy_loss_weight = policy_loss_weight
        self.self_play_iterations = self_play_iterations
        self.scheduler = secheduler

    def iterate(self, iterations=10):

        for i in range(iterations):
            print(f"Iteration {i}")
            print("Self play...")
            self.self_play(i)
            print("Learning...")
            (
                value_losses,
                policy_losses,
                total_losses,
                value_sims,
            ) = self.learn()

            self.writer.add_scalar("Loss/Value_loss", np.mean(value_losses), i)
            self.writer.add_scalar("Loss/Policy_loss", np.mean(policy_losses), i)
            self.writer.add_scalar("Loss/Value_Simularities", np.mean(value_sims), i)

            # the regularization loss is the squared l2 norm of the weights
            regularization_loss = th.tensor(0.0, device=self.agent.model.device)

            for param in self.agent.model.parameters():
                regularization_loss += th.sum(th.square(param))
            self.writer.add_scalar(
                "Loss/Regularization_loss", regularization_loss, i
            )
            self.writer.add_scalar("Loss/Total_loss", np.mean(total_losses), i)

            # Log the size of the replay buffer
            self.writer.add_scalar("Replay_Buffer/Size", len(self.replay_buffer), i)

            if i % self.checkpoint_interval == 0:
                print(f"Saving model at iteration {i}")
                self.agent.model.save_model(f"{self.run_dir}/checkpoint.pth")

            if self.scheduler is not None:
                self.scheduler.step()

            # if the env is CliffWalking-v0, plot the output of the value and policy networks
            assert self.env.spec is not None
            if self.env.spec.id == "CliffWalking-v0":
                assert self.env.observation_space is not None
                show_model_in_tensorboard(self.env.observation_space, self.agent.model, self.writer, i)



        # save the final model
        self.agent.model.save_model(f"{self.run_dir}/final_model.pth")

        # close the writer
        self.writer.close()

    def self_play(self, global_step):
        """Play games in parallel and store the data in the replay buffer."""
        self.agent.model.eval()

        # Collect results from each process
        results = []

        # Number of processes - you can customize this
        tim = datetime.datetime.now()
        # Create a pool of processes
        if self.self_play_workers > 1:
            with multiprocessing.Pool() as pool:
                # Generate tasks for each episode
                tasks = [
                    (
                        self.agent,
                        self.env,
                        self.tree_evaluation_policy,
                        self.compute_budget,
                        self.max_episode_length,
                    )
                    for _ in range(self.self_play_iterations)
                ]
                # Run the tasks using map
                results = pool.map(run_episode_process, tasks)
        else:
            for _ in tqdm(range(self.self_play_iterations)):
                results.append(
                    run_episode(
                        self.agent,
                        self.env,
                        self.tree_evaluation_policy,
                        self.compute_budget,
                        self.max_episode_length,
                    )
                )
        tot_tim = datetime.datetime.now() - tim

        # Process the results
        rewards, time_steps, entropies = [], [], []
        for trajectory in results:
            self.replay_buffer.add(trajectory)

            episode_rewards = trajectory["rewards"].sum().item()
            rewards.append(episode_rewards)

            timesteps = trajectory["mask"].sum().item()
            time_steps.append(timesteps)
            assert isinstance(self.env.action_space, gym.spaces.Discrete)
            epsilon = 1e-8
            entropy = - th.sum(trajectory["policy_distributions"] * th.log(trajectory["policy_distributions"] + epsilon), dim=-1) * trajectory["mask"] / np.log(self.env.action_space.n)
            entropies.append(th.sum(entropy).item() / timesteps)



        # Calculate statistics
        mean_reward = np.mean(rewards)
        reward_variance = np.var(rewards, ddof=1)


        # Log the statistics
        self.writer.add_scalar("Self_Play/Mean_Reward", mean_reward, global_step)
        self.writer.add_scalar(
            "Self_Play/Reward_STD", np.sqrt(reward_variance), global_step
        )
        self.writer.add_scalar(
            "Self_Play/Mean_Timesteps", np.mean(time_steps), global_step
        )
        self.writer.add_scalar(
            "Self_Play/Timesteps_STD", np.sqrt(np.var(time_steps, ddof=1)), global_step
        )
        self.writer.add_scalar(
            "Self_Play/Runtime_per_Timestep",
            tot_tim.microseconds / np.sum(time_steps),
            global_step,
        )

        self.writer.add_scalar(
            "Self_Play/Mean_Entropy", np.mean(entropies), global_step
        )

        return mean_reward

    def learn(self):
        value_losses = []
        policy_losses = []
        # regularization_losses = []
        total_losses = []
        value_sims = []
        self.agent.model.train()
        for j in tqdm(range(self.training_epochs)):
            # sample a batch from the replay buffer
            trajectories = self.replay_buffer.sample()

            values, policies = self.agent.model.forward(trajectories["observations"])

            # compute the value targets via TD learning
            # the target should be the reward + the value of the next state
            # if the next state is terminal, the value of the next state is 0
            # the indexation is a bit tricky here, since we want to ignore the last state in the trajectory
            # the observation at index i is the state at time step i
            # the reward at index i is the reward obtained by taking action i
            # the terminal at index i is True if we stepped into a terminal state by taking action i
            # the policy at index i is the policy we used to take action i


            with th.no_grad():
                # this value estimates how on policy the trajectories are. If the trajectories are on policy, this value should be close to 1
                value_simularities = th.exp(-th.sum((trajectories["mask"] * (1 - trajectories["root_values"] / values)) ** 2, dim=-1) / trajectories["mask"].sum(dim=-1))

            # the target value is the reward we got + the value of the next state if it is not terminal
            targets = n_step_value_targets(trajectories["rewards"], values.detach(), trajectories["terminals"], self.discount_factor, self.n_steps_learning)
            # the td error is the difference between the target and the current value
            td = targets - values[:, :-1]
            mask = trajectories["mask"][:, :-1]
            # compute the value loss
            if self.value_sim_loss:
                value_loss = th.sum(th.sum((td * mask) ** 2, dim=-1) * value_simularities) / th.sum(mask)
            else:
                value_loss = th.sum((td * mask) ** 2) / th.sum(mask)



            # compute the policy loss
            epsilon = 1e-8
            step_loss = -th.sum(
                trajectories["policy_distributions"] * th.log(policies + epsilon),
                dim=-1,
            )

            # we do not want to consider terminal states
            policy_loss = th.sum(step_loss * trajectories["mask"]) / th.sum(trajectories["mask"])


            loss = (
                self.value_loss_weight * value_loss
                + self.policy_loss_weight * policy_loss
            )
            # backup
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            value_losses.append(value_loss.item())
            policy_losses.append(policy_loss.item())
            # regularization_losses.append(regularization_loss.item())
            total_losses.append(loss.item())
            value_sims.append(value_simularities.mean().item())

        return value_losses, policy_losses, total_losses, value_sims



def train_alphazero():
    # set seed
    np.random.seed(0)
    # env_id = "CartPole-v1"
    env_id = "CliffWalking-v0"
    # env_id = "FrozenLake-v1"
    env = gym.make(env_id)

    selection_policy = PUCT(c=1)
    tree_evaluation_policy = DefaultTreeEvaluator()

    iterations = 100
    discount_factor = .99

    model = AlphaZeroModel(env, hidden_dim=2**8, layers=1, pref_gpu=False)
    agent = AlphaZeroMCTS(selection_policy=selection_policy, model=model, discount_factor=discount_factor, expansion_policy=DefaultExpansionPolicy())
    regularization_weight = 1e-4
    optimizer = th.optim.Adam(model.parameters(), lr=1e-4, weight_decay=regularization_weight)
    scheduler = th.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99, verbose=True)
    workers = multiprocessing.cpu_count()

    self_play_games_per_iteration = workers
    replay_buffer_size = 20 * self_play_games_per_iteration
    sample_batch_size = replay_buffer_size // 5

    replay_buffer = TensorDictReplayBuffer(storage=LazyTensorStorage(replay_buffer_size), batch_size=sample_batch_size)
    controller = AlphaZeroController(
        env,
        agent,
        optimizer,
        replay_buffer = replay_buffer,
        max_episode_length=200,
        compute_budget=100,
        training_epochs=100,
        value_loss_weight=1.0,
        policy_loss_weight=1.0,
        self_play_iterations=self_play_games_per_iteration,
        tree_evaluation_policy=tree_evaluation_policy,
        self_play_workers=workers,
        secheduler=scheduler,
        discount_factor=discount_factor,
        n_steps_learning=1,
    )
    controller.iterate(iterations)

    env.close()


if __name__ == "__main__":
    train_alphazero()
