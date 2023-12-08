import copy
import datetime
from typing import List, Tuple
import gymnasium as gym
import numpy as np
from tqdm import tqdm
from environment import obs_to_tensor
from mcts import MCTS
from node import Node
from policies import PUCT, UCT, DefaultExpansionPolicy, DefaultTreeEvaluator, Policy
import torch as th
from torchrl.data import ReplayBuffer, ListStorage, LazyTensorStorage, Storage, TensorDictReplayBuffer
from runner import run_episode
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np




class AlphaZeroModel(th.nn.Module):
    """
    The point of this class is to make sure the model is compatible with MCTS:
    The model should take in an observation and return a value and a policy. Check that
    - Input is flattened with shape of the observation space
    - The output is a tuple of (value, policy)
    - the policy is a vector of proabilities of the same size as the action space
    """
    value_head: th.nn.Module
    policy_head: th.nn.Module
    device: th.device


    def __init__(
        self,
        env: gym.Env,
        hidden_dim: int,
        layers: int,
        pref_gpu=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # check if cuda is available
        if not pref_gpu:
            self.device = th.device("cpu")
        elif th.cuda.is_available():
            self.device = th.device("cuda")
        elif th.backends.mps.is_available():
            self.device = th.device("mps")

        self.env = env
        self.state_dim = gym.spaces.flatdim(env.observation_space)
        self.action_dim = gym.spaces.flatdim(env.action_space)

        self.layers = th.nn.ModuleList()
        self.layers.append(th.nn.Linear(self.state_dim, hidden_dim))
        for _ in range(layers):
            self.layers.append(th.nn.Linear(hidden_dim, hidden_dim))

        # the value head should be two layers
        self.value_head = th.nn.Sequential(
            th.nn.Linear(hidden_dim, hidden_dim),
            th.nn.ReLU(),
            th.nn.Linear(hidden_dim, 1),
        )

        # the policy head should be two layers
        self.policy_head = th.nn.Sequential(
            th.nn.Linear(hidden_dim, hidden_dim),
            th.nn.ReLU(),
            th.nn.Linear(hidden_dim, self.action_dim),
        )
        self.to(self.device)

    def forward(self, x: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        # run the layers
        for layer in self.layers:
            x = th.nn.functional.relu(layer(x))
        # run the heads
        value = self.value_head(x)
        policy = self.policy_head(x)
        # apply softmax to the policy
        policy = th.nn.functional.softmax(policy, dim=-1)
        return value.squeeze(-1), policy

    # def forward_state(self, state: gym.spaces.Space) -> Tuple[th.Tensor, th.Tensor]:
    #     # gymnasium.spaces.utils.flatten
    #     flat_state = gym.spaces.flatten(self.env.observation_space, state)

    #     # convert to tensor
    #     flat_state = th.from_numpy(flat_state).float()

    #     # run the model
    #     value, policy = self.forward(flat_state)

    #     return value, policy


"""
- update so we expand all nodes at once?
- prior distribution on parent or float on child?

"""


class AlphaZeroMCTS(MCTS):
    model: AlphaZeroModel

    def __init__(self, model: AlphaZeroModel, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model

    @th.no_grad()
    def value_function(
        self,
        node: Node,
        env: gym.Env,
    ):
        observation = node.observation
        # flatten the observation
        assert observation is not None
        # run the model
        # convert observation from int to tensor float 1x1 tensor
        tensor_obs = obs_to_tensor(env.observation_space, observation, device=self.model.device)
        value, policy = self.model.forward(tensor_obs)
        # store the policy
        node.prior_policy = policy
        # return float 32 value
        return value.item()


    def handle_all(self, node: Node, env: gym.Env):
        """
        should do the same as
    def handle_single(
        self,
        node: Node[ObservationType],
        env: gym.Env[ObservationType, np.int64],
        action: np.int64,
    ):
        eval_node = self.expand(node, env, action)
        # evaluate the node
        value = self.value_function(eval_node, env)
        # backupagate the value
        eval_node.value_evaluation = value
        eval_node.backup(value)

    def handle_all(
        self, node: Node[ObservationType], env: gym.Env[ObservationType, np.int64]
    ):
        for action in range(node.action_space.n):
            self.handle_single(node, copy.deepcopy(env), np.int64(action))

        """
        observations = []
        all_actions = np.arange(node.action_space.n)
        for i, action in enumerate(all_actions):
            if i == len(all_actions) - 1:
                # we can mess up the env on the last action
                new_node = self.expand(node, env, action)
            else:
                new_env = copy.deepcopy(env)
                new_node = self.expand(node, new_env, action)
            observations.append(gym.spaces.flatten(env.observation_space, new_node.observation))


        tensor_obs = th.tensor(observations, dtype=th.float32, device=self.model.device) # actions x obs_dim tensor
        values, policies = self.model.forward(tensor_obs)

        value_to_backup = np.float32(0.0)
        for action, value, policy in zip(all_actions, values, policies):
            new_node = node.children[action]
            new_node.prior_policy = policy
            new_node.visits = 1
            new_node.value_evaluation = np.float32(value.item())
            new_node.subtree_sum = new_node.reward + new_node.value_evaluation
            value_to_backup += new_node.subtree_sum

        # backup
        # the value to backup from the parent should be the sum of the value and the reward for all children
        node.backup(value_to_backup, len(all_actions))


    # def backup_all_children(self, parent: Node, values: th.Tensor):



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
        storage: Storage =LazyTensorStorage(50),
        training_epochs=10,
        batch_size=32,
        tree_evaluation_policy=DefaultTreeEvaluator(),
        compute_budget=100,
        max_episode_length=500,
        tensorboard_dir="./tensorboard_logs",
        runs_dir="./runs",
        checkpoint_interval=10,
        value_loss_weight=1.0,
        policy_loss_weight=1.0,
        regularization_weight=1e-4,
        self_play_iterations=10,
    ) -> None:
        self.replay_buffer = TensorDictReplayBuffer(storage=storage)
        self.training_epochs = training_epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.agent = agent
        self.env = env
        self.tree_evaluation_policy = tree_evaluation_policy
        self.compute_budget = compute_budget
        self.max_episode_length = max_episode_length
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_name = f"{env_id}_{current_time}"
        log_dir = f"{tensorboard_dir}/{self.run_name}"
        self.writer = SummaryWriter(log_dir=log_dir)
        self.run_dir = f"{runs_dir}/{self.run_name}"
        # create run dir if it does not exist
        os.makedirs(self.run_dir, exist_ok=True)

        self.checkpoint_interval = checkpoint_interval
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
        self.regularization_weight = regularization_weight
        self.self_play_iterations = self_play_iterations

    def iterate(self, iterations=10):
        for i in tqdm(range(iterations)):
            print(f"Iteration {i}")
            print("Self play...")
            self.self_play(i)
            print("Learning...")
            (
                value_losses,
                policy_losses,
                regularization_losses,
                total_losses,
            ) = self.learn()

            self.writer.add_scalar("Loss/Value_loss", np.mean(value_losses), i)
            self.writer.add_scalar("Loss/Policy_loss", np.mean(policy_losses), i)
            self.writer.add_scalar(
                "Loss/Regularization_loss", np.mean(regularization_losses), i
            )
            self.writer.add_scalar("Loss/Total_loss", np.mean(total_losses), i)

            # Log the size of the replay buffer
            self.writer.add_scalar("Replay_Buffer/Size", len(self.replay_buffer), i)

            # save the model every 10 iterations
            if i % self.checkpoint_interval == 0:
                th.save(self.agent.model.state_dict(), f"{self.run_dir}/checkpoint.pth")

        # save the final model
        th.save(self.agent.model.state_dict(), f"{self.run_dir}/final_model.pth")

        # close the writer
        self.writer.close()

    def self_play(self, global_step):
        """play a game and store the data in the replay buffer"""
        self.agent.model.eval()
        rewards = []
        time_steps = []
        runtime = []
        time_step = []
        for _ in tqdm(range(self.self_play_iterations)):
            time_start = datetime.datetime.now()
            new_trajectory = run_episode(
                    self.agent,
                    self.env,
                    self.tree_evaluation_policy,
                    compute_budget=self.compute_budget,
                    max_steps=self.max_episode_length,
                )
            time_end = datetime.datetime.now()
            self.replay_buffer.add(new_trajectory)
            episode_rewards = new_trajectory['rewards']
            rewards.append(episode_rewards.sum())

            timesteps = new_trajectory['mask'].sum()
            time_steps.append(timesteps)

            runtime.append((time_end - time_start).total_seconds())
            time_step.append((time_end - time_start).total_seconds() / timesteps)



        mean_reward = np.mean(rewards)
        reward_variance = np.var(rewards, ddof=1)


        # Log the mean and variance of rewards
        self.writer.add_scalar("Self_Play/Mean_Reward", mean_reward, global_step)
        self.writer.add_scalar("Self_Play/Reward_STD", np.sqrt(reward_variance), global_step)

        self.writer.add_scalar("Self_Play/Mean_Timesteps", np.mean(time_steps), global_step)
        self.writer.add_scalar("Self_Play/Timesteps_STD", np.sqrt(np.var(time_steps, ddof=1)), global_step)

        self.writer.add_scalar("Self_Play/Mean_Runtime", np.mean(runtime), global_step)
        self.writer.add_scalar("Self_Play/Runtime_STD", np.sqrt(np.var(runtime, ddof=1)), global_step)

        self.writer.add_scalar("Self_Play/Mean_Runtime_per_Timestep", np.mean(time_step), global_step)
        self.writer.add_scalar("Self_Play/Runtime_per_Timestep_STD", np.sqrt(np.var(time_step, ddof=1)), global_step)

        return mean_reward





    def learn(self):
        value_losses = []
        policy_losses = []
        regularization_losses = []
        total_losses = []
        self.agent.model.train()
        for j in tqdm(range(self.training_epochs)):
            # sample a batch from the replay buffer
            trajectories = self.replay_buffer.sample(
                batch_size=min(self.batch_size, len(self.replay_buffer))
            )


            values, policies = self.agent.model.forward(trajectories["observations"])

            # compute the value targets via TD learning
            # the target should be the reward + the value of the next state
            # if the next state is terminal, the value of the next state is 0
            t_values = ~trajectories["terminals"] * values.squeeze(-1)
            targets = trajectories["rewards"][:, :-1] + t_values[:, 1:]
            td = targets.detach() - values[:, :-1]
            mask = trajectories["mask"][:, :-1]
            # compute the value loss
            value_loss = th.sum((td * mask) ** 2) / th.sum(mask)


            # compute the policy loss
            epsilon = 1e-8
            step_loss = - th.sum(trajectories["policy_distributions"] * th.log(policies + epsilon), dim=-1)
            policy_loss = th.sum(step_loss * trajectories["mask"]) / th.sum(trajectories["mask"])


            # the regularization loss is the squared l2 norm of the weights
            regularization_loss = th.tensor(0.0, device=self.agent.model.device)
            if True:  # self.regularization_weight > 0:
                # just fun to keep track of the regularization loss
                for param in self.agent.model.parameters():
                    regularization_loss += th.sum(th.square(param))
            loss = (
                self.value_loss_weight * value_loss
                + self.policy_loss_weight * policy_loss
                + self.regularization_weight * regularization_loss
            )
            # backup
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print(f"{j}. Loss: {loss.item():.2f}")
            value_losses.append(value_loss.item())
            policy_losses.append(policy_loss.item())
            regularization_losses.append(regularization_loss.item())
            total_losses.append(loss.item())

        return value_losses, policy_losses, regularization_losses, total_losses


import cProfile
import pstats

def profile():
    cProfile.runctx("controller.iterate(10)", globals(), locals(), "Profile.prof")

    s = pstats.Stats("Profile.prof")
    s.strip_dirs().sort_stats("time").print_stats()

if __name__ == "__main__":
    actType = np.int64
    env_id = "CartPole-v1"
    # env_id = "CliffWalking-v0"
    # env_id = "FrozenLake-v1"
    # env_id = "Taxi-v3"
    env = gym.make(env_id)

    selection_policy = PUCT(c=1)
    tree_evaluation_policy = DefaultTreeEvaluator()

    model = AlphaZeroModel(env, hidden_dim=128, layers=2, pref_gpu=False)
    agent = AlphaZeroMCTS(selection_policy=selection_policy, model=model)
    optimizer = th.optim.Adam(model.parameters(), lr=1e-3)
    controller = AlphaZeroController(
        env,
        agent,
        optimizer,
        max_episode_length=500,
        batch_size=20,
        compute_budget=50,
        training_epochs=10,
        regularization_weight=0.0,
        value_loss_weight=1.0,
        policy_loss_weight=1.0,
        self_play_iterations=5,
    )
    controller.iterate(100)

    env.close()
