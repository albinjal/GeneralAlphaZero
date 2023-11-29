import datetime
from typing import Tuple
import gymnasium as gym
import numpy as np
from tqdm import tqdm
from mcts import MCTS
from node import Node
from policies import UCB, DefaultExpansionPolicy, DefaultTreeEvaluator, Policy
import torch as th
from torchrl.data import ReplayBuffer, ListStorage
from runner import run_episode
from torch.utils.tensorboard import SummaryWriter
import os

class AlphaZeroModel(th.nn.Module):
    """
    The point of this class is to make sure the model is compatible with MCTS:
    The model should take in an observation and return a value and a policy. Check that
    - Input is flattened with shape of the observation space
    - The output is a tuple of (value, policy)
    - the policy is a vector of proabilities of the same size as the action space
    """

    def __init__(self, env: gym.Env, hidden_dim: int, layers: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env = env
        self.state_dim = gym.spaces.flatdim(env.observation_space)
        self.action_dim = gym.spaces.flatdim(env.action_space)

        self.layers = th.nn.ModuleList()
        self.layers.append(th.nn.Linear(self.state_dim, hidden_dim))
        for _ in range(layers):
            self.layers.append(th.nn.Linear(hidden_dim, hidden_dim))

        self.value_head = th.nn.Linear(hidden_dim, 1)
        self.policy_head = th.nn.Linear(hidden_dim, self.action_dim)

    def forward(self, x: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:

        # run the layers
        for layer in self.layers:
            x = th.nn.functional.relu(layer(x))
        # run the heads
        value = self.value_head(x)
        policy = self.policy_head(x)
        # apply softmax to the policy
        policy = th.nn.functional.softmax(policy, dim=-1)
        return value, policy

    # def forward_state(self, state: gym.spaces.Space) -> Tuple[th.Tensor, th.Tensor]:
    #     # gymnasium.spaces.utils.flatten
    #     flat_state = gym.spaces.flatten(self.env.observation_space, state)

    #     # convert to tensor
    #     flat_state = th.from_numpy(flat_state).float()

    #     # run the model
    #     value, policy = self.forward(flat_state)

    #     return value, policy




class AlphaNode(Node):
    # also has a prior_policy
    prior_policy: np.ndarray



"""
- update so we expand all nodes at once?
- prior distribution on parent or float on child?

"""


class AlphaZeroMCTS(MCTS):
    model: th.nn.Module
    def __init__(self, model: th.nn.Module, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model


    @th.no_grad()
    def value_function(
        self,
        node: AlphaNode,
        env: gym.Env,
    ) -> float:

        observation = node.observation
        # flatten the observation
        assert observation is not None
        # run the model
        # convert observation from int to tensor float 1x1 tensor
        tensor_obs = th.tensor(gym.spaces.flatten(env.observation_space, observation), dtype=th.float32)
        value, policy = self.model.forward(tensor_obs)
        # store the policy
        node.prior_policy = policy
        return value




class AlphaZeroController:
    """
    The Controller will be responsible for orchistrating the training of the model. With self play and training.
    """

    replay_buffer: ReplayBuffer
    training_epochs: int
    model: AlphaZeroModel

    def __init__(self,
                 env: gym.Env,
                 agent: AlphaZeroMCTS,
                 optimizer: th.optim.Optimizer,
                 storage = ListStorage(1000),
                 training_epochs = 10,
                 batch_size = 32,
                 tree_evaluation_policy = DefaultTreeEvaluator(),
                 compute_budget = 100,
                 max_episode_length = 500,
                 tensorboard_dir ='./tensorboard_logs',
                 runs_dir = './runs',
                 checkpoint_interval = 10,
                 ) -> None:
        self.replay_buffer = ReplayBuffer(storage=storage)
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
        log_dir = f'{tensorboard_dir}/{self.run_name}'
        self.writer = SummaryWriter(log_dir=log_dir)
        self.run_dir = f"{runs_dir}/{self.run_name}"
        # create run dir if it does not exist
        os.makedirs(self.run_dir, exist_ok=True)

        self.checkpoint_interval = checkpoint_interval
        # Log the model
        self.writer.add_graph(self.agent.model, th.tensor([gym.spaces.flatten(self.env.observation_space, self.env.reset()[0])], dtype=th.float32))



    def iterate(self, iterations = 10):
        for i in tqdm(range(iterations)):
            print(f"Iteration {i}")
            print("Self play...")
            total_reward = self.self_play()
            # Log the total reward
            self.writer.add_scalar('Self_Play/Total_Reward', total_reward, i)
            print("Learning...")
            self.learn(it=i)


            # Log the size of the replay buffer
            self.writer.add_scalar('Replay_Buffer/Size', len(self.replay_buffer), i)

            # save the model every 10 iterations
            if i % self.checkpoint_interval == 0:
                th.save(self.agent.model.state_dict(), f"{self.run_dir}/checkpoint.pth")

        # save the final model
        th.save(self.agent.model.state_dict(), f"{self.run_dir}/final_model.pth")

        # close the writer
        self.writer.close()


    def self_play(self):
        """play a game and store the data in the replay buffer"""
        self.agent.model.eval()
        new_training_data, total_reward = run_episode(self.agent, self.env, self.tree_evaluation_policy, compute_budget=self.compute_budget,
                                        max_steps=self.max_episode_length, verbose=True)
        self.replay_buffer.extend(new_training_data)

        return total_reward


    def learn(self, it: int):
        self.agent.model.train()
        for j in tqdm(range(self.training_epochs)):
            # sample a batch from the replay buffer
            observations, policy_dists, v_targets  = self.replay_buffer.sample(batch_size=self.batch_size)
            tensor_obs = th.tensor([gym.spaces.flatten(env.observation_space, observation) for observation in observations], dtype=th.float32)
            value, policy = self.agent.model.forward(tensor_obs)

            # calculate the loss
            value_loss = th.nn.functional.mse_loss(value, v_targets)
            # note that kl div expect log probabilities for input
            policy_loss = th.nn.functional.kl_div(th.log(policy), policy_dists, reduction='batchmean', log_target=False)
            loss = value_loss + policy_loss
            # backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            i = it * self.training_epochs + j
            print(f"{i}. Loss: {loss.item():.2f}")
            self.writer.add_scalar('Loss/Value_loss', value_loss.item(), i)
            self.writer.add_scalar('Loss/Policy_loss', policy_loss.item(), i)
            self.writer.add_scalar('Loss/Total_loss', loss.item(), i)



if __name__ == "__main__":
    actType = np.int64
    env_id = "CliffWalking-v0"
    # env_id = "FrozenLake-v1"
    # env_id = "Taxi-v3"
    env = gym.make(env_id, render_mode="ansi")

    selection_policy = UCB(c=3)
    tree_evaluation_policy = DefaultTreeEvaluator()


    model = AlphaZeroModel(env, hidden_dim=32, layers=2)
    agent = AlphaZeroMCTS(selection_policy=selection_policy, model=model)
    optimizer = th.optim.Adam(model.parameters(), lr=1e-3)
    controller = AlphaZeroController(env, agent, optimizer, max_episode_length=200, batch_size=200, storage = ListStorage(1000))
    controller.iterate(100)


    env.close()
