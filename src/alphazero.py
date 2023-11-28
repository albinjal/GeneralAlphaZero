from typing import Tuple
import gymnasium as gym
from tqdm import tqdm
from mcts import MCTS
from node import Node
from policies import DefaultExpansionPolicy, DefaultTreeEvaluator, Policy
import torch as th
from torchrl.data import ReplayBuffer, LazyTensorStorage

from runner import run_episode

# class AlphaZeroModel(th.nn.Module):
#     """
#     The point of this class is to make sure the model is compatible with MCTS:
#     The model should take in an observation and return a value and a policy. Check that
#     - Input is flattened with shape of the observation space
#     - The output is a tuple of (value, policy)
#     - the policy is a vector of proabilities of the same size as the action space
#     """

#     def __init__(self, core_network: th.nn.Module, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.core_network = core_network # the core network is the middle of the network

#     def forward(self, state: gym.spaces.) -> Tuple[float, np.ndarray]:
#         # flatten the state
#         state = state.is
class AlphaNode(Node):
    # also has a prior_policy
    pass



"""
- update so we expand all nodes at once?
- prior distribution on parent or float on child?

"""


class AlphaZeroMCTS(MCTS):
    model: th.nn.Module
    def __init__(self, model: th.nn.Module, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model


    def value_function(
        self,
        node: Node,
        env: gym.Env,
    ) -> float:
        observation = node.observation
        # flatten the observation
        assert observation is not None
        observation = observation.flatten()
        # run the model
        value, policy = self.model(observation)
        # store the policy
        node.policy = policy
        pass




class AlphaZeroController:
    """
    The Controller will be responsible for orchistrating the training of the model. With self play and training.
    """

    replay_buffer: ReplayBuffer
    training_epochs: int
    model: th.nn.Module

    def __init__(self,
                 env: gym.Env,
                 agent: AlphaZeroMCTS,
                 optimizer: th.optim.Optimizer,
                 storage = LazyTensorStorage(1000),
                 training_epochs = 10,
                 batch_size = 32,
                 tree_evaluation_policy = DefaultTreeEvaluator(),
                 compute_budget = 1000,
                 max_episode_length = 500,
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


    def iterate(self, iterations = 10):
        for i in range(iterations):
            print(f"Iteration {i}")
            print("Self play...")
            self.self_play()
            print("Learning...")
            self.learn()


    def self_play(self):
        """play a game and store the data in the replay buffer"""
        self.agent.model.eval()
        new_training_data = run_episode(self.agent, self.env, self.tree_evaluation_policy, compute_budget=self.compute_budget,
                                        max_steps=self.max_episode_length, verbose=True)
        self.replay_buffer.extend(new_training_data)


    def learn(self):
        self.agent.model.train()
        for i in tqdm(range(self.training_epochs)):
            # sample a batch from the replay buffer
            observation, policy_dist, v_target  = self.replay_buffer.sample(batch_size=self.batch_size)
            value, policy = self.agent.model(observation)

            # calculate the loss
            value_loss = th.nn.functional.mse_loss(value, v_target)
            policy_loss = th.nn.functional.kl_div(policy, policy_dist)
            loss = value_loss + policy_loss
            # backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            print(f"{i}. Loss: {loss.item():.2f}")
