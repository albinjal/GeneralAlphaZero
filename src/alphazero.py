from typing import Tuple
import gymnasium as gym
from mcts import MCTS
from node import Node
from policies import DefaultExpansionPolicy, Policy, SelectionPolicy
import torch as th
from torchrl.data import ReplayBuffer, LazyTensorStorage

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

    def __init__(self, env: gym.Env, agent: AlphaZeroMCTS, storage = LazyTensorStorage(1000)) -> None:
        self.replay_buffer = ReplayBuffer(storage=storage)


    def self_play(self):
        pass


    def train(self):
        pass
