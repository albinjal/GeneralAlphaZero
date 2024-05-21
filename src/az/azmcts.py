import torch as th

from core.mcts import MCTS
from az.model import AlphaZeroModel
from core.node import Node


class AlphaZeroMCTS(MCTS):
    """
    Implementation of the AlphaZero Monte Carlo Tree Search (MCTS) algorithm.

    Attributes:
        model (AlphaZeroModel): The AlphaZero model used for value and policy prediction.
        dir_epsilon (float): The epsilon value for adding Dirichlet noise to the prior policy.
        dir_alpha (float): The alpha value for the Dirichlet distribution.

    Args:
        model (AlphaZeroModel): The AlphaZero model used for value and policy prediction.
        *args: Variable length argument list.
        dir_epsilon (float, optional): The epsilon value for adding Dirichlet noise to the prior policy. Defaults to 0.0.
        dir_alpha (float, optional): The alpha value for the Dirichlet distribution. Defaults to 0.3.
        **kwargs: Arbitrary keyword arguments.
    """

    model: AlphaZeroModel
    dir_epsilon: float
    dir_alpha: float

    def __init__(
        self, model: AlphaZeroModel, *args, dir_epsilon=0.0, dir_alpha=0.3, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.model = model
        self.dir_epsilon = dir_epsilon
        self.dir_alpha = dir_alpha

    @th.no_grad()
    def value_function(self, node: Node) -> float:
        """
        Computes the value function for a given node.

        Args:
            node (Node): The node for which to compute the value function.

        Returns:
            float: The computed value function.
        """
        if node.is_terminal():
            return 0.0
        observation = node.observation
        # flatten the observation
        assert observation is not None
        # run the model
        # convert observation from int to tensor float 1x1 tensor
        assert node.env is not None

        value, policy = self.model.single_observation_forward(observation)
        # if root and dir_epsilon > 0.0, add dirichlet noise to the prior policy
        if node.parent is None and self.dir_epsilon > 0.0:
            if self.dir_alpha is not None:
                noise = th.distributions.dirichlet.Dirichlet(
                    th.ones_like(policy) * self.dir_alpha
                ).sample()
            else:
                # uniform distribution
                noise = th.ones_like(policy) / policy.numel()
            node.prior_policy = (
                1 - self.dir_epsilon
            ) * policy + self.dir_epsilon * noise
        else:
            node.prior_policy = policy

        return value

    # @th.no_grad()
    # def value_funciton_multiple(self, nodes: List[Node]):
    #     pass

    # @th.no_grad()
    # def handle_all(self, node: Node):
    #     all_actions = np.arange(node.action_space.n)
    #     assert node.env is not None
    #     obs_space = node.env.observation_space

    #     terminal_included = False
    #     for action in all_actions:
    #         new_node = self.expand(node, action)
    #         if new_node.is_terminal():
    #             terminal_included = True

    #     children = node.children

    #     # if there is a terminal node along the new nodes, simply do the regular backup
    #     if terminal_included:
    #         for child in children.values():
    #             value = self.value_function(child)
    #             # backupagate the value
    #             child.value_evaluation = value
    #             child.backup(value)

    #         return

    #     tensor_obs = th.stack([obs_to_tensor(obs_space, children[action].observation, device=self.model.device, dtype=th.float32)
    #                            for action in all_actions])  # actions x obs_dim tensor
    #     values, policies = self.model.forward(tensor_obs)

    #     value_to_backup = np.float32(0.0)
    #     for action, value, policy in zip(all_actions, values, policies):
    #         new_node = node.children[action]
    #         new_node.prior_policy = policy
    #         new_node.visits = 1
    #         new_node.value_evaluation = np.float32(value.item())
    #         new_node.subtree_sum = new_node.reward + new_node.value_evaluation
    #         value_to_backup += new_node.subtree_sum

    #     # backup
    #     # the value to backup from the parent should be the sum of the value and the reward for all children
    #     node.backup(value_to_backup, len(all_actions))

    # def backup_all_children(self, parent: Node, values: th.Tensor):
