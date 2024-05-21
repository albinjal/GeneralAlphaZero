from typing import Tuple
import torch as th
import gymnasium as gym

from environments.observation_embeddings import DefaultEmbedding, ObservationEmbedding


activation_function_dict = {
    "relu": th.nn.ReLU,
    "sigmoid": th.nn.Sigmoid,
    "tanh": th.nn.Tanh,
    "leakyrelu": th.nn.LeakyReLU,
}
norm_dict = {
    "batch": th.nn.BatchNorm1d,
    "layer": th.nn.LayerNorm,
    "none": None,
}


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
    observation_embedding: ObservationEmbedding

    def __init__(
        self,
        env: gym.Env,
        hidden_dim: int,
        nlayers: int,
        *args,
        observation_embedding: ObservationEmbedding | None = None,
        pref_gpu=False,
        activation_fn=th.nn.ReLU,
        norm_layer=None,
        value_bias: None | float = None,
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

        self.to(self.device)

        self.env = env
        self.hidden_dim = hidden_dim
        self.observation_embedding = (
            observation_embedding
            if observation_embedding is not None
            else DefaultEmbedding(env.observation_space)
        )
        self.state_dim = self.observation_embedding.obs_dim()
        self.action_dim = gym.spaces.flatdim(env.action_space)
        self.nlayers = nlayers
        self.activation_fn = activation_fn
        self.norm_layer = norm_layer
        self.core = None
        self.create_model()
        if value_bias is not None:
            th.nn.init.constant_(self.value_head[-1].bias, value_bias)
        self.print_stats()

    def create_model(self):
        """
        Creates a model for the AlphaZero algorithm.

        This method should be implemented by subclasses to define the specific model architecture.

        Raises:
            NotImplementedError: This method should be implemented by subclasses.
        """
        raise NotImplementedError

    def print_stats(self):
        """
        Prints the model parameters and the total number of trainable parameters.

        Returns:
            None
        """
        # print the model parameters
        print(f"Model initialized on {self.device} with the following parameters:")
        total_params = 0
        for name, param in self.named_parameters():
            print(name, param.numel())
            total_params += param.numel()
        print(f"Total number of trainable parameters: {total_params}")

    def forward(self, x: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing the value and policy tensors.
        """
        # run the layers
        if self.core is not None:
            x = self.core(x)

        # run the heads
        value = self.value_head(x)
        policy = self.policy_head(x)
        return value.squeeze(-1), policy

    def single_observation_forward(self, observation) -> Tuple[float, th.Tensor]:
        """
        Forward pass for a single observation.

        Args:
            observation: The input observation.

        Returns:
            A tuple containing the value and policy tensors.
        """
        tensor_obs = self.observation_embedding.obs_to_tensor(
            observation,
            device=self.device,
            dtype=th.float32,
        ).unsqueeze(0)
        value, policy = self.forward(tensor_obs)
        return value.item(), policy.squeeze(0)

    def save_model(self, filename: str):
        """
        Save the model to a file.

        Args:
            filename (str): The name of the file to save the model to.

        Returns:
            None
        """
        model_info = {
            "state_dict": self.state_dict(),
            "input_dimensions": self.state_dim,
            "output_dimensions": self.action_dim,
            "hidden_dim": self.hidden_dim,
            "layers": self.nlayers,
            # Add other relevant model configuration here
        }
        th.save(model_info, filename)

    @classmethod
    def load_model(
        cls, filename: str, env: gym.Env, pref_gpu=False, default_hidden_dim=128
    ):
        """
        Load a saved model from a file and create a new instance of the model with the saved specifications.

        Args:
            cls (class): The class of the model.
            filename (str): The path to the saved model file.
            env (gym.Env): The environment associated with the model.
            pref_gpu (bool, optional): Whether to prefer GPU for model computations. Defaults to False.
            default_hidden_dim (int, optional): The default hidden dimension if not found in the saved model information. Defaults to 128.

        Returns:
            model: The newly created model with the loaded state dict.
        """
        # Load the saved model information
        model_info = th.load(filename)

        # Get hidden_dim, use a default if not found
        hidden_dim = model_info.get("hidden_dim", default_hidden_dim)

        # Create a new instance of the model with the saved specifications
        model = cls(
            env=env,
            hidden_dim=hidden_dim,
            nlayers=model_info["layers"],
            pref_gpu=pref_gpu,
        )

        # Load the state dict into the newly created model
        model.load_state_dict(model_info["state_dict"])

        return model


def create_layers(state_dim, nlayers, hidden_dim, activation_fn, norm_layer):
    """
    Create a list of layers for a neural network model.

    Args:
        state_dim (int): The dimension of the input state.
        nlayers (int): The number of hidden layers.
        hidden_dim (int): The dimension of the hidden layers.
        activation_fn (function): The activation function to use for the hidden layers.
        norm_layer (function or None): The normalization layer to use for the hidden layers, or None if no normalization is desired.

    Returns:
        list: A list of layers for the neural network model.

    """
    layers = []
    layers.append(th.nn.Linear(state_dim, hidden_dim))
    layers.append(th.nn.ReLU())

    for _ in range(nlayers):
        layers.append(th.nn.Linear(hidden_dim, hidden_dim))
        if norm_layer is not None:
            layers.append(norm_layer(hidden_dim))
        layers.append(activation_fn())

    return layers


class UnifiedModel(AlphaZeroModel):
    """
    A class representing the unified model used in AlphaZero.

    This model consists of a core network, a value head, and a policy head.
    The core network is responsible for processing the input state.
    The value head predicts the value of the state, while the policy head predicts the probability distribution over actions.

    Attributes:
        state_dim (int): The dimension of the input state.
        nlayers (int): The number of layers in the core network.
        hidden_dim (int): The dimension of the hidden layers in the core network.
        activation_fn (torch.nn.Module): The activation function used in the core network.
        norm_layer (torch.nn.Module): The normalization layer used in the core network.
        action_dim (int): The dimension of the action space.

    Methods:
        create_model(): Creates the unified model by initializing the core network, value head, and policy head.
    """

    def create_model(self):
        self.core = th.nn.Sequential(
            *create_layers(
                self.state_dim,
                self.nlayers,
                self.hidden_dim,
                self.activation_fn,
                self.norm_layer,
            )
        )

        # the value head should be two layers
        self.value_head = th.nn.Sequential(
            th.nn.Linear(self.hidden_dim, 1),
        )

        # the policy head should be two layers
        self.policy_head = th.nn.Sequential(
            th.nn.Linear(self.hidden_dim, self.action_dim),
            th.nn.Softmax(dim=-1),
        )


class SeperatedModel(AlphaZeroModel):
    """
    A class representing a model with separate value and policy heads.

    Inherits from the AlphaZeroModel class.
    """

    def create_model(self):
        """
        Creates the model architecture with separate value and policy heads.
        """
        self.core = None

        # the value head should be two layers
        self.value_head = th.nn.Sequential(
            *create_layers(
                self.state_dim,
                self.nlayers,
                self.hidden_dim,
                self.activation_fn,
                self.norm_layer,
            ),
            th.nn.Linear(self.hidden_dim, 1),
        )

        # the policy head should be two layers
        self.policy_head = th.nn.Sequential(
            *create_layers(
                self.state_dim,
                self.nlayers,
                self.hidden_dim,
                self.activation_fn,
                self.norm_layer,
            ),
            th.nn.Linear(self.hidden_dim, self.action_dim),
            th.nn.Softmax(dim=-1),
        )


models_dict = {
    "unified": UnifiedModel,
    "seperated": SeperatedModel,
}
