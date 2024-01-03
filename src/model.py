from typing import Tuple
import torch as th
import gymnasium as gym

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
        self.hidden_dim = hidden_dim
        self.state_dim = gym.spaces.flatdim(env.observation_space)
        self.action_dim = gym.spaces.flatdim(env.action_space)

        self.layers = th.nn.ModuleList()
        self.layers.append(th.nn.Linear(self.state_dim, hidden_dim))
        self.layers.append(th.nn.ReLU())

        for _ in range(layers):
            self.layers.append(th.nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(th.nn.ReLU())

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
        self.nlayers = layers

        # print the model parameters
        print(f"Model initialized on {self.device} with the following parameters:")
        total_params = 0
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            print(name, param.numel())
            total_params += param.numel()
        print(f"Total number of trainable parameters: {total_params}")


    def forward(self, x: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        # run the layers
        for layer in self.layers:
            x = layer(x)
        # run the heads
        value = self.value_head(x)
        policy = self.policy_head(x)
        # apply softmax to the policy
        policy = th.nn.functional.softmax(policy, dim=-1)
        return value.squeeze(-1), policy


    def save_model(self, filename: str):
        model_info = {
            "state_dict": self.state_dict(),
            "input_dimensions": self.state_dim,
            "output_dimensions": self.action_dim,
            "hidden_dim": self.hidden_dim,
            "layers": self.nlayers,
            # Add other relevant model configuration here
        }
        th.save(model_info, filename)


    @staticmethod
    def load_model(filename: str, env: gym.Env, pref_gpu=False, default_hidden_dim=128):
        # Load the saved model information
        model_info = th.load(filename)

        # Get hidden_dim, use a default if not found
        hidden_dim = model_info.get("hidden_dim", default_hidden_dim)

        # Create a new instance of the model with the saved specifications
        model = AlphaZeroModel(
            env=env,
            hidden_dim=hidden_dim,
            layers=model_info["layers"],  # Subtracting 1 because the first layer is added by default
            pref_gpu=pref_gpu
        )

        # Load the state dict into the newly created model
        model.load_state_dict(model_info["state_dict"])

        return model
