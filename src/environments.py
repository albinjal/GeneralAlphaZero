import gymnasium as gym
from torchrl.envs.libs.gym import GymEnv, GymWrapper
import torchrl as trl
from torchrl.data.tensor_specs import DiscreteBox

if __name__ == '__main__':
    env = GymEnv('CliffWalking-v0')

    # test all different functionalities of the env

    print("reset")
    res = env.reset()
    print(res)
    print(res['observation'])

    print("action spec")
    print(env.action_spec)

    action = env.action_spec.rand()
    print("action")
    print(action)

    print("action dimensions")
    print(env.action_spec.shape)

    print("observation spec")
    print(env.observation_spec)

    print("action spec")
    space = env.action_spec.space
    # type should be torchrl.data.tensor_specs.DiscreteBox
    assert isinstance(space, DiscreteBox)

    print("I need to be able to crete a tensor of all possible actions")
    print(env.action_spec.
