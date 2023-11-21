import gymnasium as gym

import time
from gymenv import GymEnv


if __name__ == "__main__":
    # test the environment
    env: GymEnv = gym.make("CliffWalking-v0", render_mode="human")
    env.reset()
    while True:
        env.render()
        action = env.action_space.sample()
        state, reward, done, info, _ = env.step(action)
        print(f"action: {action}, state: {state}, reward: {reward}, done: {done}, info: {info}")
        # sleep for .5
        time.sleep(.1)
        if done:
            break
    env.close()
