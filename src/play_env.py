import gymnasium as gym
from gymnasium.utils.play import PlayPlot, play


def callback(obs_t, obs_tp1, action, rew, terminated, truncated, info):
       print(f"O: {obs_t}, A: {action}, R: {rew}, T: {terminated}")

if __name__ == "__main__":
    play(gym.make("CliffWalking-v0", render_mode="rgb_array"), callback=callback,
         keys_to_action={ # Define keys to actions
            (ord('s'),): 2,
            (ord('d'),): 1,
            (ord('w'),): 0,
            (ord('a'),): 3,
         }, fps=10, seed=0, noop=2
         )
