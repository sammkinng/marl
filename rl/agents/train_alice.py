import numpy as np
import gymnasium as gym
from rl.qkd_rl import QKDEnv

class AliceSingleAgentEnv(gym.Env):
    """
    Single-agent wrapper for training Alice only.
    Bob and Eve use fixed/random strategies.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, sim_config):
        super().__init__()
        self.env = QKDEnv(sim_config)
        self.action_space = self.env.action_space["Alice"]
        self.observation_space = self.env.observation_space
        self.current_step=0
    
    def set_ppe(self, new_ppe):
        self.env.set_ppe(new_ppe)

    def set_reward_scale(self, new_scale):
        self.env.set_reward_scale(new_scale)

    def reset(self, *, seed=None, options=None):
        obs, _ = self.env.reset(seed=seed, options=options)
        return obs, {}

    def step(self, action):
        # start = time.time()
        # if self.current_step % 100 == 0:
        #     print(f"--- Step {self.current_step} ---")
        # Bob: fixed strategy
        bob_action = np.array([0.5, 1.0], dtype=np.float32)

        # Eve: random attack
        eve_action = self.env.action_space["Eve"].sample() * 0.3

        actions = {"Alice": action, "Bob": bob_action, "Eve": eve_action}
        obs, rewards, terminated, truncated, info = self.env.step(actions)
        reward = rewards["Alice"]
        done = terminated or truncated
        # elapsed = time.time() - start
        # if self.current_step % 10 == 0:  # print every 10 steps
        #     print(f"Step {self.current_step}: took {elapsed:.4f}s")
        # self.current_step += 1
        return obs, reward, done, False, info

    


    def render(self):
        self.env.render()

    def close(self):
        self.env.close()


