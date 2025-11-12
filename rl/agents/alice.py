import numpy as np
import gymnasium as gym

from rl.qkd_rl import QKDEnv


class AliceSingleAgentEnv(gym.Env):
    """Single-agent wrapper for training Alice only."""
    metadata = {"render_modes": ["human"]}

    def __init__(self, sim_config):
        super().__init__()
        self.env = QKDEnv(sim_config)
        self.action_space      = self.env.action_space["Alice"]
        self.observation_space = self.env.observation_space

    def set_ppe(self, new_ppe):
        self.env.sim.set_ppe(new_ppe)

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return obs, info

    def step(self, action):
        # Bob fixed, Eve light random noise to keep policy robust
        bob_action = np.array([0.5, 1.0], dtype=np.float32)
        eve_action = 0.1 * self.env.action_space["Eve"].sample()  # small noise in [-0.1,0.1]
        obs, rewards, terminated, truncated, info = self.env.step({
            "Alice": action, "Bob": bob_action, "Eve": eve_action
        })
        reward = rewards["Alice"]
        done = terminated or truncated
        return obs, reward, done, False, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()
