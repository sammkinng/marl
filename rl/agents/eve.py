import numpy as np
import gymnasium as gym
from rl.qkd_rl import QKDEnv

class EveSingleAgentEnv(gym.Env):
    """Single-agent wrapper for training Eve only (Alice, Bob fixed)."""
    metadata = {"render_modes": ["human"]}

    def __init__(self, sim_config, alice_fixed=None, bob_fixed=(0.5, 1.0)):
        super().__init__()
        self.env = QKDEnv(sim_config)
        self.action_space      = self.env.action_space["Eve"]
        self.observation_space = self.env.observation_space
        # Optional fixed Alice action in [-1,1] space
        self.alice_fixed = np.array(alice_fixed if alice_fixed is not None else [0.0, 0.0, 0.0], dtype=np.float32)
        self.bob_fixed   = np.array(bob_fixed, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        obs, rewards, terminated, truncated, info = self.env.step({
            "Alice": self.alice_fixed,  # fixed Alice in normalized space
            "Bob":   self.bob_fixed,
            "Eve":   action,
        })
        reward = rewards["Eve"]
        done = terminated or truncated
        return obs, reward, done, False, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()