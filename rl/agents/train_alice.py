import numpy as np
import gymnasium as gym
from rl.qkdenv import QKDEnv
from gymnasium import spaces

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
        # self.observation_space = self.env.observation_space
        obs_low = np.zeros(5, dtype=np.float32)
        obs_high = np.ones(5, dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        self.current_step=0
    
    def set_ppe(self, new_ppe):
        self.env.set_ppe(new_ppe)

    def reset(self, *, seed=None, options=None):
        obs, _ = self.env.reset(seed=seed, options=options)
        return obs[:5], {}

    def step(self, action):
        # Bob: fixed strategy
        bob_action = np.array([0.5, 1.0], dtype=np.float32)

        # Eve: random attack
        eve_action = self.env.action_space["Eve"].sample() * 0.3

        actions = {"Alice": action, "Bob": bob_action, "Eve": eve_action}
        obs, rewards, terminated, truncated, info = self.env.step(actions)
        reward = rewards["Alice"]
        done = terminated or truncated
        
        return obs[:5], reward, done, False, info


    def render(self):
        self.env.render()

    def close(self):
        self.env.close()


