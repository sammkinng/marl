import gymnasium as gym
import numpy as np
from gymnasium import spaces

class _DummyObsEnv(gym.Env):
    def __init__(self, obs_dim, act_space):
        super().__init__()
        self.observation_space = spaces.Box(0, 1, (obs_dim,), dtype=np.float32)
        self.action_space = act_space

    def reset(self, *, seed=None, options=None):
        return np.zeros(self.observation_space.shape, np.float32), {}

    def step(self, action):
        return (
            np.zeros(self.observation_space.shape, np.float32),
            0.0,
            True,
            False,
            {}
        )