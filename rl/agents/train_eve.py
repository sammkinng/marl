import gymnasium as gym
import numpy as np
from rl.qkdenv import QKDEnv


class EveSingleAgentEnv(gym.Env):
    """
    Single-agent environment for training Eve only.
    Alice and Bob use fixed strategies.
    Eve's objective = minimize SKR (so reward = -SKR or shaped variant).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, sim_config):
        super().__init__()

        # Underlying full QKD environment
        self.env = QKDEnv(sim_config)

        # Eve is the only learning agent
        self.action_space = self.env.action_space["Eve"]
        self.observation_space = self.env.observation_space

        self.current_step = 0

    # Optional: expose PPE / reward scaling controls
    def set_ppe(self, new_ppe):
        self.env.set_ppe(new_ppe)

    def set_reward_scale(self, new_scale):
        self.env.set_reward_scale(new_scale)

    # ----------------------
    # Reset
    # ----------------------
    def reset(self, *, seed=None, options=None):
        obs, _ = self.env.reset(seed=seed, options=options)
        return obs, {}

    # ----------------------
    # Step
    # ----------------------
    def step(self, action):
        # Alice fixed strategy
        alice_action = np.array([0.6, 0.1, 0.7,0.0], dtype=np.float32)

        # Bob fixed strategy
        bob_action = np.array([0.5, 1.0], dtype=np.float32)

        # Eve is controlled by RL (your 5-dim vector)
        eve_action = action

        actions = {
            "Alice": alice_action,
            "Bob": bob_action,
            "Eve": eve_action,
        }

        obs, rewards, terminated, truncated, info = self.env.step(actions)

        # Eve receives negative reward of Alice/Bob
        reward = rewards["Eve"]

        done = terminated or truncated

        return obs, reward, done, False, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()
