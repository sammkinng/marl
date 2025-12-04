import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from rl.dummyenv import _DummyObsEnv
from rl.env import QKDEnv
from gymnasium import spaces

import os

from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize


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
        # self.observation_space = self.env.observation_space
        obs_low = np.zeros(5, dtype=np.float32)
        obs_high = np.ones(5, dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        self.current_step = 0

        obs_low = np.zeros(7, dtype=np.float32)
        obs_high = np.ones(7, dtype=np.float32)
        self._last_obs = spaces.Box(obs_low, obs_high, dtype=np.float32)
        self.load_models()

    def load_models(self):
        LOGDIR = "./rl"
        LOGDIRb = "./logs_bob"

        MODEL_PATHa = os.path.join(LOGDIR, "ppo_alice.zip")
        VECNORMa   = os.path.join(LOGDIR, "vecnormalize_alice.pkl")

        MODEL_PATHb = os.path.join(LOGDIRb, "ppo_bobr5.zip")
        VECNORMb   = os.path.join(LOGDIRb, "vecnormalize_bobr5.pkl")

        # Load env & stats

        enva = DummyVecEnv([lambda: _DummyObsEnv(5, self.env.action_space["Alice"])])
        enva = VecNormalize.load(VECNORMa, enva)
        enva.training = False
        # enva.norm_reward = False

        alice_model = PPO.load(MODEL_PATHa,env=enva)

        envb = DummyVecEnv([lambda: _DummyObsEnv(5, self.env.action_space["Bob"])])
        envb = VecNormalize.load(VECNORMb, envb)
        envb.training = False
        # envb.norm_reward = False

        bob_model = PPO.load(MODEL_PATHb,env=envb)

        self.alice_model = alice_model
        self.bob_model = bob_model


    # Optional: expose PPE / reward scaling controls
    def set_ppe(self, new_ppe):
        self.env.set_ppe(new_ppe)


    # ----------------------
    # Reset
    # ----------------------
    def reset(self, *, seed=None, options=None):
        obs, _ = self.env.reset(seed=seed, options=options)
        tobs = np.concatenate([obs[:3], obs[5:]])
        self._last_obs = obs.copy()
        return tobs, {}

    # ----------------------
    # Step
    # ----------------------
    def step(self, action):
        alice_obs = bob_obs = self._last_obs[:5]
        alice_action,_ = self.alice_model.predict(alice_obs, deterministic=True)

        
        bob_action,_ = self.bob_model.predict(bob_obs, deterministic=True)

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

        tobs = np.concatenate([obs[:3], obs[5:]])

        return tobs, reward, done, False, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()
