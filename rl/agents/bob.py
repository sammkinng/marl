import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO  # or whatever you used

import os
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
import copy

from rl.dummyenv import _DummyObsEnv
from rl.env import QKDEnv


# from your_project.simulator import CoreSimulator

class BobTrainEnv(gym.Env):
    """
    Single-agent env for training Bob.
    - Alice: pretrained + frozen (SB3 model)
    - Eve:   pretrained + frozen (SB3 model)
    - Bob:   RL agent being trained
    """
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        sim_config: dict
    ):
        super().__init__()

        self.env = QKDEnv(sim_config)

        # ---- Bob action space: [basis_prob, detector_gain] ----
        self.action_space = self.env.action_space["Bob"]

        # ---- Observation space (same 5-dim obs you used) ----
        # [detection_rate, qber, est_Y0, est_Y1_lower, est_e1_upper]
        low  = np.array([0, 0, 0, 0, 0], dtype=np.float32)
        high = np.array([1, 1, 1, 1, 1], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        # We’ll reuse last obs as input for Alice & Eve policies
        obs_low = np.zeros(7, dtype=np.float32)
        obs_high = np.ones(7, dtype=np.float32)
        self._last_obs = spaces.Box(obs_low, obs_high, dtype=np.float32)

        self.current_step = 0

        self.load_models(sim_config)
        
    def load_models(self,SIM_CFG):
        LOGDIR = "./rl"
        LOGDIRe = "./logs_eve"
        MODEL_PATH = os.path.join(LOGDIRe, "ppo_ever4.zip")
        VECNORM   = os.path.join(LOGDIRe, "vecnormalize_ever4.pkl")

        
        MODEL_PATHa = os.path.join(LOGDIR, "ppo_alice.zip")
        VECNORMa   = os.path.join(LOGDIR, "vecnormalize_alice.pkl")

        # Load env & stats
        enve = DummyVecEnv([lambda: _DummyObsEnv(5, spaces.Box(
            low=np.array([-5.0, -5.0, -5.0, -5.0], dtype=np.float32),
            high=np.array([5.0, 5.0, 5.0, 5.0], dtype=np.float32),
            dtype=np.float32,
        ))])
        enve = VecNormalize.load(VECNORM, enve)
        enve.training = False
        enve.norm_reward = False

        self.eve_model = PPO.load(MODEL_PATH,env=enve)

        enva = DummyVecEnv([lambda: _DummyObsEnv(5, spaces.Box(
            low=np.array([-5.0, -5.0, -5.0, -5.0], dtype=np.float32),
            high=np.array([5.0, 5.0, 5.0, 5.0], dtype=np.float32),
            dtype=np.float32,
        ))])
        enva = VecNormalize.load(VECNORMa, enva)
        enva.training = False
        enva.norm_reward = False

        self.alice_model = PPO.load(MODEL_PATHa,env=enva)


    # ---------------- Gym API ----------------

    def reset(self, *, seed=None, options=None):
        obs, _ = self.env.reset(seed=seed, options=options)
        self._last_obs = obs.copy()
        return obs[:5], {}

    def step(self, bob_action):
        # Ensure correct dtype/shape
        bob_action = np.asarray(bob_action, dtype=np.float32)

        # --- Get Alice and Eve actions from frozen models ---
        alice_obs = self._last_obs[:5]
        eve_obs   = np.concatenate([self._last_obs[:3], self._last_obs[5:]])

        alice_action, _ = self.alice_model.predict(alice_obs, deterministic=True)
        eve_action, _   = self.eve_model.predict(eve_obs, deterministic=True)

        alice_action = np.asarray(alice_action, dtype=np.float32)
        eve_action   = np.asarray(eve_action, dtype=np.float32)

        actions = {"Alice": alice_action, "Bob": bob_action, "Eve": eve_action}
        obs, rewards, terminated, truncated, info = self.env.step(actions)
        reward = rewards["Bob"]
        done = terminated or truncated

        self._last_obs = obs.copy()
        
        return obs[:5], reward, done, False, info


    def render(self, mode="human"):
        pass
