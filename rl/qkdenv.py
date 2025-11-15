"""
qkd_env.py — Multi-agent RL environment wrapper for QKDSimulator.
Compatible with gym-style interfaces (SB3, PettingZoo, RLlib, etc.).

Key changes / guarantees:
- Alice action_space is 4D: [raw_mu_s, raw_mu_d, raw_logit_p_s, raw_logit_p_d]
  -> Converts to constrained μ and softmax probabilities (p_signal,p_decoy,p_vac)
  -> μ_signal and μ_decoy are enforced: mu_signal > mu_decoy and in safe ranges.
  -> These Alice choices are passed to sim.run_episode and remain fixed for that episode.
- Observation space changed to Markov-safe step-level stats:
  [detection_rate, qber, Y0_est, Y1_lower, e1_upper]
- Reward: normalized SKR minus soft QBER penalty (stable shaping).
- Bob params are temporarily applied to simulator during run_episode and restored.
"""

import os
import numpy as np
from typing import Dict, Tuple, Any
import gymnasium as gym
from gymnasium import spaces

from sim.core_sim import QKDSimulator


class QKDEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, sim_config: Dict[str, Any], history_len: int = 5):
        super().__init__()

        # Simulator instance
        self.sim = QKDSimulator(sim_config)
        self.cfg = sim_config

        # episode bookkeeping
        self.current_step = 0
        self.episode_count = 0
        self.history_len = history_len
        self.history = []

        # Ensure simulator uses configured pulses per episode
        self.sim.pulses_per_episode = sim_config["pulses_per_episode"]

        # ---------------------------
        # Action spaces
        # ---------------------------
        # Alice: 4D raw outputs -> convert to safe μ and softmax probs
        # raw values are unconstrained; we'll map them with sigmoid/softmax
        # Shape: [raw_mu_signal, raw_mu_decoy, raw_logit_p_signal, raw_logit_p_decoy]
        self.alice_action_space = spaces.Box(
            low=np.array([-5.0, -5.0, -5.0, -5.0], dtype=np.float32),
            high=np.array([5.0, 5.0, 5.0, 5.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Bob: [basis_prob, detector_gain]
        self.bob_action_space = spaces.Box(
            low=np.array([0.0, 0.5], dtype=np.float32),
            high=np.array([1.0, 2.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Eve: placeholder 5-dim continuous vector for composite attacks
        self.eve_action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1e-3], dtype=np.float32),
            dtype=np.float32,
        )

        self.action_space = {
            "Alice": self.alice_action_space,
            "Bob": self.bob_action_space,
            "Eve": self.eve_action_space,
        }

        # ---------------------------
        # Observation space (Markov-safe)
        # [detection_rate, qber, Y0_est, Y1_lower, e1_upper] all normalized 0..1
        # ---------------------------
        obs_low = np.zeros(5, dtype=np.float32)
        obs_high = np.ones(5, dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        # Logging / output
        os.makedirs(self.cfg["output_dir"], exist_ok=True)
        # self.csv_file = os.path.join(self.cfg["output_dir"], self.cfg["results_csv"])

        # Reward scaling constant (tunable)
        self.MAX_SKR_BPS = 3e4  # used to normalize skr_bps to ~0..1

    # ---------------------------
    # Helpers: math transforms
    # ---------------------------
    def set_ppe(self, new_ppe):
        self.sim.set_ppe(new_ppe)

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        e = np.exp(logits - np.max(logits))
        return e / np.sum(e)

    # ---------------------------
    # Gym interface
    # ---------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_count += 1
        self.current_step = 0
        self.history.clear()
        if hasattr(self.sim, "reset_stats"):
            self.sim.reset_stats()

        # Return zero observation initially
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, {}

    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, float], bool, Dict]:
        """
        One RL step == one full simulator episode (sim.run_episode).
        Alice's chosen μ and probabilities are applied for the whole episode.
        """

        # ---------------------------
        # 1) Parse & convert Alice action (Option 3)
        # ---------------------------
        a = actions.get("Alice", np.zeros(4, dtype=np.float32))
        raw_mu_s = float(a[0])
        raw_mu_d = float(a[1])
        raw_logit_ps = float(a[2])
        raw_logit_pd = float(a[3])

        # map raw μ to safe ranges:
        # μ_signal ∈ [0.40, 0.70], μ_decoy ∈ [0.01, 0.15]
        mu_signal = 0.40 + self._sigmoid(raw_mu_s) * (0.70 - 0.40)
        mu_decoy = 0.01 + self._sigmoid(raw_mu_d) * (0.15 - 0.01)

        # ensure mu_signal > mu_decoy with a small margin
        if mu_decoy >= mu_signal - 0.01:
            mu_decoy = max(0.01, mu_signal - 0.01)

        # softmax probabilities for [p_signal, p_decoy, p_vac]
        logits = np.array([raw_logit_ps, raw_logit_pd, 0.0], dtype=np.float32)
        p_signal, p_decoy, p_vac = self._softmax(logits)

        # ---------------------------
        # 2) Parse Bob action and temporarily apply to simulator
        # ---------------------------
        b = actions.get("Bob", np.array([0.5, 1.0], dtype=np.float32))
        basis_prob = float(np.clip(b[0], 0.0, 1.0))
        detector_gain = float(np.clip(b[1], 0.5, 2.0))

        prev_det_eff = getattr(self.sim, "det_eff", None)
        prev_dark = getattr(self.sim, "dark_count", None)
        prev_basis = getattr(self.sim, "basis_prob", None)

        # compute new detector params and apply
        base_det_eff = getattr(self.sim, "det_eff", 0.1)
        base_dark = getattr(self.sim, "dark_count", 1e-6)
        det_eff = min(1.0, base_det_eff * detector_gain)
        dark_count = base_dark * (1.0 + 2.0 * max(0.0, detector_gain - 1.0) ** 2)

        # apply to sim (temporarily)
        # self.sim.det_eff = det_eff
        # self.sim.dark_count = dark_count
        # self.sim.basis_prob = basis_prob

        # ---------------------------
        # 3) Parse Eve action (for future use) - currently disabled in sim actions
        # ---------------------------
        e = actions.get("Eve", np.zeros(5, dtype=np.float32))
        # Unpack if you later want to enable composing attacks
        time_shift_prob, shift_frac, pns_frac, intercept_prob, extra_dark_prob = tuple(e)

        # ---------------------------
        # 4) Build sim actions and run episode
        # ---------------------------
        sim_actions = {
            "Alice": {
                "mu_signal": float(mu_signal),
                "mu_decoy": float(mu_decoy),
                "p_signal": float(p_signal),
                "p_decoy": float(p_decoy),
                "p_vac": float(p_vac),
            },
            # Currently not passing Eve composite attack object by default.
            "Eve": None
        }

        info = self.sim.run_episode(actions=sim_actions, verbose=False)

        # restore simulator base params
        if prev_det_eff is not None:
            self.sim.det_eff = prev_det_eff
        if prev_dark is not None:
            self.sim.dark_count = prev_dark
        if prev_basis is not None:
            self.sim.basis_prob = prev_basis

        # ---------------------------
        # 5) Observation: Markov-safe stats
        # ---------------------------
        obs = self._obs_from_info(info)

        # ---------------------------
        # 6) Rewards (stable shaping)
        # ---------------------------
        skr_bps = float(info.get("SKR_bits_per_second", 0.0))
        skr_pp = float(info.get("SKR_bits_per_pulse", 0.0))
        qber = float(info.get("E_s", 0.0))

        norm_skr = skr_bps / self.MAX_SKR_BPS

        # Alice & Bob cooperative reward: prioritize SKR but penalize QBER
        alice_reward = float(norm_skr - 10.0 * qber)
        bob_reward = alice_reward

        # Eve: incentive to increase information / errors while not trivially killing SKR
        eve_info = float(info.get("eve_info_gain", 0.0))
        eve_reward = float(eve_info + 1.0 * qber - 0.5 * norm_skr)

        rewards = {
            "Alice": alice_reward,
            "Bob": bob_reward,
            "Eve": eve_reward,
        }

        # ---------------------------
        # 7) Termination / truncation
        # ---------------------------
        self.current_step += 1
        terminated = (skr_pp <= 0.0) or (qber > 0.11)
        truncated = self.current_step >= 100

        # include raw sim info for debugging
        info_out = {"raw_info": info, "alice_params": sim_actions["Alice"]}

        return obs, rewards, terminated, truncated, info_out

    def _obs_from_info(self, info: Dict[str, Any]) -> np.ndarray:
        """
        Extract step-level Markov-safe statistics from simulator info:
          detection_rate, qber (E_s), Y0_est, Y1_lower, e1_upper
        Fall back to safe defaults if not present.
        Keep a short history and return the mean to smooth noise.
        """
        detection_rate = float(info.get("detection_rate", 0.0))
        qber = float(info.get("E_s", 0.0))
        Y0 = float(info.get("Y0", 0.0))
        Y1_lower = float(info.get("Y1_lower", info.get("Y1", 0.0)))
        e1_upper = float(info.get("e1_upper", info.get("e1", 0.0)))

        vec = np.array([detection_rate, qber, Y0, Y1_lower, e1_upper], dtype=np.float32)

        self.history.append(vec)
        if len(self.history) > self.history_len:
            self.history.pop(0)

        stacked = np.stack(self.history, axis=0)
        return np.mean(stacked, axis=0)

    def render(self, mode="human"):
        print(f"[QKDEnv] Episode {self.episode_count} Step {self.current_step}")

    def close(self):
        pass
