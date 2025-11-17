# qkd_marl_env.py

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple

from sim.core_sim import QKDSimulator


class QKDMARLEnv(gym.Env):
    """
    Multi-agent QKD environment with asymmetric observations.
    - Shared simulator physics (one run_episode() per step)
    - Alice: rich observation space (new obs)
    - Eve: restricted observation (old obs / realistic adversary)
    - Bob: optional (cooperative)

    step() returns:
        obs = {"Alice": obs_A, "Bob": obs_B, "Eve": obs_E}
        rewards = {"Alice": rA, "Bob": rB, "Eve": rE}
        terminated, truncated, info
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, sim_config: Dict[str, Any], history_len: int = 1):

        super().__init__()
        self.sim = QKDSimulator(sim_config)
        self.cfg = sim_config
        self.history_len = history_len
        self.step_idx = 0

        # ==== Action spaces ====
        # Alice: choose intensities + p_signal
        self.alice_action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            shape=(3,),
        )

        # Bob (optional cooperative agent)
        self.bob_action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32), 
            high=np.array([1.0], dtype=np.float32),
            shape=(1,),
        )

        # Eve: choose attack intensities (time-shift, pns, intercept, dark)
        self.eve_action_space = spaces.Box(
            low=np.zeros(4, dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1e-3], dtype=np.float32),
            shape=(4,),
        )

        self.action_space = {
            "Alice": self.alice_action_space,
            "Bob":   self.bob_action_space,
            "Eve":   self.eve_action_space,
        }

        # ==== Observation spaces (ASYMMETRIC!) ====

        # Alice gets rich “new obs” (better for learning)
        # Example new-obs vector:
        # [detection_rate, qber_soft, Y0_est, log_channel_loss, SKR_last]
        self.ob_space_alice = spaces.Box(
            low=np.array([0, 0, 0, -10, 0], dtype=np.float32),
            high=np.array([1, 0.5, 1, 0, 1], dtype=np.float32),
        )

        # Eve gets restricted “old obs”
        # [Q_s, E_s, SKR, Q1_est]
        self.ob_space_eve = spaces.Box(
            low=np.zeros(4, dtype=np.float32),
            high=np.array([1.0, 0.5, 1.0, 1.0], dtype=np.float32),
        )

        # Bob (optional)
        self.ob_space_bob = spaces.Box(
            low=np.zeros(3, dtype=np.float32),
            high=np.ones(3, dtype=np.float32),
        )

        self.observation_space = {
            "Alice": self.ob_space_alice,
            "Bob":   self.ob_space_bob,
            "Eve":   self.ob_space_eve,
        }

        # Episode counter
        self.ep = 0


    # ============================================================
    # Helpers: observation extractors
    # ============================================================

    def _alice_obs(self, info: Dict[str, Any]) -> np.ndarray:
        """Rich new-obs for Alice."""
        det_rate = info["Q_s"]   # approx detection probability on signal
        qber     = info["E_s"]
        Y0       = info["Q_v"]   # vac gain = dark count surrogate
        log_eta  = np.log10(self.sim.eta + 1e-12)
        skr      = info["SKR_bits_per_pulse"]

        return np.array([det_rate, qber, Y0, log_eta, skr], dtype=np.float32)


    def _eve_obs(self, info: Dict[str, Any]) -> np.ndarray:
        """Restricted obs for Eve (realistic / no aggregated decoy internals)."""
        Qs  = info["Q_s"]
        Es  = info["E_s"]
        skr = info["SKR_bits_per_pulse"]
        Q1  = info["Q1"]

        return np.array([Qs, Es, skr, Q1], dtype=np.float32)


    def _bob_obs(self, info: Dict[str, Any]) -> np.ndarray:
        """Optional — signal gain, decoy gain, qber."""
        return np.array([
            info["Q_s"], info["Q_d"], info["E_s"]
        ], dtype=np.float32)


    # ============================================================
    # Core Gym API
    # ============================================================

    def reset(self, *, seed=None, options=None):
        self.step_idx = 0
        if hasattr(self.sim, "reset_stats"):
            self.sim.reset_stats()

        dummy_info = {
            "Q_s": 0.0, "Q_d": 0.0, "Q_v": 0.0,
            "E_s": 0.0,
            "Q1":  0.0,
            "SKR_bits_per_pulse": 0.0,
        }

        return {
            "Alice": self._alice_obs(dummy_info),
            "Bob":   self._bob_obs(dummy_info),
            "Eve":   self._eve_obs(dummy_info),
        }, {}


    def step(self, actions: Dict[str, np.ndarray]):
        """
        Each step runs one full QKD episode: 100k pulses, aggregation, SKR.
        """

        # === 1. Parse Alice actions ===
        a = actions["Alice"]
        mu_signal = float(np.clip(a[0], 0.0, 1.0))
        mu_decoy  = float(np.clip(a[1], 0.0, 1.0))
        p_signal  = float(np.clip(a[2], 0.0, 1.0))
        p_decoy   = 1 - p_signal - 0.1
        p_vac     = max(0.0, 1 - p_signal - p_decoy)

        alice_actions = {
            "mu_signal": mu_signal,
            "mu_decoy":  mu_decoy,
            "p_signal":  p_signal,
            "p_decoy":   p_decoy,
            "p_vac":     p_vac,
        }

        # === 2. Parse Eve actions ===
        e = actions["Eve"]
        eve_dict = {
            "type": "composite",
            "sub_attacks": [
                {"type": "time_shift",       "attack_prob": float(e[0]), "shift_frac": 0.1, "timing_qber_delta": 0.01},
                {"type": "pns",              "pns_frac":    float(e[1])},
                {"type": "intercept_resend", "intercept_prob": float(e[2]), "resend_eff": 0.8, "resend_error_prob": 0.1},
                {"type": "dark_count",       "extra_dark_prob": float(e[3])},
            ]
        }

        # Bob (optional)
        # Here Bob has no effect unless you expose det_eff/basis to him.
        # Keeping minimal for future extension.
        bob_actions = {}

        sim_actions = {
            "Alice": alice_actions,
            "Eve":   eve_dict,
            "Bob":   bob_actions,
        }

        # === 3. Run episode ===
        info = self.sim.run_episode(sim_actions, verbose=False)

        # === 4. Build asymmetric observations ===
        obs = {
            "Alice": self._alice_obs(info),
            "Bob":   self._bob_obs(info),
            "Eve":   self._eve_obs(info),
        }

        # === 5. Rewards ===
        skr = info["SKR_bits_per_pulse"]
        qber = info["E_s"]

        alice_reward = skr - 8.0 * qber         # cooperative
        bob_reward   = alice_reward             # same objective
        eve_reward   = qber + 0.5 * (skr < 0.001)   # destructive / adversarial

        rewards = {
            "Alice": float(alice_reward),
            "Bob":   float(bob_reward),
            "Eve":   float(eve_reward),
        }

        # === 6. Termination ===
        terminated = False
        truncated = False
        self.step_idx += 1

        # allow early termination if SKR collapses
        if skr <= 0 or qber > 0.15:
            terminated = True

        return obs, rewards, terminated, truncated, {"raw": info}


    def render(self):
        print(f"[QKD-MARL] Step {self.step_idx}")

    def close(self):
        pass
