"""
qkd_env.py — Multi-agent RL environment wrapper for QKDSimulator.
Compatible with gym-style interfaces (SB3, PettingZoo, RLlib, etc.).
"""

import numpy as np
from typing import Dict, Tuple, Any
import gym
from gym import spaces
from sim.core_sim import QKDSimulator

# Import your simulator
# from core_simulator import QKDSimulator


class QKDEnv(gym.Env):
    """
    Multi-agent Gym-like environment for QKD with Alice, Bob, and Eve agents.
    Each step corresponds to one simulator.run_episode(actions).

    Observation: global vector [SKR, Q_s, E_s, Y1, e1, Q1]
    Reward: cooperative (Alice+Bob maximize SKR), adversarial (Eve minimizes SKR)
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, sim_config: Dict[str, Any], pulses_per_episode: int = None, history_len: int = 5):
        super().__init__()
        self.sim = QKDSimulator(sim_config)

        if pulses_per_episode is not None:
            self.sim.pulses_per_episode = pulses_per_episode

        # Define action spaces
        self.alice_action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0]),   # mu_signal, mu_decoy, p_signal
            high=np.array([2.0, 1.0, 1.0]),
            dtype=np.float32
        )
        self.bob_action_space = spaces.Box(
            low=np.array([0.0, 0.5]),        # basis_prob, detector_gain
            high=np.array([1.0, 2.0]),
            dtype=np.float32
        )
        self.eve_action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1e-3]),
            dtype=np.float32
        )

        self.action_space = {
            "Alice": self.alice_action_space,
            "Bob": self.bob_action_space,
            "Eve": self.eve_action_space,
        }

        # Observation space: [SKR, Q_s, E_s, Y1, e1, Q1]
        obs_low = np.zeros(6, dtype=np.float32)
        obs_high = np.array([10.0, 1.0, 0.5, 1.0, 0.5, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        self.history_len = history_len
        self.history = []
        self.episode_count = 0

    # ------------------------------------------------------------
    # Core Gym interface
    # ------------------------------------------------------------
    def reset(self) -> np.ndarray:
        self.episode_count += 1
        if hasattr(self.sim, "reset_stats"):
            self.sim.reset_stats()
        self.history.clear()
        return np.zeros(6, dtype=np.float32)

    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, float], bool, Dict]:
        """Run one simulator episode using actions from all agents."""

        # --- 1. Alice ---
        a = actions.get("Alice", np.zeros(3, dtype=np.float32))
        mu_signal = float(np.clip(a[0], 0.0, 2.0))
        mu_decoy = float(np.clip(a[1], 0.0, 1.0))
        p_signal = float(np.clip(a[2], 0.0, 1.0))
        p_decoy = max(0.0, 1.0 - p_signal - 0.1)
        p_vac = max(0.0, 1.0 - p_signal - p_decoy)

        # --- 2. Bob ---
        b = actions.get("Bob", np.array([0.5, 1.0], dtype=np.float32))
        basis_prob = float(np.clip(b[0], 0.0, 1.0))
        detector_gain = float(np.clip(b[1], 0.0, 2.0))

        base_det_eff = getattr(self.sim, "det_eff", 0.1)
        base_dark = getattr(self.sim, "dark_count", 1e-6)
        det_eff = min(1.0, base_det_eff * detector_gain)
        dark_count = base_dark * (1.0 + 2.0 * max(0.0, detector_gain - 1.0) ** 2)

        # --- 3. Eve ---
        e = actions.get("Eve", np.zeros(5, dtype=np.float32))
        time_shift_prob, shift_frac, pns_frac, intercept_prob, extra_dark_prob = e

        sim_actions = {
            "Alice": {
                "mu_signal": mu_signal,
                "mu_decoy": mu_decoy,
                "p_signal": p_signal,
                "p_decoy": p_decoy,
                "p_vac": p_vac,
            },
            "Eve": {
                "type": "composite",
                "sub_attacks": [
                    {"type": "time_shift", "attack_prob": float(time_shift_prob), "shift_frac": float(shift_frac), "timing_qber_delta": 0.01},
                    {"type": "pns", "pns_frac": float(pns_frac)},
                    {"type": "intercept_resend", "intercept_prob": float(intercept_prob), "resend_eff": 0.8, "resend_error_prob": 0.1},
                    {"type": "dark_count", "extra_dark_prob": float(extra_dark_prob)},
                ],
            },
        }

        # --- Apply Bob params temporarily ---
        prev_det_eff = getattr(self.sim, "det_eff", None)
        prev_dark = getattr(self.sim, "dark_count", None)
        prev_basis = getattr(self.sim, "basis_prob", None)

        self.sim.det_eff = det_eff
        self.sim.dark_count = dark_count
        self.sim.basis_prob = basis_prob

        # --- Run simulator episode ---
        info = self.sim.run_episode(actions=sim_actions, verbose=False)

        # --- Restore base values ---
        if prev_det_eff is not None:
            self.sim.det_eff = prev_det_eff
        if prev_dark is not None:
            self.sim.dark_count = prev_dark
        if prev_basis is not None:
            self.sim.basis_prob = prev_basis

        # --- Observation & reward ---
        obs = self._obs_from_info(info)
        skr = float(info.get("SKR_bits_per_pulse", 0.0))
        rewards = {"Alice": skr, "Bob": skr, "Eve": -skr}

        done = False
        extras = {"raw_info": info}
        return obs, rewards, done, extras

    def _obs_from_info(self, info: Dict[str, Any]) -> np.ndarray:
        vec = np.array([
            float(info.get("SKR_bits_per_pulse", 0.0)),
            float(info.get("Q_s", 0.0)),
            float(info.get("E_s", 0.0)),
            float(info.get("Y1", 0.0)),
            float(info.get("e1", 0.0)),
            float(info.get("Q1", 0.0)),
        ], dtype=np.float32)
        self.history.append(vec)
        if len(self.history) > self.history_len:
            self.history.pop(0)
        return np.mean(np.stack(self.history, axis=0), axis=0)

    def render(self, mode="human"):
        print(f"[QKDEnv] Episode {self.episode_count}")

    def close(self):
        pass
