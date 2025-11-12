import os
import numpy as np
from typing import Dict, Tuple, Any
import gymnasium as gym
from gymnasium import spaces

from sim.core_sim import QKDSimulator  # your simulator

class QKDEnv(gym.Env):
    """
    Multi-agent Gym-like environment for QKD with Alice, Bob, and Eve agents.
    Each step corresponds to one simulator.run_episode(actions).

    Observation: running mean of [SKR_pp, Q_s, E_s, Y1, e1, Q1]
    Rewards: cooperative (Alice+Bob maximize SKR), adversarial (Eve minimizes SKR)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, sim_config: Dict[str, Any], history_len: int = 5):
        super().__init__()
        self.sim = QKDSimulator(sim_config)
        self.history_len = history_len
        self.history = []
        self.current_step = 0
        self.episode_count = 0
        self.cfg = sim_config

        # Action spaces (policy outputs are assumed in [-1, 1])
        # We map to physical ranges inside step().
        self.alice_action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.bob_action_space   = spaces.Box(low=np.array([0.0, 0.0], dtype=np.float32),
                                             high=np.array([1.0, 2.0], dtype=np.float32), dtype=np.float32)
        self.eve_action_space   = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)

        self.action_space = {
            "Alice": self.alice_action_space,
            "Bob":   self.bob_action_space,
            "Eve":   self.eve_action_space,
        }

        # Observation space: [SKR_pp, Q_s, E_s, Y1, e1, Q1]
        obs_low  = np.zeros(6, dtype=np.float32)
        obs_high = np.array([10.0, 1.0, 0.5, 2.0, 0.5, 2.0], dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        os.makedirs(self.cfg.get("output_dir", "./results"), exist_ok=True)

    # ---------------- Core Gym API ----------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_count += 1
        self.current_step = 0
        self.history.clear()
        if hasattr(self.sim, "reset_stats"):
            self.sim.reset_stats()
        obs = np.zeros(6, dtype=np.float32)
        info = {}
        return obs, info

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

    def step(self, actions: Dict[str, np.ndarray]):
        """
        Stable step() with:
        - Normalized actions (policies act in [-1, 1])
        - Physically sensible mappings to QKD params
        - Rewards centered on SKR_bits_per_pulse (strong signal)
        - QBER penalty (physics-aligned)
        - No early terminate on SKR=0 (let PPO explore)
        """
        # --------------------------- 0) helpers ---------------------------
        def nn01(x):  # map [-1,1] -> [0,1]
            return 0.5 * (float(np.clip(x, -1.0, 1.0)) + 1.0)

        # --------------------------- 1) Alice actions ---------------------
        # Raw policy outputs in [-1,1]
        a_raw = actions.get("Alice", np.zeros(3, dtype=np.float32))
        a0, a1, a2 = [float(np.clip(x, -1.0, 1.0)) for x in a_raw]

        # Map to *sensible* BB84 decoy ranges
        # Tuned to keep the agent near known-good operating regions, but with room to optimize
        mu_signal = 0.70 + 0.20 * a0        # [0.50, 0.90]
        mu_decoy  = 0.12 + 0.10 * a1        # [0.02, 0.22] (low-intensity decoy)
        p_signal  = 0.75 + 0.10 * a2        # [0.65, 0.85]
        p_decoy   = 0.20                     # fixed modest decoy share
        p_vac     = max(0.0, 1.0 - p_signal - p_decoy)  # auto-adjust vacuum to keep sum=1

        # Probability validity reward term (will be 1.0 if exactly 1)
        prob_sum = p_signal + p_decoy + p_vac
        valid_prob_r = 1.0 - abs(prob_sum - 1.0)

        # Soft prior toward a typical optimum for mu_signal (kept *weak*)
        mu_target = 0.75
        mu_target_r = float(np.exp(-((mu_signal - mu_target) ** 2) / (2 * 0.08 ** 2)))  # wide, gentle bump

        # --------------------------- 2) Bob actions -----------------------
        b_raw = actions.get("Bob", np.array([0.5, 1.0], dtype=np.float32))
        basis_prob    = float(np.clip(b_raw[0], 0.0, 1.0))
        detector_gain = float(np.clip(b_raw[1], 0.0, 2.0))

        base_det_eff = getattr(self.sim, "det_eff", 0.2)
        base_dark    = getattr(self.sim, "dark_count", 1e-6)

        det_eff   = min(1.0, base_det_eff * detector_gain)
        dark_count = base_dark * (1.0 + 2.0 * max(0.0, detector_gain - 1.0) ** 2)

        # --------------------------- 3) Eve actions -----------------------
        e_raw = actions.get("Eve", np.zeros(5, dtype=np.float32))
        time_shift_prob = nn01(e_raw[0])
        shift_frac      = nn01(e_raw[1])
        pns_frac        = nn01(e_raw[2])
        intercept_prob  = nn01(e_raw[3])
        extra_dark_prob = 1e-5 * nn01(e_raw[4])  # small absolute scale

        # (Optional) light attack "effort" cost (prevents trivially maxing all attacks)
        attack_effort = (
            time_shift_prob +
            pns_frac +
            intercept_prob +
            (extra_dark_prob / 1e-5)  # normalize back to [0,1]
        ) / 4.0  # average

        sim_actions = {
            "Alice": {
                "mu_signal": mu_signal,
                "mu_decoy":  mu_decoy,
                "p_signal":  p_signal,
                "p_decoy":   p_decoy,
                "p_vac":     p_vac,
            },
            "Eve": {
                "type": "composite",
                "sub_attacks": [
                    {"type": "time_shift",       "attack_prob": float(time_shift_prob), "shift_frac": float(shift_frac), "timing_qber_delta": 0.01},
                    {"type": "pns",              "pns_frac":    float(pns_frac)},
                    {"type": "intercept_resend", "intercept_prob": float(intercept_prob), "resend_eff": 0.8, "resend_error_prob": 0.1},
                    {"type": "dark_count",       "extra_dark_prob": float(extra_dark_prob)},
                ],
            },
        }

        # --------------------------- 4) Apply Bob params temporarily ------
        prev_det_eff = getattr(self.sim, "det_eff", None)
        prev_dark    = getattr(self.sim, "dark_count", None)
        prev_basis   = getattr(self.sim, "basis_prob", None)

        self.sim.det_eff    = det_eff
        self.sim.dark_count = dark_count
        self.sim.basis_prob = basis_prob

        # --------------------------- 5) Run simulator ---------------------
        info = self.sim.run_episode(actions=sim_actions, verbose=False)

        # Restore base values
        if prev_det_eff is not None: self.sim.det_eff    = prev_det_eff
        if prev_dark    is not None: self.sim.dark_count = prev_dark
        if prev_basis   is not None: self.sim.basis_prob = prev_basis

        # --------------------------- 6) Observation -----------------------
        obs = self._obs_from_info(info)

        skr_pp = float(info.get("SKR_bits_per_pulse",  0.0))  # << primary signal
        skr_ps = float(info.get("SKR_bits_per_second", 0.0))
        qber   = float(info.get("E_s", 0.0))

        # --------------------------- 7) Rewards ---------------------------
        # Core idea:
        #   - Use SKR *per pulse* as the dominant term (strong weight)
        #   - Penalize QBER (physics-aligned; abort ~11%)
        #   - Keep constraint terms weak so they guide but don't dominate
        #   - Don't reward Eve for *killing* SKR entirely (anti-degenerate)
        #
        # Typical scales in your runs:
        #   skr_pp ~ 0.0005 .. 0.05   (bits/pulse)
        #   qber   ~ 0.0 .. 0.1
        #
        # Weights chosen to yield reward magnitudes O(1..10) and good gradients.
        W_SKR   = 400.0   # Alice: ~400 * skr_pp  (e.g., 0.01 → +4.0)
        W_QBERA = 30.0    # Alice: -30 * qber     (e.g., 0.05 → -1.5)
        W_CONS  = 0.5     # Alice: constraints (valid prob & mu target), each up to ~1

        W_QBERE = 50.0    # Eve:   +50 * qber
        W_SKRE  = 300.0   # Eve:   -300 * skr_pp (discourage total SKR kill)
        W_COSTE = 1.0     # Eve:   - effort to max all attacks

        alice_reward = (
            + W_SKR   * skr_pp
            - W_QBERA * qber
            + W_CONS  * valid_prob_r
            + W_CONS  * mu_target_r
        )

        # Eve gets rewarded for raising QBER, but is penalized if SKR collapses (to avoid degenerate policies)
        eve_reward = (
            + W_QBERE * qber
            - W_SKRE  * skr_pp
            - W_COSTE * attack_effort
        )

        rewards = {
            "Alice": float(alice_reward),
            "Bob":   float(alice_reward),
            "Eve":   float(eve_reward),
        }

        # --------------------------- 8) Episode end -----------------------
        self.current_step += 1
        truncated  = self.current_step >= 100
        # Abort only at very high QBER (let SKR=0 episodes exist for exploration)
        terminated = bool(qber > 0.20)

        return obs, rewards, terminated, truncated, {"raw_info": info}

    def render(self, mode="human"):
        print(f"[QKDEnv] Episode {self.episode_count}")

    def close(self):
        pass