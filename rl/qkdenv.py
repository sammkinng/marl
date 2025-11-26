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

import numpy as np
from typing import Dict, Tuple, Any
import gymnasium as gym
from gymnasium import spaces

from sim.core_sim import QKDSimulator
from sim.utils import  qkd_simulation


class QKDEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, sim_config: Dict[str, Any], history_len: int = 1):
        super().__init__()

        # Simulator instance
        # self.sim = QKDSimulator(sim_config)
        self.cfg = sim_config

        # episode bookkeeping
        self.current_step = 0
        self.episode_count = 0
        self.history_len = history_len
        self.history = []

        # Ensure simulator uses configured pulses per episode
        # self.sim.pulses_per_episode = sim_config["pulses_per_episode"]

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

        ## Eve (Option E1): 4D logits for attack probabilities + dark boost strength
        self.eve_action_space = spaces.Box(
            low=np.array([-5.0, -5.0, -5.0, -5.0], dtype=np.float32),
            high=np.array([5.0, 5.0, 5.0, 5.0], dtype=np.float32),
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
        obs_low = np.zeros(7, dtype=np.float32)
        obs_high = np.ones(7, dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        # Reward scaling constant (tunable)
        self.MAX_SKR_BPS = 2e5  # used to normalize skr_bps to ~0..1

    # ---------------------------
    # Helpers: math transforms
    # ---------------------------
    # def set_ppe(self, new_ppe):
    #     self.sim.set_ppe(new_ppe)

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
        # if hasattr(self.sim, "reset_stats"):
        #     self.sim.reset_stats()

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

        s = p_signal + p_decoy + p_vac
        if s > 0:
            self.p_signal, self.p_decoy, self.p_vac = p_signal/s, p_decoy/s, p_vac/s


        # ---------------------------
        # 2) Parse Bob action and temporarily apply to simulator
        # ---------------------------
        # b = actions.get("Bob", np.array([0.5, 1.0], dtype=np.float32))
        # basis_prob = float(np.clip(b[0], 0.0, 1.0))
        # detector_gain = float(np.clip(b[1], 0.5, 2.0))

        # prev_det_eff = getattr(self.sim, "det_eff", None)
        # prev_dark = getattr(self.sim, "dark_count", None)
        # prev_basis = getattr(self.sim, "basis_prob", None)

        # compute new detector params and apply
        # base_det_eff = getattr(self.sim, "det_eff", 0.1)
        # base_dark = getattr(self.sim, "dark_count", 1e-6)
        # det_eff = min(1.0, base_det_eff * detector_gain)
        # dark_count = base_dark * (1.0 + 2.0 * max(0.0, detector_gain - 1.0) ** 2)

        # apply to sim (temporarily)
        # self.sim.det_eff = det_eff
        # self.sim.dark_count = dark_count
        # self.sim.basis_prob = basis_prob

        # ---------------------------
        # 3) Parse Eve action (for future use) - currently disabled in sim actions
        # ---------------------------
        
        e = actions.get("Eve", np.zeros(4, dtype=np.float32))
        raw_ir, raw_pns, raw_ts, raw_dark = map(float, e)

        # # Softmax for attack-type probabilities
        logits = np.array([raw_ir, raw_pns, raw_ts], dtype=np.float32)
        p=self._softmax(logits)
        p_ir, p_pns, p_ts = p.tolist()

        # # Sigmoid for dark-boost
        dark_boost = self._sigmoid(raw_dark)
        max_extra_dark = 5e-6
        extra_dark_prob = dark_boost * max_extra_dark


        # ---------------------------
        # 4) Build sim actions and run episode
        # ---------------------------
        # sim_actions = {
        #     "Alice": {
        #         "mu_signal": float(mu_signal),
        #         "mu_decoy": float(mu_decoy),
        #         "p_signal": float(p_signal),
        #         "p_decoy": float(p_decoy),
        #         "p_vac": float(p_vac),
        #     },
        #     # Currently not passing Eve composite attack object by default.
        #     "Eve": None
        # }

        # sim_actions["Alice"]={
        #         "mu_signal": 0.6,
        #         "mu_decoy": 0.1,
        #         "p_signal": 0.6,
        #         "p_decoy": 0.3,
        #         "p_vac": 0.1,
        #     }

#         sim_actions["Eve"] = {"type": "composite",
#                                "sub_attacks":[
#     {
#         "type": "intercept_resend",
#         "intercept_prob": p_ir,
#         "resend_eff": 0.8,
#         "resend_error_prob": 0.05,
#     },
#     {
#         "type": "pns",
#         "pns_frac": p_pns,
#     },
#     {
#         "type": "time_shift",
#         "attack_prob": p_ts,
#         "shift_frac": 0.2,
#         "timing_qber_delta": 0.01,
#     },
#     {
#         "type": "dark_count",
#         "extra_dark_prob": extra_dark_prob,
#     }
# ]}


        # info = self.sim.run_episode(actions=sim_actions, verbose=False)

        info=qkd_simulation(self.cfg,
                            aa={},
        #                     aa={
        #     "mus": mu_signal,
        #     "mud": mu_decoy,
        #     "ps": self.p_signal,
        #     "pd": self.p_decoy
        # },
        # attacks={})
                            
                            attacks = {
    "intercept_resend": p_ir,   # probability Eve IR attack
    "pns": p_pns,                # Eve performs photon number splitting
    "time_shift": p_ts,         # Eve time-shifts to cause detector bias
    "darkcount_increase": dark_boost  # Eve artificially raises the dark count
} 
                            )
        


        # eta=10 ** (self.cfg["fiber_loss_db_per_km"] * float(self.cfg.get("distance_km", 10.0)) / 10.0)
        # info= calculate_qkd_asymptotic_performance(
        #         mu_signal=mu_signal, mu_decoy=mu_decoy, mu_vac=0.0,
        #         p_signal=p_signal, p_decoy=p_decoy, p_vac=p_vac,
        #         eta_ch=eta, eta_det=self.cfg["det_eff"], P_d_base=float(self.cfg["dark_count"]), 
        #         e_0_base=self.cfg["baseline_qber"], basis_prob=self.cfg["basis_prob"], 
        #         recon_eff=self.cfg["recon_eff"], pulse_rate=float(self.cfg["pulse_rate"]),
        #         pulses_per_episode=self.cfg["pulses_per_episode"],
        #         eve_action=None, cfg=self.cfg
        #     )

        # # restore simulator base params
        # if prev_det_eff is not None:
        #     self.sim.det_eff = prev_det_eff
        # if prev_dark is not None:
        #     self.sim.dark_count = prev_dark
        # if prev_basis is not None:
        #     self.sim.basis_prob = prev_basis

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
        # eig=float(info.get("eve_info_gain", 0.0))

        norm_skr = skr_bps / self.MAX_SKR_BPS

        # Alice & Bob cooperative reward: prioritize SKR but penalize QBER
        alice_reward = float(norm_skr - 10.0 * qber)
        bob_reward = alice_reward

        QBER_ABORT = 0.05

        ε = 0.00001     # (or 0.005, 0.02 — depends on tuning)
        # 20 and 0.006 - 0.0117(w/o norm) with norm 0.02-------------------------0----------.0118 when eig
#  1e4 and 1.0/0.0001  - 0.0163
# 1e4 and 10 /0.0001- 0.0191(w/o norm) with 0.0202--------------------------------1----------------(167 in 300k training)

# 1e2 + 2+3e2+5e3+0-------0.02(in 300k) ---------------0.0164 with im and 5e5 norm upto 5e3 it was 0------its not saved yet
# 0.2 soemthing for new ir and 10.0 10.0 3 and 5 and 0(300k)

# best was i guess 7.06 and 3.0
        multi_frac = float(info.get("P_multi_mix", 0.0))
        eta_mis = float(info.get("eta_mismatch", 0.0))

        ts_penalty  = p_ts  * (1 - eta_mis)
        pns_penalty = p_pns * (1 - multi_frac)
        dark_penalty = dark_boost * (1 - (1 - float(info.get("Q_v", 0.0))))  # punish dark boost when Y0 big

# 0.02 for deualt reward of new system
# 0.0239 best -screenshot at 23:30
# 0.0247 best at 23:40 -5.5
# now test scenario is hight info gain- best till now is 0.4872  with 20 and 7.5
# 0.4912 fro 30 and 7.5
# .4940 for 100 and 20
        eve_reward = (
    #         + (1 - norm_skr) * 10.0
    #         +20.0*qber
    #         # +10.0 * eig                         # reward for info gain
    # + 4.0 * min(qber, QBER_ABORT - ε)   # reward for pushing QBER up to edge
    # - 7.06 * (qber >= QBER_ABORT)        # big penalty if she crosses threshold
    # # - 5.0 * (norm_skr <= 0.0001)  
    #                 # penalty for killing channel completely
    #     )
    # (
        + 100.0 * info.get("info_gain_per_pulse",0.0)      # Eve wants information
        -20.0*norm_skr            # lower SKR is better
        - 2.0 * qber            # high QBER reveals Eve
    #     + 2.0 * min(qber, QBER_ABORT - ε)   # reward for pushing QBER up to edge
    # - 5.0 * (qber >= QBER_ABORT)        # big penalty if she crosses threshold
        - 1.2 * ts_penalty      # TS only if eta-mismatch high
        - 1.0 * pns_penalty     # PNS only if multi-photon high
        - 0.5 * dark_penalty    # dark-boost only when Y0 small
    )


        rewards = {
            "Alice": alice_reward,
            "Bob": bob_reward,
            "Eve": eve_reward,
        }

        # ---------------------------
        # 7) Termination / truncation
        # ---------------------------
        self.current_step += 1
        terminated = (skr_pp <= 0.00001) or (qber > 0.11)
        truncated = self.current_step >= 100

        # include raw sim info for debugging
        info_out = {"raw_info": info, "alice_params": {}}

        return obs, rewards, terminated, truncated, info_out

    def _obs_from_info(self, info: Dict[str, Any]) -> np.ndarray:
        """
        Extract step-level Markov-safe statistics from simulator info:
          detection_rate, qber (E_s), Y0_est, Y1_lower, e1_upper
        Fall back to safe defaults if not present.
        Keep a short history and return the mean to smooth noise.
        """
        detection_rate = float(info.get("Q_s", 0.0))
        qber = float(info.get("E_s", 0.0))
        Y0 = float(info.get("Q_v", 0.0))
        Y1_lower = float(info.get("Y1_lower", info.get("Y1", 0.0)))
        e1_upper = float(info.get("e1_upper", info.get("e1", 0.0)))
        multi_frac = float(info.get("P_multi_mix", 0.0))
        eta_mis = float(info.get("eta_mismatch", 0.0))
       
        vec = np.array([detection_rate, qber, Y0, Y1_lower, e1_upper,multi_frac,eta_mis], dtype=np.float32)

        self.history.append(vec)
        if len(self.history) > self.history_len:
            self.history.pop(0)

        stacked = np.stack(self.history, axis=0)
        return np.mean(stacked, axis=0)

    def render(self, mode="human"):
        print(f"[QKDEnv] Episode {self.episode_count} Step {self.current_step}")

    def close(self):
        pass
