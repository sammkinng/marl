# env_qkd_rl.py
import numpy as np
from gymnasium import Env, spaces
from sim.core_sim import QKDSimulator

class QKDRLEnv(Env):
    """RL-ready environment wrapping QKDSimulator."""
    metadata = {"render_modes": []}

    def __init__(self, config):
        super().__init__()
        self.sim = QKDSimulator(config)
        self.pulses_per_step = config.get("pulses_per_step", 5000)

        # Action: [mu_signal, basis_prob]
        self.action_space = spaces.Box(low=np.array([0.0, 0.0]),
                                       high=np.array([1.0, 1.0]),
                                       dtype=np.float32)

        # Observation: [Q_s, Q_d, Q_v, E_s, E_d, E_v]
        self.observation_space = spaces.Box(low=0.0, high=1.0,
                                            shape=(6,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.sim.reset_stats()
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        obs = []
        for lab in ["signal", "decoy", "vac"]:
            tot = max(1, self.sim.counts[lab]["total"])
            clicks = self.sim.counts[lab]["clicks"]
            errs = self.sim.counts[lab]["errors"]
            Q = clicks / tot
            E = errs / clicks if clicks > 0 else 0.0
            obs.extend([Q, E])
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        mu_signal, basis_prob = map(float, action)
        actions = {"Alice": {"mu_signal": mu_signal},
                   "Bob": {"basis_prob": basis_prob}}
        info = self.sim.run_episode(actions=actions)
        reward = info["SKR_bits_per_pulse"]
        obs = self._get_obs()
        done = True
        return obs, reward, done, False, info

    def render(self):
        pass
