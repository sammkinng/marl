"""
sims/time_shift_eve.py

Simple Time-Shift Eve module for the BB84 + decoy environment.

Usage:
    from sims.time_shift_eve import TimeShiftEve
    eve = TimeShiftEve(attack_fraction=0.5, time_shift_amount=0.5, attack_prob=1.0, seed=42)
    action = eve.act()   # returns dictionary {'time_shift': ..., 'attack_fraction': ..., 'attack': True/False}
    obs, reward, done, info = env.step({'Alice': ..., 'Bob': ..., 'Eve': action})

Design notes:
- The module is intentionally simple and returns a single dict describing Eve's global
  behavior for the episode. The env (env.py) uses that dict to modify effective detection
  efficiency per pulse as a function of the attack parameters.
- Parameters:
    attack_fraction: fraction of pulses Eve attempts to attack (0..1).
    time_shift_amount: fraction by which attacked pulses' effective detection efficiency
                       is reduced (0..1). Example: 0.5 reduces efficiency by 50%.
    attack_prob: probability that Eve chooses to attack this episode at all. (Supports
                 stochastic opponents / curriculum.)
- You can extend the class to produce per-pulse actions (a list) if you later adapt the env
  to accept per-pulse Eve choices.
"""

from dataclasses import dataclass
import numpy as np
from typing import Dict, Any


@dataclass
class TimeShiftEve:
    attack_fraction: float = 0.5       # fraction of pulses to attack (0..1)
    time_shift_amount: float = 0.5     # how much to reduce detection efficiency (0..1)
    attack_prob: float = 1.0           # probability to perform attack this episode
    seed: int | None = None

    def __post_init__(self):
        assert 0.0 <= self.attack_fraction <= 1.0, "attack_fraction in [0,1]"
        assert 0.0 <= self.time_shift_amount <= 1.0, "time_shift_amount in [0,1]"
        assert 0.0 <= self.attack_prob <= 1.0, "attack_prob in [0,1]"
        self.rng = np.random.RandomState(self.seed)

    def act(self) -> Dict[str, Any]:
        """
        Return an action dict consumable by env.step({'Eve': action}).
        Format matches expectations of the `env.py` simulate_pulse signature:
            {'time_shift': <float>, 'attack_fraction': <float>, 'attack': <bool>}
        'attack' indicates whether Eve chooses to attack this episode (stochastic).
        """
        do_attack = self.rng.rand() < self.attack_prob
        if not do_attack:
            return {'time_shift': 0.0, 'attack_fraction': 0.0, 'attack': False}

        return {
            'time_shift': float(self.time_shift_amount),
            'attack_fraction': float(self.attack_fraction),
            'attack': True
        }

    def set_params(self, attack_fraction: float = None, time_shift_amount: float = None, attack_prob: float = None):
        """Update parameters on the fly."""
        if attack_fraction is not None:
            assert 0.0 <= attack_fraction <= 1.0
            self.attack_fraction = attack_fraction
        if time_shift_amount is not None:
            assert 0.0 <= time_shift_amount <= 1.0
            self.time_shift_amount = time_shift_amount
        if attack_prob is not None:
            assert 0.0 <= attack_prob <= 1.0
            self.attack_prob = attack_prob

    def sample_per_pulse_mask(self, n_pulses: int) -> np.ndarray:
        """
        (Optional helper) Return a boolean mask of length n_pulses indicating which pulses
        are attacked (True). Useful if you later move to per-pulse Eve actions.
        """
        if self.attack_fraction <= 0:
            return np.zeros(n_pulses, dtype=bool)
        k = int(np.round(self.attack_fraction * n_pulses))
        mask = np.zeros(n_pulses, dtype=bool)
        if k > 0:
            indices = self.rng.choice(n_pulses, size=k, replace=False)
            mask[indices] = True
        return mask


# -------------------------
# Quick demo / sanity check
# -------------------------
if __name__ == "__main__":
    # Demo: run a small env step with and without Eve; print SKR difference.
    import sys
    import os
    # if repo root has env.py, add its parent to path
    here = os.path.dirname(__file__)
    repo_root = os.path.abspath(os.path.join(here, ".."))
    sys.path.insert(0, repo_root)

    try:
        from prev_versions.env import BB84Env
    except Exception as e:
        print("Could not import BB84Env from env.py in repo root. Make sure env.py is at repo root.")
        raise

    # baseline env
    env = BB84Env(pulses_per_episode=10000,
                  mu_signal=0.5, mu_decoy=0.1, mu_vac=0.0,
                  p_signal=0.6, p_decoy=0.3, p_vac=0.1,
                  eta=0.1, dark_count=1e-6, det_eff=0.6,
                  baseline_QBER=0.01, reconciliation_eff=1.16,
                  basis_prob=0.5, seed=1)

    env.reset()
    # baseline run (no Eve)
    obs, reward, done, info = env.step({'Alice': {}, 'Bob': {}, 'Eve': None})
    print("Baseline run:")
    print(f"  SKR = {info['SKR_bits_per_pulse']:.6f}, Q_s = {info['Q_s']:.6e}, E_s = {info['E_s']:.6f}")

    # run with TimeShiftEve
    eve = TimeShiftEve(attack_fraction=0.5, time_shift_amount=0.5, attack_prob=1.0, seed=2)
    env.reset()
    eve_action = eve.act()
    obs_e, reward_e, done_e, info_e = env.step({'Alice': {}, 'Bob': {}, 'Eve': eve_action})
    print("With Time-Shift Eve:")
    print(f"  Eve action: {eve_action}")
    print(f"  SKR = {info_e['SKR_bits_per_pulse']:.6f}, Q_s = {info_e['Q_s']:.6e}, E_s = {info_e['E_s']:.6f}")

    print("\nRelative change (Eve vs baseline):")
    base_skr = info['SKR_bits_per_pulse']
    atk_skr = info_e['SKR_bits_per_pulse']
    pct = (atk_skr - base_skr) / base_skr * 100 if base_skr > 0 else float('inf')
    print(f"  SKR change: {pct:.2f}%")
