"""
env.py — Minimal BB84 + decoy-state environment skeleton.

Usage:
    python env.py

This implements:
- BB84 prepare/measure with two decoys + vacuum (3 intensities).
- Poisson photon-number sampling for coherent pulses.
- Channel with transmissivity eta and dark count probability per gate.
- Simple detectors (on-off), detection window placeholder (gate width).
- 3-intensity decoy estimation (asymptotic) for single-photon yield and error.
- Asymptotic SKR formula:
    R >= q * [ -Q_mu * f(E_mu) * H2(E_mu) + Q1 * (1 - H2(e1)) ]

Notes:
- This is for simulation & RL environment skeleton. Not optimized for speed.
"""

import numpy as np
from math import exp, log2
from collections import deque

# -------------------------
# Utility functions
# -------------------------
def H2(x):
    if x <= 0 or x >= 1:
        return 0.0 if x == 0 else 1.0
    return -(x * log2(x) + (1-x) * log2(1-x))

def poisson_pmf(k, mu):
    return np.exp(-mu) * mu**k / np.math.factorial(k)

# -------------------------
# Decoy estimation (asymptotic 3-intensity)
# -------------------------
def decoy_estimates(mu_s, mu_d, mu_v, Q_s, Q_d, Q_v, E_s, E_d, E_v):
    """
    Return lower bound on Y1 (single-photon yield) and upper bound on e1 (single-photon error rate),
    using standard 3-intensity decoy formulas (asymptotic).
    Inputs:
        mu_s, mu_d, mu_v : intensities (mu_v usually 0)
        Q_* : observed gains (counts/pulse) for each intensity
        E_* : observed QBER for each intensity (error rate conditioned on detection)
    Returns: (Y1_lower, e1_upper)
    """
    # Avoid division by zero
    if mu_s == mu_d or mu_d == mu_v or mu_s == mu_v:
        raise ValueError("Decoy intensities must be distinct.")

    # Solve for Y1 lower bound (from Ma et al. PRA 2005 style)
    # Using formulas:
    #   S = (mu_s * e^{mu_s} * Q_d - mu_d * e^{mu_d} * Q_s) / (mu_s * mu_d * (mu_s - mu_d))
    # but safer to implement standard decoy linear algebra approach.
    # We'll use a commonly used bound (simple form).
    exp_s = np.exp(mu_s)
    exp_d = np.exp(mu_d)
    exp_v = np.exp(mu_v)

    # This implementation follows common bounding (note: ensure numerical stability)
    numerator = (mu_s * exp_s * Q_d - mu_d * exp_d * Q_s)
    denom = mu_s * mu_d * (mu_s - mu_d)
    Y1_lower = max(0.0, numerator / denom)

    # Alternative correction using vacuum:
    # Better bound (if vacuum data available)
    # Reference-like bound:
    try:
        Y1_alt = ((mu_s * exp_s * Q_d - mu_d * exp_d * Q_s) - (mu_s - mu_d) * exp_v * Q_v) / (mu_s * mu_d * (mu_s - mu_d))
        Y1_lower = max(Y1_lower, 0.0, Y1_alt)
    except Exception:
        pass

    # Estimate Q1 = e^{-mu} * mu * Y1 (for signal intensity)
    Q1_lower = np.exp(-mu_s) * mu_s * Y1_lower

    # e1 upper bound:
    # E_s * Q_s = e0 * Y0 * e^{-mu_s} + e1 * Q1 + contributions from multi-photon...
    # Use conservative bound:
    if Q1_lower > 0:
        # Upper bound using signal stats
        e1_upper = min(1.0, (E_s * Q_s) / Q1_lower)
    else:
        e1_upper = 0.5  # max ignorance

    return Y1_lower, e1_upper, Q1_lower

# -------------------------
# Main environment
# -------------------------
class BB84Env:
    def __init__(self,
                 pulses_per_episode=20000,
                 mu_signal=0.5,
                 mu_decoy=0.1,
                 mu_vac=0.0,
                 p_signal=0.7,
                 p_decoy=0.2,
                 p_vac=0.1,
                 eta=0.1,                # channel transmittance
                 dark_count=1e-6,        # dark count prob per gate (per detector)
                 det_eff=0.6,            # detector efficiency (intrinsic)
                 baseline_QBER=0.0,
                 reconciliation_eff=1.15,
                 basis_prob=0.5,
                 seed=None):
        self.rng = np.random.RandomState(seed)
        self.pulses_per_episode = pulses_per_episode
        # intensities & probabilities
        self.mu_s = mu_signal
        self.mu_d = mu_decoy
        self.mu_v = mu_vac
        self.p_signal = p_signal
        self.p_decoy = p_decoy
        self.p_vac = p_vac
        assert abs(p_signal + p_decoy + p_vac - 1.0) < 1e-6

        # channel / detectors
        self.eta = eta
        self.dark = dark_count
        self.det_eff = det_eff
        self.baseline_QBER = baseline_QBER
        self.recon_eff = reconciliation_eff
        self.basis_prob = basis_prob  # prob of Z basis (key basis)
        # stats
        self.reset_stats()

    def reset_stats(self):
        self.counts = {
            'signal': {'clicks': 0, 'errors': 0},
            'decoy': {'clicks': 0, 'errors': 0},
            'vac': {'clicks': 0, 'errors': 0},
        }
        self.total_pulses = {'signal': 0, 'decoy': 0, 'vac': 0}
        self.qber_window = deque(maxlen=10000)

    def reset(self):
        self.reset_stats()
        return self.get_obs()

    def get_obs(self):
        # Observations: recent rates & QBERs
        obs = {}
        for label in ['signal', 'decoy', 'vac']:
            tp = self.total_pulses[label]
            clicks = self.counts[label]['clicks']
            errs = self.counts[label]['errors']
            obs[f'{label}_rate'] = clicks / tp if tp > 0 else 0.0
            obs[f'{label}_qber'] = (errs / clicks) if clicks > 0 else 0.0
        obs['total_pulses'] = sum(self.total_pulses.values())
        return obs

    def sample_intensity_label(self):
        r = self.rng.rand()
        if r < self.p_signal:
            return 'signal', self.mu_s
        elif r < self.p_signal + self.p_decoy:
            return 'decoy', self.mu_d
        else:
            return 'vac', self.mu_v

    def simulate_pulse(self, mu, alice_basis, alice_bit, bob_basis, eve_action=None):
        """
        Simulate single pulse transmission and detection.
        eve_action: dict or None. For P0 we allow time-shift paramization:
            {'time_shift': shift_fraction, 'attack': bool}
        Returns: (click, detected_bit or None, is_error (bool))
        """
        # Photon number
        # sample k from Poisson(mu)
        k = self.rng.poisson(mu)
        # If Eve is doing time-shift, she may alter detection efficiency for this pulse:
        eff = self.det_eff * self.eta
        if eve_action and eve_action.get('time_shift', 0.0) > 0:
            # simple model: attacked fraction reduces/increases effective efficiency for certain time bins
            if eve_action.get('attack', False) and self.rng.rand() < eve_action.get('attack_fraction', 1.0):
                eff *= (1.0 - eve_action['time_shift'])  # reduce effective efficiency
        # Detection: clicks if at least one photon survives
        photon_survival_prob = 1.0 - (1.0 - eff)**k if k > 0 else 0.0
        click_from_photons = self.rng.rand() < photon_survival_prob
        # dark counts (per gate), two detectors — but treat as combined single-click probability
        dark_click = self.rng.rand() < (1 - (1 - self.dark)**2)
        click = click_from_photons or dark_click

        if not click:
            return False, None, False

        # If click, decide which detector and bit value — for on-off simple model, assume correct detector with prob (1 - baseline_QBER)
        if photon_survival_prob > 0:
            # photon-induced click
            detected_bit = alice_bit if self.rng.rand() > self.baseline_QBER else 1 - alice_bit
            is_error = (detected_bit != alice_bit)
        else:
            # dark-click: random bit
            detected_bit = self.rng.randint(0,2)
            is_error = (detected_bit != alice_bit)

        # If bases mismatch, Bob's detection is not correlated — in BB84 real detector measures in chosen basis.
        if alice_basis != bob_basis:
            # measurement in wrong basis: randomize result completely (50% error)
            is_error = self.rng.rand() < 0.5

        return True, detected_bit, is_error

    def step(self, actions):
        """
        actions: dict with keys 'Alice', 'Bob', 'Eve' actions.
        For initial baseline, pass None for Eve.
        Alice actions may contain updated intensities/probabilities.
        Bob actions may contain gate_width / alarm_threshold (not used in base simulation).
        Returns: observation, reward_dict (placeholder), done, info (includes SKR summary at episode end)
        """
        # apply parameter updates if present
        A = actions.get('Alice', {})
        B = actions.get('Bob', {})
        E = actions.get('Eve', {})

        # optional updates
        self.mu_s = A.get('mu_s', self.mu_s)
        self.mu_d = A.get('mu_d', self.mu_d)
        self.mu_v = A.get('mu_v', self.mu_v)
        # normalize probabilities if provided
        if 'p_signal' in A or 'p_decoy' in A or 'p_vac' in A:
            p_sig = A.get('p_signal', self.p_signal)
            p_dec = A.get('p_decoy', self.p_decoy)
            p_vac = A.get('p_vac', self.p_vac)
            s = p_sig + p_dec + p_vac
            if s > 0:
                self.p_signal, self.p_decoy, self.p_vac = p_sig/s, p_dec/s, p_vac/s

        # simulate episode pulses
        # for i in range(self.pulses_per_episode):
        #     label, mu = self.sample_intensity_label()
        #     # random basis choices
        #     alice_basis = 0 if self.rng.rand() < self.basis_prob else 1
        #     bob_basis = 0 if self.rng.rand() < self.basis_prob else 1
        #     alice_bit = self.rng.randint(0,2)

        #     click, detected_bit, is_error = self.simulate_pulse(mu, alice_basis, alice_bit, bob_basis, eve_action=E)
        #     self.total_pulses[label] += 1
        #     if click:
        #         self.counts[label]['clicks'] += 1
        #         if is_error:
        #             self.counts[label]['errors'] += 1
        #             self.qber_window.append(1)
        #         else:
        #             self.qber_window.append(0)
        # simulate episode pulses
        for i in range(self.pulses_per_episode):
            label, mu = self.sample_intensity_label()
            # random basis choices
            alice_basis = 0 if self.rng.rand() < self.basis_prob else 1
            bob_basis   = 0 if self.rng.rand() < self.basis_prob else 1
            alice_bit   = self.rng.randint(0, 2)

            click, detected_bit, is_error = self.simulate_pulse(
                mu, alice_basis, alice_bit, bob_basis, eve_action=E
            )
            self.total_pulses[label] += 1

            # ✅ only count when Alice and Bob used the same basis (sifting)
            if click and alice_basis == bob_basis:
                self.counts[label]['clicks'] += 1
                if is_error:
                    self.counts[label]['errors'] += 1
                    self.qber_window.append(1)
                else:
                    self.qber_window.append(0)


        # After the episode, compute gains & QBERs per intensity
        Q_s = self.counts['signal']['clicks'] / max(1, self.total_pulses['signal'])
        Q_d = self.counts['decoy']['clicks'] / max(1, self.total_pulses['decoy'])
        Q_v = self.counts['vac']['clicks'] / max(1, self.total_pulses['vac'])

        E_s = (self.counts['signal']['errors'] / self.counts['signal']['clicks']) if self.counts['signal']['clicks'] > 0 else 0.0
        E_d = (self.counts['decoy']['errors'] / self.counts['decoy']['clicks']) if self.counts['decoy']['clicks'] > 0 else 0.0
        E_v = (self.counts['vac']['errors'] / self.counts['vac']['clicks']) if self.counts['vac']['clicks'] > 0 else 0.0

        Y1, e1, Q1 = decoy_estimates(self.mu_s, self.mu_d, self.mu_v, Q_s, Q_d, Q_v, E_s, E_d, E_v)
        skr = self.compute_skr(Q_s, E_s, Q1, e1)

        obs = self.get_obs()
        done = True  # episode done after pulses_per_episode
        reward = {'Alice': skr, 'Bob': skr, 'Eve': -skr}  # placeholder: team reward = SKR
        info = {
            'Q_s': Q_s, 'E_s': E_s, 'Q_d': Q_d, 'Q_v': Q_v,
            'Y1_lower': Y1, 'e1_upper': e1, 'Q1_lower': Q1,
            'SKR_bits_per_pulse': skr
        }
        return obs, reward, done, info

    def compute_skr(self, Q_mu, E_mu, Q1, e1):
        """
        Asymptotic key rate per pulse (Devetak-Winter style, with decoy-state single-photon estimate):
        R >= q * [ - Q_mu * f(E_mu) * H2(E_mu) + Q1 * (1 - H2(e1)) ]
        where q is basis sifting (we set q = basis_prob if only Z basis used for key; otherwise 0.5).
        """
        q = self.basis_prob  # approximate sifting factor
        f = self.recon_eff
        # Ensure numeric bounds
        E_mu = max(0.0, min(0.5, E_mu))
        e1 = max(0.0, min(0.5, e1))
        term1 = - Q_mu * f * H2(E_mu)
        term2 = Q1 * (1 - H2(e1))
        R = q * (term1 + term2)
        # No negative SKR
        return max(0.0, R)

# -------------------------
# Quick test
# -------------------------
if __name__ == "__main__":
    # env = BB84Env(pulses_per_episode=100000,
    #               mu_signal=0.8,
    #               mu_decoy=0.1,
    #               mu_vac=0.0,
    #               p_signal=0.8, p_decoy=0.15, p_vac=0.05,
    #               eta=0.2, dark_count=1e-7,
    #               det_eff=0.6,
    #               baseline_QBER=0.0,
    #               reconciliation_eff=1.16,
    #               basis_prob=0.5,
    #               seed=42)

    env = BB84Env(
    pulses_per_episode=200000,
    mu_signal=0.6,
    mu_decoy=0.1,
    mu_vac=0.0,
    p_signal=0.6, p_decoy=0.3, p_vac=0.1,
    eta=0.1, dark_count=1e-7, det_eff=0.6,
    baseline_QBER=0.005, reconciliation_eff=1.16,
    basis_prob=0.5, seed=12345
)



    env.reset()
    # static baseline actions (no Eve)
    actions = {'Alice': {}, 'Bob': {}, 'Eve': None}
    obs, reward, done, info = env.step(actions)
    print("Results (baseline):")
    for k,v in info.items():
        print(f"  {k}: {v}")
    print("Estimated SKR (bits/pulse):", info['SKR_bits_per_pulse'])
