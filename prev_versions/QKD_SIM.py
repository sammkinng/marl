# -*- coding: utf-8 -*-
"""
qkd_env_configurable.py

A configurable BB84 / decoy-state simulator that loads parameters from a YAML (or JSON) config file.
Designed to be modular and easy to extend for MARL experiments.
- Supports: WCP (Poisson), channel loss (dB/km), detector efficiency, dark counts, decoy estimation (3-intensity),
  simple attacks placeholders (time-shift, PNS fraction), and episode-level batching.
- If PyYAML is not installed, JSON configs are accepted as fallback.
- Save results and prints summary info.

Usage:
    1) Install dependencies:
         pip install numpy pyyaml  # pyyaml optional if you use JSON config
    2) Prepare a YAML config (example written by main() below)
    3) Run this script or import QKDSimulator into your project.

This file will write an example config 'qkd_config_example.yaml' and run a short demo episode.
"""

import math
import json
import numpy as np
from collections import deque
from typing import Dict, Tuple

# --- Config loader (YAML optional) ---
def load_config(path: str) -> Dict:
    """
    Load YAML or JSON config. If PyYAML not installed and file is YAML, fall back to a simple parser
    for a limited subset (key: value, lists as [a,b]). But recommended: pip install pyyaml.
    """
    try:
        import yaml  # type: ignore
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except Exception:
        # Fallback: try JSON
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load config: {e}")


# --- Utilities ---
def H2(x: float) -> float:
    x = max(0.0, min(1.0, x))
    if x == 0.0 or x == 1.0:
        return 0.0
    return -(x * math.log2(x) + (1 - x) * math.log2(1 - x))


# Decoy estimation helper (same conservative/asymptotic idea)
def decoy_estimates(mu_s, mu_d, mu_v, Q_s, Q_d, Q_v, E_s, E_d, E_v):
    # Use simple linear bounds (suitable for demonstration)
    if abs(mu_s - mu_d) < 1e-12:
        return 0.0, 0.5, 0.0
    exp_s, exp_d, exp_v = math.exp(mu_s), math.exp(mu_d), math.exp(mu_v)
    numerator = (mu_s * exp_s * Q_d - mu_d * exp_d * Q_s)
    denom = mu_s * mu_d * (mu_s - mu_d)
    Y1_lower = max(0.0, numerator / denom) if denom != 0.0 else 0.0
    # Alternative using vacuum
    try:
        Y1_alt = ((mu_s * exp_s * Q_d - mu_d * exp_d * Q_s) - (mu_s - mu_d) * exp_v * Q_v) / (mu_s * mu_d * (mu_s - mu_d))
        Y1_lower = max(Y1_lower, 0.0, Y1_alt)
    except Exception:
        pass
    Q1_lower = math.exp(-mu_s) * mu_s * Y1_lower
    if Q1_lower > 0:
        e1_upper = min(1.0, (E_s * Q_s) / Q1_lower)
    else:
        e1_upper = 0.5
    return Y1_lower, e1_upper, Q1_lower


# --- Core simulator class ---
class QKDSimulator:
    def __init__(self, config: Dict):
        self.cfg = config
        self._seed_rng(self.cfg.get("seed", 1234))
        self._load_parameters()
        self.reset_stats()

    def _seed_rng(self, seed):
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def _load_parameters(self):
        c = self.cfg
        # protocol / pulses
        self.pulses_per_episode = int(c.get("pulses_per_episode", 200000))
        self.pulse_rate = float(c.get("pulse_rate", 10e6))  # pulses per second for converts
        # intensities
        self.mu_signal = float(c.get("mu_signal", 0.5))
        self.mu_decoy = float(c.get("mu_decoy", 0.1))
        self.mu_vac = float(c.get("mu_vac", 0.0))
        self.p_signal = float(c.get("p_signal", 0.7))
        self.p_decoy = float(c.get("p_decoy", 0.2))
        self.p_vac = float(c.get("p_vac", 0.1))
        # channel
        # Accept either eta directly or fiber_alpha (dB/km) + distance
        if "eta" in c:
            self.eta = float(c["eta"])
        else:
            alpha_db_per_km = float(c.get("fiber_loss_db_per_km", 0.2))
            L_km = float(c.get("distance_km", 10.0))
            self.eta = 10 ** (-alpha_db_per_km * L_km / 10.0)
        # detectors
        self.det_eff = float(c.get("det_eff", 0.6))
        self.dark_count = float(c.get("dark_count", 1e-7))
        self.baseline_qber = float(c.get("baseline_qber", 0.0))
        self.recon_eff = float(c.get("recon_eff", 1.15))
        self.basis_prob = float(c.get("basis_prob", 0.5))
        # attack config (placeholders)
        self.attack_cfg = c.get("attack", {"type": "none"})
        # bookkeeping
        self.labels = ["signal", "decoy", "vac"]
        # limits
        self.max_double_clicks_tolerated = int(c.get("max_double_clicks_tolerated", 1000))

    def reset_stats(self):
        self.counts = {lab: {"clicks": 0, "errors": 0, "total": 0} for lab in self.labels}
        self.qber_window = deque(maxlen=100000)
        self.episode = 0

    def sample_label(self) -> Tuple[str, float]:
        r = self.rng.rand()
        if r < self.p_signal:
            return "signal", self.mu_signal
        elif r < self.p_signal + self.p_decoy:
            return "decoy", self.mu_decoy
        else:
            return "vac", self.mu_vac

    def simulate_pulse(self, mu: float, alice_basis: int, alice_bit: int, bob_basis: int, eve_action: Dict = None):
        """
        Simulate one pulse with WCP photon number sampling and simple detection/dark count.
        Returns: click (bool), detected_bit (0/1 or None), is_error (bool)
        """
        # sample photon number k ~ Poisson(mu)
        k = self.rng.poisson(mu)
        # effective detection efficiency including channel loss
        eff = self.det_eff * self.eta
        # allow Eve to modify eff or to perform PNS (attack simulation)
        if eve_action and isinstance(eve_action, dict):
            if eve_action.get("type") == "time_shift":
                # time_shift: reduce eff by fraction when attack flag set
                if self.rng.rand() < eve_action.get("attack_prob", 0.0):
                    eff *= (1.0 - float(eve_action.get("shift_frac", 0.2)))
            elif eve_action.get("type") == "pns":
                # pns: if k>1 and Eve chooses to hold one photon with prob pns_frac, Bob sees k-1 photons
                pns_frac = float(eve_action.get("pns_frac", 0.0))
                if k > 1 and self.rng.rand() < pns_frac:
                    # Eve steals one photon -> decrease k by 1 (approximate)
                    k = max(0, k - 1)

        # photon-induced click probability (assuming on-off detectors and independent photons)
        photon_survival_prob = 1.0 - (1.0 - eff) ** k if k > 0 else 0.0
        click_from_photons = self.rng.rand() < photon_survival_prob if photon_survival_prob > 0 else False
        # dark counts (two detectors -> combined probability approx)
        dark_click = self.rng.rand() < (1.0 - (1.0 - self.dark_count) ** 2)
        click = click_from_photons or dark_click

        if not click:
            return False, None, False

        # Decide detected bit
        if click_from_photons:
            detected_bit = alice_bit if self.rng.rand() > self.baseline_qber else 1 - alice_bit
            is_error = (detected_bit != alice_bit)
        else:
            # dark click: random bit value
            detected_bit = int(self.rng.randint(0, 2))
            is_error = (detected_bit != alice_bit)

        # wrong-basis handling: if bases differ, measurement outcome is random (50% error on average)
        if alice_basis != bob_basis:
            if self.rng.rand() < 0.5:
                is_error = True
                detected_bit = 1 - alice_bit if self.rng.rand() < 0.5 else detected_bit
            else:
                is_error = False
                detected_bit = alice_bit if self.rng.rand() < 0.5 else detected_bit

        return True, detected_bit, is_error

    def run_episode(self, actions: Dict = None, verbose: bool = False) -> Dict:
        """
        Run one episode (pulses_per_episode). actions: dict containing 'Alice','Bob','Eve' updates (optional).
        Returns info dict with per-intensity gains and SKR estimate.
        """
        self.reset_stats()
        self.episode += 1
        # update params if present
        if actions and "Alice" in actions:
            A = actions["Alice"]
            # allow dynamic update of mu and probabilities
            if "mu_signal" in A: self.mu_signal = float(A["mu_signal"])
            if "mu_decoy" in A: self.mu_decoy = float(A["mu_decoy"])
            if "p_signal" in A or "p_decoy" in A or "p_vac" in A:
                p_sig = float(A.get("p_signal", self.p_signal))
                p_dec = float(A.get("p_decoy", self.p_decoy))
                p_vac = float(A.get("p_vac", self.p_vac))
                s = p_sig + p_dec + p_vac
                if s > 0:
                    self.p_signal, self.p_decoy, self.p_vac = p_sig/s, p_dec/s, p_vac/s
        eve_action = (actions or {}).get("Eve", None)

        # simulate pulses
        for i in range(self.pulses_per_episode):
            label, mu = self.sample_label()
            alice_basis = 0 if self.rng.rand() < self.basis_prob else 1
            bob_basis = 0 if self.rng.rand() < self.basis_prob else 1
            alice_bit = int(self.rng.randint(0, 2))
            click, detected_bit, is_error = self.simulate_pulse(mu, alice_basis, alice_bit, bob_basis, eve_action=eve_action)
            self.counts[label]["total"] += 1
            # sifting: only count when bases match and click occurred
            if click and alice_basis == bob_basis:
                self.counts[label]["clicks"] += 1
                if is_error:
                    self.counts[label]["errors"] += 1
                    self.qber_window.append(1)
                else:
                    self.qber_window.append(0)

        # compute gains and QBERs
        Q = {}
        E = {}
        for lab in self.labels:
            tot = max(1, self.counts[lab]["total"])
            clicks = self.counts[lab]["clicks"]
            errs = self.counts[lab]["errors"]
            Q[lab] = clicks / tot
            E[lab] = (errs / clicks) if clicks > 0 else 0.0

        Y1, e1, Q1 = decoy_estimates(self.mu_signal, self.mu_decoy, self.mu_vac, Q["signal"], Q["decoy"], Q["vac"], E["signal"], E["decoy"], E["vac"])
        skr = self.compute_skr(Q["signal"], E["signal"], Q1, e1)
        info = {
            "Q_s": Q["signal"], "E_s": E["signal"],
            "Q_d": Q["decoy"], "Q_v": Q["vac"],
            "Y1_lower": Y1, "e1_upper": e1, "Q1_lower": Q1,
            "SKR_bits_per_pulse": skr,
            "SKR_bits_per_second": skr * self.pulse_rate
        }
        if verbose:
            print(f"[Episode {self.episode}] SKR={info['SKR_bits_per_pulse']:.6e} bits/pulse, SKR={info['SKR_bits_per_second']:.3f} bits/s")
        return info

    def compute_skr(self, Q_mu, E_mu, Q1, e1, cfg=None):
        """
        Compute the Secure Key Rate (SKR) with optional finite-key correction.
        
        Args:
            Q_mu (float): Overall gain for signal states
            E_mu (float): Overall QBER for signal states
            Q1 (float): Single-photon gain
            e1 (float): Single-photon error rate
            cfg (dict, optional): Configuration dictionary containing:
                - finite_key (bool): Whether to apply finite-key correction
                - pulses_per_episode (int): Number of pulses per episode
                - confidence_level (float): Confidence level for finite-key correction

        Returns:
            float: Corrected SKR per pulse
        """
        q = self.basis_prob
        f = self.recon_eff

        # Clamp error rates to valid range
        E_mu = max(0.0, min(0.5, E_mu))
        e1 = max(0.0, min(0.5, e1))

        # Standard SKR formula
        term1 = - Q_mu * f * H2(E_mu)
        term2 = Q1 * (1 - H2(e1))
        R = q * (term1 + term2)
        R = max(0.0, R)

        # Apply finite-key correction if enabled
        if self.cfg is not None and self.cfg.get('finite_key', False):
            N = self.cfg.get('pulses_per_episode', 1)
            eps = 1 - self.cfg.get('confidence_level', 0.99)
            # Simple finite-key scaling (sqrt bound, can be refined later)
            finite_factor = max(0.0, 1 - (np.log(2/eps)/N)**0.5)
            R *= finite_factor

        return R

