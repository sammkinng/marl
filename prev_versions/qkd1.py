# core_simulator.py
# -*- coding: utf-8 -*-
"""
Core QKD Simulator (refactored Eve/attack modularization)
------------------
Implements decoy-state BB84 with:
- WCP (Poissonian) photon statistics
- Fiber attenuation or direct eta
- Detector efficiency, dark counts
- Modular attack classes for time-shift, PNS, intercept-resend, dark-count noise
- Decoy estimation (Y1, e1) and SKR computation

This module contains only physics & simulation logic.
"""

import numpy as np
from numba import njit, prange

from collections import deque

from sim.attacks import Attack, CompositeAttack
from .utils import H2, decoy_estimates
from typing import Dict, Tuple, Optional, List, Union



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

    def _can_vectorize_eve_action(self, eve_action):
        """
        Return True if eve_action is a dict or list-of-dicts composed only of
        supported simple attack types that can be applied in vectorized form.
        Otherwise return False (fall back to scalar loop).
        """
        if eve_action is None:
            return True
        # Accept either dict or list/tuple of dicts
        if isinstance(eve_action, dict):
            actions = [eve_action]
        elif isinstance(eve_action, (list, tuple)):
            actions = list(eve_action)
        else:
            # If it's an Attack instance or unknown, we cannot vectorize safely
            return False

        supported = {"time_shift", "pns", "intercept_resend", "dark_count", "composite", "none"}
        for a in actions:
            if not isinstance(a, dict):
                return False
            t = a.get("type", "none")
            if t not in supported:
                return False
            if t == "composite":
                # make sure all sub_attacks are dicts and supported
                subs = a.get("sub_attacks", [])
                for s in subs:
                    if not isinstance(s, dict) or s.get("type", "") not in supported:
                        return False
        return True


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
        # attack config (placeholders)  -- kept for backward compatibility
        self.attack_cfg = c.get("attack", {"type": "none"})
        # bookkeeping
        self.labels = ["signal", "decoy", "vac"]
        # limits
        self.max_double_clicks_tolerated = int(c.get("max_double_clicks_tolerated", 1000))

    def reset_stats(self):
        # existing per-label counts (signal/decoy/vac)
        self.counts = {lab: {"clicks": 0, "errors": 0, "total": 0} for lab in self.labels}
        # NEW: bucket for Eve-resend pulses (separate from legitimate labels)
        self.counts["eve_resend"] = {"clicks": 0, "errors": 0, "total": 0}
        # NEW: keep per-original-label counts of Eve resends (useful to subtract if needed)
        self.eve_resend_per_label = {lab: {"total": 0, "clicks": 0, "errors": 0} for lab in self.labels}
        # run-level counters for easy logging
        self.eve_resend_total = 0
        self.eve_resend_clicks = 0
        self.eve_resend_errors = 0

        # existing state
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

    def _parse_eve_action(self, eve_action: Optional[Union[Dict, Attack, List[Attack]]]) -> Optional[Attack]:
        """
        Convert eve_action (dict or Attack or list) into an Attack instance.
        """
        if eve_action is None:
            return None
        if isinstance(eve_action, Attack):
            return eve_action
        if isinstance(eve_action, list) or isinstance(eve_action, tuple):
            # assume list of Attack objects or dicts
            parsed = []
            for ea in eve_action:
                if isinstance(ea, Attack):
                    parsed.append(ea)
                elif isinstance(ea, dict):
                    parsed.append(Attack.from_dict(ea, self.rng))
                else:
                    raise ValueError("Unsupported element in eve_action list")
            return CompositeAttack(parsed, rng=self.rng)
        if isinstance(eve_action, dict):
            return Attack.from_dict(eve_action, self.rng)
        raise ValueError("Unsupported eve_action type")


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


    @njit(parallel=True, fastmath=True)
    def compute_photon_clicks(mu_arr, det_eff_arr, dark_count, extra_dark_arr, rng_vals_ph, rng_vals_dark, k_arr):
        """
        Vectorized photon & dark click computation under JIT.
        Returns click_mask (bool array).
        """
        n = len(mu_arr)
        click_mask = np.zeros(n, dtype=np.bool_)
        for i in prange(n):
            k = k_arr[i]
            eff = det_eff_arr[i]
            if k > 0:
                photon_survival_prob = 1.0 - (1.0 - eff) ** k
            else:
                photon_survival_prob = 0.0
            click_from_photon = rng_vals_ph[i] < photon_survival_prob

            combined_dark = 1.0 - (1.0 - dark_count - extra_dark_arr[i]) ** 2
            dark_click = rng_vals_dark[i] < combined_dark

            if click_from_photon or dark_click:
                click_mask[i] = True
        return click_mask


    @njit(parallel=True, fastmath=True)
    def compute_detected_bits(click_mask, dark_click, click_from_photons, alice_bits, bob_basis_arr,
                            alice_basis_arr, baseline_qber, timing_qber_delta_arr, source_is_eve,
                            eve_sent_bit_arr, rng_vals_qber, rng_vals_darkbit):
        """
        Compute detected bits and error array.
        """
        n = len(click_mask)
        detected_bits = np.zeros(n, dtype=np.int8)
        is_error_arr = np.zeros(n, dtype=np.int8)

        for i in prange(n):
            if not click_mask[i]:
                continue

            # dark clicks -> random bit
            if dark_click[i]:
                detected_bits[i] = 1 if rng_vals_darkbit[i] > 0.5 else 0
            # photon clicks
            elif click_from_photons[i]:
                if source_is_eve[i] and eve_sent_bit_arr[i] != -1:
                    detected_bits[i] = eve_sent_bit_arr[i]
                else:
                    local_qber = min(max(baseline_qber + timing_qber_delta_arr[i], 0.0), 0.5)
                    if rng_vals_qber[i] > local_qber:
                        detected_bits[i] = alice_bits[i]
                    else:
                        detected_bits[i] = 1 - alice_bits[i]

            # Wrong basis randomization can be handled outside for simplicity
            is_error_arr[i] = 1 if detected_bits[i] != alice_bits[i] else 0

        return detected_bits, is_error_arr


    # ======================
    # 2. Main run_episode()
    # ======================
    def run_episode(self, actions: dict = None, verbose: bool = False) -> dict:
        self.reset_stats()
        self.episode += 1

        # (same Alice/Eve action handling as before)
        if actions and "Alice" in actions:
            A = actions["Alice"]
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
        if not self._can_vectorize_eve_action(eve_action):
            # fallback path unchanged
            ...
            return info

        n = int(self.pulses_per_episode)
        rng = self.rng

        # --- same setup code as your current vectorized version ---
        r = rng.rand(n)
        signal_mask = r < self.p_signal
        decoy_mask = (r >= self.p_signal) & (r < self.p_signal + self.p_decoy)
        vac_mask = ~(signal_mask | decoy_mask)

        mu_arr = np.empty(n, dtype=np.float64)
        mu_arr[signal_mask] = self.mu_signal
        mu_arr[decoy_mask] = self.mu_decoy
        mu_arr[vac_mask] = self.mu_vac

        labels_idx = np.empty(n, dtype=np.int8)
        labels_idx[signal_mask] = 0
        labels_idx[decoy_mask] = 1
        labels_idx[vac_mask] = 2

        alice_basis_arr = (rng.rand(n) < self.basis_prob).astype(np.int8)
        bob_basis_arr   = (rng.rand(n) < self.basis_prob).astype(np.int8)
        alice_bits      = rng.randint(0, 2, size=n).astype(np.int8)
        k_arr           = rng.poisson(mu_arr)

        eff_base = self.det_eff * self.eta
        det_eff_arr = np.full(n, eff_base)
        extra_dark_arr = np.zeros(n)
        timing_qber_delta_arr = np.zeros(n)
        eve_sent_bit_arr = np.full(n, -1, dtype=np.int8)
        source_is_eve = np.zeros(n, dtype=np.bool_)

        # (same Eve attack application logic as before)
        if eve_action is not None:
            if isinstance(eve_action, dict):
                self._apply_vectorized_attack_dict(eve_action, n, rng,
                                                det_eff_arr, extra_dark_arr,
                                                k_arr, alice_bits, alice_basis_arr,
                                                eve_sent_bit_arr, source_is_eve, timing_qber_delta_arr)
            elif isinstance(eve_action, (list, tuple)):
                for ad in eve_action:
                    self._apply_vectorized_attack_dict(ad, n, rng,
                                                    det_eff_arr, extra_dark_arr,
                                                    k_arr, alice_bits, alice_basis_arr,
                                                    eve_sent_bit_arr, source_is_eve, timing_qber_delta_arr)

        # --- Pre-generate RNG arrays for JIT section ---
        rng_vals_ph = rng.rand(n)
        rng_vals_dark = rng.rand(n)
        rng_vals_qber = rng.rand(n)
        rng_vals_darkbit = rng.rand(n)

        # ---  JIT accelerated photon+dark click calc ---
        click_mask = self.compute_photon_clicks(mu_arr, det_eff_arr, self.dark_count,
                                        extra_dark_arr, rng_vals_ph, rng_vals_dark, k_arr)

        click_from_photons = rng_vals_ph < (1.0 - np.power(1.0 - det_eff_arr, k_arr))
        dark_click = rng_vals_dark < (1.0 - np.power(1.0 - self.dark_count - extra_dark_arr, 2))

        # ---  JIT accelerated detection + error calc ---
        detected_bits, is_error_arr = self.compute_detected_bits(
            click_mask, dark_click, click_from_photons,
            alice_bits, bob_basis_arr, alice_basis_arr,
            self.baseline_qber, timing_qber_delta_arr,
            source_is_eve, eve_sent_bit_arr, rng_vals_qber, rng_vals_darkbit
        )

        # (rest identical: counts, sifted stats, SKR calc, etc.)
        # ↓ reuse your existing end-part unchanged ↓
        totals = np.bincount(labels_idx, minlength=3)
        for idx_label, lab in enumerate(self.labels):
            self.counts[lab]["total"] += int(totals[idx_label])

        sift_mask = click_mask & (alice_basis_arr == bob_basis_arr)
        sift_idxs = np.nonzero(sift_mask)[0]
        for idx in sift_idxs:
            lab = self.labels[labels_idx[idx]]
            self.counts[lab]["clicks"] += 1
            self.counts[lab]["errors"] += int(is_error_arr[idx])
            self.qber_window.append(int(is_error_arr[idx]))

        Q = {}
        E = {}
        for lab in self.labels:
            tot = max(1, self.counts[lab]["total"])
            clicks = self.counts[lab]["clicks"]
            errs = self.counts[lab]["errors"]
            Q[lab] = clicks / tot
            E[lab] = (errs / clicks) if clicks > 0 else 0.0

        Y1, e1, Q1 = decoy_estimates(
            self.mu_signal, self.mu_decoy, self.mu_vac,
            Q["signal"], Q["decoy"], Q["vac"],
            E["signal"], E["decoy"], E["vac"]
        )
        skr = self.compute_skr(Q["signal"], E["signal"], Q1, e1)

        info = {
            "Q_s": Q["signal"], "E_s": E["signal"],
            "Q_d": Q["decoy"], "Q_v": Q["vac"],
            "Y1_lower": Y1, "e1_upper": e1,
            "Q1_lower": Q1, "SKR_bits_per_pulse": skr,
            "SKR_bits_per_second": skr * self.pulse_rate
        }

        if verbose:
            print(f"[Episode {self.episode}] SKR={info['SKR_bits_per_pulse']:.6e} bits/pulse, SKR={info['SKR_bits_per_second']:.3f} bits/s")
        return info
