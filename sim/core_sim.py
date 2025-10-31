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

    def simulate_pulse(self, mu: float, alice_basis: int, alice_bit: int, bob_basis: int, eve_action: Optional[Union[Dict, Attack, List[Attack]]] = None):
        # sample photon number k ~ Poisson(mu)
        k = self.rng.poisson(mu)
        # effective detection efficiency including channel loss
        eff = self.det_eff * self.eta

        # preserve true Alice bit for error calculation
        orig_alice_bit = int(alice_bit)

        # parse and get attack object (if any)
        attack_obj = self._parse_eve_action(eve_action)

        extra_dark_prob = 0.0
        source = "Alice"
        eve_sent_bit = None  # will hold bit resent by Eve if any

        if attack_obj is not None:
            res = attack_obj.modify_state(k=k, eff=eff, alice_basis=alice_basis,
                                          alice_bit=alice_bit, bob_basis=bob_basis)
            # update local variables from attack output but DO NOT overwrite orig_alice_bit
            k = int(res.get("k", k))
            eff = float(res.get("eff", eff))
            source = res.get("source", source)
            extra_dark_prob = float(res.get("extra_dark_prob", 0.0))
            label_override = res.get("label_override", None)

            # retrieve eve's resent bit if present (do not set alice_bit)
            if "eve_bit" in res:
                eve_sent_bit = int(res["eve_bit"])
            timing_qber_delta = float(res.get("timing_qber_delta", 0.0))
        else:
            timing_qber_delta = 0.0

        # photon-induced click probability (assuming on-off detectors and independent photons)
        photon_survival_prob = 1.0 - (1.0 - eff) ** k if k > 0 else 0.0
        click_from_photons = self.rng.rand() < photon_survival_prob if photon_survival_prob > 0 else False

        # dark counts (two detectors -> combined probability approx)
        combined_dark = 1.0 - (1.0 - self.dark_count - extra_dark_prob) ** 2
        combined_dark = min(max(0.0, combined_dark), 1.0)
        dark_click = self.rng.rand() < combined_dark
        click = click_from_photons or dark_click

        if not click:
            return False, None, False,None

        # Decide detected bit
        if click_from_photons:
            if source == "Eve" and eve_sent_bit is not None:
                # Use the bit Eve actually resent as the detected bit (no "baseline_qber" for Eve's prepared pulse).
                detected_bit = eve_sent_bit
            else:
                # Photons come from Alice (no Eve resend) — apply baseline QBER noise to the real Alice bit
                local_qber = min(0.5, max(0.0, self.baseline_qber + timing_qber_delta))
                detected_bit = orig_alice_bit if self.rng.rand() > local_qber else 1 - orig_alice_bit
        else:
            # dark click: random bit value
            detected_bit = int(self.rng.randint(0, 2))

        # Compute is_error against original Alice bit (this is the correct definition of QBER)
        is_error = (detected_bit != orig_alice_bit)

        # wrong-basis handling: if bases differ we still consider this event a click that will be sifted out later.
        # If you want to model wrong-basis outcomes separately, keep the above error calc (which compares to orig_alice_bit).
        # Note: sifting (alice_basis == bob_basis) is applied later in run_episode.
        if alice_basis != bob_basis:
            # Optionally adjust detected_bit/is_error in a way that reflects measurement randomness —
            # but keep is_error as comparing to orig_alice_bit so QBER stays correct.
            if self.rng.rand() < 0.5:
                # randomize detected bit outcome when wrong basis
                detected_bit = 1 - detected_bit if self.rng.rand() < 0.5 else detected_bit
                is_error = (detected_bit != orig_alice_bit)
            else:
                # keep as-is
                is_error = (detected_bit != orig_alice_bit)

        return True, detected_bit, is_error, label_override



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

            # NEW: unpack 4-tuple (simulate_pulse returns label_override now)
            click, detected_bit, is_error, label_override = self.simulate_pulse(mu, alice_basis, alice_bit, bob_basis, eve_action=eve_action)

            # increment total pulses for the original label (counts["signal"]["total"], etc.)
            self.counts[label]["total"] += 1

            # Only consider clicks where bases match (sifting)
            if click and alice_basis == bob_basis:
                # If attack explicitly marked this pulse as an Eve-resend, count it separately
                if label_override == "eve_resend":
                    # route to eve_resend bucket only (do NOT add into counts[label])
                    self.counts["eve_resend"]["total"] += 1
                    self.counts["eve_resend"]["clicks"] += 1
                    if is_error:
                        self.counts["eve_resend"]["errors"] += 1
                        self.eve_resend_errors += 1
                        self.qber_window.append(1)
                    else:
                        self.qber_window.append(0)
                    # per-label bookkeeping (how many Eve resends originated from this label)
                    self.eve_resend_per_label[label]["total"] += 1
                    self.eve_resend_per_label[label]["clicks"] += 1
                    # run-level counters
                    self.eve_resend_total += 1
                    self.eve_resend_clicks += 1
                else:
                    # Normal, legitimate pulse — count under its label
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
        "Q_s": Q["signal"],
        "E_s": E["signal"],
        "Q_d": Q["decoy"],
        "Q_v": Q["vac"],
        "Y1_lower": Y1,
        "e1_upper": e1,
        "Q1_lower": Q1,
        # — explicit internals for logging/diagnosis —
        "Y1": Y1,
        "e1": e1,
        "Q1": Q1,
        "Q_s_signal": Q["signal"],
        # Eve-resend stats (per-run)
        "eve_resend_total": self.eve_resend_total,
        "eve_resend_clicks": self.eve_resend_clicks,
        "eve_resend_errors": self.eve_resend_errors,
        # keep SKR outputs
        "SKR_bits_per_pulse": skr,
        "SKR_bits_per_second": skr * self.pulse_rate
    }

        info["eve_resend_from_signal_total"] = self.eve_resend_per_label["signal"]["total"]
        info["eve_resend_from_decoy_total"]  = self.eve_resend_per_label["decoy"]["total"]
        info["eve_resend_from_vac_total"]    = self.eve_resend_per_label["vac"]["total"]




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
