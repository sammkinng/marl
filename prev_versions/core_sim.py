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
from numba import njit

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

    def __simulate_pulse(self, mu: float, alice_basis: int, alice_bit: int, bob_basis: int, eve_action: Optional[Union[Dict, Attack, List[Attack]]] = None):
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


    def simulate_pulse(self, mu: float, alice_basis: int, alice_bit: int, bob_basis: int,
                    eve_action: Optional[Union[Dict, Attack, List[Attack]]] = None):
        """
        Wrapper that uses a Numba-jitted core when possible (no Attack object present).
        Returns: click (bool), detected_bit (0/1 or None), is_error (bool), label_override (str or None)
        """
        # --- 1) Quick path: if there's an Attack *object* (custom Python instance),
        # fall back to original scalar implementation (unmodified) to preserve behavior.
        if isinstance(eve_action, Attack) or isinstance(eve_action, CompositeAttack):
            # Use original code path (as before): call attack object's modify_state etc.
            # We'll reuse original implementation exactly — keep for backwards compatibility.
            # (Original implementation copied here.)
            # --- BEGIN original code ---
            k = self.rng.poisson(mu)
            eff = self.det_eff * self.eta
            orig_alice_bit = int(alice_bit)
            attack_obj = self._parse_eve_action(eve_action)

            extra_dark_prob = 0.0
            source = "Alice"
            eve_sent_bit = None
            label_override = None

            if attack_obj is not None:
                res = attack_obj.modify_state(k=k, eff=eff, alice_basis=alice_basis,
                                            alice_bit=alice_bit, bob_basis=bob_basis)
                k = int(res.get("k", k))
                eff = float(res.get("eff", eff))
                source = res.get("source", source)
                extra_dark_prob = float(res.get("extra_dark_prob", 0.0))
                label_override = res.get("label_override", None)
                if "eve_bit" in res:
                    eve_sent_bit = int(res["eve_bit"])
                timing_qber_delta = float(res.get("timing_qber_delta", 0.0))
            else:
                timing_qber_delta = 0.0

            photon_survival_prob = 1.0 - (1.0 - eff) ** k if k > 0 else 0.0
            click_from_photons = self.rng.rand() < photon_survival_prob if photon_survival_prob > 0 else False

            combined_dark = 1.0 - (1.0 - self.dark_count - extra_dark_prob) ** 2
            combined_dark = min(max(0.0, combined_dark), 1.0)
            dark_click = self.rng.rand() < combined_dark
            click = click_from_photons or dark_click

            if not click:
                return False, None, False, None

            if click_from_photons:
                if source == "Eve" and eve_sent_bit is not None:
                    detected_bit = eve_sent_bit
                else:
                    local_qber = min(0.5, max(0.0, self.baseline_qber + timing_qber_delta))
                    detected_bit = orig_alice_bit if self.rng.rand() > local_qber else 1 - orig_alice_bit
            else:
                detected_bit = int(self.rng.randint(0, 2))

            is_error = (detected_bit != orig_alice_bit)

            if alice_basis != bob_basis:
                if self.rng.rand() < 0.5:
                    detected_bit = 1 - detected_bit if self.rng.rand() < 0.5 else detected_bit
                    is_error = (detected_bit != orig_alice_bit)
                else:
                    is_error = (detected_bit != orig_alice_bit)

            return True, detected_bit, is_error, label_override
            # --- END original code ---

        # --- 2) Fast JIT path (most common): no Attack instance, or eve_action is a simple dict/list we pre-apply ---
        # Pre-apply simple eve_action modifications in Python (so the core gets only numeric values).
        # Start with defaults:
        k = self.rng.poisson(mu)
        eff = self.det_eff * self.eta
        orig_alice_bit = int(alice_bit)

        extra_dark_prob = 0.0
        source = "Alice"
        eve_sent_bit = -1
        label_override = None

        timing_qber_delta = 0.0

        # If eve_action is a dict/list of dicts, apply simple modifications (same logic used elsewhere)
        # NOTE: this is a lightweight Python application (cheap compared to numeric core).
        if eve_action is not None:
            # allow lists or single dict
            actions_list = eve_action if isinstance(eve_action, (list, tuple)) else [eve_action]
            for a in actions_list:
                t = a.get("type", "none")
                if t == "time_shift":
                    ap = float(a.get("attack_prob", 0.0))
                    sf = float(a.get("shift_frac", 0.0))
                    tq = float(a.get("timing_qber_delta", 0.0))
                    if self.rng.rand() < ap:
                        eff = eff * (1.0 - sf)
                        timing_qber_delta += tq
                elif t == "pns":
                    pns_frac = float(a.get("pns_frac", 0.0))
                    if k > 1 and self.rng.rand() < pns_frac:
                        k = max(0, k - 1)
                elif t == "intercept_resend":
                    intercept_prob = float(a.get("intercept_prob", 0.0))
                    resend_eff = float(a.get("resend_eff", 1.0))
                    resend_error_prob = float(a.get("resend_error_prob", 0.0))
                    if self.rng.rand() < intercept_prob:
                        eve_basis = 0 if self.rng.rand() < 0.5 else 1
                        if eve_basis == alice_basis:
                            measured_bit = orig_alice_bit
                        else:
                            measured_bit = int(self.rng.randint(0, 2))
                        if self.rng.rand() < resend_error_prob:
                            measured_bit = 1 - measured_bit
                        eve_sent_bit = int(measured_bit)
                        k = 1
                        eff = eff * resend_eff
                        source = "Eve"
                        label_override = "eve_resend"
                elif t == "dark_count":
                    extra_dark_prob += float(a.get("extra_dark_prob", 0.0))
                elif t == "composite":
                    for s in a.get("sub_attacks", []):
                        # re-run this loop for sub-attack dict
                        stype = s.get("type", "none")
                        # handle subtypes simply (avoid recursion complexity)
                        if stype == "time_shift":
                            if self.rng.rand() < float(s.get("attack_prob", 0.0)):
                                eff = eff * (1.0 - float(s.get("shift_frac", 0.0)))
                                timing_qber_delta += float(s.get("timing_qber_delta", 0.0))
                        elif stype == "pns":
                            if k > 1 and self.rng.rand() < float(s.get("pns_frac", 0.0)):
                                k = max(0, k - 1)
                        elif stype == "intercept_resend":
                            if self.rng.rand() < float(s.get("intercept_prob", 0.0)):
                                eve_basis = 0 if self.rng.rand() < 0.5 else 1
                                if eve_basis == alice_basis:
                                    measured_bit = orig_alice_bit
                                else:
                                    measured_bit = int(self.rng.randint(0, 2))
                                if self.rng.rand() < float(s.get("resend_error_prob", 0.0)):
                                    measured_bit = 1 - measured_bit
                                eve_sent_bit = int(measured_bit)
                                k = 1
                                eff = eff * float(s.get("resend_eff", 1.0))
                                source = "Eve"
                                label_override = "eve_resend"
                        elif stype == "dark_count":
                            extra_dark_prob += float(s.get("extra_dark_prob", 0.0))
                        # else ignore
        # sample randoms to pass into JIT core
        r_photon = self.rng.rand()
        r_dark = self.rng.rand()
        r_qber = self.rng.rand()
        r_wrong_choice = self.rng.rand()
        r_wrong_flip = self.rng.rand()
        r_rand_bit_int = int(self.rng.randint(0, 2))

        # call jitted core
        click_flag, detected_bit, is_error_flag, label_flag = self._simulate_pulse_core_jit(
            k, eff, self.dark_count, extra_dark_prob,
            self.baseline_qber, orig_alice_bit,
            alice_basis, bob_basis,
            eve_sent_bit, 1 if source == "Eve" else 0,
            timing_qber_delta,
            r_photon, r_dark, r_qber, r_wrong_choice, r_wrong_flip, r_rand_bit_int
        )

        if click_flag == 0:
            return False, None, False, None

        # convert results
        click = True
        detected_bit = int(detected_bit)
        is_error = bool(is_error_flag)
        # map label_flag or wrapper-set label_override string
        if label_override is not None:
            # e.g. set by intercept_resend path above
            lo = label_override
        else:
            lo = "eve_resend" if label_flag == 1 else None

        return click, detected_bit, is_error, lo


    @njit(cache=True, fastmath=True)
    def _simulate_pulse_core_jit(k, eff, dark_count, extra_dark_prob,
                                baseline_qber, orig_alice_bit,
                                alice_basis, bob_basis,
                                eve_sent_bit, source_is_eve,
                                timing_qber_delta,
                                r_photon, r_dark, r_qber, r_wrong_choice, r_wrong_flip, r_rand_bit_int):
        """
        Numba-jitted scalar core. All inputs are simple numeric types.
        Returns: click (0/1), detected_bit (0/1 if click else -1), is_error (0/1), label_flag (0 normal, 1 eve_resend)
        """
        # photon-induced click probability
        if k > 0:
            # (1 - eff)^k
            one_minus_eff = 1.0 - eff
            # pow
            survival = 1.0 - one_minus_eff ** k
            photon_survival_prob = survival
        else:
            photon_survival_prob = 0.0

        click_from_photons = 1 if (r_photon < photon_survival_prob) else 0

        combined_dark = 1.0 - (1.0 - dark_count - extra_dark_prob) ** 2
        if combined_dark < 0.0:
            combined_dark = 0.0
        elif combined_dark > 1.0:
            combined_dark = 1.0

        dark_click = 1 if (r_dark < combined_dark) else 0

        click_flag = 1 if (click_from_photons == 1 or dark_click == 1) else 0

        if click_flag == 0:
            return 0, -1, 0, 0

        # Decide detected bit
        if click_from_photons == 1:
            if source_is_eve and eve_sent_bit >= 0:
                detected_bit = eve_sent_bit
            else:
                local_qber = baseline_qber + timing_qber_delta
                if local_qber < 0.0:
                    local_qber = 0.0
                elif local_qber > 0.5:
                    local_qber = 0.5
                # r_qber in [0,1)
                if r_qber > local_qber:
                    detected_bit = orig_alice_bit
                else:
                    detected_bit = 1 - orig_alice_bit
        else:
            # dark click: use provided random int 0/1
            detected_bit = int(r_rand_bit_int)

        # Compute is_error against original Alice bit
        is_error_flag = 1 if (detected_bit != orig_alice_bit) else 0

        # wrong-basis handling
        label_flag = 0
        if alice_basis != bob_basis:
            # if r_wrong_choice < 0.5 -> randomize outcome sometimes
            if r_wrong_choice < 0.5:
                # with 50% flip chance
                if r_wrong_flip < 0.5:
                    detected_bit = 1 - detected_bit
                is_error_flag = 1 if (detected_bit != orig_alice_bit) else 0
            else:
                is_error_flag = 1 if (detected_bit != orig_alice_bit) else 0
        # label_flag is left 0; if wrapper set eve_resend, wrapper will override
        return 1, int(detected_bit), int(is_error_flag), label_flag


    def __run_episode(self, actions: Dict = None, verbose: bool = False) -> Dict:
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


    # @njit(parallel=True, fastmath=True)
    # def _simulate_pulses_numba(
    #     k_arr, det_eff_arr, dark_count, extra_dark_arr,
    #     alice_bits, alice_basis_arr, bob_basis_arr,
    #     eve_sent_bit_arr, source_is_eve_arr,
    #     baseline_qber, timing_qber_delta_arr,
    #     rng_vals_photon, rng_vals_dark, rng_vals_qber, rng_vals_random
    # ):
    #     """
    #     Vectorized, JIT-compiled simulation of photon clicks and bit errors.
    #     All arrays are 1D numpy arrays (float64 or int8/boolean).
    #     Returns (click_mask, detected_bits, is_error_arr)
    #     """

    #     n = len(k_arr)
    #     detected_bits = np.zeros(n, dtype=np.int8)
    #     is_error_arr = np.zeros(n, dtype=np.int8)
    #     click_mask = np.zeros(n, dtype=np.bool_)

    #     for i in prange(n):
    #         k = k_arr[i]
    #         eff = det_eff_arr[i]
    #         extra_dark = extra_dark_arr[i]

    #         # photon survival prob
    #         photon_survival_prob = 0.0
    #         if k > 0:
    #             photon_survival_prob = 1.0 - (1.0 - eff) ** k

    #         click_from_photons = rng_vals_photon[i] < photon_survival_prob
    #         combined_dark = 1.0 - (1.0 - dark_count - extra_dark) ** 2
    #         combined_dark = min(max(0.0, combined_dark), 1.0)
    #         dark_click = rng_vals_dark[i] < combined_dark

    #         click = click_from_photons or dark_click
    #         click_mask[i] = click
    #         if not click:
    #             continue

    #         # Decide detected bit
    #         if click_from_photons:
    #             if source_is_eve_arr[i] == 1 and eve_sent_bit_arr[i] >= 0:
    #                 detected_bit = eve_sent_bit_arr[i]
    #             else:
    #                 local_qber = baseline_qber + timing_qber_delta_arr[i]
    #                 if local_qber < 0.0:
    #                     local_qber = 0.0
    #                 elif local_qber > 0.5:
    #                     local_qber = 0.5
    #                 detected_bit = alice_bits[i] if rng_vals_qber[i] > local_qber else 1 - alice_bits[i]
    #         else:
    #             detected_bit = int(rng_vals_random[i] > 0.5)

    #         # wrong-basis randomization
    #         if alice_basis_arr[i] != bob_basis_arr[i]:
    #             if rng_vals_random[i] < 0.25:  # ≈50% randomization chance
    #                 detected_bit = 1 - detected_bit

    #         detected_bits[i] = detected_bit
    #         is_error_arr[i] = 1 if detected_bit != alice_bits[i] else 0

    #     return click_mask, detected_bits, is_error_arr


    def run_episode(self, actions: Dict = None, verbose: bool = False) -> Dict:
        """
        Run one episode (pulses_per_episode). actions: dict containing 'Alice','Bob','Eve' updates (optional).
        Vectorized implementation when possible (much faster). Falls back to scalar loop otherwise.
        """
        self.reset_stats()
        self.episode += 1

        # Update Alice params if present
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

        # If we cannot safely vectorize (e.g., a custom Attack instance), fall back to original
        if not self._can_vectorize_eve_action(eve_action):
            # Original scalar loop path (unchanged)
            for i in range(self.pulses_per_episode):
                label, mu = self.sample_label()
                alice_basis = 0 if self.rng.rand() < self.basis_prob else 1
                bob_basis = 0 if self.rng.rand() < self.basis_prob else 1
                alice_bit = int(self.rng.randint(0, 2))

                click, detected_bit, is_error, label_override = self.simulate_pulse(mu, alice_basis, alice_bit, bob_basis, eve_action=eve_action)

                self.counts[label]["total"] += 1
                if click and alice_basis == bob_basis:
                    if label_override == "eve_resend":
                        self.counts["eve_resend"]["total"] += 1
                        self.counts["eve_resend"]["clicks"] += 1
                        if is_error:
                            self.counts["eve_resend"]["errors"] += 1
                            self.eve_resend_errors += 1
                            self.qber_window.append(1)
                        else:
                            self.qber_window.append(0)
                        self.eve_resend_per_label[label]["total"] += 1
                        self.eve_resend_per_label[label]["clicks"] += 1
                        self.eve_resend_total += 1
                        self.eve_resend_clicks += 1
                    else:
                        self.counts[label]["clicks"] += 1
                        if is_error:
                            self.counts[label]["errors"] += 1
                            self.qber_window.append(1)
                        else:
                            self.qber_window.append(0)

        else:
            # ----------------------------
            # Vectorized path (fast)
            # ----------------------------
            n = int(self.pulses_per_episode)
            rng = self.rng  # local alias

            # 1) Sample labels (signal/decoy/vac)
            r = rng.rand(n)
            signal_mask = r < self.p_signal
            decoy_mask = (r >= self.p_signal) & (r < self.p_signal + self.p_decoy)
            vac_mask = ~(signal_mask | decoy_mask)

            # Map mu per pulse
            mu_arr = np.empty(n, dtype=float)
            mu_arr[signal_mask] = self.mu_signal
            mu_arr[decoy_mask] = self.mu_decoy
            mu_arr[vac_mask] = self.mu_vac

            labels_idx = np.empty(n, dtype=np.int8)  # 0=signal,1=decoy,2=vac
            labels_idx[signal_mask] = 0
            labels_idx[decoy_mask] = 1
            labels_idx[vac_mask] = 2
            labels = np.array(self.labels)  # ["signal","decoy","vac"]

            # 2) Sample bases and bits
            alice_basis_arr = (rng.rand(n) >= (1.0 - self.basis_prob)).astype(np.int8)  # basis_prob => prob 0
            # Above keeps semantics: 0 if rand < basis_prob else 1 — but we can map properly:
            alice_basis_arr = (rng.rand(n) < self.basis_prob).astype(np.int8)
            bob_basis_arr = (rng.rand(n) < self.basis_prob).astype(np.int8)
            alice_bits = rng.randint(0, 2, size=n).astype(np.int8)

            # 3) Sample photon numbers (Poisson) with per-pulse mu
            k_arr = rng.poisson(mu_arr)

            # 4) Precompute constants
            eff_base = self.det_eff * self.eta
            det_eff_arr = np.full(n, eff_base, dtype=float)
            extra_dark_arr = np.zeros(n, dtype=float)
            source_arr = np.full(n, "Alice", dtype=object)
            eve_sent_bit_arr = np.full(n, -1, dtype=np.int8)  # -1 means no eve bit
            label_override_arr = np.full(n, "", dtype=object)
            timing_qber_delta_arr = np.zeros(n, dtype=float)

            # --- initialize accumulators for vectorized attacks ---
            extra_dark_arr = np.zeros(n)
            timing_qber_arr = np.zeros(n)


            # 5) Apply vectorized Eve attacks (sequence as in composite)
            def _apply_single_attack_dict(a):
                nonlocal extra_dark_arr, timing_qber_arr
                t = a.get("type", "none")
                if t == "time_shift":
                    ap = float(a.get("attack_prob", 0.0))
                    sf = float(a.get("shift_frac", 0.0))
                    tq = float(a.get("timing_qber_delta", 0.0))
                    mask = (rng.rand(n) < ap)
                    det_eff_arr[mask] = det_eff_arr[mask] * (1.0 - sf)
                    timing_qber_delta_arr[mask] += tq
                elif t == "pns":
                    pns_frac = float(a.get("pns_frac", 0.0))
                    mask = (rng.rand(n) < pns_frac) & (k_arr > 1)
                    k_arr[mask] = np.maximum(0, k_arr[mask] - 1)
                elif t == "intercept_resend":
                    intercept_prob = float(a.get("intercept_prob", 0.0))
                    resend_eff = float(a.get("resend_eff", 1.0))
                    resend_error_prob = float(a.get("resend_error_prob", 0.0))
                    mask = rng.rand(n) < intercept_prob
                    if mask.any():
                        # Eve chooses random measurement basis per masked pulse
                        eve_basis = (rng.rand(mask.sum()) < 0.5).astype(np.int8)
                        # For slots where eve_basis == alice_basis -> measured_bit = alice_bit; else random
                        idxs = np.nonzero(mask)[0]
                        for ii_idx, ii in enumerate(idxs):
                            eb = eve_basis[ii_idx]
                            if eb == alice_basis_arr[ii]:
                                measured_bit = alice_bits[ii]
                            else:
                                measured_bit = int(rng.randint(0,2))
                            # with resend error prob, flip bit
                            if rng.rand() < resend_error_prob:
                                measured_bit = 1 - measured_bit
                            eve_sent_bit_arr[ii] = measured_bit
                        # set k=1 for those pulses and reduce eff
                        k_arr[mask] = 1
                        det_eff_arr[mask] = det_eff_arr[mask] * resend_eff
                        source_arr[mask] = "Eve"
                        label_override_arr[mask] = "eve_resend"
                elif t == "dark_count":
                    extra = float(a.get("extra_dark_prob", 0.0))
                    extra_dark_arr += extra
                elif t == "composite":
                    subs = a.get("sub_attacks", [])
                    for s in subs:
                        _apply_single_attack_dict(s)
                # 'none' -> do nothing

            # If eve_action is dict or list, apply
            if eve_action is None:
                pass
            elif isinstance(eve_action, dict):
                _apply_single_attack_dict(eve_action)
            elif isinstance(eve_action, (list, tuple)):
                for ad in eve_action:
                    _apply_single_attack_dict(ad)

            # 6) Photon survival prob and clicks
            # Avoid pow with negative eff/k combos: compute 1-(1-eff)^k
            # For k==0 -> photon_survival_prob = 0
            photon_survival_prob = np.zeros(n, dtype=float)
            nonzero_mask = k_arr > 0
            # Use np.power in vectorized fashion
            photon_survival_prob[nonzero_mask] = 1.0 - np.power(1.0 - det_eff_arr[nonzero_mask], k_arr[nonzero_mask])
            rand_photon = rng.rand(n)
            click_from_photons = rand_photon < photon_survival_prob

            # dark counts: combined_dark = 1 - (1 - dark_count - extra_dark_prob)^2
            combined_dark = 1.0 - np.power(np.clip(1.0 - self.dark_count - extra_dark_arr, 0.0, 1.0), 2)
            rand_dark = rng.rand(n)
            dark_click = rand_dark < combined_dark

            click_mask = click_from_photons | dark_click

            # If no click, we can skip further processing for those indices
            any_clicks = np.any(click_mask)
            if not any_clicks:
                # nothing clicked; compute empty Q/E as before -> will be zeros
                pass
            else:
                # 7) Decide detected bits
                detected_bits = np.zeros(n, dtype=np.int8)
                # dark-click indices -> random bit
                dark_idxs = np.nonzero(dark_click)[0]
                if dark_idxs.size:
                    detected_bits[dark_idxs] = rng.randint(0, 2, size=dark_idxs.size).astype(np.int8)

                # photon-click indices (may overlap with dark_click; our model prioritizes photon when click_from_photons True)
                photon_idxs = np.nonzero(click_from_photons)[0]
                if photon_idxs.size:
                    # For positions where source == 'Eve' and eve_sent_bit_arr != -1:
                    eve_mask_idx = photon_idxs[(source_arr[photon_idxs] == "Eve") & (eve_sent_bit_arr[photon_idxs] != -1)]
                    if eve_mask_idx.size:
                        # If several indices selected, assign their eve_sent_bit
                        detected_bits[eve_mask_idx] = eve_sent_bit_arr[eve_mask_idx]
                    # For remaining photon indices from Alice:
                    alice_photon_mask = np.setdiff1d(photon_idxs, eve_mask_idx, assume_unique=True)
                    if alice_photon_mask.size:
                        local_qber = np.clip(self.baseline_qber + timing_qber_delta_arr[alice_photon_mask], 0.0, 0.5)
                        # draw random numbers and compare to local_qber
                        r_local = rng.rand(alice_photon_mask.size)
                        # if r_local > local_qber -> correct bit, else flipped
                        correct_mask = r_local > local_qber
                        idxs = alice_photon_mask
                        detected_bits[idxs[correct_mask]] = alice_bits[idxs[correct_mask]]
                        detected_bits[idxs[~correct_mask]] = 1 - alice_bits[idxs[~correct_mask]]

                # 8) Wrong-basis handling (randomization preserving QBER calc)
                wrong_basis_mask = (alice_basis_arr != bob_basis_arr)
                if wrong_basis_mask.any():
                    wb_idxs = np.nonzero(wrong_basis_mask & click_mask)[0]
                    if wb_idxs.size:
                        # For half of wb events, randomize outcome
                        rand_choice = rng.rand(wb_idxs.size)
                        for jj, idx in enumerate(wb_idxs):
                            if rand_choice[jj] < 0.5:
                                # randomize detected_bit outcome when wrong basis
                                if rng.rand() < 0.5:
                                    detected_bits[idx] = 1 - detected_bits[idx]
                                # recompute is_error implicitly later

                # 9) Compute is_error against original alice_bits
                is_error_arr = (detected_bits != alice_bits).astype(np.int8)

                # 10) Update counts vectorized
                # Total pulses per label:
                # note: counts[label]["total"] increments per original label, independent of clicks
                totals = np.bincount(labels_idx, minlength=3)
                for idx_label, lab in enumerate(self.labels):
                    self.counts[lab]["total"] += int(totals[idx_label])

                # Now handle sifted events: clicks AND alice_basis == bob_basis
                sift_mask = click_mask & (alice_basis_arr == bob_basis_arr)
                sift_idxs = np.nonzero(sift_mask)[0]
                if sift_idxs.size:
                    # Those that were marked as eve_resend go into eve_resend bucket
                    eve_resend_idxs = sift_idxs[label_override_arr[sift_idxs] == "eve_resend"]
                    if eve_resend_idxs.size:
                        self.counts["eve_resend"]["total"] += int(eve_resend_idxs.size)
                        self.counts["eve_resend"]["clicks"] += int(eve_resend_idxs.size)
                        errs = int(np.sum(is_error_arr[eve_resend_idxs]))
                        self.counts["eve_resend"]["errors"] += errs
                        self.eve_resend_errors += errs
                        self.qber_window.extend(list(is_error_arr[eve_resend_idxs]))
                        # per-label bookkeeping: the original label for these indices:
                        orig_labels = labels_idx[eve_resend_idxs]
                        for lval in [0,1,2]:
                            sel = (orig_labels == lval)
                            if sel.any():
                                labname = self.labels[lval]
                                self.eve_resend_per_label[labname]["total"] += int(np.sum(sel))
                                self.eve_resend_per_label[labname]["clicks"] += int(np.sum(sel))
                        self.eve_resend_total += int(eve_resend_idxs.size)
                        self.eve_resend_clicks += int(eve_resend_idxs.size)

                    # Normal legitimate clicks: those sifted and not eve_resend
                    normal_idxs = sift_idxs[label_override_arr[sift_idxs] != "eve_resend"]
                    if normal_idxs.size:
                        # Count clicks/errors per original label
                        orig_labels = labels_idx[normal_idxs]
                        for lval in [0,1,2]:
                            sel = (orig_labels == lval)
                            if sel.any():
                                labname = self.labels[lval]
                                added_clicks = int(np.sum(sel))
                                self.counts[labname]["clicks"] += added_clicks
                                errs = int(np.sum(is_error_arr[normal_idxs][sel]))
                                self.counts[labname]["errors"] += errs
                                # push per-event qber window bits
                                for v in is_error_arr[normal_idxs][sel].tolist():
                                    self.qber_window.append(int(v))

            # end vectorized processing

        # compute gains and QBERs (same as before)
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
            "Y1": Y1,
            "e1": e1,
            "Q1": Q1,
            "Q_s_signal": Q["signal"],
            "eve_resend_total": self.eve_resend_total,
            "eve_resend_clicks": self.eve_resend_clicks,
            "eve_resend_errors": self.eve_resend_errors,
            "SKR_bits_per_pulse": skr,
            "SKR_bits_per_second": skr * self.pulse_rate
        }

        info["eve_resend_from_signal_total"] = self.eve_resend_per_label["signal"]["total"]
        info["eve_resend_from_decoy_total"]  = self.eve_resend_per_label["decoy"]["total"]
        info["eve_resend_from_vac_total"]    = self.eve_resend_per_label["vac"]["total"]

        if verbose:
            print(f"[Episode {self.episode}] SKR={info['SKR_bits_per_pulse']:.6e} bits/pulse, SKR={info['SKR_bits_per_second']:.3f} bits/s")
        return info
