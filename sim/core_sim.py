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

    def set_ppe(self, new_ppe):
        """Dynamically update pulses per episode."""
        self.pulses_per_episode = int(new_ppe)

        # ===============================
    # Dynamic parameter control (for RL tests)
    # ===============================

    def set_fiber_length(self, L_km: float):
        """
        Dynamically adjust channel transmittance based on fiber length.
        Updates self.eta accordingly using fiber loss coefficient in cfg.
        """
        alpha_db_per_km = float(self.cfg.get("fiber_loss_db_per_km", 0.2))
        self.cfg["distance_km"] = float(L_km)
        self.eta = 10 ** (-alpha_db_per_km * L_km / 10.0)

    def get_fiber_length(self) -> float:
        """Return the current fiber length in km."""
        return float(self.cfg.get("distance_km", 0.0))

    def set_detector_efficiency(self, eta_det: float):
        """Set detector quantum efficiency (0–1)."""
        self.det_eff = float(np.clip(eta_det, 0.0, 1.0))
        self.cfg["det_eff"] = self.det_eff

    def get_detector_efficiency(self) -> float:
        """Return current detector efficiency."""
        return float(self.det_eff)

    def set_dark_count(self, dc_prob: float):
        """Set dark count probability per detector per pulse."""
        self.dark_count = float(np.clip(dc_prob, 0.0, 1.0))
        self.cfg["dark_count"] = self.dark_count

    def get_dark_count(self) -> float:
        """Return dark count probability."""
        return float(self.dark_count)

    def set_distance_and_recompute_eta(self, L_km: float):
        """Convenience alias for backward compatibility."""
        self.set_fiber_length(L_km)


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

        # print(f"Total clicks: {sum([self.counts[l]['clicks'] for l in self.labels])}")
        # print(f"Sifted clicks: {sum([self.counts[l]['clicks'] for l in self.labels if l != 'eve_resend'])}")
        # print(f"eff={self.det_eff*self.eta}")




        if verbose:
            print(f"[Episode {self.episode}] SKR={info['SKR_bits_per_pulse']:.6e} bits/pulse, SKR={info['SKR_bits_per_second']:.3f} bits/s")
        return info
