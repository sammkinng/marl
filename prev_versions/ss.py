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
from ..sim.utils import H2, decoy_estimates
from typing import Dict, Tuple, Optional, Any, List, Union

# --- Attack framework -------------------------------------------------------

class Attack:
    """
    Base class for attacks. Subclasses should implement modify_state.

    modify_state inputs:
        k: int (photon number sampled from Poisson)
        eff: float (effective detection efficiency det_eff * eta)
        alice_basis: int
        alice_bit: int
        bob_basis: int
        rng: numpy RandomState

    modify_state returns dict with any of:
        k (int), eff (float), alice_bit (int), source (str), extra_dark_prob (float)
    """
    def __init__(self, rng: Optional[np.random.RandomState] = None):
        self.rng = rng

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any], rng: np.random.RandomState):
        """
        Factory to produce a concrete Attack from a dict configuration.
        Keeps backward compatibility with previous dict-based 'eve_action'.
        """
        t = cfg.get("type", "none")
        if t == "time_shift":
            return TimeShiftAttack(attack_prob=float(cfg.get("attack_prob", 0.0)),
                                   shift_frac=float(cfg.get("shift_frac", 0.2)),
                                   rng=rng)
        elif t == "pns":
            return PNSAttack(pns_frac=float(cfg.get("pns_frac", 0.0)), rng=rng)
        elif t == "intercept_resend":
            return InterceptResendAttack(intercept_prob=float(cfg.get("intercept_prob", 0.0)),
                                         resend_mu=float(cfg.get("resend_mu", 1.0)),
                                         rng=rng)
        elif t == "dark_count":
            return DarkCountAttack(extra_dark_prob=float(cfg.get("extra_dark_prob", 0.0)), rng=rng)
        elif t == "composite":
            # cfg expected to contain a list of sub-attack dicts under 'sub_attacks'
            subs = cfg.get("sub_attacks", [])
            sub_objs = [Attack.from_dict(s, rng) for s in subs]
            return CompositeAttack(sub_objs, rng=rng)
        else:
            return NoAttack(rng=rng)

    def modify_state(self, k: int, eff: float, alice_basis: int, alice_bit: int, bob_basis: int):
        # default: no change
        return {"k": k, "eff": eff, "alice_bit": alice_bit, "source": "Alice", "extra_dark_prob": 0.0}


class NoAttack(Attack):
    pass


class CompositeAttack(Attack):
    def __init__(self, attacks: List[Attack], rng: Optional[np.random.RandomState] = None):
        super().__init__(rng)
        self.attacks = attacks

    def modify_state(self, k, eff, alice_basis, alice_bit, bob_basis):
        out = {"k": k, "eff": eff, "alice_bit": alice_bit, "source": "Alice", "extra_dark_prob": 0.0}
        for a in self.attacks:
            # ensure each attack uses the same RNG (already passed)
            res = a.modify_state(out["k"], out["eff"], alice_basis, out["alice_bit"], bob_basis)
            # merge outputs
            out["k"] = res.get("k", out["k"])
            out["eff"] = res.get("eff", out["eff"])
            out["alice_bit"] = res.get("alice_bit", out["alice_bit"])
            # mark source if changed by attack
            if res.get("source") is not None:
                out["source"] = res.get("source")
            out["extra_dark_prob"] = out.get("extra_dark_prob", 0.0) + res.get("extra_dark_prob", 0.0)
        return out


class TimeShiftAttack(Attack):
    def __init__(self, attack_prob: float = 0.0, shift_frac: float = 0.2, rng: Optional[np.random.RandomState] = None):
        super().__init__(rng)
        self.attack_prob = float(attack_prob)
        self.shift_frac = float(shift_frac)

    def modify_state(self, k, eff, alice_basis, alice_bit, bob_basis):
        if self.rng.rand() < self.attack_prob:
            eff = eff * (1.0 - self.shift_frac)
            return {"k": k, "eff": eff, "alice_bit": alice_bit, "source": "Alice", "extra_dark_prob": 0.0}
        return {"k": k, "eff": eff, "alice_bit": alice_bit, "source": "Alice", "extra_dark_prob": 0.0}


class PNSAttack(Attack):
    """
    Photon-number splitting approximation:
    if k > 1 and random < pns_frac, Eve steals one photon -> reduce k by 1.
    """
    def __init__(self, pns_frac: float = 0.0, rng: Optional[np.random.RandomState] = None):
        super().__init__(rng)
        self.pns_frac = float(pns_frac)

    def modify_state(self, k, eff, alice_basis, alice_bit, bob_basis):
        if k > 1 and self.rng.rand() < self.pns_frac:
            k = max(0, k - 1)
            return {"k": k, "eff": eff, "alice_bit": alice_bit, "source": "Alice", "extra_dark_prob": 0.0}
        return {"k": k, "eff": eff, "alice_bit": alice_bit, "source": "Alice", "extra_dark_prob": 0.0}


class InterceptResendAttack(Attack):
    """
    Intercept-Resend:
    With probability intercept_prob, Eve measures in a random basis.
    If she measures, she resends a single-photon pulse (k=1) representing her measurement result.
    This is a simple, approximate model to introduce errors when Eve's basis != Alice's.
    """
    def __init__(self, intercept_prob: float = 0.0, resend_mu: float = 1.0, rng: Optional[np.random.RandomState] = None):
        super().__init__(rng)
        self.intercept_prob = float(intercept_prob)
        self.resend_mu = float(resend_mu)

    def modify_state(self, k, eff, alice_basis, alice_bit, bob_basis):
        if self.rng.rand() < self.intercept_prob:
            # Eve chooses a random measurement basis
            eve_basis = 0 if self.rng.rand() < 0.5 else 1
            # If bases match, Eve obtains alice_bit with baseline_qber=0 (we assume perfect Eve)
            if eve_basis == alice_basis:
                measured_bit = alice_bit
            else:
                # if basis mismatch, measurement is random
                measured_bit = int(self.rng.randint(0, 2))
            # Eve resends a single-photon (approximate): set k -> 1 and mark source as 'Eve'
            return {"k": 1, "eff": eff, "alice_bit": measured_bit, "source": "Eve", "extra_dark_prob": 0.0}
        else:
            return {"k": k, "eff": eff, "alice_bit": alice_bit, "source": "Alice", "extra_dark_prob": 0.0}


class DarkCountAttack(Attack):
    """
    Attack that increases dark-count probability while active. It returns extra_dark_prob, which the simulator
    will add to the normal dark_count for the calculation of dark clicks for that pulse.
    """
    def __init__(self, extra_dark_prob: float = 0.0, rng: Optional[np.random.RandomState] = None):
        super().__init__(rng)
        self.extra_dark_prob = float(extra_dark_prob)

    def modify_state(self, k, eff, alice_basis, alice_bit, bob_basis):
        return {"k": k, "eff": eff, "alice_bit": alice_bit, "source": "Alice", "extra_dark_prob": self.extra_dark_prob}


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
        """
        Simulate one pulse with WCP photon number sampling and modular attack hooks.
        Returns: click (bool), detected_bit (0/1 or None), is_error (bool)
        """
        # sample photon number k ~ Poisson(mu)
        k = self.rng.poisson(mu)
        # effective detection efficiency including channel loss
        eff = self.det_eff * self.eta

        # parse and get attack object (if any)
        attack_obj = self._parse_eve_action(eve_action)

        extra_dark_prob = 0.0
        source = "Alice"

        if attack_obj is not None:
            res = attack_obj.modify_state(k=k, eff=eff, alice_basis=alice_basis, alice_bit=alice_bit, bob_basis=bob_basis)
            # update local variables from attack output
            k = int(res.get("k", k))
            eff = float(res.get("eff", eff))
            alice_bit = int(res.get("alice_bit", alice_bit))
            source = res.get("source", source)
            extra_dark_prob = float(res.get("extra_dark_prob", 0.0))

        # photon-induced click probability (assuming on-off detectors and independent photons)
        photon_survival_prob = 1.0 - (1.0 - eff) ** k if k > 0 else 0.0
        click_from_photons = self.rng.rand() < photon_survival_prob if photon_survival_prob > 0 else False

        # dark counts (two detectors -> combined probability approx)
        # combine simulator dark_count with any attack-driven extra dark probability
        combined_dark = 1.0 - (1.0 - self.dark_count - extra_dark_prob) ** 2
        # ensure bounds
        combined_dark = min(max(0.0, combined_dark), 1.0)
        dark_click = self.rng.rand() < combined_dark
        click = click_from_photons or dark_click

        if not click:
            return False, None, False

        # Decide detected bit
        if click_from_photons:
            # If the photons originate from Eve (source == 'Eve'), we assume Eve sent a clean prepared bit (no baseline_qber),
            # otherwise apply baseline_qber
            if source == "Eve":
                detected_bit = alice_bit  # Eve resends a state aligned to her measurement
                is_error = (detected_bit != alice_bit)
            else:
                detected_bit = alice_bit if self.rng.rand() > self.baseline_qber else 1 - alice_bit
                is_error = (detected_bit != alice_bit)
        else:
            # dark click: random bit value
            detected_bit = int(self.rng.randint(0, 2))
            is_error = (detected_bit != alice_bit)

        # wrong-basis handling: if bases differ, measurement outcome is random (50% error on average).
        # We keep same logic but ensure it is applied to 'is_error' and 'detected_bit'.
        if alice_basis != bob_basis:
            if self.rng.rand() < 0.5:
                is_error = True
                # random flip or keep some randomness
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


# ---------------------------
# Usage examples (small)
# ---------------------------
if __name__ == "__main__":
    # basic simulator
    cfg = {
        "pulses_per_episode": 50000,
        "mu_signal": 0.5,
        "mu_decoy": 0.1,
        "p_signal": 0.7,
        "p_decoy": 0.2,
        "p_vac": 0.1,
        "det_eff": 0.6,
        "dark_count": 1e-7,
        "seed": 42
    }
    sim = QKDSimulator(cfg)

    # Example 1: time-shift attack (dict form, kept for backward compat)
    actions = {"Eve": {"type": "time_shift", "attack_prob": 0.2, "shift_frac": 0.3}}
    info_ts = sim.run_episode(actions=actions, verbose=True)
    print("Time shift info:", info_ts)

    # Example 2: composite attack using Attack objects directly
    ts = TimeShiftAttack(attack_prob=0.2, shift_frac=0.3, rng=sim.rng)
    ir = InterceptResendAttack(intercept_prob=0.01, resend_mu=1.0, rng=sim.rng)
    dc = DarkCountAttack(extra_dark_prob=1e-6, rng=sim.rng)
    comp = CompositeAttack([ts, ir, dc], rng=sim.rng)
    info_comp = sim.run_episode(actions={"Eve": comp}, verbose=True)
    print("Composite attack info:", info_comp)
