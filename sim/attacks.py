import numpy as np

from typing import Dict, Optional, Any, List


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
    def __init__(self, attack_prob: float = 0.0, shift_frac: float = 0.2,
                 timing_qber_delta: float = 0.0, rng: Optional[np.random.RandomState] = None):
        super().__init__(rng)
        self.attack_prob = float(attack_prob)
        self.shift_frac = float(shift_frac)
        # Delta to baseline QBER for pulses shifted into the attack time slot
        self.timing_qber_delta = float(timing_qber_delta)

    def modify_state(self, k, eff, alice_basis, alice_bit, bob_basis):
        if self.rng.rand() < self.attack_prob:
            eff = eff * (1.0 - self.shift_frac)
            # Tell simulator to increase local QBER by timing_qber_delta for this pulse
            return {
                "k": k,
                "eff": eff,
                "alice_bit": alice_bit,
                "source": "Alice",
                "extra_dark_prob": 0.0,
                "timing_qber_delta": self.timing_qber_delta
            }
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
    def __init__(self, intercept_prob: float = 0.0, resend_mu: float = 1.0,
                 resend_eff: float = 1.0, resend_error_prob: float = 0.0,
                 rng: Optional[np.random.RandomState] = None):
        super().__init__(rng)
        self.intercept_prob = float(intercept_prob)
        self.resend_mu = float(resend_mu)
        self.resend_eff = float(resend_eff)            # fraction of eff for Eve's resent pulse
        self.resend_error_prob = float(resend_error_prob)  # prob Eve resends a wrong bit (imperfections)

    def modify_state(self, k, eff, alice_basis, alice_bit, bob_basis):
        if self.rng.rand() < self.intercept_prob:
            eve_basis = 0 if self.rng.rand() < 0.5 else 1
            if eve_basis == alice_basis:
                measured_bit = alice_bit
            else:
                measured_bit = int(self.rng.randint(0, 2))
            # With some resend error probability, Eve might misprepare the bit
            if self.rng.rand() < self.resend_error_prob:
                measured_bit = 1 - measured_bit
            # Eve resends a single-photon; we reduce eff to represent imperfect injection / coupling
            # Return eve_bit and modified eff for the resent pulse (resend_eff * eff)
                return {
                "k": 1,
                "eff": eff * self.resend_eff,
                "eve_bit": int(measured_bit),
                "source": "Eve",
                "extra_dark_prob": 0.0,
                "label_override": "eve_resend"    # <-- ONE-LINER: mark this pulse as Eve-resend
            }

        return {"k": k, "eff": eff, "source": "Alice", "extra_dark_prob": 0.0}



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
