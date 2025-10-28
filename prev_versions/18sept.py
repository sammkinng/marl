
from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import List, Tuple

import netsquid as ns
from netsquid.qubits import qubitapi as qapi
from netsquid.qubits import operators as ops


# --------------------------- Utilities ---------------------------------

def h2(x: float) -> float:
    """Binary entropy in bits; safe for edge cases."""
    x = max(0.0, min(1.0, x))
    if x == 0.0 or x == 1.0:
        return 0.0
    return -x*math.log2(x) - (1-x)*math.log2(1-x)


def apply_depolarizing_noise(qubit, p: float) -> None:
    """Apply depolarizing noise to a NetSquid qubit by randomly applying X, Y, or Z with prob p/3 each.
    With probability (1-p) do nothing."""
    if p <= 0:
        return
    r = random.random()
    if r < p:
        # choose which Pauli
        choice = random.choice(['X', 'Y', 'Z'])
        if choice == 'X':
            qapi.operate(qubit, ops.X)
        elif choice == 'Y':
            qapi.operate(qubit, ops.Y)
        else:
            qapi.operate(qubit, ops.Z)


def prepare_bb84_state(bit: int, basis: int):
    """Prepare |0>,|1> in Z basis or |+>,|-> in X basis using NetSquid.
    basis: 0 -> Z, 1 -> X
    bit:   0 or 1
    Returns: (qubit, expected_bit)
    """
    q, = qapi.create_qubits(1)
    # Start in |0>
    if bit == 1:
        qapi.operate(q, ops.X)  # -> |1>
    if basis == 1:
        qapi.operate(q, ops.H)  # map to X-basis (|+> if bit=0, |-> if bit=1)
    return q


def measure_in_basis(qubit, basis: int) -> int:
    """Measure the qubit in the specified basis (0=Z, 1=X) and return classical bit 0/1."""
    if basis == 1:
        qapi.operate(qubit, ops.H)
    # Measure in Z
    m, prob = qapi.measure(qubit, observable=ops.Z, discard=True)
    # NetSquid returns m in {0,1}
    return int(m)


@dataclass
class BB84Results:
    total_pulses: int
    sifted_bits: int
    sifting_fraction: float
    qber: float
    skr_per_pulse: float
    skr_total_bits: float


# --------------------------- Core Simulation ---------------------------

def bb84_simulate(
    pulses: int = 10000,
    p_depol: float = 0.02,
    alice_basis_bias: float = 0.5,
    bob_basis_bias: float = 0.5,
    f_ec: float = 1.0,
    seed: int | None = 42,
) -> BB84Results:
    """Run a BB84 Monte Carlo simulation with depolarizing noise.

    Args:
        pulses: number of signals sent by Alice.
        p_depol: depolarizing probability per qubit (0..1).
        alice_basis_bias: P(basis=Z). So P(X)=1-bias.
        bob_basis_bias:   P(basis=Z) at Bob.
        f_ec: error-correction inefficiency (≥1). Use 1.0 for ideal.
        seed: RNG seed for reproducibility.

    Returns:
        BB84Results with SKR computed from measured QBER and sifting fraction.
    """
    if seed is not None:
        random.seed(seed)
        ns.set_random_state(seed)

    sifted = 0
    errors = 0

    for _ in range(pulses):
        # Alice chooses bit and basis
        a_bit = random.getrandbits(1)
        a_basis = 0 if random.random() < alice_basis_bias else 1  # 0=Z,1=X

        q = prepare_bb84_state(a_bit, a_basis)
        apply_depolarizing_noise(q, p_depol)

        # Bob chooses basis and measures
        b_basis = 0 if random.random() < bob_basis_bias else 1
        b_bit = measure_in_basis(q, b_basis)

        # Public discussion of bases (sifting)
        if a_basis == b_basis:
            sifted += 1
            if b_bit != a_bit:
                errors += 1

    q = (errors / sifted) if sifted > 0 else 0.0

    # Sifting fraction q_factor ≈ sifted / pulses (could be ~0.5 for unbiased)
    q_factor = sifted / pulses if pulses > 0 else 0.0

    # Asymptotic SKR per pulse. Two common forms:
    #   (A) ideal EC:         R = q * max(0, 1 - 2*h2(Q))
    #   (B) with inefficiency: R = q * max(0, 1 - (1+f_ec)*h2(Q))
    # We'll use (B) with f_ec as provided (f_ec=1.0 -> (A)).
    R_per_pulse = q_factor * max(0.0, 1.0 - (1.0 + f_ec) * h2(q))

    return BB84Results(
        total_pulses=pulses,
        sifted_bits=sifted,
        sifting_fraction=q_factor,
        qber=q,
        skr_per_pulse=R_per_pulse,
        skr_total_bits=R_per_pulse * pulses,
    )


# --------------------------- Parameter Sweep ---------------------------

def sweep_noise_example():
    """Run a small sweep over p_depol and print a compact summary."""
    print("p_depol\tQBER\tq_sift\tSKR/pulse")
    for p in [0.0, 0.01, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15]:
        res = bb84_simulate(pulses=20000, p_depol=p, seed=123)
        print(f"{p:.3f}\t{res.qber:.3f}\t{res.sifting_fraction:.3f}\t{res.skr_per_pulse:.4f}")


if __name__ == "__main__":
    # Quick demo run
    results = bb84_simulate(
        pulses=50000,
        p_depol=0.02,       # 2% depolarizing noise
        alice_basis_bias=0.5,
        bob_basis_bias=0.5,
        f_ec=1.1,           # 10% EC inefficiency
        seed=7,
    )

    print("--- BB84 Simulation (Depolarizing Channel) ---")
    print(f"Total pulses:      {results.total_pulses}")
    print(f"Sifted bits:       {results.sifted_bits}")
    print(f"Sifting fraction:  {results.sifting_fraction:.3f}")
    print(f"QBER (sifted):     {results.qber:.3%}")
    print(f"SKR per pulse:     {results.skr_per_pulse:.5f} bits")
    print(f"Total secret bits: {results.skr_total_bits:.1f}")

    # Optional: show a tiny sweep table in stdout
    # sweep_noise_example()
