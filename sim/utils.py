# utils.py
import math

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


import math

def __decoy_estimates(mu_s, mu_d, mu_v, Q_s, Q_d, Q_v, E_s, E_d, E_v):
    """
    Proper vacuum + weak decoy estimation following:
    Ma, Qi, Zhao & Lo, PRA 72, 012326 (2005)

    Inputs:
        mu_s : signal intensity
        mu_d : decoy intensity
        mu_v : vacuum intensity (should be 0)
        Q_s, Q_d, Q_v : observed gains
        E_s, E_d, E_v : observed QBERs

    Returns:
        Y1_lower : lower bound on single-photon yield
        e1_upper : upper bound on single-photon error rate
        Q1_lower : lower bound on single-photon gain
    """

    # Vacuum yield (dark count contribution)
    Y0 = Q_v * math.exp(mu_v)  # usually Q_v with mu_v=0

    # Precompute exponentials
    es = math.exp(mu_s)
    ed = math.exp(mu_d)

    # Denominator used in Ma 2005 eq. (41)
    denom = (mu_s * mu_d - mu_d**2)
    if denom <= 0:
        # Degenerate case: identical intensities or impossible setting
        return 0.0, 0.5, 0.0

    # Lower-bound numerator for Y1
    numerator = (
        Q_d * ed
        - Q_s * es * (mu_d**2 / mu_s**2)
        - ((mu_s**2 - mu_d**2) / (mu_s**2)) * Y0
    )

    Y1_lower = (mu_s / denom) * numerator
    Y1_lower = max(0.0, Y1_lower)  # enforce physicality

    # --- Compute single-photon gain Q1 ---
    Q1_lower = Y1_lower * mu_s * math.exp(-mu_s)

    # --- e1 upper bound ---
    if Q1_lower > 0:
        # Correct error estimator from Ma eq. (44)
        e1_upper = (
            E_s * Q_s - 0.5 * Y0 * math.exp(-mu_s)
        ) / Q1_lower
        e1_upper = min(max(e1_upper, 0.0), 1.0)
    else:
        e1_upper = 0.5  # pessimistic

    return 2*Y1_lower, e1_upper, 2*Q1_lower


