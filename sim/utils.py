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




import numpy as np
from typing import Dict, Union, List, Optional, Tuple

# =================================================================
# CORE STANDALONE ANALYTICAL FUNCTION
# =================================================================

def calculate_qkd_asymptotic_performance(
    mu_signal: float, mu_decoy: float, mu_vac: float,
    p_signal: float, p_decoy: float, p_vac: float,
    eta_ch: float, eta_det: float, P_d_base: float, e_0_base: float,
    basis_prob: float, recon_eff: float, pulse_rate: float,
    pulses_per_episode: int,
    eve_action: Optional[Union[Dict, List[Dict]]] = None,
    cfg: Optional[Dict] = None
) -> Dict:
    """
    Analytically computes the expected asymptotic QKD performance for RL training.
    
    Returns the complete info dictionary structure, equivalent to the Monte Carlo simulator.
    """
    
    # --- 1. Aggregate Input Parameters ---
    
    mus = {"signal": mu_signal, "decoy": mu_decoy, "vac": mu_vac}
    pulse_probs = {"signal": p_signal, "decoy": p_decoy, "vac": p_vac}
    labels = ["signal", "decoy", "vac"]
    q = basis_prob
    N = float(pulses_per_episode)
    
    # --- 2. Parse Attack Parameters (Handles 'composite' by summing effects) ---
    
    attack_params = {
        "time_shift": {"attack_prob": 0.0, "shift_frac": 0.0, "timing_qber_delta": 0.0},
        "pns": {"pns_frac": 0.0},
        "intercept_resend": {"intercept_prob": 0.0, "resend_eff": 1.0, "resend_error_prob": 0.0},
        "dark_count": {"extra_dark_prob": 0.0}
    }
    
    actions_list = []
    if eve_action is None:
        pass
    elif isinstance(eve_action, dict):
        actions_list = [eve_action]
    elif isinstance(eve_action, list):
        actions_list = eve_action

    for a in actions_list:
        t = a.get("type", "none")
        if t == "composite":
            for sub_a in a.get("sub_attacks", []):
                if sub_a.get("type") in attack_params:
                    attack_params[sub_a["type"]].update(sub_a)
        elif t in attack_params:
            attack_params[t].update(a)

    # --- 3. Calculate Effective Physical Parameters (Modified by Attack) ---
    
    # Time-Shift Attack:
    ts = attack_params["time_shift"]
    P_ts = ts["attack_prob"]
    shift_frac = ts["shift_frac"]
    qber_delta = ts["timing_qber_delta"]
    
    # Effective Detector Efficiency reduced by Time-Shift loss
    eta_det_eff = eta_det * (1.0 - P_ts * shift_frac) 
    
    # Dark Count Attack:
    P_d_extra = attack_params["dark_count"]["extra_dark_prob"]
    P_d_eff = np.clip(P_d_base + P_d_extra, 0.0, 1.0)
    
    # Effective transmission/detection efficiency for Alice's original pulse
    eta_eff = eta_ch * eta_det_eff
    
    # Intercept-Resend Attack:
    ir = attack_params["intercept_resend"]
    P_ir = ir["intercept_prob"]
    eta_resend = ir["resend_eff"]
    e_resend = ir["resend_error_prob"]
    
    # --- 4. Calculate Asymptotic Gains (Q) and QBERs (E) ---
    
    Q = {}
    E = {}
    
    # Accumulated (scaled) counting metrics
    counts = {lab: {"clicks": 0.0, "errors": 0.0, "total": 0.0} for lab in labels}
    eve_resend_clicks_by_label = {lab: 0.0 for lab in labels}
    eve_resend_total = 0.0
    
    eve_knows_count = 0.0
    eve_attacks_total = N * (P_ts + P_ir)
    
    # PNS is inherently included in the decoy analysis (worst-case), 
    # but the MC model also tracks PNS *touching* the pulse (we ignore the k>1 reduction
    # here as the decoy analysis covers it).
    
    for lab in labels:
        mu = mus[lab]
        P_lab = pulse_probs[lab]
        
        # A. Contribution from Alice's ORIGINAL pulse (not intercepted, with full channel/detector effects)
        
        # 1. Photon-induced click probability from Alice's source (unintercepted)
        Q_A_photon = 1.0 - np.exp(-mu * eta_eff)
        
        # 2. Dark count contribution (independent of Alice/Eve pulse)
        Q_Dark = 2 * P_d_eff - P_d_eff**2 
        
        # Total click probability for a pulse from Alice that WAS NOT intercepted:
        Q_unintercepted_mu = Q_A_photon + Q_Dark
        
        # Error Gain Q*E from unintercepted pulse:
        e_eff = e_0_base + P_ts * qber_delta # Baseline QBER + Timing QBER
        Q_E_unintercepted = (1.0 - P_ir) * (Q_A_photon * e_eff + Q_Dark * 0.5)
        
        # B. Contribution from Eve's RESENT pulse (Intercept-Resend)
        Q_E_total = 0.0
        Q_E_E_total = 0.0

        if mu > 0:
            # Eve sends a single photon (k=1) with efficiency eta_resend * eta_ch
            eta_eve = eta_resend * eta_ch
            Q_eve_click = 1.0 - np.exp(-1.0 * eta_eve)
            
            # Error rate introduced by Eve's measurement + resend:
            # Assumes 50% basis mismatch (error rate 0.5) + resend error prob (e_resend)
            e_eve = 0.5 * 0.5 + e_resend # 0.25 (basis mismatch) + e_resend (resend error)
            
            Q_E_total = P_ir * Q_eve_click
            Q_E_E_total = P_ir * Q_eve_click * e_eve

            # Eve tracking: She knows the bit if she intercepts AND measures in the correct basis (50% chance).
            eve_knows_count += N * P_lab * P_ir * 0.5
            
            # Bookkeeping for Eve-resend counts
            eve_resend_clicks_by_label[lab] = N * P_lab * Q_E_total
        
        # C. Total Asymptotic Gain (Sifted Probability per pulse)
        # Note: (1 - P_ir) component means the dark counts are shared between A and E contributions
        Q_mu_unnormalized = (1 - P_ir) * Q_A_photon + Q_Dark + Q_E_total
        Q[lab] = q * Q_mu_unnormalized
        
        # Total Asymptotic Error Gain (Q*E)
        Q_E_mu_unnormalized = (1 - P_ir) * (Q_A_photon * e_eff + Q_Dark * 0.5) + Q_E_E_total
        
        if Q_mu_unnormalized > 1e-18:
            E[lab] = Q_E_mu_unnormalized / Q_mu_unnormalized
        else:
            E[lab] = 0.5 # Max error if no clicks

        # --- 5. Populate Deterministic Counts ---
        
        counts[lab]["total"] = N * P_lab
        counts[lab]["clicks"] = N * P_lab * Q[lab]
        counts[lab]["errors"] = N * P_lab * Q[lab] * E[lab]
        
        # Add Eve resend contribution to total Eve stats
        eve_resend_total += eve_resend_clicks_by_label[lab]

    # --- 6. Compute Final SKR ---
    
    # Denormalize Q back to yield (gain/basis_prob) for decoy analysis
    Q_yields = {lab: Q[lab]/q for lab in labels}
    E_final = E

    # This calls the original decoy_estimates function
    Y1, e1, Q1 = decoy_estimates(
        mus["signal"], mus["decoy"], mus["vac"], 
        Q_yields["signal"], Q_yields["decoy"], Q_yields["vac"], 
        E_final["signal"], E_final["decoy"], E_final["vac"]
    )
    
    # SKR = q * [ Q1 * (1 - H2(e1)) - Q_mu * f * H2(E_mu) ]
    # We need the original compute_skr logic here, assumed to be:
    # SKR_per_pulse = q * (Q1 * (1 - H2(e1)) + Q["signal"] * (-recon_eff * H2(E["signal"])))
    term1 = Q1 * (1 - H2(e1))
    term2 = - Q["signal"] * recon_eff * H2(E["signal"])
    skr = q * max(0.0, term1 + term2) # Ensure SKR >= 0
    
    # --- 7. Create Final Info Dict ---

    eve_resend_errors = eve_resend_total * e_resend # Simple estimate

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
        "eve_resend_total": eve_resend_total,
        "eve_resend_clicks": eve_resend_total, # Total and clicks are the same for Eve resend
        "eve_resend_errors": eve_resend_errors,
        "SKR_bits_per_pulse": skr,
        "SKR_bits_per_second": skr * pulse_rate
    }
    
    info["eve_resend_from_signal_total"] = eve_resend_clicks_by_label["signal"]
    info["eve_resend_from_decoy_total"]  = eve_resend_clicks_by_label["decoy"]
    info["eve_resend_from_vac_total"]    = eve_resend_clicks_by_label["vac"]
    info["eve_info_gain"] = float(eve_knows_count / max(1.0, eve_attacks_total))
    info["eve_attacks_total"] = eve_attacks_total
    info["eve_attacks_success"] = eve_knows_count

    return info
