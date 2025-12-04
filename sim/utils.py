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


def __qkd_simulation(config, aa,attacks,bb={}):
    """
    QKD Simulator with:
    - p_s and p_d now included in gain/QBER aggregation
    - p_pns included as a reduction of single-photon yield
    """

    pulses = config["pulses_per_episode"]
    pulse_rate = float(config["pulse_rate"])

    # Decoy intensities
    mu_s = aa.get("mus",config["mu_signal"])
    mu_d = aa.get("mud",config["mu_decoy"])
    mu_v = config["mu_vac"]     # unused (mu_v = 0)
    p_s  = aa.get("ps",config["p_signal"])
    p_d  = aa.get("pd",config["p_decoy"])
    

    # Channel
    loss_db_km = config["fiber_loss_db_per_km"]
    dist_km = config["distance_km"]

    # Hardware
    det_eff = bb.get("de",config["det_eff"])
    dark_count = bb.get("dc",float(config["dark_count"]))
    baseline_qber = config["baseline_qber"]
    recon_eff = config["recon_eff"]
    basis_prob = bb.get("bp",config["basis_prob"])

    # Attacks
    p_ir   = attacks.get("intercept_resend", 0.0)
    p_pns  = attacks.get("pns", 0.0)
    p_ts   = attacks.get("time_shift", 0.0)
    p_dci  = attacks.get("darkcount_increase", 0.0)

    # -----------------------------------------------------
    # Channel transmittance
    # -----------------------------------------------------
    eta_channel = 10 ** (-(loss_db_km * dist_km) / 10)

    det_eff_0 = det_eff * (1 + p_ts)
    det_eff_1 = det_eff * (1 - p_ts)
    det_eff_avg = (det_eff_0 + det_eff_1) / 2

    # Background (dark counts)
    Y0 = 2 * dark_count * (1 + p_dci)

    # -----------------------------------------------------
    # Gain/QBER for an intensity
    # -----------------------------------------------------
    def yield_from_mu(mu):
        eta_eff = eta_channel * det_eff_avg

        Q_mu = 1 - np.exp(-mu * eta_eff) + Y0
        E_ir = 0.25 * p_ir
        E_mu = baseline_qber + E_ir

        return Q_mu, E_mu

    # Gains for individual states
    Q_s_raw, E_s = yield_from_mu(mu_s)
    Q_d_raw, E_d = yield_from_mu(mu_d)
    Q_v = Y0

    # --------------------------------------------------------
    # MIXTURE USING p_s, p_d
    # --------------------------------------------------------
    Q_s = p_s * Q_s_raw + p_d * Q_d_raw    # mixture gain
    E_s = (p_s * E_s * Q_s_raw + p_d * E_d * Q_d_raw) / Q_s  # weighted QBER

    # --------------------------------------------------------
    # Decoy-state estimation
    # --------------------------------------------------------
    num = (
        Q_d_raw * np.exp(mu_d)
        - Q_s_raw * np.exp(mu_s) * (mu_d**2 / mu_s**2)
        - (1 - (mu_d**2 / mu_s**2)) * Q_v
    )
    den = (mu_s * mu_d - mu_d**2)

    if den <= 0:
        Y1_lower = 0
    else:
        Y1_lower = max((mu_s / den) * num, 0)

    # --- Apply PNS attack: remove fraction of single photons ---
    Y1_lower *= (1 - p_pns)

    # e1 upper bound
    den_e = Y1_lower * mu_d * (1 - mu_d / mu_s)
    if den_e <= 0:
        e1_upper = 0.5
    else:
        num_e = (
            E_d * Q_d_raw * np.exp(mu_d)
            - E_s * Q_s_raw * np.exp(mu_s) * (mu_d**2 / mu_s**2)
        )
        e1_upper = np.clip(num_e / den_e, 0, 0.5)

    # Single-photon gain
    Q1_lower = Y1_lower * mu_s * np.exp(-mu_s)

    # --------------------------------------------------------
    # Intercept-resend statistics (unchanged)
    # --------------------------------------------------------
    eve_resend_total = p_ir * pulses
    eve_resend_clicks = eve_resend_total * eta_channel * det_eff_avg
    eve_resend_errors = eve_resend_clicks * 0.25

    # --------------------------------------------------------
    # Secret Key Rate
    # --------------------------------------------------------
    H2 = lambda x: -x*np.log2(x + 1e-12) - (1-x)*np.log2(1-x + 1e-12)

    SKR_bits_per_pulse = basis_prob * (
        Q1_lower * (1 - H2(e1_upper))
        - Q_s * recon_eff * H2(E_s)
    )
    SKR_bits_per_pulse = max(SKR_bits_per_pulse, 0)
    SKR_bits_per_second = SKR_bits_per_pulse * pulse_rate


        # --------------------------------------------------------
    # Eve Information Gain (per pulse)
    # --------------------------------------------------------

    # 1. IR: BB84 intercept-resend gives Eve 0.5 bits of info per intercepted bit
    # (because she guesses the basis correctly 50% of the time)
    info_ir = 0.5 * p_ir

    # 2. PNS: Eve gets full information on multi-photon pulses.
    # Multi-photon probability for the mix (signal + decoy):
    def P_multi(mu): 
        return 1 - (1 + mu) * np.exp(-mu)

    P_multi_s = P_multi(mu_s)
    P_multi_d = P_multi(mu_d)
    P_multi_mix = p_s * P_multi_s + p_d * P_multi_d

    info_pns = P_multi_mix * p_pns   # full info on fraction of pulses attacked by PNS

    # 3. Time-shift info gain: Eve gets partial info depending on detector mismatch.
    # Larger mismatch => more bias => more Eve info.
    eta0 = det_eff * (1 + p_ts)
    eta1 = det_eff * (1 - p_ts)
    eta_mismatch = abs(eta0 - eta1) / max(eta0 + eta1, 1e-12)

    # TS gives Eve biased information proportionally:
    info_ts = 0.5 * eta_mismatch * p_ts

    # 4. Dark count injection gives Eve no direct info (just masks other attacks)
    info_dark = 0.0

    # Total information gain per pulse (bounded [0,1])
    info_gain_per_pulse = np.clip(info_ir + info_pns + info_ts + info_dark, 0.0, 1.0)

    # Total info gain per second
    eve_info_gain_total = info_gain_per_pulse * pulse_rate

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------
    info = {
        "Q_s": Q_s,
        "E_s": E_s,
        "Q_d": Q_d_raw,
        "Q_v": Q_v,

        "Y1_lower": Y1_lower,
        "e1_upper": e1_upper,
        "Q1_lower": Q1_lower,

        "eve_resend_total": eve_resend_total,
        "eve_resend_clicks": eve_resend_clicks,
        "eve_resend_errors": eve_resend_errors,

        "SKR_bits_per_pulse": SKR_bits_per_pulse,
        "SKR_bits_per_second": SKR_bits_per_second
    }

    info["info_gain_per_pulse"] = info_gain_per_pulse
    info["eve_info_gain_total"] = eve_info_gain_total
    info["eta_mismatch"] = eta_mismatch
    info["P_multi_mix"] = P_multi_mix

    # Mutual information Bob gets
    I_AB = 1.0 - H2(E_s)


    info["I_AB"] = max(min(I_AB, 1.0), 0.0)



    return info

def qkd_simulation(config, aa, attacks, bb={}):
    """
    Fast analytical QKD simulator (decoy-state BB84) with:
    - Alice controls: mus, mud, ps, pd via 'aa' dict
    - Eve controls: intercept_resend, pns, time_shift, darkcount_increase via 'attacks' dict
    - Bob controls (realistic):
        * effective detector efficiency 'de'   (0..1)
        * effective dark count      'dc'       (>0)
      Basis probability is taken from the protocol config and is NOT dynamically
      controlled by Bob (that would be unphysical for BB84).
    """

    pulses = config["pulses_per_episode"]
    pulse_rate = float(config["pulse_rate"])

    # ----------------------------
    # Alice parameters (aa)
    # ----------------------------
    mu_s = aa.get("mus", config["mu_signal"])
    mu_d = aa.get("mud", config["mu_decoy"])
    mu_v = config["mu_vac"]     # unused (mu_v = 0)
    p_s  = aa.get("ps",  config["p_signal"])
    p_d  = aa.get("pd",  config["p_decoy"])

    # ----------------------------
    # Channel
    # ----------------------------
    loss_db_km = config["fiber_loss_db_per_km"]
    dist_km    = config["distance_km"]

    # ----------------------------
    # Hardware (base)
    # ----------------------------
    base_det_eff  = config["det_eff"]
    base_dark_cnt = float(config["dark_count"])
    baseline_qber = config["baseline_qber"]
    recon_eff     = config["recon_eff"]
    # In real BB84, basis probability is fixed by the protocol
    basis_prob    = config["basis_prob"]

    # ----------------------------
    # Bob parameters (bb) – realistic knobs
    # ----------------------------
    # Bob can effectively change the detector efficiency (threshold/gain)
    det_eff = float(bb.get("de", base_det_eff))
    # Clip into [0,1] physical range
    det_eff = float(np.clip(det_eff, 0.0, 1.0))

    # Bob can effectively modify dark count (looser/tighter threshold)
    dark_count = float(bb.get("dc", base_dark_cnt))
    # Ensure non-negative
    dark_count = max(dark_count, 0.0)

    # How much Bob has changed efficiency relative to baseline
    rel_det_gain = det_eff / max(base_det_eff, 1e-12)

    # ----------------------------
    # Eve attacks
    # ----------------------------
    p_ir  = attacks.get("intercept_resend",   0.0)
    p_pns = attacks.get("pns",                0.0)
    p_ts  = attacks.get("time_shift",         0.0)
    p_dci = attacks.get("darkcount_increase", 0.0)

    # -----------------------------------------------------
    # Channel transmittance
    # -----------------------------------------------------
    eta_channel = 10 ** (-(loss_db_km * dist_km) / 10)

    det_eff_0 = det_eff * (1 + p_ts)
    det_eff_1 = det_eff * (1 - p_ts)
    det_eff_avg = (det_eff_0 + det_eff_1) / 2.0

    # Background (dark counts)
    Y0 = 2.0 * dark_count * (1.0 + p_dci)

    # -----------------------------------------------------
    # Gain/QBER for an intensity
    # -----------------------------------------------------
    def yield_from_mu(mu):
        eta_eff = eta_channel * det_eff_avg

        Q_mu = 1.0 - np.exp(-mu * eta_eff) + Y0
        E_ir = 0.25 * p_ir
        E_mu = baseline_qber + E_ir

        return Q_mu, E_mu

    # Gains for individual states
    Q_s_raw, E_s = yield_from_mu(mu_s)
    Q_d_raw, E_d = yield_from_mu(mu_d)
    Q_v          = Y0

    # --------------------------------------------------------
    # MIXTURE USING p_s, p_d
    # --------------------------------------------------------
    Q_s = p_s * Q_s_raw + p_d * Q_d_raw    # mixture gain
    Q_s = max(Q_s, 1e-12)                  # avoid division by 0

    E_s = (p_s * E_s * Q_s_raw + p_d * E_d * Q_d_raw) / Q_s  # weighted QBER

    # --------------------------------------------------------
    # Decoy-state estimation
    # --------------------------------------------------------
    num = (
        Q_d_raw * np.exp(mu_d)
        - Q_s_raw * np.exp(mu_s) * (mu_d**2 / mu_s**2)
        - (1.0 - (mu_d**2 / mu_s**2)) * Q_v
    )
    den = (mu_s * mu_d - mu_d**2)

    if den <= 0:
        Y1_lower = 0.0
    else:
        Y1_lower = max((mu_s / den) * num, 0.0)

    # --- Apply PNS attack: remove fraction of single photons ---
    Y1_lower *= (1.0 - p_pns)

    # e1 upper bound
    den_e = Y1_lower * mu_d * (1.0 - mu_d / mu_s)
    if den_e <= 0:
        e1_upper = 0.5
    else:
        num_e = (
            E_d * Q_d_raw * np.exp(mu_d)
            - E_s * Q_s_raw * np.exp(mu_s) * (mu_d**2 / mu_s**2)
        )
        e1_upper = float(np.clip(num_e / den_e, 0.0, 0.5))

    # Single-photon gain
    Q1_lower = Y1_lower * mu_s * np.exp(-mu_s)

    # --------------------------------------------------------
    # Intercept-resend statistics (unchanged analytically)
    # --------------------------------------------------------
    eve_resend_total   = p_ir * pulses
    eve_resend_clicks  = eve_resend_total * eta_channel * det_eff_avg
    eve_resend_errors  = eve_resend_clicks * 0.25

    # --------------------------------------------------------
    # Secret Key Rate
    # --------------------------------------------------------
    H2 = lambda x: -x*np.log2(x + 1e-12) - (1.0 - x)*np.log2(1.0 - x + 1e-12)

    SKR_bits_per_pulse = basis_prob * (
        Q1_lower * (1.0 - H2(e1_upper))
        - Q_s * recon_eff * H2(E_s)
    )
    SKR_bits_per_pulse = max(SKR_bits_per_pulse, 0.0)
    SKR_bits_per_second = SKR_bits_per_pulse * pulse_rate

    # --------------------------------------------------------
    # Eve Information Gain (per pulse)
    # --------------------------------------------------------

    # 1. IR: BB84 intercept-resend gives Eve ~0.5 bits per intercepted pulse.
    # We keep the same functional form so old results remain comparable,
    # and Bob does NOT unrealistically change this via basis_prob.
    info_ir = 0.5 * p_ir

    # 2. PNS: Eve gets full information on multi-photon pulses.
    def P_multi(mu):
        return 1.0 - (1.0 + mu) * np.exp(-mu)

    P_multi_s   = P_multi(mu_s)
    P_multi_d   = P_multi(mu_d)
    P_multi_mix = p_s * P_multi_s + p_d * P_multi_d

    info_pns = P_multi_mix * p_pns

    # 3. Time-shift info gain: Eve gets partial info depending on detector mismatch.
    # Original model: info_ts_base = 0.5 * |p_ts| * p_ts (independent of det_eff).
    # To give Bob a realistic lever, we scale TS info by rel_det_gain:
    #  - Lower det_eff (tighter threshold) -> TS harder to exploit -> less info_ts
    #  - Higher det_eff -> TS slightly more exploitable.
    eta0 = det_eff * (1.0 + p_ts)
    eta1 = det_eff * (1.0 - p_ts)
    eta_mismatch = abs(eta0 - eta1) / max(eta0 + eta1, 1e-12)  # ~ |p_ts|

    info_ts_base = 0.5 * eta_mismatch * p_ts
    info_ts = info_ts_base * rel_det_gain  # Bob’s det_eff scales TS info

    # 4. Dark count injection gives Eve no direct info (just masks other attacks)
    # But *Bob's* own dark_count choice can dilute Eve's effective information:
    # higher dark_count -> more random clicks -> Eve's fraction of "useful" bits drops.
    info_dark = 0.0

    # Base total information gain per pulse
    info_gain_per_pulse = info_ir + info_pns + info_ts + info_dark

    # Noise dilution: if Bob pushes dark_count above baseline, reduce EveInfo more than SKR.
    noise_excess = max(0.0, (dark_count - base_dark_cnt) / max(base_dark_cnt, 1e-12))
    noise_excess = np.clip(noise_excess, 0.0, 1.0)
    # At most 50% reduction in EveInfo for very noisy detectors
    info_gain_per_pulse *= (1.0 - 0.5 * noise_excess)

    # Bound in [0,1]
    info_gain_per_pulse = float(np.clip(info_gain_per_pulse, 0.0, 1.0))

    # Total info gain per second
    eve_info_gain_total = info_gain_per_pulse * pulse_rate

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------
    info = {
        "Q_s": Q_s,
        "E_s": E_s,
        "Q_d": Q_d_raw,
        "Q_v": Q_v,

        "Y1_lower": Y1_lower,
        "e1_upper": e1_upper,
        "Q1_lower": Q1_lower,

        "eve_resend_total":  eve_resend_total,
        "eve_resend_clicks": eve_resend_clicks,
        "eve_resend_errors": eve_resend_errors,

        "SKR_bits_per_pulse":  SKR_bits_per_pulse,
        "SKR_bits_per_second": SKR_bits_per_second,
    }

    info["info_gain_per_pulse"] = info_gain_per_pulse
    info["eve_info_gain_total"] = eve_info_gain_total
    info["eta_mismatch"]        = eta_mismatch
    info["P_multi_mix"]         = P_multi_mix

    # Mutual information Bob gets
    I_AB = 1.0 - H2(E_s)
    info["I_AB"] = max(min(I_AB, 1.0), 0.0)

    # For debugging/analysis
    info["det_eff"]      = det_eff
    info["dark_count"]   = dark_count
    info["rel_det_gain"] = rel_det_gain

    return info

