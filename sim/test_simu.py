from sim.core_sim import QKDSimulator

import numpy as np

SIM_CFG = {
        "pulses_per_episode": 100000,
        "output_dir": "./results",
        "results_csv": "alice_results.csv",
        "fiber_loss_db_per_km": 0.2,
        "distance_km": 50.0,
        "det_eff": 0.2,
        "dark_count": 1e-6,
        "baseline_qber": 0.01,
    }

sim_actions = {
                "Alice": {
                    "mu_signal": 0.6,
                    "mu_decoy": 0.1,
                    "p_signal": 0.6,
                    "p_decoy": 0.3,
                    "p_vac": 0.1,
                },
                "Eve": None
            }


raw_ts, raw_pns, raw_ir=[0.563,0.449,-1.056]
raw_dark=-0.00855

logits = np.array([raw_ir, raw_pns, raw_ts], dtype=np.float32)
exp_logits = np.exp(logits - np.max(logits))
p = exp_logits / np.sum(exp_logits)
p_ir, p_pns, p_ts = p.tolist()

# # Sigmoid for dark-boost
dark_boost = 1.0 / (1.0 + np.exp(-raw_dark))

# p_ir, p_pns, p_ts = [1.0,0.0,0.0]
# dark_boost = 1.0

print(p_ts,p_pns,p_ir,dark_boost)
print(p_ts+p_pns+p_ir)

max_extra_dark = 5e-6
extra_dark_prob = dark_boost * max_extra_dark

sim_actions["Eve"] = {"type": "composite",
                            "sub_attacks":[
    {
        "type": "intercept_resend",
        "intercept_prob": p_ir,
        "resend_eff": 0.8,
        "resend_error_prob": 0.05,
    },
    {
        "type": "pns",
        "pns_frac": p_pns,
    },
    {
        "type": "time_shift",
        "attack_prob": p_ts,
        "shift_frac": 0.2,
        "timing_qber_delta": 0.01,
    },
    {
        "type": "dark_count",
        "extra_dark_prob": extra_dark_prob,
    }
]}

sim=QKDSimulator(SIM_CFG)

info = sim.run_episode(actions=sim_actions)

print(info)