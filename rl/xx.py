import numpy as np
import matplotlib.pyplot as plt
from sim.core_sim import QKDSimulator   # adapt import if needed
import copy

base_cfg = {
    "pulses_per_episode": 100000,
    "output_dir": "./results",
    "fiber_loss_db_per_km": 0.2,
    "distance_km": 50.0,   # change to test different distances
    "det_eff": 0.2,
    "dark_count": 1e-6,
    "baseline_qber": 0.01,
}

def sweep_mu(distance_km, mu_list):
    cfg = copy.deepcopy(base_cfg)
    cfg["distance_km"] = distance_km
    sim = QKDSimulator(cfg)
    skr_pps, skr_pss, qbers, y1s, q1s, e1s, detect_probs = [], [], [], [], [], [], []
    for mu in mu_list:
        actions = {
            "Alice": {"mu_signal": float(mu), "mu_decoy": 0.1, "p_signal": 0.7, "p_decoy":0.2, "p_vac":0.1},
            "Eve": {"type": "none"}   # disable Eve
        }
        info = sim.run_episode(actions=actions, verbose=False)
        skr_pps.append(info.get("SKR_bits_per_pulse", 0.0))
        skr_pss.append(info.get("SKR_bits_per_second", 0.0))
        qbers.append(info.get("E_s", 0.0))
        y1s.append(info.get("Y1", 0.0))
        q1s.append(info.get("Q1", 0.0))
        e1s.append(info.get("e1", 0.0))
        detect_probs.append(info.get("Q_s", 0.0) + info.get("Q_d", 0.0) + info.get("Q_v",0.0))
    return skr_pps, skr_pss, qbers, y1s, q1s, e1s, detect_probs

# run for a few distances
mus = np.linspace(0.05, 1.5, 25)
for d in [20.0, 50.0, 80.0]:
    skr_pp, skr_ps, qber, y1, q1, e1, dp = sweep_mu(d, mus)
    plt.figure(figsize=(8,4))
    plt.plot(mus, skr_pp, '-o', label='SKR (bits/pulse)')
    plt.plot(mus, qber, '-x', label='QBER')
    plt.title(f'distance={d} km')
    plt.xlabel('mu_signal')
    plt.legend()
    plt.savefig(f'sweep_mu_distance_{int(d)}km.png')
plt.close('all')
