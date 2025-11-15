from sim.core_sim import QKDSimulator


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


sim=QKDSimulator(SIM_CFG)

info = sim.run_episode(actions=sim_actions, verbose=True)
print(info)

