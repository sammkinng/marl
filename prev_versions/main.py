from prev_versions.QKD_SIM import QKDSimulator, load_config
import os
import csv
import numpy as np

# ------------------------- Example usage & sample config -------------------------
EXAMPLE_CONFIG = {
    "pulses_per_episode": 50000,
    "pulse_rate": 10e6,
    "mu_signal": 0.6,
    "mu_decoy": 0.1,
    "mu_vac": 0.0,
    "p_signal": 0.7,
    "p_decoy": 0.25,
    "p_vac": 0.05,
    # Option: provide eta directly OR provide fiber loss + distance
    # "eta": 0.1,
    "fiber_loss_db_per_km": 0.2,
    "distance_km": 20.0,
    "det_eff": 0.6,
    "dark_count": 1e-7,
    "baseline_qber": 0.005,
    "recon_eff": 1.16,
    "basis_prob": 0.5,
    "seed": 2024,
    # simple attack example (comment out or set type: none)
    "attack": {"type": "time_shift", "attack_prob": 0.0, "shift_frac": 0.2}
}


def main_demo():
    # Write example YAML (if pyyaml installed) or JSON file for user
    # conf_path_yaml = "qkd_config_example.yaml"
    # conf_path_json = "qkd_config_example.json"
    # try:
    #     import yaml  # type: ignore
    #     with open(conf_path_yaml, "w") as f:
    #         yaml.safe_dump(EXAMPLE_CONFIG, f)
    #     used_path = conf_path_yaml
    # except Exception:
    #     with open(conf_path_json, "w") as f:
    #         json.dump(EXAMPLE_CONFIG, f, indent=2)
    #     used_path = conf_path_json

    # print(f"Wrote example config to {used_path}")

    used_path = "config.yaml"  # specify your config path here

    # Load config and run a demo episode (short)
    cfg = load_config(used_path)

    # Prepare results CSV
    os.makedirs(cfg['output_dir'], exist_ok=True)
    csv_file = os.path.join(cfg['output_dir'], 'baseline_results.csv')
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['distance_km', 'mu_signal', 'seed', 'SKR_bits_per_s', 'SKR_bits_per_pulse', 'QBER'])

    # Outer loops: distance -> mu -> seeds
    for distance in cfg.get('sweep_distance_km', [cfg['distance_km']]):
        cfg['distance_km'] = distance
        for mu in cfg.get('sweep_mu', [cfg['mu_signal']]):
            cfg['mu_signal'] = mu
            skr_list = []
            qber_list = []

            for seed in cfg.get('seeds', [cfg.get('seed', 42)]):
                cfg['seed'] = seed
                # Run a single episode
                sim = QKDSimulator(cfg)
                info = sim.run_episode(actions=None, verbose=True)
                result=info

                # Save metrics for statistics
                skr_list.append(result['SKR_bits_per_second'])
                qber_list.append(result['E_s'])

                # Write raw result to CSV
                with open(csv_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([distance, mu, seed, result['SKR_bits_per_second'], 
                                    result['SKR_bits_per_pulse'], result['E_s']])

            # Compute mean/std over seeds for this (distance, mu) combination
            mean_skr = np.mean(skr_list)
            std_skr = np.std(skr_list)
            mean_qber = np.mean(qber_list)
            std_qber = np.std(qber_list)

            print(f"Distance: {distance} km | μ: {mu} | SKR: {mean_skr:.2f}±{std_skr:.2f} bits/s | "
                f"QBER: {mean_qber:.4f}±{std_qber:.4f}")






    # used_path = "config.yaml"  # specify your config path here

    # # Load config and run a demo episode (short)
    # cfg = load_config(used_path)



    # seeds = cfg.get('seeds', [cfg.get('seed', 42)])  # default if no seeds

    # # results = []

    # for seed in seeds:
    #     print(f"Running episode with seed {seed}")
    #     cfg['seed'] = seed
        
    #     sim = QKDSimulator(cfg)
    #     info = sim.run_episode(actions=None, verbose=True)

    #     # results.append(info)
        
    #     if cfg.get('log_per_episode', False):
    #         os.makedirs(cfg['output_dir'], exist_ok=True)
    #         with open(f"{cfg['output_dir']}/log_seed_{seed}.txt", "w") as f:
    #             f.write(str(info))


    # for k, v in info.items():
    #     if isinstance(v, float):
    #         print(f"  {k}: {v:.6e}")
    #     else:
    #         print(f"  {k}: {v}")

if __name__ == "__main__":
    main_demo()

"""
Notes for extension:
- Hook this simulator to a Gym/Stable-Baselines environment wrapper: observations = recent Q_s, E_s, rolling QBER, and past actions.
- Define actions for Alice (mu_signal, p_signal), Bob (gate width, basis bias), Eve (attack type + parameters).
- Add logging callbacks and experiment manifest runner (CSV) for reproducibility.
- For NetSquid-level fidelity, wrap NetSquid circuits behind the same API (simulate_pulse) and use the same obs/action interfaces.
"""
