# run_experiments.py
import os
import csv
import numpy as np
from sim.core_sim import QKDSimulator

def run_experiment(cfg):
    os.makedirs(cfg["output_dir"], exist_ok=True)
    csv_file = os.path.join(cfg["output_dir"], "results.csv")

    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["distance_km", "mu_signal", "seed",
                         "SKR_bits_per_second", "SKR_bits_per_pulse", "QBER"])

    for distance in cfg.get("sweep_distance_km", [cfg["distance_km"]]):
        cfg["distance_km"] = distance
        for mu in cfg.get("sweep_mu", [cfg["mu_signal"]]):
            cfg["mu_signal"] = mu
            skr_list, qber_list = [], []
            for seed in cfg.get("seeds", [cfg.get("seed", 42)]):
                cfg["seed"] = seed
                sim = QKDSimulator(cfg)
                info = sim.run_episode(verbose=True)
                skr_list.append(info["SKR_bits_per_second"])
                qber_list.append(info["E_s"])
                with open(csv_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([distance, mu, seed,
                                     info["SKR_bits_per_second"],
                                     info["SKR_bits_per_pulse"], info["E_s"]])
            print(f"Distance {distance} km | μ={mu:.2f} | "
                  f"SKR={np.mean(skr_list):.2f} ± {np.std(skr_list):.2f} bits/s | "
                  f"QBER={np.mean(qber_list):.4f} ± {np.std(qber_list):.4f}")

if __name__ == "__main__":
    import yaml
    cfg = yaml.safe_load(open("configs/config.yaml"))
    run_experiment(cfg)
