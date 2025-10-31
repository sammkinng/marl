# run_experiments.py
import os
import csv
import numpy as np
from sim.core_sim import QKDSimulator,TimeShiftAttack,InterceptResendAttack,DarkCountAttack,CompositeAttack

def run_experiment(cfg):
    os.makedirs(cfg["output_dir"], exist_ok=True)
    csv_file = os.path.join(cfg["output_dir"], cfg["results_csv"])

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

                # info = sim.run_episode(verbose=True)

                # ts = TimeShiftAttack(attack_prob=0.2, shift_frac=0.3, rng=sim.rng)
                # ir = InterceptResendAttack(intercept_prob=0.01, resend_mu=1.0, rng=sim.rng)
                # dc = DarkCountAttack(extra_dark_prob=1e-6, rng=sim.rng)
                # comp = CompositeAttack([ts, ir, dc], rng=sim.rng)
                # info = sim.run_episode(actions={"Eve": comp}, verbose=True)




                # build attack(s) from cfg["attack"] dynamically
                attack_cfg = cfg.get("attack", None)
                def _make_attack(ac):
                    t = ac.get("type")
                    params = {k: v for k, v in ac.items() if k != "type"}
                    # always provide simulator rng if available
                    params["rng"] = sim.rng
                    if t == "time_shift":
                        return TimeShiftAttack(**params)
                    if t == "intercept_resend":
                        return InterceptResendAttack(**params)
                    if t == "dark_count":
                        return DarkCountAttack(**params)
                    raise ValueError(f"Unknown attack type: {t}")

                if attack_cfg is None:
                    actions = {}
                elif attack_cfg.get("type") == "composite":
                    subs = attack_cfg.get("sub_attacks", [])
                    sub_objs = [_make_attack(sub) for sub in subs]
                    comp = CompositeAttack(sub_objs, rng=sim.rng)
                    actions = {"Eve": comp}
                else:
                    actions = {"Eve": _make_attack(attack_cfg)}

                info = sim.run_episode(actions=actions, verbose=True)



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
