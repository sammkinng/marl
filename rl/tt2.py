import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from rl.agents.alice import AliceSingleAgentEnv
from stable_baselines3.common.vec_env import DummyVecEnv


import copy

# === Load model ===
model = PPO.load("ppo_supernrs.zip")  # adjust path


# === Base simulator config ===
base_cfg = {
    "pulses_per_episode": 1000,
    "output_dir": "./results",
    "results_csv": "test_results.csv",
    "fiber_loss_db_per_km": 0.2,
    "distance_km": 50.0,
    "det_eff": 0.2,
    "dark_count": 1e-6,
    "baseline_qber": 0.01,
}

# === Helper to run one evaluation ===
def evaluate_model(env, model, n_episodes=3):
    skr_values = []
    actions = []
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, term, info = env.step(action)
            done = term
            skr = info[0]["raw_info"].get("SKR_bits_per_pulse", 0)
            skr_values.append(skr)
            actions.append(action)
    mean_skr = np.mean(skr_values)
    mean_action = np.mean(np.array(actions), axis=0)
    return mean_skr, mean_action


# === Parameter sweeps ===
param_sweeps = {
    "distance_km": np.linspace(10, 120, 10),
    "det_eff": np.linspace(0.1, 0.3, 5),
    "dark_count": np.linspace(1e-6, 1e-4, 5),
    "baseline_qber": np.linspace(0.01, 0.1, 10),
}

results = {}

for param, values in param_sweeps.items():
    print(f"\nTesting sensitivity to {param}...")
    skr_list, mu_signal_list, mu_decoy_list, p_signal_list = [], [], [], []

    for v in values:
        cfg = copy.deepcopy(base_cfg)
        cfg[param] = float(v)
        env = AliceSingleAgentEnv(cfg)
        env = DummyVecEnv([lambda: env])
        mean_skr, mean_action = evaluate_model(env, model)
        env.close()

        mean_action = np.atleast_1d(mean_action)
        if mean_action.shape[0] == 1:
            mean_action = np.array([mean_action[0], mean_action[0], mean_action[0]])


        skr_list.append(mean_skr)
        mu_signal_list.append(mean_action[0])
        mu_decoy_list.append(mean_action[1])
        p_signal_list.append(mean_action[2])

        

        print(f"  {param}={v:.4f} | SKR={mean_skr:.5f} | actions={mean_action}")

    results[param] = {
        "values": values,
        "skr": skr_list,
        "mu_signal": mu_signal_list,
        "mu_decoy": mu_decoy_list,
        "p_signal": p_signal_list,
    }

# print(results)
# === Plot results ===
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
axes = axes.flatten()

for i, param in enumerate(param_sweeps.keys()):
    r = results[param]
    ax = axes[i]
    ax2 = ax.twinx()
    ax.plot(r["values"], r["mu_signal"], "-o", label="μ_signal")
    ax.plot(r["values"], r["mu_decoy"], "-o", label="μ_decoy")
    ax.plot(r["values"], r["p_signal"], "-o", label="p_signal")
    ax2.plot(r["values"], r["skr"], "--s", color="black", label="Mean SKR")

    ax.set_title(f"Policy Adaptation vs {param}")
    ax.set_xlabel(param)
    ax.set_ylabel("Action Values")
    ax2.set_ylabel("Mean SKR (bits/pulse)")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")

plt.tight_layout()
plt.savefig("kagglenrssensitivity_analysis_alice_actions.png")
plt.close()
