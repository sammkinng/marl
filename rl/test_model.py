

import os
import copy
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl.agents.train_alice import AliceSingleAgentEnv



SIM_CFG = {
    "pulses_per_episode": 100000,
    "results_csv": "alice_results.csv",
    "output_dir": "./results",
    "fiber_loss_db_per_km": 0.2,
    "distance_km": 50.0,
    "det_eff": 0.2,
    "dark_count": 1e-6,
    "baseline_qber": 0.01,
}

LOGDIR = "./logs_alice"
MODEL_PATH = os.path.join(LOGDIR, "ppo_alice.zip")
VECNORM   = os.path.join(LOGDIR, "vecnormalize_alice.pkl")

# Load env & stats
base_env = DummyVecEnv([lambda: AliceSingleAgentEnv(copy.deepcopy(SIM_CFG))])
base_env = VecNormalize.load(VECNORM, base_env)
base_env.training = False
base_env.norm_reward = False

model = PPO.load(MODEL_PATH,env=base_env)

param_sweeps = {
    "distance_km": np.linspace(10, 120, 10),
    "det_eff":     np.linspace(0.1, 0.3, 5),
    "dark_count":  np.linspace(1e-6, 1e-4, 5),
    "baseline_qber": np.linspace(0.01, 0.1, 10),
}

results = {}

for param, values in param_sweeps.items():
    skr_list, a0_list, a1_list, a2_list = [], [], [], []
    for v in values:
        cfg = copy.deepcopy(SIM_CFG)
        cfg[param] = float(v)
        env = DummyVecEnv([lambda: AliceSingleAgentEnv(cfg)])
        env = VecNormalize.load(VECNORM, env)
        env.training = False
        env.norm_reward = False

        # single rollout
        obs = env.reset()
        done = False
        ep_skr = []
        acts = []
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done,  info = env.step(action)
            skr = info[0]["raw_info"].get("SKR_bits_per_pulse", 0.0)
            ep_skr.append(skr)
            acts.append(action[0])
        env.close()
        skr_list.append(np.mean(ep_skr))
        a_mean = np.mean(np.array(acts), axis=0)
        a0_list.append(a_mean[0]); a1_list.append(a_mean[1]); a2_list.append(a_mean[2])

    results[param] = {"values": values, "skr": skr_list, "a0": a0_list, "a1": a1_list, "a2": a2_list}

# Plot
fig, axes = plt.subplots(2, 2, figsize=(12, 9)); axes = axes.flatten()
for i, param in enumerate(param_sweeps.keys()):
    r = results[param]
    ax = axes[i]; ax2 = ax.twinx()
    ax.plot(r["values"], r["a0"], "-o", label="a0 (mu_signal)")
    ax.plot(r["values"], r["a1"], "-o", label="a1 (mu_decoy)")
    ax.plot(r["values"], r["a2"], "-o", label="a2 (p_signal)")
    ax2.plot(r["values"], r["skr"], "--s", color="black", label="Mean SKR")
    ax.set_title(f"Policy Adaptation vs {param}")
    ax.set_xlabel(param); ax.set_ylabel("Action (normalized)"); ax2.set_ylabel("Mean SKR (bits/pulse)")
    ax.legend(loc="upper left"); ax2.legend(loc="upper right")
plt.tight_layout(); plt.savefig("sensitivity_alice_ummmm.png"); plt.close()