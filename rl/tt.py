import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

# ====== Import your environment ======
from rl.agents.train_alice import AliceSingleAgentEnv
from stable_baselines3.common.vec_env import DummyVecEnv


# ====== Load trained PPO model ======
model = PPO.load("ppo_alice_qkdppp100k.zip")  # adjust path

# ====== Helper function to evaluate ======
def evaluate_model(env, model, n_episodes=5):
    rewards, skr_values = [], []
    for _ in range(n_episodes):
        obs= env.reset()
        done, ep_reward, ep_skr = False, 0.0, []
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done,  info = env.step(action)
            ep_reward += reward
            skr_values.append(info[0].get("raw_info", {}).get("SKR_bits_per_pulse", 0))
        rewards.append(ep_reward)
        skr_values.append(np.mean(skr_values))
    return np.mean(rewards), np.mean(skr_values)

# ====== Test ranges ======
test_ranges = {
    "distance_km": np.linspace(10, 120, 12),
    "det_eff": np.linspace(0.1, 0.3, 5),
    "dark_count": np.logspace(-6, -4, 10),
    "baseline_qber": np.linspace(0.01, 0.1, 10),
}

# ====== Baseline parameters ======
base_params = dict(
    distance_km=50.0,
    pulses_per_episode=1000,
    output_dir="./results",
    results_csv= 'results12.csv',
    fiber_loss_db_per_km=0.2,
    det_eff=0.2,
    dark_count=1e-6,
    baseline_qber=0.01,
    mu_signal=0.6,
    mu_decoy=0.1,
    mu_vac=0.0,
    p_signal=0.7,
    p_decoy=0.2,
    p_vac=0.1,
    eta=1.0,
    recon_eff=1.1,
    basis_prob=0.5,
)

# ====== Run tests ======
results = {}

for param_name, param_values in test_ranges.items():
    skr_means, rew_means = [], []
    print(f"\nTesting sensitivity to {param_name}...")
    for val in param_values:
        params = base_params.copy()
        params[param_name] = float(val)
        # env = QKDEnv(**params)
        env = AliceSingleAgentEnv(params)
        env = DummyVecEnv([lambda: env])
        mean_rew, mean_skr = evaluate_model(env, model)
        skr_means.append(mean_skr)
        rew_means.append(mean_rew)
        print(f"  {param_name}={val:.4f} | Mean SKR={mean_skr:.5f}, Mean Reward={mean_rew:.4f}")
    results[param_name] = (param_values, skr_means, rew_means)

# ====== Plot results ======
plt.figure(figsize=(16, 12))
for i, (param, (vals, skr, _)) in enumerate(results.items(), 1):
    plt.subplot(2, 2, i)
    plt.plot(vals, skr, marker='o', lw=2)
    plt.xlabel(param)
    plt.ylabel("Mean SKR (bits/pulse)")
    plt.title(f"Sensitivity: {param}")
    plt.grid(True)

plt.tight_layout()
plt.savefig("sensitivity_analysis_alice2.png")
plt.close()
