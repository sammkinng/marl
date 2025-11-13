from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize,DummyVecEnv
from stable_baselines3 import PPO
from rl.agents.alice import AliceSingleAgentEnv



SIM_CFG = {
    "pulses_per_episode": 100000,
    "output_dir": "./results",
    "fiber_loss_db_per_km": 0.2,
    "distance_km": 50.0,
    "det_eff": 0.2,
    "dark_count": 1e-6,
    "baseline_qber": 0.01,
}


env = AliceSingleAgentEnv(SIM_CFG)
env = DummyVecEnv([lambda: env])
model = PPO.load("ppo_alice_qkdnbest.zip", env=env)



# ==============================
# RL-QKD Convergence Diagnostics
# ==============================

import numpy as np
import matplotlib.pyplot as plt


# ------------------------------
# 1. Load model and environment
# ------------------------------
# Replace these with your actual model and env creation
# from stable_baselines3 import PPO
# from qkd_env import QKDSimEnv   # <-- your custom env

# env = QKDSimEnv()
# model = PPO.load("trained_alice_model.zip", env=env)

# If your env is not vectorized, wrap it:
# if not hasattr(env, "num_envs"):
#     env = DummyVecEnv([lambda: env])

# ------------------------------
# 2. Evaluate mean and std reward
# ------------------------------
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=30, deterministic=True)
print("=== Policy Evaluation ===")
print(f"Mean Reward over 30 episodes: {mean_reward:.3f}")
print(f"Std Dev of Reward: {std_reward:.3f}\n")

# Interpretation:
# - If std_reward << mean_reward → stable behavior
# - If std_reward ≈ mean_reward or larger → unstable / not converged


# ------------------------------
# 3. Entropy / Action Dispersion
# ------------------------------
# Checks how uncertain or random the policy still is.
# (works for continuous-action models like PPO, SAC)
try:
    log_std = model.policy.log_std.detach().cpu().numpy().flatten()
    plt.plot(log_std)
    plt.title("Policy log-std (Entropy proxy)")
    plt.xlabel("Action Dimension Index")
    plt.ylabel("Log Std")
    plt.show()

    print(f"Average log std: {np.mean(log_std):.4f}")
    print("Expect: Decreasing but non-zero values for converged model.\n")
except Exception as e:
    print("Could not extract log_std; may not apply for your algorithm:", e)


# ------------------------------
# 4. Action Stability Across Episodes
# ------------------------------
# Collect actions over several evaluation runs and visualize
actions = []
for ep in range(10):
    obs = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        actions.append(np.array(action).flatten())
        obs, reward, done, info = env.step(action)
actions = np.array(actions)

plt.figure()
plt.boxplot(actions)
plt.title("Action Distribution (10 eval episodes)")
plt.xlabel("Action Dimension")
plt.ylabel("Action Value")
plt.show()

# Interpretation:
# - Narrow, consistent distributions → stable policy
# - Extremely broad / multimodal → still exploring or unstable


# ------------------------------
# 5. Multi-seed Robustness Check
# ------------------------------
# Tests if performance consistent under different RNG seeds
def test_agent(model, env, seed, n_episodes=5):
    env.seed(seed)
    total_rewards = []
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
        total_rewards.append(ep_reward)
    return np.mean(total_rewards)

seeds = [0, 42, 123, 2025, 9999]
mean_rewards = [test_agent(model, env, s) for s in seeds]

plt.figure()
plt.bar(range(len(seeds)), mean_rewards, tick_label=seeds)
plt.title("Cross-seed Reward Stability")
plt.xlabel("Seed")
plt.ylabel("Mean Episode Reward")
plt.show()

print("=== Cross-Seed Rewards ===")
for s, r in zip(seeds, mean_rewards):
    print(f"Seed {s}: Mean Reward = {r:.3f}")

print("\nExpect roughly similar rewards across seeds if converged.\n")


# ------------------------------
# 6. Learning Progress Proxy (Optional)
# ------------------------------
# If you saved checkpoints during training, you can reload earlier ones
# and compare their mean rewards here to see if learning stabilized.
# Example:
"""
checkpoints = ['model_100k.zip', 'model_500k.zip', 'model_final.zip']
rewards = []
for ckpt in checkpoints:
    m = PPO.load(ckpt, env=env)
    mr, _ = evaluate_policy(m, env, n_eval_episodes=10)
    rewards.append(mr)
plt.plot(checkpoints, rewards, marker='o')
plt.title('Reward Progress Across Checkpoints')
plt.ylabel('Mean Reward')
plt.show()
"""

