# ===============================
# RL–QKD Physical Alignment Tests
# ===============================

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize,DummyVecEnv
from stable_baselines3 import PPO
from rl.agents.alice import AliceSingleAgentEnv
import numpy as np
import matplotlib.pyplot as plt
import os
import copy

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


LOGDIR = "./logs_alice"
MODEL_PATH = os.path.join(LOGDIR, "best_model.zip")
VECNORM   = os.path.join(LOGDIR, "vecnormalize_alice.pkl")

# Load env & stats
base_env = DummyVecEnv([lambda: AliceSingleAgentEnv(copy.deepcopy(SIM_CFG))])
base_env = VecNormalize.load(VECNORM, base_env)
base_env.training = False
base_env.norm_reward = False

env = DummyVecEnv([lambda: AliceSingleAgentEnv(SIM_CFG)])
env = VecNormalize.load(VECNORM, env)
env.training = False
env.norm_reward = False

model = PPO.load(MODEL_PATH,env=base_env)




# ------------------------------
# 1. Evaluate Agent Over Channel Conditions (multi-wrapper safe)
# ------------------------------
print("=== Running Physics Alignment Evaluation ===")

fiber_lengths = [10, 20, 40, 60, 80, 100]  # km
mean_skr, mean_qber, mean_actions = [], [], []

# --- unwrap nested wrappers automatically ---
def get_qkd_simulator(env):
    """
    Recursively unwrap DummyVecEnv / custom wrappers to reach QKDSimulator.
    Works for: DummyVecEnv -> AgentAlice -> QKDEnv -> QKDSimulator
    """
    sim = env
    # unwrap vectorized environments
    if hasattr(sim, "envs"):
        sim = sim.envs[0]
    # unwrap AgentAlice -> QKDEnv
    if hasattr(sim, "env"):
        sim = sim.env
    # unwrap QKDEnv -> QKDSimulator
    if hasattr(sim, "sim"):
        sim = sim.sim
    return sim

def run_physics_alignment_tests(env, model, fiber_lengths,mean_skr, mean_qber, mean_actions):
    sim = get_qkd_simulator(env)
    print(f"Detected simulator class: {type(sim).__name__}")

    if not hasattr(sim, "set_fiber_length"):
        print("⚠️ Simulator missing set_fiber_length(). Running single-condition test.")
        fiber_lengths = [0]

    for L in fiber_lengths:
        if hasattr(sim, "set_fiber_length"):
            sim.set_fiber_length(L)
            print(f"→ Testing at {L} km (eta={sim.eta:.3e})")

        total_skr, total_qber, total_actions = [], [], []

        for _ in range(10):
            obs = env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)

                total_actions.append(np.array(action).flatten())

                # SB3 can return tuple (obs, rewards, dones, infos) from VecEnv
                if isinstance(info, (list, tuple)):
                    info = info[0]['raw_info']

                # try to extract SKR/QBER from info
                if isinstance(info, dict):
                    if "SKR_bits_per_pulse" in info:
                        total_skr.append(info["SKR_bits_per_pulse"])
                    elif "skr" in info:
                        total_skr.append(info["skr"])
                    if "E_s" in info:
                        total_qber.append(info["E_s"])
                    elif "qber" in info:
                        total_qber.append(info["qber"])

            # safeguard if missing
            if len(total_skr) == 0:
                total_skr.append(0.0)
            if len(total_qber) == 0:
                total_qber.append(0.0)

        mean_skr.append(np.mean(total_skr))
        mean_qber.append(np.mean(total_qber))
        mean_actions.append(np.mean(total_actions, axis=0))
        

    print("SKR samples collected:", mean_skr)
    print("QBER samples collected:", mean_qber)
    return mean_skr, mean_qber, mean_actions


mean_skr,mean_qber,mean_actions=run_physics_alignment_tests(   env,model,fiber_lengths,mean_skr,mean_qber,mean_actions)


# ------------------------------
# 2. Plot SKR and QBER vs Fiber Length
# ------------------------------
plt.figure()
plt.plot(fiber_lengths, mean_skr, marker='o')
plt.xlabel("Fiber Length (km)")
plt.ylabel("Mean SKR (bps or bits/pulse)")
plt.title("SKR vs Fiber Length")
plt.grid(True)
plt.savefig("skr_vs_distancextralastest.png")

plt.figure()
plt.plot(fiber_lengths, mean_qber, marker='o')
plt.xlabel("Fiber Length (km)")
plt.ylabel("Mean QBER")
plt.title("QBER vs Fiber Length")
plt.grid(True)
plt.savefig("qber_vs_distancextralastest.png")

# ------------------------------
# 3. Plot Agent's Action Trends vs Distance
# ------------------------------
mean_actions = np.array(mean_actions)
n_actions = mean_actions.shape[1]

for i in range(n_actions):
    plt.figure()
    plt.plot(fiber_lengths, mean_actions[:, i], marker='s')
    plt.xlabel("Fiber Length (km)")
    plt.ylabel(f"Action {i+1} value")
    plt.title(f"Action {i+1} trend vs distance")
    plt.grid(True)
    plt.savefig(f"action{i+1}_vs_distancextralastest.png")



# ------------------------------
# 4. Perturbation Sensitivity Test (multi-wrapper safe)
# ------------------------------

def perturb_and_test(env, model, param_name, delta, trials=5):
    """
    Perturb a simulator parameter (like detector efficiency or dark count)
    by ±delta and check if SKR changes logically.
    Automatically finds the QKDSimulator inside nested wrappers.
    """
    sim = get_qkd_simulator(env)

    getter_name = f"get_{param_name}"
    setter_name = f"set_{param_name}"

    get_func = getattr(sim, getter_name, None)
    set_func = getattr(sim, setter_name, None)

    if get_func is None or set_func is None:
        print(f"⚠️ Simulator missing getter/setter for {param_name}")
        return None

    base_value = get_func()
    print(f"🔬 Testing sensitivity of {param_name} around {base_value:.4f} ± {delta}")

    skr_values = []

    for d in [-delta, 0, +delta]:
        perturbed_value = base_value + d
        set_func(perturbed_value)
        print(f" → Setting {param_name} = {perturbed_value:.4f}")

        total_skr = []
        for _ in range(trials):
            obs = env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)

                # print(info)

                if isinstance(info, (list, tuple)):
                    info = info[0]['raw_info']
                


                if isinstance(info, dict):
                    if "SKR_bits_per_pulse" in info:
                        # print("here")
                        total_skr.append(info["SKR_bits_per_pulse"])
                    elif "skr" in info:
                        total_skr.append(info["skr"])
            # print(total_skr)

        skr_values.append(np.mean(total_skr))

    # restore base value
    set_func(base_value)
    return skr_values



# Example test for detector efficiency sensitivity:
skr_det_eff = perturb_and_test(env,model,"detector_efficiency", 0.05)
if skr_det_eff:
    plt.figure()
    plt.plot([-0.05, 0, +0.05], skr_det_eff, marker='o')
    plt.title("SKR sensitivity to detector efficiency ±5%")
    plt.xlabel("Perturbation Δη")
    plt.ylabel("SKR")
    plt.grid(True)
    plt.savefig("skr_vs_detector_efficiencyxtralastest.png")



# Run the test (if not done yet)
skr_dark = perturb_and_test(env, model, "dark_count", 1e-6)

if skr_dark is None or len(skr_dark) == 0:
    print("⚠️ No SKR data returned — check perturb_and_test implementation.")
else:
    # The x-axis perturbations correspond to [-Δ, 0, +Δ]
    delta = 1e-6
    perturbations = np.array([-delta, 0, +delta])
    
    plt.figure(figsize=(6,4))
    plt.plot(perturbations, skr_dark, marker='o', linewidth=2)
    plt.title("SKR Sensitivity to Dark Count Probability ±1e-6")
    plt.xlabel("Δ dark_count (per detector per pulse)")
    plt.ylabel("Mean SKR (bits/pulse)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("skr_vs_dark_countxtralastest.png", dpi=300)
    plt.show()

    # Optional: print numerical results for clarity
    print("Perturbations:", perturbations)
    print("SKR values:", [f"{v:.4e}" for v in skr_dark])


# ------------------------------
# 5. Cross-check with Theoretical Expectation
# ------------------------------
# You can compare RL's SKR to theoretical decoy-state BB84 estimates.

def theoretical_skr(eta, qber, mu=0.5):
    """Very rough analytical SKR estimate (Ma 2005-style simplified)"""
    from math import log2
    H2 = lambda x: -x * np.log2(x) - (1 - x) * np.log2(1 - x) if 0 < x < 1 else 0
    return eta * (1 - H2(qber)) * mu  # simplistic: scales linearly with η and (1-H2(QBER))

eta_values = [10 ** (-0.2 * L / 10) for L in fiber_lengths]  # fiber loss ≈ 0.2 dB/km
skr_theory = [theoretical_skr(eta, q) for eta, q in zip(eta_values, mean_qber)]

plt.figure()
plt.plot(fiber_lengths, mean_skr, 'o-', label='RL Agent')
plt.plot(fiber_lengths, skr_theory, 's--', label='Theoretical (approx)')
plt.xlabel("Fiber Length (km)")
plt.ylabel("SKR (normalized)")
plt.title("RL vs Theoretical SKR Trend")
plt.legend()
plt.grid(True)
plt.savefig("rl_vs_theory_skrxtralastest.png")

print("✅ Physics-alignment tests complete. Plots saved for inspection.")


# =======================================
# 2D Heatmap: SKR vs (Fiber Length, Detector Efficiency)
# =======================================


sim = get_qkd_simulator(env)

# Parameter ranges
fiber_lengths = np.linspace(10, 100, 10)         # km
det_efficiencies = np.linspace(0.4, 0.9, 10)     # detector efficiency (40–90%)

skr_grid = np.zeros((len(det_efficiencies), len(fiber_lengths)))

for i, eta_det in enumerate(det_efficiencies):
    sim.set_detector_efficiency(eta_det)
    for j, L in enumerate(fiber_lengths):
        sim.set_fiber_length(L)

        total_skr = []
        for _ in range(5):  # 5 episodes for averaging
            obs = env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                if isinstance(info, (list, tuple)):
                    info = info[0]['raw_info']
                if "SKR_bits_per_pulse" in info:
                    total_skr.append(info["SKR_bits_per_pulse"])
        skr_grid[i, j] = np.mean(total_skr)

# Plot
plt.figure(figsize=(7,5))
plt.imshow(
    skr_grid,
    origin="lower",
    aspect="auto",
    extent=[fiber_lengths[0], fiber_lengths[-1], det_efficiencies[0], det_efficiencies[-1]],
    cmap="viridis"
)
plt.colorbar(label="Mean SKR (bits/pulse)")
plt.xlabel("Fiber Length (km)")
plt.ylabel("Detector Efficiency")
plt.title("SKR Heatmap: Fiber Length × Detector Efficiency")
plt.tight_layout()
plt.savefig("skr_heatmap_length_vs_efficiencyxtralastest.png")
plt.show()


# =======================================
# Compare Learned μ vs Theoretical μ_opt(L)
# =======================================

import numpy as np
import matplotlib.pyplot as plt

fiber_lengths = np.linspace(10, 100, 10)
alpha_db_per_km = 0.2  # fiber loss
eta_det = 0.6          # typical detector efficiency
eta_values = 10 ** (-alpha_db_per_km * fiber_lengths / 10) * eta_det

# --- Theoretical approximate μ_opt(L) ---
# empirical form from Ma et al. (2005) or Xu et al. (2020) analysis:
# μ_opt ≈ min(0.8, 0.5 + 0.3 * exp(-L / 40))
mu_theory = np.minimum(0.8, 0.5 + 0.3 * np.exp(-fiber_lengths / 40))

# --- Get agent's learned μ (Action 1) ---
agent_mu = []

for L in fiber_lengths:
    sim.set_fiber_length(L)
    obs = env.reset()
    action, _ = model.predict(obs, deterministic=True)
    agent_mu.append(float(np.array(action).flatten()[0]))  # Action 1

plt.figure()
plt.plot(fiber_lengths, agent_mu, "o-", label="RL Learned μ (Action 1)")
plt.plot(fiber_lengths, mu_theory, "s--", label="Theoretical μₒₚₜ")
plt.xlabel("Fiber Length (km)")
plt.ylabel("Mean Photon Number μ")
plt.title("RL vs Theoretical Optimal μ vs Distance")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("mu_vs_theoryxtralastest.png")
plt.show()

