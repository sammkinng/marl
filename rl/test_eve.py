# test_eve.py
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
import numpy as np
from stable_baselines3 import PPO
from rl.agents.train_eve import EveSingleAgentEnv

def evaluate_eve(
    eve_model_path,
    sim_cfg,
    alice_mode="fixed_optimal",
    num_episodes=50
):
    """
    Tests Eve's performance by:
    - Comparing SKR/QBER with and without Eve.
    - Checking whether Eve increases disturbance.
    - Checking attack-rate patterns.
    """

    print("Loading environment & model...")
    import os
    import copy

    LOGDIR = "./logs_eve"
    MODEL_PATH = os.path.join(LOGDIR, "ppo_eveprebob.zip")
    # MODEL_PATH="ppo_eve_checkpoint_780000_steps.zip"

    VECNORM   = os.path.join(LOGDIR, "vecnormalize_eveprebob.pkl")

    # Load env & stats
    env = DummyVecEnv([lambda: EveSingleAgentEnv(copy.deepcopy(sim_cfg))])
    env = VecNormalize.load(VECNORM, env)
    env.training = False
    env.norm_reward = False

    model = PPO.load(MODEL_PATH,env=env)

    results_with_eve  = []
    results_no_eve    = []
    attack_stats      = []

    # === Helper to generate Alice action ===
    def alice_policy():
        if alice_mode == "fixed_optimal":
            return np.array([0.6, 0.1, 0.6, 0.0], dtype=np.float32)
        elif alice_mode == "random":
            return env.alice_action_space.sample()
        else:
            raise ValueError("Unknown alice_mode")

    # ======= BASELINE: NO-EVE RUNS ========
    # print("\nRunning NO-EVE baseline...")
    # for _ in range(num_episodes):
    #     obs = env.reset()

    #     alice_a = alice_policy()
    #     bob_a   = np.zeros(1)
    #     eve_a   = None  # no attack

    #     _, _, _, _, info = env.step(eve_a)
    #     base = info["raw"]
    #     results_no_eve.append(base["SKR_bits_per_pulse"])

    baseline_mean_skr = 0.002
    baseline_qber     = 0.012
    print(f"Baseline SKR: {baseline_mean_skr:.5e}")
    print(f"Baseline QBER: {baseline_qber:.5f}")

    # ======= EVE MODEL RUNS ========
    print("\nRunning trained Eve model...")
    for _ in range(num_episodes):
        obs= env.reset()

        alice_a = alice_policy()
        bob_a = np.zeros(1)

        eve_obs = obs
        eve_a, _ = model.predict(eve_obs, deterministic=True)

        obs2, reward,  _, info = env.step(eve_a)

        raw = info[0]
        results_with_eve.append(raw["SKR_bits_per_pulse"])
        eve_a=eve_a[0]

        attack_stats.append({
            "p_timeshift": eve_a[2],
            "p_pns": eve_a[1],
            "p_ir": eve_a[0],
            "extra_dark": eve_a[3],
            "QBER": raw["E_s"],
            "SKR": raw["SKR_bits_per_pulse"],
            "ig":raw["info_gain_per_pulse"]
        })

    print("\n===== Final Eve Test Results =====")
    print(f"Baseline SKR:            {baseline_mean_skr:.6e}")
    print(f"Eve SKR:                 {np.mean(results_with_eve):.6e}")
    print(f"SKR Reduction:           {baseline_mean_skr - np.mean(results_with_eve):.6e}")
    print(f"QBER with Eve:           {np.mean([d['QBER'] for d in attack_stats]):.4f}")
    print(f"IG with Eve:           {np.mean([d['ig'] for d in attack_stats]):.4f}")
    
    print(f"Mean Attack Strengths:   ")
    print(f"  Time-shift:            {np.mean([d['p_timeshift'] for d in attack_stats]):.3f}")
    print(f"  PNS:                   {np.mean([d['p_pns'] for d in attack_stats]):.3f}")
    print(f"  Intercept-resend:      {np.mean([d['p_ir'] for d in attack_stats]):.3f}")
    print(f"  Extra-dark:            {np.mean([d['extra_dark'] for d in attack_stats]):.5f}")

    return {
        "baseline_skr": baseline_mean_skr,
        "eve_skr": np.mean(results_with_eve),
        "eve_qber": np.mean([d["QBER"] for d in attack_stats]),
        "attack_stats": attack_stats,
    }



if __name__ == "__main__":

    import yaml

    SIM_CFG = yaml.safe_load(open("configs/config.yaml"))

    evaluate_eve(
        eve_model_path="./logs_eve/ppo_eve.zip",
        sim_cfg=SIM_CFG,
        alice_mode="fixed_optimal",
        num_episodes=50
    )