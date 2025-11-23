from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
import numpy as np
from stable_baselines3 import PPO
from rl.agents.train_eve import EveSingleAgentEnv

def test_action_stability(model, env, n_samples=1000, noise_std=1e-4):
    """
    Feed similar observations to Eve and check if actions vary smoothly.
    """
    obs = env.reset()
    base_obs = obs.copy()

    actions = []
    deltas = []

    for _ in range(n_samples):
        # add tiny noise to simulate "similar state"
        noisy_obs = base_obs + np.random.normal(0, noise_std, size=base_obs.shape).astype(np.float32)

        action, _ = model.predict(noisy_obs, deterministic=True)
        actions.append(action)

        if len(actions) > 1:
            deltas.append(np.linalg.norm(actions[-1] - actions[-2]))

    avg_delta = np.mean(deltas)
    max_delta = np.max(deltas)

    print("=== Action Stability Test ===")
    print(f"Average Δa between similar states: {avg_delta:.6f}")
    print(f"Max    Δa between similar states: {max_delta:.6f}")

    if avg_delta < 0.01:
        print("✔ Stable actions → good sign.")
    else:
        print("✖ Actions vary too much → might be random or unstable.")


# Example:
# test_action_stability(eve_model, env)
def test_determinism(model, env, repeats=200):
    """
    Feed identical observation repeatedly and verify the action is identical.
    """
    obs = env.reset()
    actions = []

    for _ in range(repeats):
        action, _ = model.predict(obs, deterministic=True)
        actions.append(action)

    actions = np.array(actions)
    diffs = np.max(actions.max(axis=0) - actions.min(axis=0))

    print("\n=== Determinism Test ===")
    print(f"Max difference between repeated actions: {diffs:.10f}")

    if diffs < 1e-6:
        print("✔ Deterministic → good.")
    else:
        print("✖ Non-deterministic output → check seed or exploration ON.")
def test_softmax_behavior(model, env, n_samples=1000):
    """
    Check if softmax outputs valid probabilities and non-degenerate behavior.
    """
    probs = []

    for _ in range(n_samples):
        obs = env.reset()
        action, _ = model.predict(obs, deterministic=True)

        raw_ir, raw_pns, raw_ts = action[0][:3]
        logits = np.array([raw_ir, raw_pns, raw_ts], dtype=np.float32)
        exp_logits = np.exp(logits - np.max(logits))
        p = exp_logits / exp_logits.sum()  # softmax

        probs.append(p)

    probs = np.array(probs)
    sums = probs.sum(axis=1)
    mean_prob = probs.mean(axis=0)
    std_prob = probs.std(axis=0)

    print("\n=== Softmax Probability Behavior Test ===")
    print(f"Probability sums (mean): {np.mean(sums):.6f}, (max dev): {np.max(np.abs(sums - 1)):.6f}")
    print(f"Mean attack probabilities: {mean_prob}")
    print(f"Std of attack probabilities: {std_prob}")

    if np.max(np.abs(sums - 1)) < 1e-4:
        print("✔ Softmax outputs valid distributions.")
    else:
        print("✖ Softmax sum deviates → check overflow/underflow issues.")

    # degeneracy tests:
    if np.allclose(mean_prob, [1/3]*3, atol=0.05):
        print("⚠ Warning: Probabilities look almost uniform → model might be random.")

    if np.any(mean_prob < 0.02):
        print("⚠ Warning: Some attacks never selected → possible dead neuron.")

if __name__=="__main__":
    import os
    import copy
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
    LOGDIR = "./logs_eve"
    MODEL_PATH = os.path.join(LOGDIR, "ppo_eveskr.zip")
    # MODEL_PATH="ppo_eve_checkpoint_780000_steps.zip"

    VECNORM   = os.path.join(LOGDIR, "vecnormalize_eveskr.pkl")

    # Load env & stats
    env = DummyVecEnv([lambda: EveSingleAgentEnv(copy.deepcopy(SIM_CFG))])
    env = VecNormalize.load(VECNORM, env)
    env.training = False
    env.norm_reward = False

    model = PPO.load(MODEL_PATH,env=env)
    test_action_stability(model, env)
    test_determinism(model, env)
    test_softmax_behavior(model, env)
