from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
import numpy as np
from stable_baselines3 import PPO
from rl.agents.train_eve import EveSingleAgentEnv

def get_attack_probs(action):
    raw_ir, raw_pns, raw_ts = action[0][:3]
    logits = np.array([raw_ir, raw_pns, raw_ts], dtype=np.float32)
    exp_logits = np.exp(logits - np.max(logits))
    p = exp_logits / exp_logits.sum()
    return p  # p_ir, p_pns, p_ts

def modify_obs_vec(obs, detection_rate=None, qber=None, Y0=None, Y1=None, e1=None):
    """
    Modify observation (5-vector). Returns new obs.
    """
    new = obs[0].copy()

    if detection_rate is not None:
        new[0] = detection_rate
    if qber is not None:
        new[1] = qber
    if Y0 is not None:
        new[2] = Y0
    if Y1 is not None:
        new[3] = Y1
    if e1 is not None:
        new[4] = e1
    
    return [new]



def test_clean_channel_aggression(model, env):
    print("\n=== TEST A: Clean Channel Aggression ===")

    obs = env.reset()

    clean_obs = modify_obs_vec(
        obs,
        detection_rate=0.95,   # high detection → low loss
        qber=0.01,             # very clean
        Y0=0.01,               # low dark counts
        Y1=0.95,               # good single-photon yield
        e1=0.01                # low single-photon error
    )

    action, _ = model.predict(clean_obs, deterministic=True)
    p_ir, p_pns, p_ts = get_attack_probs(action)

    print(f"Clean: IR={p_ir:.4f}, PNS={p_pns:.4f}, TS={p_ts:.4f}")
    print("Expected: IR↑, PNS↑, TS↓")

def test_noisy_channel_conservatism(model, env):
    print("\n=== TEST B: Noisy Channel Conservatism ===")

    obs = env.reset()

    noisy_obs = modify_obs_vec(
        obs,
        detection_rate=0.30,   # heavy loss
        qber=0.08,             # high QBER → dangerous
        Y0=0.30,               # high dark count component
        Y1=0.40,               # poor single-photon yield
        e1=0.15                # high single-photon error
    )

    action, _ = model.predict(noisy_obs, deterministic=True)
    p_ir, p_pns, p_ts = get_attack_probs(action)

    print(f"Noisy: IR={p_ir:.4f}, PNS={p_pns:.4f}, TS={p_ts:.4f}")
    print("Expected: IR↓, PNS↓, TS small or ↓")

def compare_clean_vs_noisy(model, env):
    print("\n=== TEST A vs B Comparison ===")

    obs = env.reset()

    clean_obs = modify_obs_vec(obs, detection_rate=0.95, qber=0.01, Y0=0.01, Y1=0.95, e1=0.01)
    noisy_obs = modify_obs_vec(obs, detection_rate=0.30, qber=0.08, Y0=0.30, Y1=0.40, e1=0.15)

    a_clean, _ = model.predict(clean_obs, deterministic=True)
    a_noisy, _ = model.predict(noisy_obs, deterministic=True)

    p_clean = get_attack_probs(a_clean)
    p_noisy = get_attack_probs(a_noisy)

    print(f"Clean: IR={p_clean[0]:.4f} PNS={p_clean[1]:.4f} TS={p_clean[2]:.4f}")
    print(f"Noisy: IR={p_noisy[0]:.4f} PNS={p_noisy[1]:.4f} TS={p_noisy[2]:.4f}")

    if p_clean[0] > p_noisy[0]:
        print("✔ IR decreases on noisy → correct physics.")
    if p_clean[1] > p_noisy[1]:
        print("✔ PNS decreases on noisy → correct physics.")
    if p_clean[2] > p_noisy[2]:
        print("✔ TS decreases on noisy → expected (TS is risky).")

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
    test_clean_channel_aggression(model, env)
    test_noisy_channel_conservatism(model, env)
    compare_clean_vs_noisy(model, env)
