import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize,SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback

from rl.agents.train_alice import AliceSingleAgentEnv
from stable_baselines3.common.monitor import Monitor


SIM_CFG = {
    "pulses_per_episode": 1000,
    "output_dir": "./results",
    "results_csv": "alice_results.csv",
    "fiber_loss_db_per_km": 0.2,
    "distance_km": 50.0,
    "det_eff": 0.2,
    "dark_count": 1e-6,
    "baseline_qber": 0.01,
}

LOGDIR = "./logs_alice"
MODEL_PATH = os.path.join(LOGDIR, "ppo_alice.zip")

os.makedirs(LOGDIR, exist_ok=True)




def make_env(rank, log_dir=LOGDIR):
    def _init():
        env = AliceSingleAgentEnv(SIM_CFG)
        # Create per-worker log folder
        worker_log = os.path.join(log_dir, f"env_{rank}")
        os.makedirs(worker_log, exist_ok=True)
        env = Monitor(env, filename=os.path.join(worker_log, "monitor.csv"))
        return env
    return _init


if __name__ == "__main__":
    # ---- Create parallel envs ----
    n_envs = 8
    env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True,clip_reward=10.0)

    # ---- Eval env (single-thread) ----
    eval_env = DummyVecEnv([
    lambda: Monitor(AliceSingleAgentEnv(SIM_CFG), "./logs_alice/eval_monitor.csv")
])
    eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=True)
    callback = EvalCallback(eval_env, best_model_save_path=LOGDIR, log_path=LOGDIR,
                            eval_freq=10_000, deterministic=True, render=False)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=5e-4,
        n_steps=4096,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,
        vf_coef=0.5,
        verbose=1,
    )

    model.learn(total_timesteps=1_000_000, callback=callback)
    model.save(MODEL_PATH)

    # Save VecNormalize stats
    env.save(os.path.join(LOGDIR, "vecnormalize_alice.pkl"))