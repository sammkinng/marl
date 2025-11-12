

import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize,SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback

from rl.agents.eve import EveSingleAgentEnv



SIM_CFG = {
    "pulses_per_episode": 1000,
    "output_dir": "./results",
    "fiber_loss_db_per_km": 0.2,
    "distance_km": 50.0,
    "det_eff": 0.2,
    "dark_count": 1e-6,
    "baseline_qber": 0.01,
}

LOGDIR = "./logs_eve"
MODEL_PATH = os.path.join(LOGDIR, "ppo_eve.zip")

os.makedirs(LOGDIR, exist_ok=True)

def make_env(rank):
    def _init():
        return EveSingleAgentEnv(SIM_CFG)
    return _init

if __name__ == "__main__":
    # ---- Create parallel envs ----
    n_envs = 8
    env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True,clip_reward=10.0)

    # ---- Eval env (single-thread) ----
    eval_env = DummyVecEnv([lambda: EveSingleAgentEnv(SIM_CFG)])
    eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=True)
    callback = EvalCallback(eval_env, best_model_save_path=LOGDIR, log_path=LOGDIR,
                            eval_freq=20_000, deterministic=True, render=False)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        verbose=1,
    )

    model.learn(total_timesteps=1_000_000, callback=callback)
    model.save(MODEL_PATH)

    env.save(os.path.join(LOGDIR, "vecnormalize_eve.pkl"))

