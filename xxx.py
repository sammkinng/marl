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
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20)
print(mean_reward, std_reward)
