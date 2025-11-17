
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize

import yaml
import numpy as np
from rl.agents.train_alice import AliceSingleAgentEnv

from stable_baselines3 import PPO

from rl.agents.train_eve import EveSingleAgentEnv


sim_config = yaml.safe_load(open("configs/config.yaml"))



if __name__ == "__main__":
   

    import os
    import copy

    LOGDIR = "./logs_eve"
    MODEL_PATH = os.path.join(LOGDIR, "ppo_eve.zip")
    VECNORM   = os.path.join(LOGDIR, "vecnormalize_eve.pkl")

    # Load env & stats
    env = DummyVecEnv([lambda: EveSingleAgentEnv(copy.deepcopy(sim_config))])
    env = VecNormalize.load(VECNORM, env)
    env.training = False
    env.norm_reward = False

    model = PPO.load(MODEL_PATH,env=env)



    
    # Test the trained agent
    obs = env.reset()
    for _ in range(100):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done,  info = env.step(action)
        r_mean = np.mean(reward)
        skr_mean = np.mean([i['raw_info'].get('SKR_bits_per_pulse', 0) for i in info])
        print(f"Mean Reward={r_mean:.6f}, Mean SKR={skr_mean:.6f}")
        # obs = env.reset()

    