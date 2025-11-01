import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback,CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import time
import os
from rl.qkd_rl import QKDEnv

# from qkd_env import QKDEnv   # <-- your Gymnasium-compatible env


from stable_baselines3.common.callbacks import BaseCallback

from stable_baselines3.common.callbacks import BaseCallback

class DynamicPPECallback(BaseCallback):
    def __init__(self, env, start_ppe=5_000, end_ppe=500_000, total_timesteps=1_000_000, verbose=0):
        super().__init__(verbose)
        self.env = env
        self.start_ppe = start_ppe
        self.end_ppe = end_ppe
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        # Compute progress (0 → 1)
        progress = min(self.num_timesteps / self.total_timesteps, 1.0)
        # Linear increase in PPE
        new_ppe = int(self.start_ppe + progress * (self.end_ppe - self.start_ppe))
        # Apply to all subprocess envs
        self.env.env_method("set_ppe", new_ppe)
        if self.verbose > 0 and self.num_timesteps % 10000 == 0:
            print(f"[DynamicPPECallback] Updated PPE to {new_ppe}")
        return True

class DynamicRewardScaleCallback(BaseCallback):
    def __init__(self, env, start_scale=100.0, end_scale=1.0, total_timesteps=1_000_000):
        super().__init__()
        self.env = env
        self.start_scale = start_scale
        self.end_scale = end_scale
        self.total_timesteps = total_timesteps

    def _on_step(self):
        progress = min(self.num_timesteps / self.total_timesteps, 1.0)
        new_scale = self.start_scale + progress * (self.end_scale - self.start_scale)
        self.env.env_method("set_reward_scale", new_scale)
        return True


class AliceSingleAgentEnv(gym.Env):
    """
    Single-agent wrapper for training Alice only.
    Bob and Eve use fixed/random strategies.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, sim_config):
        super().__init__()
        self.env = QKDEnv(sim_config)
        self.action_space = self.env.action_space["Alice"]
        self.observation_space = self.env.observation_space
        self.current_step=0
    
    def set_ppe(self, new_ppe):
        self.env.set_ppe(new_ppe)

    def set_reward_scale(self, new_scale):
        self.env.set_reward_scale(new_scale)

    def reset(self, *, seed=None, options=None):
        obs, _ = self.env.reset(seed=seed, options=options)
        return obs, {}

    def step(self, action):
        # start = time.time()
        # if self.current_step % 100 == 0:
        #     print(f"--- Step {self.current_step} ---")
        # Bob: fixed strategy
        bob_action = np.array([0.5, 1.0], dtype=np.float32)

        # Eve: random attack
        eve_action = self.env.action_space["Eve"].sample() * 0.3

        actions = {"Alice": action, "Bob": bob_action, "Eve": eve_action}
        obs, rewards, terminated, truncated, info = self.env.step(actions)
        reward = rewards["Alice"]
        done = terminated or truncated
        # elapsed = time.time() - start
        # if self.current_step % 10 == 0:  # print every 10 steps
        #     print(f"Step {self.current_step}: took {elapsed:.4f}s")
        # self.current_step += 1
        return obs, reward, done, False, info

    


    def render(self):
        self.env.render()

    def close(self):
        self.env.close()


# -----------------------------------------------------------
# Main training section
# -----------------------------------------------------------
if __name__ == "__main__":
    # sim_config = {
    #     "distance_km": 10,
    #     "mu_signal": 0.5,
    #     "mu_decoy": 0.1,
    #     "det_eff": 0.1,
    #     "dark_count": 1e-6,
    #     "fiber_loss_db_per_km": 0.2,
    #     "pulses_per_episode": 1000,
    # }
    import yaml
    sim_config = yaml.safe_load(open("configs/config.yaml"))



    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

    # Create multiple envs in parallel
    def make_env(rank):
        def _init():
            env = AliceSingleAgentEnv(sim_config)
            return env
        return _init

    n_envs = 8  # adjust based on CPU cores
    env = SubprocVecEnv([make_env(i) for i in range(n_envs)])

    # Optional normalization
    env = VecNormalize(env, norm_obs=False, norm_reward=True)



    # env = AliceSingleAgentEnv(sim_config)
    # env = DummyVecEnv([lambda: env])
    # Add this line:
    # env = VecNormalize(env, norm_obs=False, norm_reward=True)

    # check_env(env, warn=True,skip_render_check=True)

    # start = time.time()
    # for i in range(1000):
    #     actions = env.action_space.sample()
    #     obs, reward, terminated, truncated, info = env.step(actions)
    #     if (i+1) % 100 == 0:
    #         print(f"Completed {i+1} steps")
    # end = time.time()

    # print(f"✅ 1000 environment steps took {end - start:.2f} seconds")

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        ent_coef=0.01,
        gamma=0.99,
        tensorboard_log="./logs_alice/",
    )

    checkpoint_dir = "./checkpoints/"
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
    save_freq=10000,  # Save every 10,000 timesteps
    save_path=checkpoint_dir,
    name_prefix="ppo_alice_checkpoint",  # The name of the saved model
)
    dynamic_ppe_callback = DynamicPPECallback(env, start_ppe=5_000, end_ppe=500_000, total_timesteps=1_000_000, verbose=1)
    dynamic_r_cb=DynamicRewardScaleCallback(env, start_scale=100.0, end_scale=1.0, total_timesteps=1_000_000)
    callbacks = [checkpoint_callback, dynamic_ppe_callback,dynamic_r_cb]
    print("Training started... ⏳")
    # cb=ProgressCallback(check_freq=100)
    model.learn(total_timesteps=1000_000,callback=callbacks)
    model.save("ppo_alice_qkdnight.zip")

    print("✅ Training complete. Model saved as ppo_alice_qkd.zip")

    # Test the trained agent
    obs = env.reset()
    for _ in range(10):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done,  info = env.step(action)
        r_mean = np.mean(reward)
        skr_mean = np.mean([i['raw_info'].get('SKR_bits_per_pulse', 0) for i in info])
        print(f"Mean Reward={r_mean:.6f}, Mean SKR={skr_mean:.6f}")


