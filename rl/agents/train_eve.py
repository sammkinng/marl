import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
import time
from rl.qkd_rl import QKDEnv

# from qkd_env import QKDEnv   # <-- your Gymnasium-compatible env



class ProgressCallback(BaseCallback):
    def __init__(self, check_freq=10000, verbose=1):
        super(ProgressCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.start_time = None

    def _on_training_start(self):
        self.start_time = time.time()
        print("🚀 Training started...")

    def _on_step(self):
        # Print log every N steps
        if self.n_calls % self.check_freq == 0:
            elapsed = time.time() - self.start_time
            fps = self.n_calls / elapsed if elapsed > 0 else 0
            print(f"[Step {self.n_calls:,}] "
                  f"Elapsed: {elapsed/60:.1f} min | "
                  f"FPS: {fps:.1f} | "
                  f"Latest reward: {np.mean(self.locals.get('rewards', [0])):.4f}")
        return True

    def _on_training_end(self):
        print("✅ Training finished.")


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

    def reset(self, *, seed=None, options=None):
        obs, _ = self.env.reset(seed=seed, options=options)
        return obs, {}

    def step(self, action):
        start = time.time()
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
        elapsed = time.time() - start
        if self.current_step % 10 == 0:  # print every 10 steps
            print(f"Step {self.current_step}: took {elapsed:.4f}s")
        self.current_step += 1
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


    env = AliceSingleAgentEnv(sim_config)
    check_env(env, warn=True,skip_render_check=True)

    # start = time.time()
    # for i in range(1000):
    #     actions = env.action_space.sample()
    #     obs, reward, terminated, truncated, info = env.step(actions)
    #     if (i+1) % 100 == 0:
    #         print(f"Completed {i+1} steps")
    # end = time.time()

    # print(f"✅ 1000 environment steps took {end - start:.2f} seconds")

    # model = PPO(
    #     "MlpPolicy",
    #     env,
    #     verbose=1,
    #     learning_rate=3e-4,
    #     n_steps=2048,
    #     batch_size=64,
    #     ent_coef=0.01,
    #     gamma=0.99,
    #     tensorboard_log="./logs_alice/",
    # )

    # print("Training started... ⏳")
    # cb=ProgressCallback(check_freq=100)
    # model.learn(total_timesteps=1_000,callback=cb)
    # model.save("ppo_alice_qkd")

    # print("✅ Training complete. Model saved as ppo_alice_qkd.zip")

    # Test the trained agent
    # obs, _ = env.reset()
    # for _ in range(10):
    #     action, _ = model.predict(obs, deterministic=True)
    #     obs, reward, done, _, info = env.step(action)
    #     print(f"Reward={reward:.6f}, SKR={info['raw_info'].get('SKR_bits_per_pulse', 0):.6f}")
    #     if done:
    #         obs, _ = env.reset()
