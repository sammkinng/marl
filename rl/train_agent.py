
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize,DummyVecEnv

import yaml
import numpy as np
from rl.agents.train_alice import AliceSingleAgentEnv
# from stable_baselines3.common.env_checker import check_env

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO

from rl.agents.train_eve import EveSingleAgentEnv


sim_config = yaml.safe_load(open("configs/config.yaml"))


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

# Create multiple envs in parallel
def make_env(rank):
    def _init():
        env = EveSingleAgentEnv(sim_config)
        # env = AliceSingleAgentEnv(sim_config)
        return env
    return _init

if __name__ == "__main__":
    # n_envs = 8  # adjust based on CPU cores
    # env = SubprocVecEnv([make_env(i) for i in range(n_envs)])

    # # Optional normalization
    # env = VecNormalize(env, norm_obs=False, norm_reward=True)



    # env = EveSingleAgentEnv(sim_config)
    env = AliceSingleAgentEnv(sim_config)
    env = DummyVecEnv([lambda: env])
    

    # model = PPO(
    #     "MlpPolicy",
    #     env,
    #     verbose=1,
    #     learning_rate=5e-4,
    #     n_steps=4096,
    #     batch_size=128,
    #     ent_coef=0.02,
    #     gamma=0.99,
    #     tensorboard_log="./logs_eve/",
    # )

    import os
    import copy

    LOGDIR = "./logs_alice"
    MODEL_PATH = os.path.join(LOGDIR, "best_model.zip")
    VECNORM   = os.path.join(LOGDIR, "vecnormalize_alice.pkl")

    # Load env & stats
    base_env = DummyVecEnv([lambda: AliceSingleAgentEnv(copy.deepcopy(sim_config))])
    base_env = VecNormalize.load(VECNORM, base_env)
    base_env.training = False
    base_env.norm_reward = False

    model = PPO.load(MODEL_PATH,env=base_env)



    # model = PPO.load("ppo_new_rewasystem_eve.zip", env=env)
    # model = PPO.load("logs_alice/best_model.zip", env=env)
    # model.ent_coef = 0.03   # or slightly higher

    # checkpoint_dir = "./checkpoints/"
    # os.makedirs(checkpoint_dir, exist_ok=True)

    # checkpoint_callback = CheckpointCallback(
    # save_freq=10000,  # Save every 10,000 timesteps
    # save_path=checkpoint_dir,
    # name_prefix="ppo_alice_checkpoint",  # The name of the saved model
    # )

    # dynamic_ppe_callback = DynamicPPECallback(env, start_ppe=5_000, end_ppe=500_000, total_timesteps=1_000_000, verbose=1)

    # dynamic_ppe_callback = DynamicPPECallback(env, start_ppe=44600, end_ppe=500_000, total_timesteps=920_000, verbose=1)


    # callbacks = [checkpoint_callback, dynamic_ppe_callback,dynamic_r_cb]

    # print("Training started... ⏳")
    # cb=ProgressCallback(check_freq=100)

    # model.learn(total_timesteps=1000_000)

    # model.learn(total_timesteps=200_000)

    # model.learn(total_timesteps=100_000,reset_num_timesteps=False)



    # model.save("ppo_new_rewasystem_eve.zip")

    # print("✅ Training complete. Model saved as ppo_eve.zip")

    # Test the trained agent
    obs = env.reset()
    for _ in range(100):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done,  info = env.step(action)
        r_mean = np.mean(reward)
        skr_mean = np.mean([i['raw_info'].get('SKR_bits_per_pulse', 0) for i in info])
        print(f"Mean Reward={r_mean:.6f}, Mean SKR={skr_mean:.6f}")
        # obs = env.reset()

    # obs = env.reset()
    # for step in range(1000):  # simulate 1000 steps
    #     action, _ = model.predict(obs, deterministic=True)
    #     obs, reward, done, info = env.step(action)
    #     print(f"Step {step}: Reward={reward}, SKR={info[0]['raw_info'].get('SKR_bits_per_pulse', 0)}")

    #     if step % 100 == 0:  # every 100 steps, treat it as a pseudo-episode
    #         obs = env.reset()

    # cum_reward = 0
    # for step in range(1000):
    #     action, _ = model.predict(obs, deterministic=True)
    #     obs, reward, done, info = env.step(action)
    #     cum_reward += np.mean(reward)
    #     avg_reward = cum_reward / (step + 1)
        
    #     reward_value = float(np.mean(reward))  # or reward.item() if it's a scalar array
    #     print(f"Step {step}: Instant Reward={reward_value:.3f}, Running Avg={avg_reward:.3f}")

    # episode_length=1000
    # maxp=200
    # total_reward = 0.0
    # obs = env.reset()
    # for _ in range(episode_length):
    #     action, _ = model.predict(obs, deterministic=True)
    #     obs, reward, done, info = env.step(action)
    #     total_reward += reward
    #     if done:
    #         obs = env.reset()
    # print("Average reward per step:", total_reward / episode_length)

    # rewards = []
    # for ep in range(episode_length):
    #     obs = env.reset()
    #     done = False
    #     ep_reward = 0
    #     while not done:
    #         action, _ = model.predict(obs, deterministic=True)
    #         obs, reward, done, info = env.step(action)
    #         ep_reward += reward
    #     rewards.append(ep_reward)

    # import matplotlib.pyplot as plt
    # plt.plot(rewards)
    # plt.xlabel("Episode")
    # plt.ylabel("Total Reward")
    # plt.title("PPO Evaluation Reward per Episode")
    # # plt.show()
    # plt.savefig("ppo_alice_evaluation_rewards.png")
    # plt.close()




    # ----------------------------
    # 3️⃣ Collect test episode rewards
    # ----------------------------
    # num_episodes = 50
    # episode_rewards = []

    # for ep in range(num_episodes):
    #     obs = env.reset()
    #     done = False
    #     total_reward = 0
    #     while not done:
    #         action, _ = model.predict(obs, deterministic=True)  # deterministic evaluation
    #         obs, reward, done, info = env.step(action)
    #         total_reward += reward
    #     episode_rewards.append(total_reward)

    # env.close()

    # # ----------------------------
    # # 4️⃣ Compute moving average (smoothing)
    # # ----------------------------
    # def moving_average(data, window_size=5):
    #     if len(data) < window_size:
    #         return data
    #     return [sum(data[i:i+window_size])/window_size for i in range(len(data)-window_size+1)]

    # smoothed_rewards = moving_average(episode_rewards, window_size=5)

    # # ----------------------------
    # # 5️⃣ Plot results
    # # ----------------------------
    # plt.figure(figsize=(10,5))
    # plt.plot(episode_rewards, label='Episode Reward', alpha=0.5)
    # plt.plot(range(len(smoothed_rewards)), smoothed_rewards, label='Smoothed Reward (MA)', color='red')
    # plt.xlabel('Episode')
    # plt.ylabel('Total Reward')
    # plt.title('Fine-Tuned Model Test Episode Rewards')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig("ppo_alice_finetuned_test_rewards.png")
    # plt.close()


    # n_episodes = 50
    # skr_list = []

    # for ep in range(n_episodes):
    #     obs= env.reset()
    #     done = False
    #     ep_skr = []

    #     while not done:
    #         action, _ = model.predict(obs, deterministic=True)
    #         obs, reward, done, info = env.step(action)
            
    #         # Collect SKR (update this key if different)
    #         # skr = info["raw_info"].get("SKR_bits_per_pulse", 0)
    #         skr = np.mean([i['raw_info'].get('SKR_bits_per_pulse', 0) for i in info])
    #         ep_skr.append(skr)
        
    #     mean_skr = np.mean(ep_skr)
    #     skr_list.append(mean_skr)

    # # 🔹 Plot mean SKR trend
    # plt.figure(figsize=(8,4))
    # plt.plot(skr_list, label="Mean SKR", color="royalblue")
    # plt.title("Mean SKR per Test Episode")
    # plt.xlabel("Episode")
    # plt.ylabel("Mean SKR")
    # plt.grid(True)
    # plt.legend()
    # # plt.show()
    # plt.savefig("ppo_alice_finetuned_test_skr.png")
    # plt.close()

    # # 🔹 Analyze stability (trend + variance)
    # def analyze_trend(data):
    #     x = np.arange(len(data))
    #     slope, _ = np.polyfit(x, data, 1)
    #     rel_change = (data[-1] - data[0]) / max(1e-9, abs(data[0]))
    #     cv = np.std(data) / np.mean(data)
    #     print(f"Slope: {slope:.6f}, Rel Change: {rel_change*100:.2f}%, CoeffVar: {cv:.3f}")
    #     if abs(slope) < 0.001 and abs(rel_change) < 0.01 and cv < 0.05:
    #         print("✅ SKR trend is flat — training likely converged.")
    #     elif slope > 0.001:
    #         print("⚙️  SKR still increasing — more training may help.")
    #     else:
    #         print("⚠️  Noisy or stagnant behavior — consider retraining.")
            
    # analyze_trend(skr_list)




    # import numpy as np
    # from collections import Counter
    # adapt these names/keys to your env/logging

    # 1) Print a few raw info dicts for a single episode (first few steps)
    # obs = env.reset()
    # done = False
    # step = 0
    # print("=== First episode: first 20 info dicts ===")
    # while not done and step < 20:
    #     action, _ = model.predict(obs, deterministic=True)
    #     obs, reward, done,  info = env.step(action)
    #     print(f"step {step}: keys={info}")
    #     # pretty-print any keys that look like counts or skr
    #     if 'raw_info' in info:
    #         print(" raw_info snippet:", {k: info['raw_info'].get(k) for k in list(info['raw_info'].keys())[:10]})
    #     else:
    #         print(info)
    #     step += 1

    # # 2) Collect per-episode aggregate numbers (counts/gains/QBER) over N episodes
    # N = 20
    # per_episode = []
    # for ep in range(N):
    #     obs = env.reset()
    #     done = False
    #     pulses_sent = 0
    #     detections = 0
    #     bit_errors = 0
    #     sifting_count = 0
    #     skr_logged = []  # whatever you logged
    #     while not done:
    #         action, _ = model.predict(obs, deterministic=True)
    #         obs, reward, done,  info = env.step(action)
    #         # Update these extraction lines to match your info keys
    #         # Try to find actual raw counts (detector clicks, sifted bits, errors)
    #         if 'raw_info' in info:
    #             ri = info['raw_info']
    #             # example keys to inspect:
    #             # clicks = ri.get('clicks', None)
    #             # sifted = ri.get('sifted_bits', None)
    #             # errors = ri.get('bit_errors', None)
    #             # pulses_inc = ri.get('pulses', 1)  # if available
    #             # Append whatever you find:
    #             skr_logged.append(ri.get('SKR_bits_per_pulse', None))
    #             # If you have raw click counts, accumulate them here (replace names)
    #             if 'clicks' in ri:
    #                 detections += ri['clicks']
    #             if 'pulses' in ri:
    #                 pulses_sent += ri['pulses']
    #             if 'sifted_bits' in ri:
    #                 sifting_count += ri['sifted_bits']
    #             if 'bit_errors' in ri:
    #                 bit_errors += ri['bit_errors']
    #         else:
    #             # fallback: try keys at top-level info
                
    #             skr_logged.append(info[0].get('SKR_bits_per_pulse', None))
    #     per_episode.append({
    #         'pulses_sent': pulses_sent,
    #         'detections': detections,
    #         'sifted': sifting_count,
    #         'bit_errors': bit_errors,
    #         'mean_skr_logged': np.nanmean([x for x in skr_logged if x is not None])
    #     })

    # # 3) Print summary diagnostics
    # for i, d in enumerate(per_episode):
    #     pulses = d['pulses_sent'] or 1
    #     detect_prob = d['detections'] / pulses if pulses else None
    #     qber_emp = (d['bit_errors']/d['sifted']) if d['sifted'] else None
    #     print(f"Ep {i}: pulses={pulses}, detections={d['detections']}, detect_prob={detect_prob}, sifted={d['sifted']}, qber_emp={qber_emp}, mean_skr_logged={d['mean_skr_logged']}")




