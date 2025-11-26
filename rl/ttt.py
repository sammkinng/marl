from stable_baselines3 import PPO  # or SAC/TD3/whatever you used
import numpy as np
import os
from rl.agents.train_alice import AliceSingleAgentEnv
from rl.agents.train_eve import EveSingleAgentEnv
from rl.qkdenv import QKDEnv
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
import copy

import yaml

SIM_CFG = yaml.safe_load(open("configs/config.yaml"))


def run_eval_step1(
    alice_model_path: str,
    eve_model_path: str,
    config: dict,
    n_episodes: int = 50,
):
    # 1. Create core simulator + eval env

    env = QKDEnv(SIM_CFG)

    LOGDIR = "./rl"
    MODEL_PATH = os.path.join(LOGDIR, "ppo_eveanal5.zip")
    # MODEL_PATH="ppo_eve_checkpoint_780000_steps.zip"

    VECNORM   = os.path.join(LOGDIR, "vecnormalize_eveanal5.pkl")

    
    MODEL_PATHa = os.path.join(LOGDIR, "ppo_alice.zip")
    # MODEL_PATH="ppo_eve_checkpoint_780000_steps.zip"

    VECNORMa   = os.path.join(LOGDIR, "vecnormalize_alice.pkl")

    # Load env & stats
    enve = DummyVecEnv([lambda: EveSingleAgentEnv(copy.deepcopy(SIM_CFG))])
    enve = VecNormalize.load(VECNORM, enve)
    enve.training = False
    enve.norm_reward = False

    eve_model = PPO.load(MODEL_PATH,env=enve)

    enva = DummyVecEnv([lambda: AliceSingleAgentEnv(copy.deepcopy(SIM_CFG))])
    enva = VecNormalize.load(VECNORMa, enva)
    enva.training = False
    enva.norm_reward = False

    alice_model = PPO.load(MODEL_PATHa,env=enva)

    skr_list = []
    qber_list = []
    eve_info_list = []

    for ep in range(n_episodes):
        obs,_ = env.reset()
        done = False

        # If your env is single-step, this loop will run once per episode
        ep_skr = 0.0
        ep_eve_info = 0.0

        while not done:
            # print("OBS:", obs)
            # print("TYPE:", type(obs))
            # print("SLICE1:", obs[:3])
            # print("SLICE2:", obs[5:])
            # print("TYPE1:", [type(x) for x in obs[:3]])
            # print("TYPE2:", [type(x) for x in obs[5:]])

            alice_obs = obs[:5]
            eve_obs   = np.concatenate([obs[:3], obs[5:]])

            alice_action, _ = alice_model.predict(alice_obs, deterministic=True)
            eve_action, _   = eve_model.predict(eve_obs, deterministic=True)

            actions = {
                "alice": alice_action,
                "eve": eve_action,
            }

            obs, rewards, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            ep_skr      += info["SKR_bits_per_pulse"]
            ep_eve_info += info.get("info_gain_per_pulse", 0.0)

        skr_list.append(ep_skr)
        qber_list.append(info["E_s"])
        eve_info_list.append(ep_eve_info)

        print(
            f"Episode {ep+1}/{n_episodes}: "
            f"SKR={ep_skr:.4e}, QBER={info['E_s']:.4f}, "
            f"EveInfo={ep_eve_info:.4f}"
        )

    print("\n=== Summary over", n_episodes, "episodes ===")
    print(f"Mean SKR       : {np.mean(skr_list):.4e}")
    print(f"Mean QBER      : {np.mean(qber_list):.4f}")
    print(f"Mean Eve Info  : {np.mean(eve_info_list):.4f}")

    return {
        "skr": np.array(skr_list),
        "qber": np.array(qber_list),
        "eve_info": np.array(eve_info_list),
    }


if __name__ == "__main__":
    config = {
        "max_steps": 1,          # or more if you want multi-step episodes
        "n_pulses": 10_000,      # whatever your simulator uses
        # other channel / decoy params...
    }

    results = run_eval_step1(
        alice_model_path="models/alice_clean.zip",
        eve_model_path="models/eve_vs_fixed_ab.zip",
        config=config,
        n_episodes=50,
    )
