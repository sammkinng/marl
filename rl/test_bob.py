import numpy as np
from stable_baselines3 import PPO

from rl.agents.bob import BobTrainEnv
from rl.env import QKDEnv
from sim.utils import qkd_simulation

import gymnasium as gym
import numpy as np
from gymnasium import spaces


import os
from rl.agents.train_alice import AliceSingleAgentEnv
from rl.agents.train_eve import EveSingleAgentEnv
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
import copy


# import your simulator
# from your_project.qkd_sim import qkd_simulation

# ---- You already have something like this in your envs ----


def load_models(SIM_CFG):
        LOGDIR = "./rl"
        LOGDIRb = "./logs_bob"

        MODEL_PATH = os.path.join(LOGDIR, "ppo_eveanal5.zip")
        VECNORM   = os.path.join(LOGDIR, "vecnormalize_eveanal5.pkl")

        MODEL_PATHa = os.path.join(LOGDIR, "ppo_alice.zip")
        VECNORMa   = os.path.join(LOGDIR, "vecnormalize_alice.pkl")

        MODEL_PATHb = os.path.join(LOGDIRb, "ppo_bob.zip")
        VECNORMb   = os.path.join(LOGDIRb, "vecnormalize_bob.pkl")

        # Load env & stats
        enve = DummyVecEnv([lambda: EveSingleAgentEnv(copy.deepcopy(SIM_CFG))])
        enve = VecNormalize.load(VECNORM, enve)
        enve.training = False
        # enve.norm_reward = False

        eve_model = PPO.load(MODEL_PATH,env=enve)

        enva = DummyVecEnv([lambda: AliceSingleAgentEnv(copy.deepcopy(SIM_CFG))])
        enva = VecNormalize.load(VECNORMa, enva)
        enva.training = False
        # enva.norm_reward = False

        alice_model = PPO.load(MODEL_PATHa,env=enva)

        envb = DummyVecEnv([lambda: BobTrainEnv(copy.deepcopy(SIM_CFG))])
        envb = VecNormalize.load(VECNORMb, envb)
        envb.training = False
        # envb.norm_reward = False

        bob_model = PPO.load(MODEL_PATHb,env=envb)

        return alice_model,enva,bob_model,envb,eve_model,enve


def alice_action_to_aa(alice_action):
    """
    Map Alice's 4D action to simulator aa dict.
    alice_action: [raw_mu_signal, raw_mu_decoy, raw_logit_p_signal, raw_logit_p_decoy]
    This is just an example; match it to what you used in training.
    """
    a = np.asarray(alice_action[0], dtype=np.float32)

    # print(a.shape,a)

    raw_mu_s = float(a[0])
    raw_mu_d = float(a[1])
    raw_logit_ps = float(a[2])
    raw_logit_pd = float(a[3])

    # map raw μ to safe ranges:
    # μ_signal ∈ [0.40, 0.70], μ_decoy ∈ [0.01, 0.15]
    mu_signal = 0.40 + QKDEnv._sigmoid(raw_mu_s) * (0.70 - 0.40)
    mu_decoy = 0.01 + QKDEnv._sigmoid(raw_mu_d) * (0.15 - 0.01)

    # ensure mu_signal > mu_decoy with a small margin
    if mu_decoy >= mu_signal - 0.01:
        mu_decoy = max(0.01, mu_signal - 0.01)

    # softmax probabilities for [p_signal, p_decoy, p_vac]
    logits = np.array([raw_logit_ps, raw_logit_pd, 0.0], dtype=np.float32)
    p_signal, p_decoy, p_vac = QKDEnv._softmax(logits)

    s = p_signal + p_decoy + p_vac
    if s > 0:
        ps, pd, _ = p_signal/s, p_decoy/s, p_vac/s


    return {
        "mus": float(mu_signal),
        "mud": float(mu_decoy),
        "ps":  float(ps),
        "pd":  float(pd),
    }


def eve_action_to_attacks(eve_action):
    """
    Map Eve's 4D action to attack dict.
    Example: 3 logits for [p_ir, p_pns, p_ts] + one dark boost.
    Adjust to your real mapping.
    """
    e = np.asarray(eve_action[0], dtype=np.float32)


    raw_ir, raw_pns, raw_ts, raw_dark = map(float, e)

    # # Softmax for attack-type probabilities
    logits = np.array([raw_ir, raw_pns, raw_ts], dtype=np.float32)
    p=QKDEnv._softmax(logits)
    p_ir, p_pns, p_ts = p.tolist()

    # # Sigmoid for dark-boost
    dark_boost = QKDEnv._sigmoid(raw_dark)

    return {
        "intercept_resend":   p_ir,
        "pns":                p_pns,
        "time_shift":         p_ts,
        "darkcount_increase": dark_boost,
    }


def bob_action_to_bb(bob_action,cfg):
    """
    Map Bob's 2D action to bb dict:
    bob_action: [basis_prob, detector_gain]
    """
    a0, a1 = np.asarray(bob_action[0], dtype=np.float32)

    basis_prob = float(np.clip(a0, 0.0, 1.0))
    detector_gain = float(np.clip(a1, 0.5, 2.0))

    # compute new detector params and apply
    base_det_eff = cfg.get( "det_eff", 0.1)
    base_dark = float(cfg.get( "dark_count", 1e-6))
    det_eff = min(1.0, base_det_eff * detector_gain)
    dark_count = base_dark * (1.0 + 2.0 * max(0.0, detector_gain - 1.0) ** 2)

    return {
        "bp": basis_prob,
        "de":   det_eff,
        "dc":dark_count
    }


def eval_trained_bob(
    alice_model_path: str,
    eve_model_path: str,
    bob_model_path: str,
    config: dict,
    n_episodes: int = 50,
):
    # Load models
    alice_model,enva, bob_model ,envb,eve_model   ,enve,=load_models(config)

    skr_list = []
    qber_list = []
    eve_info_list = []

    for ep in range(n_episodes):
        # simple single-step eval: fake obs as zeros (or some prior)
        obsa=enva.reset()
        obsb=envb.reset()
        obse=enve.reset()

        # Get actions
        alice_action, _ = alice_model.predict(obsa, deterministic=True)
        eve_action, _   = eve_model.predict(obse, deterministic=True)
        bob_action, _   = bob_model.predict(obsb, deterministic=True)

        # print(bob_action,eve_action)

        # Map to simulator inputs
        aa      = alice_action_to_aa(alice_action)
        attacks = eve_action_to_attacks(eve_action)
        bb      = bob_action_to_bb(bob_action,config)

        # Run simulator
        info = qkd_simulation(config, aa, attacks=attacks, bb=bb)

        skr      = info["SKR_bits_per_second"]
        qber     = info["E_s"]           # sifted QBER
        eve_info = info["info_gain_per_pulse"]

        skr_list.append(skr)
        qber_list.append(qber)
        eve_info_list.append(eve_info)

        print(
            f"Ep {ep+1}/{n_episodes}: "
            f"SKR={skr:.4e}, QBER={qber:.4f}, EveInfo={eve_info:.4e}, "
            f"basis_prob={bb['bp']:.3f}, det_gain={bb['de']:.3f}"
        )

    print("\n=== Summary with trained Bob vs trained Eve & Alice ===")
    print(f"Mean SKR      : {np.mean(skr_list):.4e}")
    print(f"Mean QBER     : {np.mean(qber_list):.4f}")
    print(f"Mean Eve info : {np.mean(eve_info_list):.4e}")

    return {
        "skr": np.array(skr_list),
        "qber": np.array(qber_list),
        "eve_info": np.array(eve_info_list),
    }


if __name__ == "__main__":
    config = {
        "pulses_per_episode": 10_000,
        "pulse_rate": 1e6,
        "mu_signal": 0.5,
        "mu_decoy": 0.1,
        "mu_vac": 0.0,
        "p_signal": 0.7,
        "p_decoy": 0.3,
        "fiber_loss_db_per_km": 0.2,
        "distance_km": 50,
        "det_eff": 0.15,
        "dark_count": 1e-6,
        "baseline_qber": 0.01,
        "recon_eff": 1.1,
        "basis_prob": 0.5,
        # any extra scaling constants used in alice_action_to_aa
        "mu_signal_scale": 0.5,
        "mu_decoy_scale": 0.1,
    }

    eval_trained_bob(
        alice_model_path="models/alice_clean.zip",
        eve_model_path="models/eve_vs_fixed_ab.zip",
        bob_model_path="models/bob_vs_trained_eve.zip",
        config=config,
        n_episodes=50,
    )
