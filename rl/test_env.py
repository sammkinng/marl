

# minimal simulator config (your existing config dict works too)
from .qkd_rl import QKDEnv


# config = {
#     "distance_km": 10,
#     "mu_signal": 0.5,
#     "mu_decoy": 0.1,
#     "det_eff": 0.1,
#     "dark_count": 1e-6,
#     "fiber_loss_db_per_km": 0.2,
#     "pulses_per_episode": 1000,
# }

import yaml
config = yaml.safe_load(open("configs/config.yaml"))

env = QKDEnv(config)

obs = env.reset()
print("Initial observation:", obs)

# random test actions
actions = {
    "Alice": env.action_space["Alice"].sample(),
    "Bob": env.action_space["Bob"].sample(),
    "Eve": env.action_space["Eve"].sample(),
}
obs, rewards, terminated, truncated, info = env.step(actions)
done = terminated or truncated

print("Obs:", obs)
print("Rewards:", rewards)
print("SKR:", info["raw_info"].get("SKR_bits_per_pulse"))
