import pandas as pd
import matplotlib.pyplot as plt

def plot_training_curve(monitor_file, label, ax=None):
    # read SB3 monitor log
    df = pd.read_csv(monitor_file, skiprows=1)  # first line is metadata

    # print(len(df))
    
    timesteps = df["t"]
    rewards = df["r"]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    df["reward_smooth"] = df["r"].rolling(window=20).mean()

    ax.plot(df["t"], df["reward_smooth"], label=label)

    # ax.plot(timesteps, rewards, linewidth=2, label=label)
    ax.set_xlabel("Total Timesteps")
    ax.set_ylabel("Episode Reward")
    ax.grid(True)
    return ax


# ============================
# Example usage
# ============================

fig, ax = plt.subplots(figsize=(10, 5))

ax = plot_training_curve("logs_eve/env_6/monitor.csv", "Bob (defender)", ax)
ax = plot_training_curve("logs_eve/env_7/monitor.csv", "Eve (attacker)", ax)

ax.legend(loc="upper left")
ax.set_title("Training Curve: Episode Reward vs Timesteps")

plt.tight_layout()
plt.savefig("training_curve.png", dpi=300)
# plt.savefig("training_curve.pdf")
plt.show()
