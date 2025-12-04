import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("logs_eve/env_7/monitor.csv", comment="#",skiprows=1)

df["r_smooth"] = df["r"].rolling(window=50).mean()



plt.figure(figsize=(10,5))
# plt.plot(df["t"], df["r"], linewidth=1)
plt.plot(df["t"], df["r_smooth"], linewidth=2, label="Smoothed Reward")
# plt.legend()
plt.xlabel("Timesteps")
plt.ylabel("Episode Reward")
plt.title("Training Curve")
plt.grid(True)
plt.tight_layout()
plt.savefig("training_curve_eve.png", dpi=300)
plt.show()
