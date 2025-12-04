import matplotlib.pyplot as plt
import numpy as np

# ---------------------------
# Bob round metrics
# ---------------------------
bob_rounds = [
    ("R0", 1.29e5, 0.0129, 4.75e-1),
    ("R1", 6.49e4, 0.0129, 2.407e-1),
    ("R2", 6.43e4, 0.0133, 2.39e-1),
    ("R3", 6.24e4, 0.0151, 2.38e-1),
    ("R4", 6.54e4, 0.0120, 2.39e-1),
    ("R5", 6.44e4, 0.0132, 2.40e-1),
]

round_labels = [r[0] for r in bob_rounds]
skr = np.array([r[1] for r in bob_rounds])
qber = np.array([r[2] for r in bob_rounds])
eve_info = np.array([r[3] for r in bob_rounds])

# ---------------------------
# Plot
# ---------------------------
fig, ax1 = plt.subplots(figsize=(9, 5))

# Left y-axis → SKR
ax1.plot(round_labels, skr, marker='o', linewidth=2, label="SKR (bits/s)")
ax1.set_xlabel("Round")
ax1.set_ylabel("SKR (bits/s)")
ax1.set_ylim(0.4e5, max(skr)*1.1)

# Right y-axis → QBER & Eve Info
ax2 = ax1.twinx()
ax2.plot(round_labels, qber, marker='s', linestyle='--', linewidth=2, label="QBER")
ax2.plot(round_labels, eve_info, marker='^', linestyle='--', linewidth=2, label="Eve Info Gain")
ax2.set_ylabel("QBER / Eve Info")

# Legend (top right)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.title("Bob Performance Across Rounds")
plt.tight_layout()

# Save
plt.savefig("bob_round_metrics.png", dpi=300)
# plt.savefig("bob_round_metrics.pdf")

plt.show()
