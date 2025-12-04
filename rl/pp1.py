import matplotlib.pyplot as plt

bp={"R0":0.5,"R1":0.0645,"R2":0.0,"R3":0.0996,"R4":0.0,"R5":0.0421}
gain={"R0":0.2,"R1":0.5,"R2":0.5,"R3":0.5,"R4":0.5,"R5":0.5}

rounds = list(bp.keys())
bp_vals = [bp[r] for r in rounds]
gain_vals = [gain[r] for r in rounds]

plt.figure(figsize=(9,5))

plt.plot(rounds, bp_vals, marker='o', label="Basis probability", linewidth=2)
plt.plot(rounds, gain_vals, marker='s', label="Detector gain", linewidth=2)

plt.xlabel("Round")
plt.ylabel("Action values")
plt.title("Bob Strategy Evolution Across Rounds (bp, gain)")
plt.legend(loc="upper right")
plt.grid(True)
plt.tight_layout()

plt.savefig("bob_strategy_evolution.png", dpi=300)
# plt.savefig("bob_strategy_evolution.pdf")
plt.show()
