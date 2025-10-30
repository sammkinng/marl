import numpy as np
import random
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output

# === Environment setup ===
GRID_SIZE = 5
ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY']
ACTION_SPACE = len(ACTIONS)

def move(pos, action):
    x, y = pos
    if action == 0 and y > 0: y -= 1
    elif action == 1 and y < GRID_SIZE-1: y += 1
    elif action == 2 and x > 0: x -= 1
    elif action == 3 and x < GRID_SIZE-1: x += 1
    return (x, y)

def random_pos(exclude=None):
    pos = (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1))
    while exclude and pos in exclude:
        pos = (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1))
    return pos

# === Q-tables for both agents ===
Q_hunter = {}
Q_prey = {}

def get_Q(Q, state):
    if state not in Q:
        Q[state] = np.zeros(ACTION_SPACE)
    return Q[state]

# === Hyperparameters ===
alpha = 0.2
gamma = 0.9
epsilon = 1.0
min_epsilon = 0.05
decay = 0.9995
episodes = 4000
max_steps = 50

# === Visualization helper ===
def render(hunter, prey, ep, rh, rp):
    clear_output(wait=True)
    grid = [["." for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    hx, hy = hunter
    px, py = prey
    grid[hy][hx] = "H"
    grid[py][px] = "P"
    print(f"Episode {ep} | Hunter reward={rh:.2f} | Prey reward={rp:.2f}")
    for row in grid:
        print(" ".join(row))
    time.sleep(0.05)

# === Training loop ===
r_hist_h, r_hist_p = [], []

for ep in range(episodes):
    hunter = random_pos()
    prey = random_pos(exclude=[hunter])

    total_h, total_p = 0, 0
    done = False

    for step in range(max_steps):
        state = (hunter, prey)

        # Actions (epsilon-greedy)
        if random.random() < epsilon:
            act_h = random.randint(0, ACTION_SPACE-1)
        else:
            act_h = np.argmax(get_Q(Q_hunter, state))

        if random.random() < epsilon:
            act_p = random.randint(0, ACTION_SPACE-1)
        else:
            act_p = np.argmax(get_Q(Q_prey, state))

        # Move both
        next_h = move(hunter, act_h)
        next_p = move(prey, act_p)
        next_state = (next_h, next_p)

        # Rewards
        if next_h == next_p:
            r_h, r_p = 1, -1   # hunter catches prey
            done = True
        else:
            r_h, r_p = -0.01, 0.01  # small time penalty/reward

        # Q-learning updates
        Q_hunter[state] = get_Q(Q_hunter, state)
        Q_prey[state] = get_Q(Q_prey, state)
        Q_hunter[next_state] = get_Q(Q_hunter, next_state)
        Q_prey[next_state] = get_Q(Q_prey, next_state)

        Q_hunter[state][act_h] += alpha * (r_h + gamma * np.max(Q_hunter[next_state]) - Q_hunter[state][act_h])
        Q_prey[state][act_p]   += alpha * (r_p + gamma * np.max(Q_prey[next_state]) - Q_prey[state][act_p])

        hunter, prey = next_h, next_p
        total_h += r_h
        total_p += r_p

        if ep % 200 == 0:
            render(hunter, prey, ep, total_h, total_p)

        if done:
            break

    epsilon = max(min_epsilon, epsilon * decay)
    r_hist_h.append(total_h)
    r_hist_p.append(total_p)

# === Results ===
clear_output(wait=True)
print("✅ Training complete!")

plt.plot(np.convolve(r_hist_h, np.ones(100)/100, mode='valid'), label="Hunter")
plt.plot(np.convolve(r_hist_p, np.ones(100)/100, mode='valid'), label="Prey")
plt.title("Adversarial MARL: Hunter vs Prey")
plt.xlabel("Episode")
plt.ylabel("Reward (100-episode avg)")
plt.legend()
plt.savefig("hunter.png")
print("Plot saved as rewards_plot.png")
