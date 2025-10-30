import numpy as np
import random
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output

# === Environment parameters ===
GRID_SIZE = 5
ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']
ACTION_SPACE = len(ACTIONS)

def move(pos, action):
    x, y = pos
    if action == 0 and y > 0: y -= 1
    elif action == 1 and y < GRID_SIZE - 1: y += 1
    elif action == 2 and x > 0: x -= 1
    elif action == 3 and x < GRID_SIZE - 1: x += 1
    return (x, y)

def random_pos(exclude=None):
    pos = (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1))
    while exclude and pos in exclude:
        pos = (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1))
    return pos

# === Q-learning setup ===
Q1, Q2 = {}, {}

def get_Q(Q, state):
    if state not in Q:
        Q[state] = np.zeros(ACTION_SPACE)
    return Q[state]

alpha = 0.1
gamma = 0.9
epsilon = 0.3
episodes = 4000

# === Visualization ===
def render(a1, a2, goal, ep, total_r):
    clear_output(wait=True)
    grid = [["." for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    gx, gy = goal
    x1, y1 = a1
    x2, y2 = a2
    grid[gy][gx] = "💰"
    grid[y1][x1] = "A1"
    grid[y2][x2] = "A2"
    print(f"Episode {ep} | Total reward: {total_r}")
    for row in grid:
        print(" ".join(row))
    time.sleep(0.05)

r_hist = []

# === Training loop ===
for ep in range(episodes):
    a1 = random_pos()
    a2 = random_pos(exclude=[a1])
    goal = random_pos(exclude=[a1, a2])
    total_r = 0

    for step in range(50):
        state = (a1, a2, goal)

        # Actions (epsilon-greedy)
        if random.random() < epsilon:
            act1 = random.randint(0, ACTION_SPACE-1)
            act2 = random.randint(0, ACTION_SPACE-1)
        else:
            act1 = np.argmax(get_Q(Q1, state))
            act2 = np.argmax(get_Q(Q2, state))

        # Move
        next_a1 = move(a1, act1)
        next_a2 = move(a2, act2)

        # Reward: only if both reach goal together
        r = 0
        if next_a1 == goal and next_a2 == goal:
            r = 1
            goal = random_pos(exclude=[next_a1, next_a2])

        next_state = (next_a1, next_a2, goal)

        # Update Q-values (shared reward)
        Q1[state] = get_Q(Q1, state)
        Q2[state] = get_Q(Q2, state)
        Q1[next_state] = get_Q(Q1, next_state)
        Q2[next_state] = get_Q(Q2, next_state)

        Q1[state][act1] += alpha * (r + gamma * np.max(Q1[next_state]) - Q1[state][act1])
        Q2[state][act2] += alpha * (r + gamma * np.max(Q2[next_state]) - Q2[state][act2])

        a1, a2 = next_a1, next_a2
        total_r += r

        if ep % 200 == 0:
            render(a1, a2, goal, ep, total_r)

    r_hist.append(total_r)
    epsilon = max(0.05, epsilon * 0.9995)

# === Results ===
clear_output(wait=True)
plt.plot(np.convolve(r_hist, np.ones(100)/100, mode='valid'))
plt.title("Cooperative MARL: Two Agents, One Goal")
plt.xlabel("Episode")
plt.ylabel("Average Shared Reward (100-ep window)")
# plt.show()
print("✅ Training complete!")
plt.savefig("coop.png")
print("Plot saved as rewards_plot.png")

