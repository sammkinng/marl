import numpy as np
import random
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output

# === Environment setup ===
GRID_SIZE = 5
ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']
ACTION_SPACE = len(ACTIONS)

def move(pos, action):
    x, y = pos
    if action == 0 and y > 0: y -= 1      # UP
    elif action == 1 and y < GRID_SIZE-1: y += 1  # DOWN
    elif action == 2 and x > 0: x -= 1    # LEFT
    elif action == 3 and x < GRID_SIZE-1: x += 1  # RIGHT
    return (x, y)

def random_pos(exclude=None):
    pos = (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1))
    while exclude and pos in exclude:
        pos = (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1))
    return pos

# === MARL setup ===
Q1 = {}  # agent 1 Q-table
Q2 = {}  # agent 2 Q-table

def get_Q(Q, state):
    if state not in Q:
        Q[state] = np.zeros(ACTION_SPACE)
    return Q[state]

alpha = 0.1
gamma = 0.9
epsilon = 0.3
episodes = 5000

# === Visualization helper ===
def render(agent1, agent2, coin, ep, r1, r2):
    clear_output(wait=True)
    grid = [["." for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    x1, y1 = agent1
    x2, y2 = agent2
    xc, yc = coin
    grid[y1][x1] = "A1"
    grid[y2][x2] = "A2"
    grid[yc][xc] = "💰"
    print(f"Episode {ep} | Reward A1={r1}, A2={r2}")
    for row in grid:
        print(" ".join(row))
    time.sleep(0.05)

# === Training ===
r_hist1, r_hist2 = [], []

for ep in range(episodes):
    a1 = random_pos()
    a2 = random_pos(exclude=[a1])
    coin = random_pos(exclude=[a1, a2])

    total_r1, total_r2 = 0, 0

    for step in range(50):
        state = (a1, a2, coin)

        # Agent 1 action (epsilon-greedy)
        if random.random() < epsilon:
            act1 = random.randint(0, ACTION_SPACE-1)
        else:
            act1 = np.argmax(get_Q(Q1, state))
        
        # Agent 2 action (epsilon-greedy)
        if random.random() < epsilon:
            act2 = random.randint(0, ACTION_SPACE-1)
        else:
            act2 = np.argmax(get_Q(Q2, state))

        # Move both agents
        next_a1 = move(a1, act1)
        next_a2 = move(a2, act2)
        next_coin = coin

        # Rewards
        r1, r2 = 0, 0
        if next_a1 == coin and next_a2 == coin:
            r1, r2 = 0.5, 0.5   # both reached simultaneously
            next_coin = random_pos(exclude=[next_a1, next_a2])
        elif next_a1 == coin:
            r1, r2 = 1, 0
            next_coin = random_pos(exclude=[next_a1, next_a2])
        elif next_a2 == coin:
            r1, r2 = 0, 1
            next_coin = random_pos(exclude=[next_a1, next_a2])

        # Next state
        next_state = (next_a1, next_a2, next_coin)

        # Q-updates
        Q1[state] = get_Q(Q1, state)
        Q2[state] = get_Q(Q2, state)
        Q1[next_state] = get_Q(Q1, next_state)
        Q2[next_state] = get_Q(Q2, next_state)

        Q1[state][act1] += alpha * (r1 + gamma * np.max(Q1[next_state]) - Q1[state][act1])
        Q2[state][act2] += alpha * (r2 + gamma * np.max(Q2[next_state]) - Q2[state][act2])

        a1, a2, coin = next_a1, next_a2, next_coin
        total_r1 += r1
        total_r2 += r2

        if ep % 200 == 0:  # render occasionally
            render(a1, a2, coin, ep, total_r1, total_r2)

    r_hist1.append(total_r1)
    r_hist2.append(total_r2)

    # epsilon decay
    epsilon = max(0.05, epsilon * 0.9995)

# === Results ===
clear_output(wait=True)
print("✅ Training complete!")
plt.plot(np.convolve(r_hist1, np.ones(100)/100, mode='valid'), label="Agent 1")
plt.plot(np.convolve(r_hist2, np.ones(100)/100, mode='valid'), label="Agent 2")
plt.title("Rewards over time")
plt.xlabel("Episode")
plt.ylabel("Reward (100-episode avg)")
plt.legend()
plt.savefig("rewards_plot.png")
print("Plot saved as rewards_plot.png")

