import numpy as np
import random
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time

# Environment setup
num_states = 5  # positions: 0 to 4
actions = [0, 1]  # 0 = left, 1 = right
num_actions = len(actions)

# Q-table: state x action
Q = np.zeros((num_states, num_actions))

# Hyperparameters
alpha = 0.1     # learning rate
gamma = 0.9     # discount factor
epsilon = 0.3   # exploration rate
episodes = 100

def step(state, action):
    """Returns next_state, reward, done"""
    if action == 0:  # move left
        next_state = max(0, state - 1)
    else:             # move right
        next_state = min(num_states - 1, state + 1)

    reward = 10 if next_state == num_states - 1 else 0
    done = next_state == num_states - 1
    return next_state, reward, done

def render(state, episode):
    """Visualize the current environment"""
    clear_output(wait=True)
    env = ["_"] * num_states
    env[-1] = "💰"  # Treasure
    env[state] = "🤖"  # Agent
    print(f"Episode: {episode}")
    print(" ".join(env))
    plt.figure(figsize=(6, 2))
    plt.bar(range(num_states), np.max(Q, axis=1))
    plt.title("Learned Q-values (max over actions)")
    plt.xlabel("State")
    plt.ylabel("Value")
    plt.ylim(0, 10)
    plt.show()
    time.sleep(0.1)

# Training loop
for ep in range(1, episodes + 1):
    state = 0
    done = False

    while not done:
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = random.choice(actions)
        else:
            action = np.argmax(Q[state])

        next_state, reward, done = step(state, action)

        # Q-learning update rule
        Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])

        state = next_state
        render(state, ep)

print("\nTraining complete 🎉")
print("Final Q-table:\n", Q)
print("Best policy (0=Left, 1=Right):", [np.argmax(Q[s]) for s in range(num_states)])
