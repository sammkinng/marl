import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
import time

# === Create the environment (with human render support) ===
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode=None)

# === Initialize Q-table ===
state_space = env.observation_space.n
action_space = env.action_space.n
Q = np.zeros((state_space, action_space))

# === Hyperparameters ===
alpha = 0.8      # learning rate
gamma = 0.95     # discount factor
epsilon = 0.1    # exploration rate
episodes = 20000  # training episodes
max_steps = 100  # per episode

# === Training ===
rewards = []

for episode in range(episodes):
    state, _ = env.reset()
    total_reward = 0
    done = False

    for step in range(max_steps):
        # Epsilon-greedy exploration
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])

        # Take action
        next_state, reward, done, truncated, info = env.step(action)

        # Q-learning update
        Q[state, action] = Q[state, action] + alpha * (
            reward + gamma * np.max(Q[next_state, :]) - Q[state, action]
        )

        state = next_state
        total_reward += reward

        if done:
            break

    rewards.append(total_reward)

# === Plot learning curve ===
window = 100
smoothed_rewards = [np.mean(rewards[i - window:i + 1]) for i in range(window, len(rewards))]
plt.plot(smoothed_rewards)
plt.title("Q-learning on FrozenLake-v1")
plt.xlabel("Episode")
plt.ylabel("Average Reward (100-episode window)")
plt.show()

# === Print learned Q-table and policy ===
print("\nLearned Q-table:")
print(Q)

policy = np.array([np.argmax(Q[s]) for s in range(state_space)])
print("\nLearned policy (0=Left, 1=Down, 2=Right, 3=Up):")
print(policy.reshape(4, 4))

# === Step 3: Watch the trained agent play ===
# Recreate the environment with human render enabled
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True, render_mode="human")

state, _ = env.reset()
done = False
total_reward = 0

print("\n🤖 Agent is playing...")

while not done:
    action = np.argmax(Q[state])
    state, reward, done, truncated, info = env.step(action)
    total_reward += reward
    time.sleep(0.7)  # slow down for visibility

print(f"\n✅ Episode finished with total reward: {total_reward}")
env.close()
