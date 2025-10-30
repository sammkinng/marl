import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from matplotlib import animation

# ==============================
# ENVIRONMENT
# ==============================

class GridCaptureEnv(gym.Env):
    def __init__(self, grid_size=5):
        super(GridCaptureEnv, self).__init__()
        self.grid_size = grid_size
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=0, high=grid_size - 1, shape=(3, 2), dtype=np.int32
        )
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.positions = {
            "A1": np.array([0, 0]),
            "A2": np.array([0, 1]),
            "B": np.array([4, 4]),
        }
        self.goal = np.array([4, 0])
        return self._get_obs(), {}

    def step(self, a1, a2, b):
        for agent, act in zip(["A1", "A2", "B"], [a1, a2, b]):
            delta = self._action_to_delta(act)
            self.positions[agent] = np.clip(self.positions[agent] + delta, 0, self.grid_size - 1)

        rewards = {"A1": -1, "A2": -1, "B": -1}
        done = False

        for a in ["A1", "A2"]:
            if np.array_equal(self.positions[a], self.goal):
                rewards["A1"] = rewards["A2"] = 10
                done = True

        for a in ["A1", "A2"]:
            if np.array_equal(self.positions[a], self.positions["B"]):
                rewards["B"] = 10
                rewards["A1"] = rewards["A2"] = -10
                done = True

        return self._get_obs(), rewards, done, False, {}

    def _action_to_delta(self, act):
        mapping = {
            0: np.array([0, 0]),
            1: np.array([-1, 0]),
            2: np.array([1, 0]),
            3: np.array([0, -1]),
            4: np.array([0, 1]),
        }
        return mapping[act]

    def _get_obs(self):
        return np.array([self.positions["A1"], self.positions["A2"], self.positions["B"]])


# ==============================
# Q-LEARNING AGENT
# ==============================

class QLearningAgent:
    def __init__(self, name, n_actions=5, lr=0.1, gamma=0.95, eps=0.1):
        self.name = name
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.eps = eps
        self.q_table = {}

    def _key(self, state):
        return tuple(state.flatten())

    def select_action(self, state):
        key = self._key(state)
        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.n_actions)
        if random.random() < self.eps:
            return random.randint(0, self.n_actions - 1)
        return np.argmax(self.q_table[key])

    def update(self, state, action, reward, next_state, done):
        key = self._key(state)
        next_key = self._key(next_state)
        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.n_actions)
        if next_key not in self.q_table:
            self.q_table[next_key] = np.zeros(self.n_actions)

        q_predict = self.q_table[key][action]
        q_target = reward + (0 if done else self.gamma * np.max(self.q_table[next_key]))
        self.q_table[key][action] += self.lr * (q_target - q_predict)


# ==============================
# VISUALIZATION HELPERS
# ==============================

def render_frame(env):
    """Draws and returns a matplotlib figure frame."""
    grid_size = env.grid_size
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.set_xticks(range(grid_size))
    ax.set_yticks(range(grid_size))
    ax.grid(True)
    ax.invert_yaxis()

    # Draw goal and agents
    ax.scatter(env.goal[1], env.goal[0], c='gold', marker='*', s=200, label="Goal")
    ax.scatter(env.positions["A1"][1], env.positions["A1"][0], c='blue', s=100, label="A1")
    ax.scatter(env.positions["A2"][1], env.positions["A2"][0], c='cyan', s=100, label="A2")
    ax.scatter(env.positions["B"][1], env.positions["B"][0], c='red', s=100, label="B")
    ax.legend(loc='upper right')
    ax.set_title("Grid Capture Game")
    return fig


# ==============================
# TRAINING + ANIMATION
# ==============================

def train_and_animate(num_episodes=200):
    env = GridCaptureEnv()
    A1, A2, B = QLearningAgent("A1"), QLearningAgent("A2"), QLearningAgent("B")
    reward_history = []

    frames = []  # collect frames for one test episode later

    for ep in range(1, num_episodes + 1):
        state, _ = env.reset()
        done = False
        total_rewards = {"A1": 0, "A2": 0, "B": 0}

        while not done:
            a1, a2, b = A1.select_action(state), A2.select_action(state), B.select_action(state)
            next_state, rewards, done, _, _ = env.step(a1, a2, b)

            A1.update(state, a1, rewards["A1"], next_state, done)
            A2.update(state, a2, rewards["A2"], next_state, done)
            B.update(state, b, rewards["B"], next_state, done)

            for k in total_rewards:
                total_rewards[k] += rewards[k]

            state = next_state

        reward_history.append((total_rewards["A1"] + total_rewards["A2"]) / 2)

    # === Create a test run for visualization ===
    state, _ = env.reset()
    done = False
    while not done:
        a1, a2, b = np.random.randint(5), np.random.randint(5), np.random.randint(5)
        next_state, rewards, done, _, _ = env.step(a1, a2, b)
        fig = render_frame(env)
        frames.append(fig)
        plt.close(fig)
        state = next_state

    # === Create Animation ===
    print("Creating animation... (may take a few seconds)")
    fig, ax = plt.subplots()

    def update(frame):
        ax.clear()
        img = frame
        ax.imshow(img.canvas.renderer.buffer_rgba())
        return [ax]

        # === Create Animation ===
    print("Creating animation... (may take a few seconds)")
    frame_images = []

    for f in frames:
        f.canvas.draw()
        canvas = f.canvas
        width, height = canvas.get_width_height()

        # Handle both modern and legacy Matplotlib canvases
        if hasattr(canvas, "tostring_rgb"):
            data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
            image = data.reshape(height, width, 3)
        else:
            # Convert ARGB → RGB
            data = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8)
            data = data.reshape(height, width, 4)
            image = data[:, :, 1:]  # drop alpha channel

        frame_images.append(image)
        plt.close(f)

    fig, ax = plt.subplots()
    ani = animation.ArtistAnimation(
        fig, [[plt.imshow(img)] for img in frame_images],
        interval=400, blit=True, repeat_delay=1000
    )
    # ani.save("marl_game_animation.mp4", writer="ffmpeg", dpi=150)
    ani.save("marl_game_animation.gif", writer="pillow", dpi=100)

    print("✅ Animation saved as 'marl_game_animation.mp4'")


    # === Plot Learning Curve ===
    plt.figure(figsize=(6,4))
    plt.plot(reward_history, label="Avg Coop Reward", color="blue")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Learning Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("marl_learning_curve.png")
    # plt.show()
    print("✅ Learning curve saved as 'marl_learning_curve.png'")

if __name__ == "__main__":
    train_and_animate(num_episodes=200)
