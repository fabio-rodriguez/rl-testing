import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# =============================
# Mountain Car Environment
# =============================
class MountainCar:
    def __init__(self):
        self.pos_min = -1.2
        self.pos_max = 0.5
        self.vel_min = -0.07
        self.vel_max = 0.07
        self.reset()

    def reset(self):
        # Start always at the bottom of the valley
        self.pos = -0.5
        self.vel = 0.0
        return np.array([self.pos, self.vel])

    def step(self, action):
        self.vel += 0.001 * action - 0.0025 * np.cos(3 * self.pos)
        self.vel = np.clip(self.vel, self.vel_min, self.vel_max)

        self.pos += self.vel
        self.pos = np.clip(self.pos, self.pos_min, self.pos_max)

        if self.pos == self.pos_min:
            self.vel = 0

        done = self.pos >= self.pos_max
        reward = -1

        return np.array([self.pos, self.vel]), reward, done

# =============================
# Tile Coding
# =============================
class TileCoder:
    def __init__(self, num_tilings=10, tiles=9):
        self.num_tilings = num_tilings
        self.tiles = tiles
        self.pos_scale = tiles / (0.5 + 1.2)
        self.vel_scale = tiles / (0.14)
        self.offsets = np.random.rand(num_tilings, 2)

    def get_features(self, state, action):
        pos, vel = state
        features = []
        for i in range(self.num_tilings):
            offset = self.offsets[i]
            p = int((pos + 1.2) * self.pos_scale + offset[0])
            v = int((vel + 0.07) * self.vel_scale + offset[1])
            features.append((i, p, v, action))
        return features

# =============================
# Sarsa(λ)
# =============================
class SarsaLambda:
    def __init__(self, alpha=0.05, gamma=1.0, lam=0.9):
        self.alpha = alpha
        self.gamma = gamma
        self.lam = lam
        self.weights = {}
        self.traces = {}

    def get_q(self, features):
        return sum(self.weights.get(f, 0.0) for f in features)

    def reset_traces(self):
        self.traces = {}

    def update(self, features, delta):
        # replacing traces
        for f in features:
            self.traces[f] = 1

        for f in list(self.traces.keys()):
            self.weights[f] = self.weights.get(f, 0.0) + self.alpha * delta * self.traces[f]
            self.traces[f] *= self.gamma * self.lam

# =============================
# Policy
# =============================
def epsilon_greedy(agent, tile, state, actions, eps=0.1):
    # exploration
    if np.random.rand() < eps:
        return np.random.choice(actions)

    qs = np.array([agent.get_q(tile.get_features(state, a)) for a in actions])
    max_q = np.max(qs)

    # random tie-breaking (VERY important)
    best_actions = [a for a, q in zip(actions, qs) if q == max_q]
    return np.random.choice(best_actions)

# =============================
# Training
# =============================
def train(episodes=200):
    env = MountainCar()
    tile = TileCoder()
    agent = SarsaLambda()

    actions = [-1, 0, 1]
    steps_per_episode = []
    cost_snapshots = {}

    for ep in range(episodes):
        state = env.reset()
        agent.reset_traces()

        action = epsilon_greedy(agent, tile, state, actions, eps=0.0)
        steps = 0

        while True:
            next_state, reward, done = env.step(action)
            next_action = epsilon_greedy(agent, tile, next_state, actions, eps=0.0)

            q_current = agent.get_q(tile.get_features(state, action))
            q_next = agent.get_q(tile.get_features(next_state, next_action))

            delta = reward + agent.gamma * q_next - q_current
            agent.update(tile.get_features(state, action), delta)

            state = next_state
            action = next_action
            steps += 1

            if done:
                break

        steps_per_episode.append(steps)

        # Save snapshots like Figure 9.10
        if ep in [0, 10, 50, 100, 199]:
            cost_snapshots[ep] = compute_cost(agent, tile)

        print(f"Episode {ep}, steps: {steps}")

    return agent, tile, steps_per_episode, cost_snapshots

# =============================
# Cost-to-go
# =============================
def compute_cost(agent, tile):
    pos = np.linspace(-1.2, 0.5, 40)
    vel = np.linspace(-0.07, 0.07, 40)
    Z = np.zeros((len(pos), len(vel)))

    actions = [-1, 0, 1]

    for i, p in enumerate(pos):
        for j, v in enumerate(vel):
            state = np.array([p, v])
            values = [agent.get_q(tile.get_features(state, a)) for a in actions]
            Z[i, j] = -max(values)

    return pos, vel, Z

# =============================
# Plot cost evolution
# =============================
def plot_cost_snapshots(cost_snapshots):
    fig = plt.figure(figsize=(12, 8))

    for idx, (ep, (pos, vel, Z)) in enumerate(cost_snapshots.items()):
        ax = fig.add_subplot(2, 3, idx+1, projection='3d')
        X, Y = np.meshgrid(pos, vel)
        ax.plot_surface(X, Y, Z.T)
        ax.set_title(f"Episode {ep}")

    plt.tight_layout()
    plt.show()

# =============================
# Plot trajectory
# =============================
def plot_trajectory(agent, tile):
    env = MountainCar()
    actions = [-1, 0, 1]

    state = env.reset()
    positions = []

    for _ in range(300):
        positions.append(state[0])
        action = epsilon_greedy(agent, tile, state, actions, eps=0.0)
        state, _, done = env.step(action)
        if done:
            break

    plt.plot(positions)
    plt.title("Car Position Over Time (Learned Policy)")
    plt.xlabel("Time step")
    plt.ylabel("Position")
    plt.show()

# =============================
# Main
# =============================
if __name__ == "__main__":
    agent, tile, steps, cost_snapshots = train(episodes=200)

    # Learning curve
    plt.plot(steps)
    plt.xlabel("Episode")
    plt.ylabel("Steps to goal")
    plt.title("Learning Curve")
    plt.show()

    # Cost evolution
    plot_cost_snapshots(cost_snapshots)

    # Trajectory after learning
    plot_trajectory(agent, tile)
