import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# -----------------------------
# Environment: Windy Gridworld
# -----------------------------
class WindyGridworld:
    def __init__(self):
        self.rows = 7
        self.cols = 10
        self.start = (3, 0)
        self.goal = (3, 7)
        self.wind = np.array([0,0,0,1,1,1,2,2,1,0])
        self.reset()

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        r, c = self.state
        
        # Actions: 0=up, 1=down, 2=left, 3=right
        if action == 0:
            r -= 1
        elif action == 1:
            r += 1
        elif action == 2:
            c -= 1
        elif action == 3:
            c += 1

        # Apply wind (upward)
        r -= self.wind[self.state[1]]

        # Boundaries
        r = min(max(r, 0), self.rows - 1)
        c = min(max(c, 0), self.cols - 1)

        self.state = (r, c)

        reward = -1
        done = (self.state == self.goal)

        return self.state, reward, done


# -----------------------------
# SARSA Implementation
# -----------------------------
def epsilon_greedy(Q, state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(4)
    else:
        return np.argmax(Q[state])


def sarsa(env, episodes=1000, alpha=0.5, epsilon=0.1, gamma=1.0, max_steps=8000):
    Q = defaultdict(lambda: np.zeros(4))
    
    time_steps = 0
    episode_count = 0
    episodes_vs_time = []

    while time_steps < max_steps:
        state = env.reset()
        action = epsilon_greedy(Q, state, epsilon)
        done = False
        
        while not done:
            next_state, reward, done = env.step(action)
            next_action = epsilon_greedy(Q, next_state, epsilon)

            Q[state][action] += alpha * (
                reward + gamma * Q[next_state][next_action] - Q[state][action]
            )

            state = next_state
            action = next_action

            time_steps += 1
            episodes_vs_time.append(episode_count)

            if done:
                episode_count += 1

            if time_steps >= max_steps:
                break

    return Q, episodes_vs_time


# -----------------------------
# Training
# -----------------------------
env = WindyGridworld()
Q, episodes_vs_time = sarsa(env, max_steps=8000)

# -----------------------------
# Plot Learning Curve
# -----------------------------
plt.figure()
plt.plot(episodes_vs_time)
plt.xlabel("Time Steps")
plt.ylabel("Episodes")
plt.title("Sarsa on Windy Gridworld")
plt.show()


# -----------------------------
# Plot Learned Greedy Policy
# -----------------------------
policy_grid = np.zeros((env.rows, env.cols), dtype=int)

for r in range(env.rows):
    for c in range(env.cols):
        if (r, c) != env.goal:
            policy_grid[r, c] = np.argmax(Q[(r, c)])

plt.figure()
for r in range(env.rows):
    for c in range(env.cols):
        if (r, c) == env.goal:
            plt.text(c, env.rows - r - 1, "G", ha='center', va='center')
        else:
            a = policy_grid[r, c]
            dx, dy = 0, 0
            if a == 0: dy = 0.4
            elif a == 1: dy = -0.4
            elif a == 2: dx = -0.4
            elif a == 3: dx = 0.4
            plt.arrow(c, env.rows - r - 1, dx, dy,
                      head_width=0.2, length_includes_head=True)

plt.xlim(-0.5, env.cols - 0.5)
plt.ylim(-0.5, env.rows - 0.5)
plt.xticks(range(env.cols))
plt.yticks(range(env.rows))
plt.grid()
plt.title("Learned Greedy Policy")
plt.show()

