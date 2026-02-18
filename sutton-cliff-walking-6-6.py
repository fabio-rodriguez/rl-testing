import numpy as np
import matplotlib.pyplot as plt

# =============================
# Environment
# =============================

HEIGHT = 4
WIDTH = 12

START = (3, 0)
GOAL = (3, 11)
CLIFF = [(3, i) for i in range(1, 11)]

ACTIONS = [
    (0, 1),   # 0: Right
    (0, -1),  # 1: Left
    (1, 0),   # 2: Down
    (-1, 0)   # 3: Up
]

ACTION_SYMBOLS = {
    0: "→",
    1: "←",
    2: "↓",
    3: "↑"
}

N_ACTIONS = 4


def step(state, action):
    if state == GOAL:
        return state, 0

    move = ACTIONS[action]
    next_state = (state[0] + move[0], state[1] + move[1])

    # Stay inside grid
    next_state = (
        max(0, min(HEIGHT - 1, next_state[0])),
        max(0, min(WIDTH - 1, next_state[1]))
    )

    reward = -1

    if next_state in CLIFF:
        reward = -100
        next_state = START

    return next_state, reward


def epsilon_greedy(Q, state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(N_ACTIONS)
    return np.argmax(Q[state[0], state[1]])


# =============================
# SARSA
# =============================

def sarsa(alpha=0.5, epsilon=0.1, episodes=500):
    Q = np.zeros((HEIGHT, WIDTH, N_ACTIONS))
    rewards = []

    for _ in range(episodes):
        state = START
        action = epsilon_greedy(Q, state, epsilon)
        total_reward = 0

        while state != GOAL:
            next_state, reward = step(state, action)
            next_action = epsilon_greedy(Q, next_state, epsilon)

            Q[state[0], state[1], action] += alpha * (
                reward
                + Q[next_state[0], next_state[1], next_action]
                - Q[state[0], state[1], action]
            )

            state = next_state
            action = next_action
            total_reward += reward

        rewards.append(total_reward)

    return Q, rewards


# =============================
# Q-Learning
# =============================

def q_learning(alpha=0.5, epsilon=0.1, episodes=500):
    Q = np.zeros((HEIGHT, WIDTH, N_ACTIONS))
    rewards = []

    for _ in range(episodes):
        state = START
        total_reward = 0

        while state != GOAL:
            action = epsilon_greedy(Q, state, epsilon)
            next_state, reward = step(state, action)

            Q[state[0], state[1], action] += alpha * (
                reward
                + np.max(Q[next_state[0], next_state[1]])
                - Q[state[0], state[1], action]
            )

            state = next_state
            total_reward += reward

        rewards.append(total_reward)

    return Q, rewards


# =============================
# Policy & Path Extraction
# =============================

def extract_policy(Q):
    policy = np.zeros((HEIGHT, WIDTH), dtype=int)
    for r in range(HEIGHT):
        for c in range(WIDTH):
            policy[r, c] = np.argmax(Q[r, c])
    return policy


def print_policy(policy, title):
    print("\n" + title)
    for r in range(HEIGHT):
        row = ""
        for c in range(WIDTH):
            if (r, c) == START:
                row += " S "
            elif (r, c) == GOAL:
                row += " G "
            elif (r, c) in CLIFF:
                row += " C "
            else:
                row += f" {ACTION_SYMBOLS[policy[r, c]]} "
        print(row)


def extract_path(Q):
    state = START
    path = [state]

    visited = set()

    while state != GOAL:
        if state in visited:
            # Avoid infinite loop
            break
        visited.add(state)

        action = np.argmax(Q[state[0], state[1]])
        next_state, _ = step(state, action)
        path.append(next_state)
        state = next_state

    return path


# =============================
# Utilities
# =============================

def smooth(data, window=10):
    return np.convolve(data, np.ones(window) / window, mode="valid")


# =============================
# Main
# =============================

if __name__ == "__main__":
    np.random.seed(0)
    episodes = 500

    Q_sarsa, sarsa_rewards = sarsa(episodes=episodes)
    Q_q, q_rewards = q_learning(episodes=episodes)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(smooth(sarsa_rewards), label="Sarsa")
    plt.plot(smooth(q_rewards), label="Q-learning")
    plt.xlabel("Episodes")
    plt.ylabel("Reward (smoothed)")
    plt.title("Cliff Walking: Sarsa vs Q-learning")
    plt.legend()
    plt.grid()
    plt.show()

    # Extract and print policies
    policy_sarsa = extract_policy(Q_sarsa)
    policy_q = extract_policy(Q_q)

    print_policy(policy_sarsa, "Learned Policy (Sarsa)")
    print_policy(policy_q, "Learned Policy (Q-learning)")

    # Print paths
    print("\nGreedy Path (Sarsa):")
    print(extract_path(Q_sarsa))

    print("\nGreedy Path (Q-learning):")
    print(extract_path(Q_q))

