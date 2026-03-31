import numpy as np
import random
import matplotlib.pyplot as plt

# -----------------------------
# Maze Environment
# -----------------------------
HEIGHT, WIDTH = 6, 9

START = (2, 0)
GOAL = (0, 8)

OBSTACLES = {
    (1,2),(2,2),(3,2),   # vertical wall
    (0,7),(1,7),(2,7),   # vertical wall
    (4,5)                # single block
}

ACTIONS = [(0,1),(0,-1),(1,0),(-1,0)]  # right, left, down, up
N_ACTIONS = len(ACTIONS)

GAMMA = 0.95
ALPHA = 0.1
EPSILON = 0.1

# -----------------------------
# Environment dynamics
# -----------------------------
def step(state, action):
    if state == GOAL:
        return START, 0

    r, c = state
    dr, dc = ACTIONS[action]
    nr, nc = r + dr, c + dc

    # check boundaries / obstacles
    if (nr < 0 or nr >= HEIGHT or
        nc < 0 or nc >= WIDTH or
        (nr, nc) in OBSTACLES):
        nr, nc = r, c

    next_state = (nr, nc)
    reward = 1 if next_state == GOAL else 0

    return next_state, reward

# -----------------------------
# Policy (epsilon-greedy)
# -----------------------------
def epsilon_greedy(Q, state):
    if random.random() < EPSILON:
        return random.randint(0, N_ACTIONS - 1)
    return np.argmax(Q[state])

# -----------------------------
# Dyna-Q Algorithm
# -----------------------------
def dyna_q(n_planning, episodes=50):
    Q = np.zeros((HEIGHT, WIDTH, N_ACTIONS))
    model = {}  # stores (state, action) -> (next_state, reward)

    steps_per_episode = []

    for ep in range(episodes):
        state = START
        steps = 0

        while state != GOAL:
            # --- Real interaction ---
            action = epsilon_greedy(Q, state)
            next_state, reward = step(state, action)

            # --- Direct RL (Q-learning) ---
            Q[state][action] += ALPHA * (
                reward + GAMMA * np.max(Q[next_state]) - Q[state][action]
            )

            # --- Model learning ---
            model[(state, action)] = (next_state, reward)

            # --- Planning ---
            for _ in range(n_planning):
                (s, a) = random.choice(list(model.keys()))
                s_next, r = model[(s, a)]

                Q[s][a] += ALPHA * (
                    r + GAMMA * np.max(Q[s_next]) - Q[s][a]
                )

            state = next_state
            steps += 1

        steps_per_episode.append(steps)

    return steps_per_episode

# -----------------------------
# Experiment (like Figure 8.5)
# -----------------------------
def run_experiment():
    episodes = 50
    runs = 20

    planning_values = [0, 5, 50]
    results = {}

    for n in planning_values:
        avg_steps = np.zeros(episodes)

        for _ in range(runs):
            steps = dyna_q(n, episodes)
            avg_steps += np.array(steps)

        avg_steps /= runs
        results[n] = avg_steps

    # -----------------------------
    # Plot results
    # -----------------------------
    plt.figure()

    for n, data in results.items():
        plt.plot(data, label=f"{n} planning steps")

    plt.xlabel("Episodes")
    plt.ylabel("Steps per episode")
    plt.title("Dyna-Q Learning (Maze Example)")
    plt.legend()
    plt.grid()

    plt.show()

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    run_experiment()