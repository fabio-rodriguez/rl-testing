import numpy as np
import matplotlib.pyplot as plt
import random

# ========== Original Cliff Walking ==========
n_rows, n_cols = 4, 12
n_states = n_rows * n_cols
n_actions = 4  # up, down, left, right
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

start = (3, 0)
goal = (3, 11)
cliff = [(3, i) for i in range(1, 11)]

def state_to_index(pos, n_cols=n_cols):
    return pos[0] * n_cols + pos[1]

def step(state, action, n_rows=n_rows, n_cols=n_cols, start=start, goal=goal, cliff=cliff):
    r, c = state
    dr, dc = actions[action]
    nr, nc = min(max(r + dr, 0), n_rows - 1), min(max(c + dc, 0), n_cols - 1)
    new_state = (nr, nc)
    if new_state in cliff:
        return start, -100, True
    elif new_state == goal:
        return new_state, 0, True
    else:
        return new_state, -1, False

# Q-learning parameters
alpha = 0.5
gamma = 1.0
epsilon = 0.1
episodes = 500

Q = np.zeros((n_states, n_actions))

def epsilon_greedy(state_idx, eps):
    if random.random() < eps:
        return random.randint(0, n_actions - 1)
    return np.argmax(Q[state_idx])

# Train on original environment
for ep in range(episodes):
    state = start
    state_idx = state_to_index(state)
    done = False
    while not done:
        action = epsilon_greedy(state_idx, epsilon)
        next_state, reward, done = step(state, action)
        next_idx = state_to_index(next_state)

        Q[state_idx, action] += alpha * (
            reward + gamma * np.max(Q[next_idx]) - Q[state_idx, action]
        )

        state, state_idx = next_state, next_idx

# ========== Policy printer ==========
arrow_map = {0: "↑", 1: "↓", 2: "←", 3: "→"}

def print_policy(Q, rows, cols, start, goal, cliff):
    policy = np.array([np.argmax(Q[s]) if s < len(Q) else -1 for s in range(rows*cols)])
    for r in range(rows):
        row_str = ""
        for c in range(cols):
            if (r, c) == start:
                row_str += " S "
            elif (r, c) == goal:
                row_str += " G "
            elif (r, c) in cliff:
                row_str += " ■ "
            else:
                s = state_to_index((r, c), cols)
                if s < len(policy):
                    row_str += f" {arrow_map[policy[s]]} "
                else:
                    row_str += " ? "
        print(row_str)

# ========== Show Original Policy ==========
print("\nPolicy learned in Original 4x12 Cliff Walking:")
print_policy(Q, n_rows, n_cols, start, goal, cliff)

# ========== Extract and Plot Path in Original Environment ==========
state = start
path = [state]
done = False
while not done and len(path) < 200:
    s_idx = state_to_index(state, n_cols=n_cols)
    action = np.argmax(Q[s_idx])
    next_state, reward, done = step(state, action)
    path.append(next_state)
    state = next_state

grid = np.zeros((n_rows, n_cols))
for (r, c) in cliff:
    grid[r, c] = -1
for (r, c) in path:
    grid[r, c] = 0.5
grid[start] = 0.8
grid[goal] = 1.0

plt.figure(figsize=(12, 4))
plt.imshow(grid, cmap="coolwarm", origin="upper")
plt.colorbar(label="State value (path/cliff)")
plt.title("Original Cliff Walking (4x12) - Learned Path")
plt.show()



# ========== Test new start and goal ==========
new_start = (3, 3)
new_goal = (3, 11)

# Run greedy policy from new start to new goal
state = new_start
path = [state]
done = False
while not done and len(path) < 200:
    s_idx = state_to_index(state)
    action = np.argmax(Q[s_idx])
    next_state, reward, done = step(state, action, start=new_start, goal=new_goal)
    path.append(next_state)
    state = next_state

# Print arrows for this run
print("\nApplying learned policy to new start/goal on same 4x12 board:")
print_policy(Q, n_rows, n_cols, new_start, new_goal, cliff)

# Plot the path
grid = np.zeros((n_rows, n_cols))
for (r, c) in cliff:
    grid[r, c] = -1
for (r, c) in path:
    grid[r, c] = 0.5
grid[new_start] = 0.8
grid[new_goal] = 1.0

plt.figure(figsize=(12, 4))
plt.imshow(grid, cmap="coolwarm", origin="upper")
plt.colorbar(label="State value (path/cliff)")
plt.title(f"New Start {new_start} -> New Goal {new_goal} (Policy Reuse)")
plt.show()