import numpy as np
import matplotlib.pyplot as plt

# Gridworld size
grid_shape = (5, 5)

# Special states
specials = {
    (0,1): {'reward': 10, 'next': (4,1)},  # A -> A'
    (0,3): {'reward': 5, 'next': (2,3)}    # B -> B'
}

# Actions: north, south, east, west
actions = ['N', 'S', 'E', 'W']
action_vectors = {'N': (-1,0), 'S': (1,0), 'E': (0,1), 'W': (0,-1)}

# Reward for hitting grid edge
edge_penalty = -1
gamma = 0.9

# Initialize value function
V = np.zeros(grid_shape)

# Policy placeholder
policy = np.full(grid_shape, '', dtype=object)

# Value Iteration Parameters
epsilon = 1e-4
max_iter = 1000

def step(state, action):
    if state in specials:
        special = specials[state]
        return special['next'], special['reward']
    # Normal move
    vec = action_vectors[action]
    next_state = (state[0]+vec[0], state[1]+vec[1])
    # Check grid boundaries
    if 0 <= next_state[0] < grid_shape[0] and 0 <= next_state[1] < grid_shape[1]:
        return next_state, 0
    else:
        return state, edge_penalty

# Value Iteration
for it in range(max_iter):
    delta = 0
    V_new = np.copy(V)
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            state = (i,j)
            action_values = []
            for a in actions:
                next_s, r = step(state, a)
                action_values.append(r + gamma * V[next_s])
            best_value = max(action_values)
            V_new[state] = best_value
            delta = max(delta, abs(V_new[state] - V[state]))
    V = V_new
    if delta < epsilon:
        break

# Extract optimal policy
for i in range(grid_shape[0]):
    for j in range(grid_shape[1]):
        state = (i,j)
        action_values = {}
        for a in actions:
            next_s, r = step(state, a)
            action_values[a] = r + gamma * V[next_s]
        best_action = max(action_values, key=action_values.get)
        policy[state] = best_action

print("Optimal Value Function (V*):\n", V)
print("\nOptimal Policy (π*):\n", policy)

plt.figure(figsize=(6,5))
plt.imshow(V, cmap='coolwarm', interpolation='nearest')
for i in range(grid_shape[0]):
    for j in range(grid_shape[1]):
        plt.text(j, i, f"{V[i,j]:.1f}", ha='center', va='center', color='black')
plt.title("Optimal Value Function V*")
plt.colorbar()
plt.show()

plt.figure(figsize=(6,5))
plt.imshow(np.zeros(grid_shape), cmap='Greys', interpolation='nearest')
for i in range(grid_shape[0]):
    for j in range(grid_shape[1]):
        plt.text(j, i, policy[i,j], ha='center', va='center', color='red', fontsize=12)
plt.title("Optimal Policy π* (N/S/E/W)")
plt.show()

