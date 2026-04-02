import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Parameters (match the book)
# ----------------------------
n_servers = 10
priorities = [1, 2, 4, 8]
p = 0.06

alpha = 0.01
beta = 0.01
epsilon = 0.1

num_steps = 2_000_000

num_priorities = len(priorities)

# Q(s,a): [free_servers, priority_index, action]
Q = np.zeros((n_servers + 1, num_priorities, 2))

# Average reward estimate
R_bar = 0.0


# ----------------------------
# Helpers
# ----------------------------
def sample_priority():
    return np.random.choice(priorities)

def priority_index(p):
    return priorities.index(p)

def step(free_servers, action, priority):
    reward = 0

    # Accept if possible
    if action == 1 and free_servers > 0:
        reward = priority
        free_servers -= 1

    # Servers freeing up
    busy = n_servers - free_servers
    freed = np.random.binomial(busy, p)
    free_servers = min(n_servers, free_servers + freed)

    next_priority = sample_priority()

    return free_servers, next_priority, reward


def choose_action(free_servers, p_idx):
    if np.random.rand() < epsilon:
        return np.random.choice([0, 1])
    else:
        return np.argmax(Q[free_servers, p_idx])


# ----------------------------
# Training
# ----------------------------
free_servers = n_servers
priority = sample_priority()

for t in range(num_steps):
    p_idx = priority_index(priority)

    action = choose_action(free_servers, p_idx)

    next_free, next_priority, reward = step(free_servers, action, priority)
    next_p_idx = priority_index(next_priority)

    # R-learning TD error
    delta = reward - R_bar + np.max(Q[next_free, next_p_idx]) - Q[free_servers, p_idx, action]

    # Update Q
    Q[free_servers, p_idx, action] += alpha * delta

    # Update average reward (only greedy)
    if action == np.argmax(Q[free_servers, p_idx]):
        R_bar += beta * delta

    free_servers = next_free
    priority = next_priority


print(f"Estimated average reward: {R_bar:.2f}")


# ----------------------------
# Extract Policy
# ----------------------------
policy = np.zeros((n_servers + 1, num_priorities))

for s in range(n_servers + 1):
    for p_idx in range(num_priorities):
        policy[s, p_idx] = np.argmax(Q[s, p_idx])  # 0=reject, 1=accept


# ----------------------------
# Extract Value Function
# ----------------------------
V = np.max(Q, axis=2)  # best action value


# ----------------------------
# Plot (replicating Figure 11.3)
# ----------------------------
fig, axs = plt.subplots(2, 1, figsize=(8, 8))

# --- POLICY PLOT ---
for i, p_val in enumerate(priorities):
    axs[0].step(range(n_servers + 1), policy[:, i], where='mid', label=f'Priority {p_val}')

axs[0].set_title("Policy")
axs[0].set_xlabel("Number of free servers")
axs[0].set_ylabel("Action (0=Reject, 1=Accept)")
axs[0].legend()
axs[0].grid()


# --- VALUE FUNCTION PLOT ---
for i, p_val in enumerate(priorities):
    axs[1].plot(range(n_servers + 1), V[:, i], label=f'priority {p_val}')

axs[1].set_title("Value Function (Value of Best Action)")
axs[1].set_xlabel("Number of free servers")
axs[1].set_ylabel("Value")
axs[1].legend()
axs[1].grid()

plt.tight_layout()
plt.show()