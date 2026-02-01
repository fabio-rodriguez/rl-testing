import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Parameters
# ----------------------------
GOAL = 100
ph = 0.4
theta = 1e-10
TOL = 1e-8       # tolerance for tie detection

# ----------------------------
# Initialization
# ----------------------------
V = np.zeros(GOAL + 1)
V[GOAL] = 1.0

# ----------------------------
# Value Iteration
# ----------------------------
while True:
    delta = 0

    for s in range(1, GOAL):
        old_v = V[s]
        actions = range(1, min(s, GOAL - s) + 1)

        q_values = []

        for a in actions:
            win_state = s + a
            lose_state = s - a

            # Correct Bellman update
            if win_state == GOAL:
                q = ph * 1.0 + (1 - ph) * V[lose_state]
            else:
                q = ph * V[win_state] + (1 - ph) * V[lose_state]

            q_values.append(q)

        V[s] = max(q_values)
        delta = max(delta, abs(old_v - V[s]))

    if delta < theta:
        break

print("Value iteration converged.")

# ----------------------------
# Policy Extraction
# ----------------------------
policy = np.zeros(GOAL + 1)

for s in range(1, GOAL):
    actions = list(range(1, min(s, GOAL - s) + 1))
    q_values = []

    for a in actions:
        win_state = s + a
        lose_state = s - a

        if win_state == GOAL:
            q = ph * 1.0 + (1 - ph) * V[lose_state]
        else:
            q = ph * V[win_state] + (1 - ph) * V[lose_state]

        q_values.append(q)

    q_values = np.array(q_values)
    best_q = np.max(q_values)

    # ---- KEY PART ----
    # Select ALL actions that are within tolerance of optimal value
    best_actions = [a for a, q in zip(actions, q_values)
                    if abs(q - best_q) < TOL]

    # Tie-breaking rule that reproduces Sutton figure:
    policy[s] = max(best_actions)

# ----------------------------
# Plot: Value Function
# ----------------------------
plt.figure(figsize=(10,4))
plt.plot(V)
plt.xlabel("Capital")
plt.ylabel("Probability of Winning")
plt.title("Optimal Value Function (Gambler's Problem)")
plt.grid(True)
plt.show()

# ----------------------------
# Plot: Policy
# ----------------------------
plt.figure(figsize=(10,4))
plt.bar(range(GOAL + 1), policy)
plt.xlabel("Capital")
plt.ylabel("Optimal Stake")
plt.title("Optimal Policy (Largest Optimal Stake Tie-Breaking)")
plt.grid(True)
plt.show()


