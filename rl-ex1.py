import numpy as np
import matplotlib.pyplot as plt

# ---------------- GRIDWORLD SETUP ----------------
rows, cols = 3, 4
terminals = {(0, 3): 1, (1, 3): -1}
wall = (1, 1)
gamma = 0.9
actions = [(-1,0),(1,0),(0,-1),(0,1)]  # up, down, left, right
arrow = {(-1,0):"↑", (1,0):"↓", (0,-1):"←", (0,1):"→"}

def step(state, action):
    r, c = state
    dr, dc = action
    nr, nc = r + dr, c + dc
    # Check boundaries
    if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) != wall:
        return (nr, nc), 0
    else:
        return (r, c), 0  # blocked, stay in same cell

# ---------------- VALUE ITERATION ----------------
V = np.zeros((rows, cols))
for _ in range(100):
    new_V = np.copy(V)
    for r in range(rows):
        for c in range(cols):
            if (r,c) in terminals or (r,c) == wall:
                continue
            values = []
            for a in actions:
                (nr,nc), reward = step((r,c), a)
                # Check if moving into terminal
                if (nr,nc) in terminals:
                    reward = terminals[(nr,nc)]
                values.append(reward + gamma*V[nr,nc])
            new_V[r,c] = max(values)
    V = new_V

# ---------------- EXTRACT GREEDY POLICY ----------------
policy = {}
for r in range(rows):
    for c in range(cols):
        if (r,c) in terminals or (r,c) == wall:
            continue
        best_value = -np.inf
        best_action = None
        for a in actions:
            (nr,nc), reward = step((r,c), a)
            if (nr,nc) in terminals:
                reward = terminals[(nr,nc)]
            val = reward + gamma*V[nr,nc]
            if val > best_value:
                best_value = val
                best_action = a
        policy[(r,c)] = best_action

# ---------------- VISUALIZATION ----------------
fig, ax = plt.subplots(figsize=(6,5))
im = ax.imshow(V, cmap="coolwarm", origin="upper", vmin=np.min(V), vmax=np.max(V))

for r in range(rows):
    for c in range(cols):
        if (r,c) == wall:
            ax.text(c,r,"■",ha='center',va='center',fontsize=20,color='black')
        elif (r,c) in terminals:
            ax.text(c,r,f"{terminals[(r,c)]:+d}",ha='center',va='center',fontsize=16,color='black')
        else:
            ax.text(c,r,arrow[policy[(r,c)]],ha='center',va='center',fontsize=20,color='black')

ax.set_xticks(np.arange(cols))
ax.set_yticks(np.arange(rows))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_title("Gridworld: Optimal Policy & Value Function")
plt.colorbar(im, ax=ax, label="Value")
plt.show()

# ---------------- PRINT RESULTS ----------------
print("Optimal Value Function:")
print(np.round(V,2))
print("\nOptimal Policy (as arrows):")
for r in range(rows):
    row_str = ""
    for c in range(cols):
        if (r,c) in terminals:
            row_str += f" {terminals[(r,c)]:+d} "
        elif (r,c) == wall:
            row_str += " ■ "
        else:
            row_str += " " + arrow[policy[(r,c)]] + " "
    print(row_str)
