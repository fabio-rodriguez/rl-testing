import numpy as np
import matplotlib.pyplot as plt

# MDP parameters
states = ['high', 'low']
actions = {
    'high': ['search', 'wait'],
    'low': ['search', 'wait', 'recharge']
}

# Parameters
alpha = 0.8  # prob high->high when searching
beta = 0.6   # prob low->low when searching
rs = 2       # reward for collecting a can while searching
rw = 1       # reward for waiting
rescue_penalty = -3
gamma = 0.9  # discount factor

# Transition function and rewards
def transitions(state, action):
    if state == 'high':
        if action == 'search':
            return [('high', alpha, rs), ('low', 1-alpha, rs)]
        elif action == 'wait':
            return [('high', 1.0, rw)]
    elif state == 'low':
        if action == 'search':
            return [('low', beta, rs), ('high', 1-beta, rescue_penalty)]
        elif action == 'wait':
            return [('low', 1.0, rw)]
        elif action == 'recharge':
            return [('high', 1.0, 0)]
    return []

# Value iteration
def value_iteration(states, actions, gamma=0.9, epsilon=1e-5):
    V = {s: 0 for s in states}
    policy = {s: None for s in states}
    iteration = 0
    
    while True:
        delta = 0
        for s in states:
            action_values = {}
            for a in actions[s]:
                total = 0
                for next_state, prob, reward in transitions(s, a):
                    total += prob * (reward + gamma * V[next_state])
                action_values[a] = total
            best_action = max(action_values, key=action_values.get)
            best_value = action_values[best_action]
            delta = max(delta, abs(best_value - V[s]))
            V[s] = best_value
            policy[s] = best_action
        iteration += 1
        if delta < epsilon:
            break
    return V, policy, iteration

# Solve MDP
V_opt, policy_opt, iters = value_iteration(states, actions, gamma)

print("Optimal Value Function:", V_opt)
print("Optimal Policy:", policy_opt)
print("Converged in iterations:", iters)

import networkx as nx

G = nx.DiGraph()

# Add state-action edges
for s in states:
    for a in actions[s]:
        for next_state, prob, reward in transitions(s, a):
            G.add_edge(f"{s}", f"{next_state}", label=f"{a}\nP={prob}, R={reward}")

pos = nx.spring_layout(G)
plt.figure(figsize=(8,6))
nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightblue', font_size=12)
edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
plt.title("Recycling Robot MDP Transition Graph")
plt.show()
