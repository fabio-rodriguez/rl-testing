import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random

# ------------------------------------------------------------
# Blackjack Environment (infinite deck)
# ------------------------------------------------------------

def draw_card():
    card = np.random.randint(1, 14)
    return min(card, 10)

def draw_hand():
    return [draw_card(), draw_card()]

def usable_ace(hand):
    return 1 in hand and sum(hand) + 10 <= 21

def hand_value(hand):
    s = sum(hand)
    if usable_ace(hand):
        return s + 10
    return s

def is_bust(hand):
    return hand_value(hand) > 21

def dealer_policy(hand):
    while hand_value(hand) < 17:
        hand.append(draw_card())
    return hand

# ------------------------------------------------------------
# Generate episode with Exploring Starts
# ------------------------------------------------------------

def generate_episode(Q, policy):
    # Exploring start
    player_sum = random.randint(12, 21)
    dealer_show = random.randint(1, 10)
    usable = random.choice([True, False])
    action = random.choice([0, 1])  # 0=stick, 1=hit

    episode = []
    state = (player_sum, dealer_show, usable)

    # Construct player's hand consistent with state
    player = []
    if usable:
        player = [1, player_sum - 11]
    else:
        player = [player_sum - 10, 10]

    dealer = [dealer_show, draw_card()]

    while True:
        episode.append((state, action))

        if action == 1:  # hit
            player.append(draw_card())
            if is_bust(player):
                return episode, -1
        else:
            break

        state = (hand_value(player), dealer_show, usable_ace(player))
        action = policy[state]

    # Dealer's turn
    dealer = dealer_policy(dealer)

    if is_bust(dealer):
        return episode, 1

    player_score = hand_value(player)
    dealer_score = hand_value(dealer)

    if player_score > dealer_score:
        reward = 1
    elif player_score < dealer_score:
        reward = -1
    else:
        reward = 0

    return episode, reward

# ------------------------------------------------------------
# Monte Carlo ES
# ------------------------------------------------------------

def monte_carlo_es(episodes=500000):
    Q = defaultdict(lambda: np.zeros(2))
    returns = defaultdict(list)

    # initial policy: stick on 20 or 21
    policy = defaultdict(int)
    for ps in range(12, 22):
        for ds in range(1, 11):
            for ua in [True, False]:
                policy[(ps, ds, ua)] = 0 if ps >= 20 else 1

    for i in range(episodes):
        episode, reward = generate_episode(Q, policy)

        visited = set()
        for state, action in episode:
            if (state, action) not in visited:
                returns[(state, action)].append(reward)
                Q[state][action] = np.mean(returns[(state, action)])
                policy[state] = np.argmax(Q[state])
                visited.add((state, action))

        if i % 100000 == 0:
            print(f"Episode {i}")

    return Q, policy

# ------------------------------------------------------------
# Extract Value Function
# ------------------------------------------------------------

def compute_value_function(Q):
    V = {}
    for state, actions in Q.items():
        V[state] = np.max(actions)
    return V

# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------

def plot_value(V, usable):
    x = np.arange(1, 11)
    y = np.arange(12, 22)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X, dtype=float)

    for i, ps in enumerate(y):
        for j, ds in enumerate(x):
            Z[i, j] = V.get((ps, ds, usable), 0)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(X, Y, Z)
    ax.set_xlabel('Dealer Showing')
    ax.set_ylabel('Player Sum')
    ax.set_zlabel('Value')
    ax.set_title(f'Value Function (Usable Ace={usable})')
    plt.show()

def plot_policy(policy, usable):
    x = np.arange(1, 11)
    y = np.arange(12, 22)
    Z = np.zeros((len(y), len(x)))

    for i, ps in enumerate(y):
        for j, ds in enumerate(x):
            Z[i, j] = policy.get((ps, ds, usable), 0)

    plt.imshow(Z, origin='lower', extent=[1,10,12,21], aspect='auto')
    plt.colorbar(label="0=Stick, 1=Hit")
    plt.xlabel("Dealer Showing")
    plt.ylabel("Player Sum")
    plt.title(f'Optimal Policy (Usable Ace={usable})')
    plt.show()

# ------------------------------------------------------------
# Run Everything
# ------------------------------------------------------------

if __name__ == "__main__":
    Q, policy = monte_carlo_es(episodes=500000)
    V = compute_value_function(Q)

    plot_value(V, usable=True)
    plot_value(V, usable=False)

    plot_policy(policy, usable=True)
    plot_policy(policy, usable=False)
