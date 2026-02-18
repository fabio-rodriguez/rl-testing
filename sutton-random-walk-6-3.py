import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Environment definition
# -----------------------

N_STATES = 5
START_STATE = 2  # C (0=A,1=B,2=C,3=D,4=E)
TRUE_VALUES = np.arange(1, 6) / 6.0
GAMMA = 1.0


def step(state):
    """Take one random walk step."""
    if np.random.rand() < 0.5:
        next_state = state - 1
    else:
        next_state = state + 1

    if next_state == -1:
        return None, 0  # left terminal
    elif next_state == N_STATES:
        return None, 1  # right terminal reward
    else:
        return next_state, 0


def generate_episode():
    """Generate one episode."""
    states = []
    rewards = []

    state = START_STATE
    while state is not None:
        states.append(state)
        next_state, reward = step(state)
        rewards.append(reward)
        state = next_state

    return states, rewards


# -----------------------
# TD(0)
# -----------------------

def td0(alpha, episodes):
    V = np.ones(N_STATES) * 0.5
    errors = []

    for _ in range(episodes):
        state = START_STATE

        while state is not None:
            next_state, reward = step(state)

            target = reward
            if next_state is not None:
                target += GAMMA * V[next_state]

            V[state] += alpha * (target - V[state])
            state = next_state

        errors.append(rms_error(V))

    return errors


# -----------------------
# Monte Carlo (constant α)
# -----------------------

def monte_carlo(alpha, episodes):
    V = np.ones(N_STATES) * 0.5
    errors = []

    for _ in range(episodes):
        states, rewards = generate_episode()

        G = sum(rewards)  # undiscounted, only terminal reward

        for state in states:
            V[state] += alpha * (G - V[state])

        errors.append(rms_error(V))

    return errors


# -----------------------
# RMS error
# -----------------------

def rms_error(V):
    return np.sqrt(np.mean((V - TRUE_VALUES) ** 2))


# -----------------------
# Run experiment
# -----------------------

def run_experiment(episodes=100, runs=100):
    td_alphas = [0.05, 0.1, 0.15]
    mc_alphas = [0.01, 0.02, 0.03, 0.04]

    td_results = {a: np.zeros(episodes) for a in td_alphas}
    mc_results = {a: np.zeros(episodes) for a in mc_alphas}

    for _ in range(runs):
        for alpha in td_alphas:
            td_results[alpha] += np.array(td0(alpha, episodes))

        for alpha in mc_alphas:
            mc_results[alpha] += np.array(monte_carlo(alpha, episodes))

    for alpha in td_alphas:
        td_results[alpha] /= runs

    for alpha in mc_alphas:
        mc_results[alpha] /= runs

    return td_results, mc_results


# -----------------------
# Plot results
# -----------------------

def plot_results(td_results, mc_results):
    plt.figure(figsize=(8, 6))

    for alpha, errors in td_results.items():
        plt.plot(errors, label=f"TD α={alpha}")

    for alpha, errors in mc_results.items():
        plt.plot(errors, linestyle="--", label=f"MC α={alpha}")

    plt.xlabel("Episodes")
    plt.ylabel("RMS Error")
    plt.legend()
    plt.title("TD(0) vs Constant-α Monte Carlo")
    plt.show()


if __name__ == "__main__":
    td_results, mc_results = run_experiment()
    plot_results(td_results, mc_results)

