import numpy as np
import heapq
import matplotlib.pyplot as plt

# =========================
# Environment
# =========================
class RodEnv:
    def __init__(self, size=10):
        self.size = size
        self.actions = [0, 1, 2, 3]  # forward, side, rotate L, rotate R
        self.nA = 4
        self.goal = (size-2, size-2)

        # Create obstacles (simple wall with gap)
        self.obstacles = set()
        for i in range(2, size-2):
            if i != size//2:  # leave a gap
                self.obstacles.add((size//2, i))

        self.reset()

    def reset(self):
        self.state = (1, 1, 0)  # x, y, theta
        return self.state

    def step(self, action):
        x, y, theta = self.state

        if action == 0:  # forward
            nx = x + int(round(np.cos(np.deg2rad(theta))))
            ny = y + int(round(np.sin(np.deg2rad(theta))))
            ntheta = theta

        elif action == 1:  # side
            nx = x + int(round(np.sin(np.deg2rad(theta))))
            ny = y - int(round(np.cos(np.deg2rad(theta))))
            ntheta = theta

        elif action == 2:  # rotate left
            nx, ny = x, y
            ntheta = (theta + 30) % 360

        else:  # rotate right
            nx, ny = x, y
            ntheta = (theta - 30) % 360

        # collision / bounds
        if (nx < 0 or nx >= self.size or
            ny < 0 or ny >= self.size or
            (nx, ny) in self.obstacles):
            nx, ny = x, y

        next_state = (nx, ny, ntheta)

        reward = -1
        done = (nx, ny) == self.goal

        if done:
            reward = 100

        self.state = next_state
        return next_state, reward, done


# =========================
# Prioritized Sweeping
# =========================
class PrioritizedSweeping:
    def __init__(self, env, alpha=0.1, gamma=0.95, theta=0.01, n=3):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.theta = theta
        self.n = n

        self.Q = {}
        self.model = {}
        self.predecessors = {}
        self.pqueue = []

    def get_Q(self, s, a):
        return self.Q.get((s, a), 0.0)

    def best_action(self, s):
        qs = [self.get_Q(s, a) for a in range(self.env.nA)]
        return int(np.argmax(qs))

    def update_predecessors(self, s_prev, a_prev, s):
        if s not in self.predecessors:
            self.predecessors[s] = set()
        self.predecessors[s].add((s_prev, a_prev))

    def train(self, episodes=60):
        for ep in range(episodes):
            if ep % 10 == 0:
                print(f"Episode {ep}")

            s = self.env.reset()
            done = False

            while not done:
                # epsilon-greedy
                if np.random.rand() < 0.1:
                    a = np.random.randint(self.env.nA)
                else:
                    a = self.best_action(s)

                s_next, r, done = self.env.step(a)

                # Model update
                self.model[(s, a)] = (r, s_next)
                self.update_predecessors(s, a, s_next)

                # Priority
                max_q = max([self.get_Q(s_next, a2) for a2 in range(self.env.nA)])
                P = abs(r + self.gamma * max_q - self.get_Q(s, a))

                if P > self.theta:
                    heapq.heappush(self.pqueue, (-P, s, a))

                # Planning updates
                for _ in range(self.n):
                    if not self.pqueue:
                        break

                    _, s_p, a_p = heapq.heappop(self.pqueue)
                    r_p, s_next_p = self.model[(s_p, a_p)]

                    max_q_p = max([self.get_Q(s_next_p, a2) for a2 in range(self.env.nA)])

                    self.Q[(s_p, a_p)] = self.get_Q(s_p, a_p) + self.alpha * (
                        r_p + self.gamma * max_q_p - self.get_Q(s_p, a_p)
                    )

                    # Predecessors
                    for s_prev, a_prev in self.predecessors.get(s_p, []):
                        r_prev, _ = self.model[(s_prev, a_prev)]

                        max_q_prev = max([self.get_Q(s_p, a2) for a2 in range(self.env.nA)])

                        P = abs(r_prev + self.gamma * max_q_prev - self.get_Q(s_prev, a_prev))

                        if P > self.theta:
                            heapq.heappush(self.pqueue, (-P, s_prev, a_prev))

                s = s_next


# =========================
# Rollout (extract path)
# =========================
def rollout(env, agent):
    s = env.reset()
    path = [s]

    for _ in range(200):
        a = agent.best_action(s)
        s, _, done = env.step(a)
        path.append(s)
        if done:
            break

    return path


# =========================
# Visualization (FINAL FIX)
# =========================
def plot_final(env, path):
    fig, ax = plt.subplots(figsize=(6,6))

    # obstacles
    for (x, y) in env.obstacles:
        ax.add_patch(plt.Rectangle((x-0.5, y-0.5), 1, 1, color='black'))

    # path
    xs = [s[0] for s in path]
    ys = [s[1] for s in path]
    ax.plot(xs, ys, linewidth=2)

    # rod orientation (like book figure)
    for (x, y, theta) in path:
        dx = 0.4 * np.cos(np.deg2rad(theta))
        dy = 0.4 * np.sin(np.deg2rad(theta))
        ax.plot([x-dx, x+dx], [y-dy, y+dy], linewidth=1)

    # start / goal
    sx, sy, _ = path[0]
    gx, gy, _ = path[-1]

    ax.scatter(sx, sy, s=100, marker='o')
    ax.text(sx, sy, "Start")

    ax.scatter(gx, gy, s=120, marker='*')
    ax.text(gx, gy, "Goal")

    ax.set_xlim(-1, env.size)
    ax.set_ylim(-1, env.size)
    ax.set_aspect('equal')
    ax.set_title("Prioritized Sweeping - Final Path")
    plt.grid(True)
    plt.show()


# =========================
# RUN
# =========================
env = RodEnv(size=10)
agent = PrioritizedSweeping(env, n=3)

agent.train(episodes=60)

path = rollout(env, agent)
plot_final(env, path)