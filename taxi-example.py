import gymnasium as gym
import numpy as np
import cv2
import time

# ---------------- SETUP ENV ----------------
env = gym.make("Taxi-v3", render_mode="rgb_array")  # render as images
nS = env.observation_space.n
nA = env.action_space.n
gamma = 0.9
theta = 1e-4

# ---------------- VALUE ITERATION ----------------
V = np.zeros(nS)

def one_step_lookahead(s, V):
    A = np.zeros(nA)
    for a in range(nA):
        for prob, next_state, reward, done in env.unwrapped.P[s][a]:
            A[a] += prob * (reward + gamma * V[next_state])
    return A

while True:
    delta = 0
    for s in range(nS):
        A = one_step_lookahead(s, V)
        best_value = np.max(A)
        delta = max(delta, abs(best_value - V[s]))
        V[s] = best_value
    if delta < theta:
        break

# ---------------- EXTRACT POLICY ----------------
policy = np.zeros(nS, dtype=int)
for s in range(nS):
    A = one_step_lookahead(s, V)
    policy[s] = np.argmax(A)

# ---------------- SIMULATE EPISODE AND RECORD FRAMES ----------------
state, _ = env.reset()
done = False
frames = []
step_count = 0
max_steps = 30

while not done and step_count < max_steps:
    action = policy[state]
    state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    frame = env.render()
    frames.append(frame)
    step_count += 1

env.close()

# ---------------- CREATE VIDEO ----------------
height, width, layers = frames[0].shape
video_filename = "taxi_simulation.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(video_filename, fourcc, 2, (width, height))  # 2 FPS

for frame in frames:
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # OpenCV uses BGR
    video.write(frame_bgr)

video.release()
print(f"Video saved as {video_filename}")
