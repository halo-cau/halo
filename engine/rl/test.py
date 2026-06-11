from sb3_contrib import MaskablePPO
import matplotlib.pyplot as plt
import numpy as np
from engine.rl.datacenter import DataCenterEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def visualize(env, title="Data Center"):

    heat = env.temp
    rack = env.rack_map
    obstacle = env.obstacle
    cooling = env.cooling_pos

    plt.figure(figsize=(6, 6))

    # 🔥 heatmap
    plt.imshow(heat, cmap="hot", origin="lower")

    # 🔴 rack (파란색)
    ys, xs = np.where(rack == 1)
    plt.scatter(xs, ys, c="blue", label="Rack", s=40)

    # ❄️ cooling (초록색 X)
    cy = cooling[:, 0]
    cx = cooling[:, 1]
    plt.scatter(cx, cy, c="green", marker="x", s=100, label="Cooling")

    # ⛔ obstacle (노란색 점)
    oy, ox = np.where(obstacle == 1)
    plt.scatter(ox, oy, c="yellow", s=10, label="Obstacle")

    plt.colorbar(label="Heat")
    plt.legend(bbox_to_anchor=(1.25, 1), loc="upper left")
    plt.title(f"{title}: {env.step_count} steps")
    plt.tight_layout()
    plt.show()


def visualize_advanced(env, step, reward=None):

    heat = env.temp
    rack = env.rack_map
    obstacle = env.obstacle
    cooling = env.cooling_pos

    plt.figure(figsize=(7, 7))

    # 🔥 heatmap
    plt.imshow(heat, cmap="hot", origin="lower", alpha=0.85)

    # =========================
    # 🌬 airflow (핵심 수정 부분)
    # =========================
    skip = 3

    X = np.arange(0, env.grid_size, skip)
    Y = np.arange(0, env.grid_size, skip)
    XX, YY = np.meshgrid(X, Y)

    plt.quiver(
        XX,
        YY,
        cmap="cool",
        scale=25,
        width=0.003,
        alpha=0.9,
    )

    # =========================
    # 🔵 rack 위치
    # =========================
    ys, xs = np.where(rack == 1)
    plt.scatter(xs, ys, c="blue", s=50, label="Rack")

    # ➡️ rack 방향
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    for y, x in zip(ys, xs):
        d = dirs[int(env.rack_dir[y, x])]
        plt.arrow(x, y, d[1] * 0.8, d[0] * 0.8, color="cyan", head_width=0.5)

    # =========================
    # ❄️ cooling
    # =========================
    cy, cx = cooling[:, 0], cooling[:, 1]
    plt.scatter(cx, cy, c="green", marker="x", s=120, label="Cooling")

    # =========================
    # ⛔ obstacle
    # =========================
    oy, ox = np.where(obstacle == 1)
    plt.scatter(ox, oy, c="yellow", s=10, label="Obstacle")

    # =========================
    # 📊 정보 표시
    # =========================
    max_temp = heat.max()
    mean_temp = heat.mean()

    title = f"Step {step} | maxT={max_temp:.2f}, meanT={mean_temp:.2f}"
    if reward is not None:
        title += f", reward={reward:.2f}"

    plt.title(title)
    plt.legend(loc="upper right")
    plt.colorbar(label="Temperature")
    plt.tight_layout()
    plt.show()


def make_env():
    def _init():
        env = DataCenterEnv(grid_size=50, rack_num=10)
        return env

    return _init


PATH = "./ppo_logs/ppo_dc_20260518_151637/"
grid = np.zeros((50, 50))
grid[:40, :] = 1
grid[:, 10:] = 1
cooler = np.array([[45, 5], [45, 6], [46, 5], [46, 6]])
options = {
    "obstacle": grid,
    "rack_num": 4,
    "cooling_pos": cooler,
}


def test():
    env = DummyVecEnv([make_env()])
    env = VecNormalize.load(f"{PATH}center_3000000_steps_vecnormalize.pkl", env)
    env.training = False
    env.norm_reward = False

    model = MaskablePPO.load(f"{PATH}center_3000000_steps", env=env)
    obs, _ = env.envs[0].reset(options=None)
    obs = env.normalize_obs(obs)
    done = False
    step = 0
    while not done:
        action_masks = env.envs[0].action_masks()
        print("Action mask sum:", action_masks.sum())
        action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
        obs, reward, done, _, _ = env.envs[0].step(action)
        # done = done[0]
        step += 1
        if step == env.envs[0].rack_num:
            visualize_advanced(env.envs[0], step)
