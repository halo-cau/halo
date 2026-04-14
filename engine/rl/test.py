from sb3_contrib import MaskablePPO
import matplotlib.pyplot as plt
import numpy as np
from datacenter_env import DataCenterEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def visualize(env, title="Data Center"):

    heat = env.heat
    rack = env.rack_pos
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


def make_env():
    def _init():
        env = DataCenterEnv(grid_size=50, rack_num=10)
        return env

    return _init


PATH = "./ppo_logs/ppo_dc_20260408_190247/"

env = DummyVecEnv([make_env()])
env = VecNormalize.load(f"{PATH}center_96000000_steps_vecnormalize.pkl", env)
env.training = False
env.norm_reward = False

model = MaskablePPO.load(f"{PATH}center_96000000_steps", env=env)
obs = env.reset()
done = False
step = 0
while not done and step < 100:
    action_masks = env.envs[0].action_masks()
    print("Action mask sum:", action_masks.sum())
    action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
    obs, reward, done, _ = env.step(action)
    done = done[0]
    step += 1
    if step == 9:
        visualize(env.envs[0], title="Trained Policy Result")
