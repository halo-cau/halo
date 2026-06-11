import os
import sys
import numpy as np
from engine.rl.datacenter import DataCenterEnv
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import threading
from backend.app.schemas.datacenter import OptimizeResponse

# Placement mode for the SERVING env. The shipped model.zip was trained with one-rack-per-step ("single"),
# so that is the default to keep its output meaningful. After training a row-placement policy
# (engine/rl/train.py uses placement_mode="row"), set HALO_RL_PLACEMENT_MODE=row so serving matches it.
PLACEMENT_MODE = os.environ.get("HALO_RL_PLACEMENT_MODE", "single").strip() or "single"


def make_env():
    return DataCenterEnv(placement_mode=PLACEMENT_MODE)


class RLService:
    def __init__(self):
        try:
            self.env = DummyVecEnv([make_env])
            self.env = VecNormalize.load("engine/rl/vecnormalize.pkl", self.env)
            self.env.training = False
            self.env.norm_reward = False
            self.env.norm_obs = True
        except Exception as e:
            print(f"Vecnormalize load failed: {e}")

        self.model = MaskablePPO.load("engine/rl/model.zip")

        self.lock = threading.Lock()

    def optimize(self, obstacle, cooling_pos, rack_num, ceiling_m=None):
        with self.lock:
            raw_env = self.env.unwrapped.envs[0]

            # 1. 원본 환경에 변수 설정 및 초기화. ceiling_m (the scanned room height from the CV->RL
            #    twin_bridge converter) reaches the 3-D thermal solve via reset; None keeps the env default.
            opts = {
                "obstacle": np.array(obstacle),
                "cooling_pos": np.array(cooling_pos),
                "rack_num": rack_num,
            }
            if ceiling_m is not None:
                opts["ceiling_m"] = float(ceiling_m)
            obs = raw_env.reset(options=opts)
            if isinstance(obs, tuple):
                obs = obs[0]

            obs = self.env.normalize_obs(np.expand_dims(obs, axis=0))

            done = False
            actions = []
            info = {}

            while not done:
                action_masks = raw_env.action_masks()

                action, _ = self.model.predict(
                    obs, action_masks=action_masks, deterministic=True
                )

                obs, _, terminated, truncated, info = raw_env.step(action[0])

                obs = self.env.normalize_obs(np.expand_dims(obs, axis=0))
                done = terminated or truncated
                actions.append(int(action[0]))

            total_energy = info.get("total_energy", 0.0)
            max_temp = info.get("temp", [])

            return {
                "total_energy": float(total_energy),
                "max_temp": max_temp,
                # Decode from the env's final rack_map/rack_dir, not the action list: one action may place a
                # whole row, so the placed racks are the ground truth regardless of placement mode.
                "data": self._decode_from_grid(raw_env),
            }

    def _decode_from_grid(self, env):
        """Read every placed rack from the environment's rack_map / rack_dir (mode-agnostic)."""
        result = []
        for gx, gy in np.argwhere(env.rack_map == 1):
            result.append({"x": int(gx), "y": int(gy), "dir": int(env.rack_dir[gx, gy])})
        return result

    def _decode(self, actions):
        result = []
        g = (
            self.env.envs[0].grid_size
            if hasattr(self.env, "envs")
            else self.env.grid_size
        )

        for a in actions:
            pos = a // 4
            d = a % 4

            x = pos // g
            y = pos % g

            result.append({"x": int(x), "y": int(y), "dir": int(d)})

        return result


rl_service = RLService()
