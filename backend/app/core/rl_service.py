import sys
import numpy as np
from engine.rl.datacenter import DataCenterEnv
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import threading


def make_env():
    return DataCenterEnv()


class RLService:
    def __init__(self):
        try:
            self.env = DummyVecEnv([make_env])
            self.env = VecNormalize.load("rl/vecnormalize.pkl", self.env)
            self.env.training = False
            self.env.norm_reward = False
        except Exception as e:
            print(f"Vecnormalize load failed: {e}")

        self.model = MaskablePPO.load("rl/model.zip")

        self.lock = threading.Lock()

    def optimize(self, obstacle, cooling_pos, rack_num):
        with self.lock:
            obs, _ = self.env.reset(
                options={
                    "obstacle": np.array(obstacle),
                    "cooling_pos": np.array(cooling_pos),
                    "rack_num": rack_num,
                }
            )

            done = False
            actions = []

            while not done:
                action_masks = self.env.action_masks()
                action, _ = self.model.predict(
                    obs, action_masks=action_masks, deterministic=True
                )
                obs, _, done, _, _ = self.env.step(action)
                actions.append(action[0])

            return self._decode(actions)

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
