from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.vec_env import VecNormalize, VecMonitor
from stable_baselines3.common.monitor import Monitor
from engine.rl.datacenter import DataCenterEnv
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from engine.rl.policy import SpatialMaskablePolicy
from datetime import datetime
import os
import gymnasium as gym
import numpy as np


class VecNormCheckPointCallback(BaseCallback):
    def __init__(self, save_freq, save_path, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            vec_env = self.model.get_vec_normalize_env()
            if vec_env is not None:
                vec_path = os.path.join(
                    self.save_path,
                    f"center_{self.num_timesteps}_steps_vecnormalize.pkl",
                )
                vec_env.save(vec_path)
                if self.verbose:
                    print(f"\n[VecNorm] Saved standard VecNormalize to: {vec_path}")
        return True


def mask_fn(env: gym.Env) -> np.ndarray:
    return env.action_masks()


def make_env(rank):
    def _init():
        env = DataCenterEnv(grid_size=50, rack_num=10, placement_mode="row")
        env.reset(seed=rank)
        env = ActionMasker(env, mask_fn)
        env = Monitor(env, filename=None)
        return env

    return _init


# if __name__ == "__main__":
def train():
    config = {
        "learning_rate": 5e-5,
        "n_steps": 1024,
        "n_epochs": 5,
        "batch_size": 256,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "target_kl": None,
        "num_envs": 16,
        "total_timesteps": 100_000_000,
        "self_play_update_freq": 400_000,
        "switch_freq": 100_000,
        "eval_freq": 2_000_000,
        "check_freq": 1_000_000,
    }

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"ppo_dc_{now}"

    log_dir = f"./ppo_logs/{run_name}"
    # os.makedirs(f"./ppo_models/{run_name}", exist_ok=True)
    os.makedirs("./ppo_logs", exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    wandb.init(
        project="datacenter_ppo",
        name=run_name,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
        config=config,
    )

    env = SubprocVecEnv([make_env(i) for i in range(config["num_envs"])])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    env = VecMonitor(env)
    # env = DummyVecEnv([make_env])

    checkpoint_callback = CheckpointCallback(
        save_freq=config["check_freq"] // config["num_envs"],
        save_path=log_dir,
        name_prefix="center",
    )
    vecnorm_callback = VecNormCheckPointCallback(
        save_freq=config["check_freq"] // config["num_envs"],
        save_path=log_dir,
        verbose=1,
    )

    # Spatial policy: resolution-preserving CNN + fully convolutional action head (engine/rl/policy.py).
    # It wires the feature extractor, the identity MLP extractor, and normalize_images itself, so it
    # needs no policy_kwargs.
    model = MaskablePPO(
        SpatialMaskablePolicy,
        env=env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=config["learning_rate"],
        # learning_rate=get_linear_fn(1e-5, 5e-6, 1.0),
        n_steps=config["n_steps"],
        n_epochs=config["n_epochs"],
        batch_size=config["batch_size"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_range=config["clip_range"],
        ent_coef=config["ent_coef"],
        target_kl=config["target_kl"],
    )

    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=[
            WandbCallback(
                gradient_save_freq=10_000, model_save_path=log_dir, verbose=2
            ),
            checkpoint_callback,
            vecnorm_callback,
        ],
    )

    model.save(f"{log_dir}/final_model")
    env.save(f"{log_dir}/final_vecnormalize.pkl")
    env.close()
