"""RL policy service — lazy-loads the MaskablePPO checkpoint on first use.

The model is intentionally *not* loaded at module import time so the backend
can boot without the checkpoint files present (e.g., before training has
completed). The first call to ``optimize`` triggers loading; if the
checkpoint is missing, ``ModelNotAvailableError`` is raised and the API
endpoint translates that to a clean HTTP 503.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from app.core.exceptions import ModelNotAvailableError
from engine.rl.datacenter import DataCenterEnv

# Resolve checkpoint paths relative to the repo root so the backend can run
# from any working directory. Repo root = parents[3] of this file:
#   backend/app/core/rl_service.py
#     parents[0] = core/
#     parents[1] = app/
#     parents[2] = backend/
#     parents[3] = repo root
_REPO_ROOT = Path(__file__).resolve().parents[3]
_MODEL_PATH = _REPO_ROOT / "engine" / "rl" / "model.zip"
_VECNORM_PATH = _REPO_ROOT / "engine" / "rl" / "vecnormalize.pkl"


def _make_env() -> DataCenterEnv:
    return DataCenterEnv()


class RLService:
    """Thread-safe wrapper around a trained MaskablePPO policy.

    Construction is cheap — heavyweight model loading is deferred to the
    first ``optimize`` call. Subsequent calls reuse the loaded model.
    """

    def __init__(self) -> None:
        self._model = None
        self._env = None
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            if not _MODEL_PATH.exists() or not _VECNORM_PATH.exists():
                missing = [
                    str(p) for p in (_MODEL_PATH, _VECNORM_PATH) if not p.exists()
                ]
                raise ModelNotAvailableError(
                    f"RL model checkpoint not deployed (missing: {', '.join(missing)})"
                )

            # Import SB3 lazily — it's a heavy dependency and we only need it
            # once the checkpoint is actually being loaded.
            from sb3_contrib import MaskablePPO
            from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

            env = DummyVecEnv([_make_env])
            env = VecNormalize.load(str(_VECNORM_PATH), env)
            env.training = False
            env.norm_reward = False
            self._env = env
            self._model = MaskablePPO.load(str(_MODEL_PATH))

    def optimize(
        self,
        obstacle: np.ndarray | list,
        cooling_pos: np.ndarray | list,
        rack_num: int,
    ) -> list[dict]:
        """Run a full placement episode and return decoded actions.

        Returns a list of ``{"x": int, "y": int, "dir": int}`` dicts, one per
        placed rack. ``dir`` is the RL exhaust direction (0=+x, 1=-x, 2=+y,
        3=-y); use ``engine.rl.thermal_bridge._DIR_TO_FACING`` to convert to
        a solver ``RackFacing`` if needed.
        """
        self._ensure_loaded()

        with self._lock:
            obs, _ = self._env.reset(
                options={
                    "obstacle": np.asarray(obstacle),
                    "cooling_pos": np.asarray(cooling_pos),
                    "rack_num": int(rack_num),
                }
            )

            done = False
            actions: list[int] = []
            while not done:
                # MaskablePPO at inference time still needs the action mask —
                # without it the policy can pick infeasible cells. Mirrors the
                # pattern used in engine/rl/test.py.
                action_masks = self._env.envs[0].action_masks()
                action, _ = self._model.predict(
                    obs, action_masks=action_masks, deterministic=True,
                )
                obs, _, done, _, _ = self._env.step(action)
                # action comes back as a (1,) array from the VecEnv wrap.
                actions.append(int(action[0]))

            return self._decode(actions)

    def _decode(self, actions: list[int]) -> list[dict]:
        """Reverse the env's action encoding: action = (gx * grid + gy) * 4 + dir."""
        g = (
            self._env.envs[0].grid_size
            if hasattr(self._env, "envs")
            else self._env.grid_size
        )
        result: list[dict] = []
        for a in actions:
            pos = a // 4
            d = a % 4
            x = pos // g
            y = pos % g
            result.append({"x": int(x), "y": int(y), "dir": int(d)})
        return result


# Module-level singleton — construction is now IO-free so this is safe at
# import time. The checkpoint is loaded lazily on first .optimize() call.
rl_service = RLService()
