"""Factory for the HALO 3D segmentation backend.

The HALO pipeline runs **mesh semantic segmentation** before voxelization so
movable objects (chairs, desks, boxes, legacy racks) are stripped from the
geometry. Only the immovable room shell and fixed infrastructure (server
racks, AC units, cable trays) survive into the voxelizer.

This project intentionally commits to a single backend: **Mask3D**.

Rationale: HALO is a capstone with a fixed demo room. The Mask3D model is
finetuned (or hard-overfit) to that room's labeled scans, which gives much
more reliable per-vertex labels than zero-shot open-vocab pipelines and runs
in a single 3D forward pass. The DINO + SAM rendering pipeline still exists
under ``engine.vision.segmentor_dino_sam`` for ad-hoc experiments via
``scripts/segment_scan.py``, but it is no longer wired into the live API.

Checkpoint resolution order:

1. The environment variable ``HALO_MASK3D_CHECKPOINT`` (overrides everything).
2. The bundled ScanNet200 checkpoint at
   ``opt/Mask3D/checkpoints/scannet200/scannet200_benchmark.ckpt``.
3. ``None``: factory returns ``None`` and logs a warning. Callers fall back
   to voxelizing the cleaned mesh as-is so the API stays functional in CI
   or on hosts without ML runtimes.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from engine.vision.segmentor_base import BaseSegmentor

logger = logging.getLogger(__name__)

_MASK3D_ENV = "HALO_MASK3D_CHECKPOINT"

# Bundled checkpoint shipped under opt/. Used when the env var is not set.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MASK3D_CHECKPOINT = (
    _PROJECT_ROOT / "opt" / "Mask3D" / "checkpoints" / "scannet200" / "scannet200_benchmark.ckpt"
)

# Module-level cache so weights are loaded once per process.
_cached: Optional[BaseSegmentor] = None
_cache_resolved: bool = False


def _resolve_checkpoint() -> Path | None:
    """Pick the Mask3D checkpoint to load, or ``None`` if none is available."""
    env_value = os.environ.get(_MASK3D_ENV)
    if env_value:
        env_path = Path(env_value)
        if env_path.exists():
            return env_path
        logger.warning("%s=%s does not exist; ignoring.", _MASK3D_ENV, env_value)

    if _DEFAULT_MASK3D_CHECKPOINT.exists():
        return _DEFAULT_MASK3D_CHECKPOINT

    return None


def get_default_segmentor(cached: bool = True) -> BaseSegmentor | None:
    """Return the configured Mask3D segmentor, or ``None`` if unavailable.

    Parameters
    ----------
    cached
        When True (default), reuse a previously constructed instance. Set
        False in tests or after rotating weight paths.
    """
    global _cached, _cache_resolved
    if cached and _cache_resolved:
        return _cached

    _cache_resolved = True
    _cached = None

    checkpoint = _resolve_checkpoint()
    if checkpoint is None:
        logger.warning(
            "No Mask3D checkpoint found. Set %s or place a checkpoint at %s. "
            "The CV pipeline will voxelize the cleaned mesh without removing "
            "movable objects.",
            _MASK3D_ENV,
            _DEFAULT_MASK3D_CHECKPOINT,
        )
        return None

    try:
        from engine.vision.segmentor_mask3d import Mask3DSegmentor

        _cached = Mask3DSegmentor(checkpoint_path=checkpoint)
        logger.info("Using Mask3D segmentor with checkpoint %s", checkpoint)
        return _cached
    except Exception as exc:  # noqa: BLE001 — surface as warning, return None
        logger.warning("Failed to construct Mask3DSegmentor: %s", exc)
        return None
