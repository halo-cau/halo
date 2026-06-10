"""Factory for the HALO 3D segmentation backend.

The HALO pipeline runs **mesh semantic segmentation** before voxelization so
movable objects (chairs, desks, boxes, legacy racks) are stripped from the
geometry. Only the immovable room shell and fixed infrastructure (server
racks, AC units, cable trays) survive into the voxelizer.

The backends, all returning a per-vertex labelled ``SegmentorResult``:

- ``geometric`` (default): a percentile-cuboid shell + DBSCAN interior
  clustering pipeline that needs no weights. Robust for the capstone demo
  while the Mask3D finetune is still being trained.
- ``mask3d``: the cvg/Mask3D model, finetuned (or hard-overfit) on the
  capstone room's labeled scans. Single 3D forward pass over the point
  cloud. Use when the checkpoint is healthy.
- ``dino_sam3``: open-vocabulary multi-view pipeline (GroundingDINO + SAM3).
  Renders the aligned mesh, runs 2D detection + segmentation per view, and
  backprojects masks to 3D vertex labels.
- ``sam3_concept``: SAM3 promptable-concept segmentation — prompts each target
  class name as text (no GroundingDINO), unions the instance masks, votes
  across views.
- ``dinov3``: DINOv3 patch features k-means clustered into *unnamed* groups,
  colored for eyeball inspection (tests feature separability with zero
  supervision; does not feed named labels to the voxel stamper).
- ``dinov3_sam3``: SAM3 seeds exemplar regions per class; DINOv3 features pooled
  inside them become prototypes; every patch is then labelled by
  nearest-prototype cosine similarity.

The SAM3 / DINOv3 backends fetch weights gated on Hugging Face — set
``HF_TOKEN`` (or run ``huggingface-cli login``) in the env before running.

Selection is controlled by the ``HALO_SEGMENTOR_BACKEND`` env var (one of
``geometric`` / ``mask3d`` / ``dino_sam3`` / ``sam3_concept`` / ``dinov3`` /
``dinov3_sam3`` / ``none``). ``none`` returns ``None`` so the pipeline voxelizes
the cleaned mesh directly — useful in CI or on hosts without ML runtimes.

Env vars per backend:
- ``HALO_MASK3D_CHECKPOINT`` — Mask3D .ckpt path
- ``HALO_DINO_CONFIG``      — GroundingDINO config .py path
- ``HALO_DINO_CHECKPOINT``  — GroundingDINO .pth path
- ``HALO_SAM3_CHECKPOINT``  — SAM3 .pt path (empty = download from HF)
- ``HALO_DINOV3_MODEL``     — DINOv3 HF model id (empty = library default)
- ``HF_TOKEN``              — Hugging Face token for the gated SAM3/DINOv3 repos

The ``DinoSam2`` variant still lives under ``engine.vision.segmentor_dino_sam``
for ad-hoc experiments via ``scripts/segment_scan.py`` but is not exposed
through this factory; the production hook is SAM3.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from engine.vision.segmentor_base import BaseSegmentor

logger = logging.getLogger(__name__)

_BACKEND_ENV = "HALO_SEGMENTOR_BACKEND"
_MASK3D_ENV = "HALO_MASK3D_CHECKPOINT"
_DINO_CONFIG_ENV = "HALO_DINO_CONFIG"
_DINO_CHECKPOINT_ENV = "HALO_DINO_CHECKPOINT"
_SAM3_CHECKPOINT_ENV = "HALO_SAM3_CHECKPOINT"
_DINOV3_MODEL_ENV = "HALO_DINOV3_MODEL"

_DEFAULT_BACKEND = "geometric"
_VALID_BACKENDS = (
    "geometric", "mask3d", "dino_sam3", "sam3_concept", "dinov3", "dinov3_sam3", "none",
)

# Bundled checkpoints shipped under opt/. Used when the env vars are unset.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MASK3D_CHECKPOINT = (
    _PROJECT_ROOT / "opt" / "Mask3D" / "checkpoints" / "scannet200" / "scannet200_benchmark.ckpt"
)
_DEFAULT_DINO_CONFIG = (
    _PROJECT_ROOT / "opt" / "checkpoints" / "groundingdino" / "GroundingDINO_SwinT_OGC.py"
)
_DEFAULT_DINO_CHECKPOINT = (
    _PROJECT_ROOT / "opt" / "checkpoints" / "groundingdino" / "groundingdino_swint_ogc.pth"
)
_DEFAULT_SAM3_CHECKPOINT = (
    _PROJECT_ROOT / "opt" / "checkpoints" / "sam3" / "sam3.pt"
)

# Module-level cache so weights are loaded once per process.
_cached: Optional[BaseSegmentor] = None
_cache_resolved: bool = False
_cache_backend: str | None = None


def _resolve_backend() -> str:
    raw = os.environ.get(_BACKEND_ENV, _DEFAULT_BACKEND).strip().lower()
    if raw not in _VALID_BACKENDS:
        logger.warning(
            "%s=%s is not one of %s; falling back to %s.",
            _BACKEND_ENV, raw, _VALID_BACKENDS, _DEFAULT_BACKEND,
        )
        return _DEFAULT_BACKEND
    return raw


def _resolve_mask3d_checkpoint() -> Path | None:
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


def _build_mask3d() -> BaseSegmentor | None:
    checkpoint = _resolve_mask3d_checkpoint()
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

        seg = Mask3DSegmentor(checkpoint_path=checkpoint)
        logger.info("Using Mask3D segmentor with checkpoint %s", checkpoint)
        return seg
    except Exception as exc:  # noqa: BLE001 — surface as warning, return None
        logger.warning("Failed to construct Mask3DSegmentor: %s", exc)
        return None


def _build_geometric() -> BaseSegmentor:
    from engine.vision.segmentor_geometric import GeometricSegmentor

    logger.info("Using geometric segmentor (percentile shell + DBSCAN clusters).")
    return GeometricSegmentor()


def _resolve_path(env_var: str, default: Path) -> Path | None:
    raw = os.environ.get(env_var)
    if raw:
        path = Path(raw)
        if path.exists():
            return path
        logger.warning("%s=%s does not exist; ignoring.", env_var, raw)
    return default if default.exists() else None


def _build_dino_sam3() -> BaseSegmentor | None:
    dino_config = _resolve_path(_DINO_CONFIG_ENV, _DEFAULT_DINO_CONFIG)
    dino_checkpoint = _resolve_path(_DINO_CHECKPOINT_ENV, _DEFAULT_DINO_CHECKPOINT)
    if dino_config is None or dino_checkpoint is None:
        logger.warning(
            "GroundingDINO config / checkpoint not found. Set %s and %s "
            "or stage files at %s and %s. Falling back to no segmentor.",
            _DINO_CONFIG_ENV, _DINO_CHECKPOINT_ENV,
            _DEFAULT_DINO_CONFIG, _DEFAULT_DINO_CHECKPOINT,
        )
        return None

    # SAM3 checkpoint is optional — the segmentor downloads from HF when None.
    sam3_path: Path | None = _resolve_path(_SAM3_CHECKPOINT_ENV, _DEFAULT_SAM3_CHECKPOINT)
    sam3_arg: str | Path | None = sam3_path  # None triggers HF download

    try:
        from engine.vision.segmentor_dino_sam import DinoSam3Segmentor

        seg = DinoSam3Segmentor(
            grounding_dino_config=str(dino_config),
            grounding_dino_checkpoint=str(dino_checkpoint),
            sam3_checkpoint=sam3_arg,
        )
        logger.info(
            "Using DINO+SAM3 segmentor (dino=%s, sam3=%s).",
            dino_checkpoint.name,
            sam3_path.name if sam3_path else "Hugging Face download",
        )
        return seg
    except Exception as exc:  # noqa: BLE001 — surface as warning, return None
        logger.warning("Failed to construct DinoSam3Segmentor: %s", exc)
        return None


def _resolve_dinov3_model() -> str | None:
    """DINOv3 HF model id from the env, or None to use the library default."""
    raw = os.environ.get(_DINOV3_MODEL_ENV)
    return raw.strip() if raw and raw.strip() else None


def _build_sam3_concept() -> BaseSegmentor | None:
    # None ⇒ download from Hugging Face (gated; needs HF_TOKEN).
    sam3_path = _resolve_path(_SAM3_CHECKPOINT_ENV, _DEFAULT_SAM3_CHECKPOINT)
    try:
        from engine.vision.segmentor_sam3_concept import Sam3ConceptSegmentor

        seg = Sam3ConceptSegmentor(sam3_checkpoint=sam3_path)
        logger.info(
            "Using SAM3 concept segmentor (sam3=%s).",
            sam3_path.name if sam3_path else "Hugging Face download",
        )
        return seg
    except Exception as exc:  # noqa: BLE001 — surface as warning, return None
        logger.warning("Failed to construct Sam3ConceptSegmentor: %s", exc)
        return None


def _build_dinov3() -> BaseSegmentor | None:
    model = _resolve_dinov3_model()
    try:
        from engine.vision.segmentor_dinov3 import Dinov3Segmentor

        seg = Dinov3Segmentor(model_name=model) if model else Dinov3Segmentor()
        logger.info(
            "Using DINOv3 clustering segmentor (unnamed groups, model=%s).",
            model or "library default",
        )
        return seg
    except Exception as exc:  # noqa: BLE001 — surface as warning, return None
        logger.warning("Failed to construct Dinov3Segmentor: %s", exc)
        return None


def _build_dinov3_sam3() -> BaseSegmentor | None:
    sam3_path = _resolve_path(_SAM3_CHECKPOINT_ENV, _DEFAULT_SAM3_CHECKPOINT)
    model = _resolve_dinov3_model()
    try:
        from engine.vision.segmentor_dinov3 import Dinov3Sam3Segmentor

        kwargs: dict = {"sam3_checkpoint": sam3_path}
        if model:
            kwargs["model_name"] = model
        seg = Dinov3Sam3Segmentor(**kwargs)
        logger.info(
            "Using DINOv3+SAM3 segmentor (sam3=%s, dinov3=%s).",
            sam3_path.name if sam3_path else "Hugging Face download",
            model or "library default",
        )
        return seg
    except Exception as exc:  # noqa: BLE001 — surface as warning, return None
        logger.warning("Failed to construct Dinov3Sam3Segmentor: %s", exc)
        return None


def get_default_segmentor(cached: bool = True) -> BaseSegmentor | None:
    """Return the configured segmentor, or ``None`` for the no-segmentor mode.

    Parameters
    ----------
    cached
        When True (default), reuse a previously constructed instance. Set
        False in tests or after rotating weight paths.
    """
    global _cached, _cache_resolved, _cache_backend

    backend = _resolve_backend()
    if cached and _cache_resolved and _cache_backend == backend:
        return _cached

    _cache_resolved = True
    _cache_backend = backend
    _cached = None

    if backend == "none":
        logger.info("%s=none: skipping segmentation.", _BACKEND_ENV)
        return None
    if backend == "mask3d":
        _cached = _build_mask3d()
        return _cached
    if backend == "geometric":
        _cached = _build_geometric()
        return _cached
    if backend == "dino_sam3":
        _cached = _build_dino_sam3()
        return _cached
    if backend == "sam3_concept":
        _cached = _build_sam3_concept()
        return _cached
    if backend == "dinov3":
        _cached = _build_dinov3()
        return _cached
    if backend == "dinov3_sam3":
        _cached = _build_dinov3_sam3()
        return _cached

    # Defensive: _resolve_backend() already normalises invalid values.
    return None
