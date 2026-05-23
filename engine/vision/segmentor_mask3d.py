"""Mask3D segmentation backend.

Uses the packaged cvg/Mask3D API (https://github.com/cvg/Mask3D) to run
3D instance segmentation directly on the aligned mesh's point cloud.
ScanNet-200 class labels are mapped to structural / non-structural categories
and the mesh is stripped accordingly.

Requirements
------------
- cvg/Mask3D installed in the active Python env::

      cd /path/to/cvg_mask3d_repo
      pip install .   # installs the `mask3d` package

- MinkowskiEngine compiled and installed (pre-compiled on research server).
- A ScanNet200 checkpoint (.ckpt).  The filename **must** start with
  ``scannet200`` so that ``get_model()`` auto-configures itself.
  e.g. ``scannet200_benchmark.ckpt``

Usage
-----
    from engine.vision.segmentor_mask3d import Mask3DSegmentor
    seg = Mask3DSegmentor(checkpoint_path="/path/to/scannet200_benchmark.ckpt")
    result = seg.run(mesh_aligned, source_path)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import open3d as o3d
import torch

from engine.vision.segmentor_base import (
    PRESERVED_LABELS,
    BaseSegmentor,
    SegmentorResult,
    label_to_color,
    strip_non_structural,
)
from engine.vision.graph_fusion import GraphCleanupConfig, apply_graph_cleanup
from engine.vision.structural_priors import (
    StructuralProtectionConfig,
    append_virtual_room_shell,
    apply_structural_protection,
    flatten_structural_labels_to_room_shell,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ScanNet-200 class index → label string
# The cvg fork's get_model() / map_output_to_pointcloud() returns integer
# labels that equal (model_class_index + 2) for non-background instances
# (background class 0 → label 1; classes 1..200 → labels 3..202).
# VALID_CLASS_IDS_200 positions (0-indexed) match the model class indices.
# Class names below follow the canonical ScanNet200 ordering.
# ---------------------------------------------------------------------------

# First 20 classes are ScanNet-20 (same order as nyu40 minus sky/person/...)
# Classes 21–199 extend to the full ScanNet-200 vocabulary.
_SCANNET200_CLASS_NAMES: list[str] = [
    # 0-indexed model class → human name
    "wall",              # 0
    "floor",             # 1
    "cabinet",           # 2
    "bed",               # 3
    "chair",             # 4
    "sofa",              # 5
    "table",             # 6
    "door",              # 7
    "window",            # 8
    "bookshelf",         # 9
    "picture",           # 10
    "counter",           # 11
    "blinds",            # 12
    "desk",              # 13
    "shelves",           # 14
    "curtain",           # 15
    "dresser",           # 16
    "pillow",            # 17
    "mirror",            # 18
    "floor mat",         # 19
    "clothes",           # 20
    "ceiling",           # 21
    "books",             # 22
    "fridge",            # 23
    "tv",                # 24
    "paper",             # 25
    "towel",             # 26
    "shower curtain",    # 27
    "box",               # 28
    "whiteboard",        # 29
    "person",            # 30
    "nightstand",        # 31
    "toilet",            # 32
    "sink",              # 33
    "lamp",              # 34
    "bathtub",           # 35
    "bag",               # 36
    "otherfurniture",    # 37
    # indices 38–199 are the ScanNet200 tail classes; anything beyond index 37
    # is an uncommonly-scanned object and treated as non-structural.
]

# label value returned by map_output_to_pointcloud → class name string
# label = class_index + 2  (background class 0 → label 1, then offset +2)
_LABEL_ID_TO_NAME: dict[int, str] = {
    1: "unknown",  # background
}
for _idx, _name in enumerate(_SCANNET200_CLASS_NAMES):
    _LABEL_ID_TO_NAME[_idx + 3] = _name   # class_index + 2 + 1 (0-index offset)


def _label_id_to_str(label_id: int) -> str:
    """Convert a map_output_to_pointcloud integer label to a category string."""
    return _LABEL_ID_TO_NAME.get(int(label_id), "otherfurniture")


# ---------------------------------------------------------------------------
# Mask3D segmentor
# ---------------------------------------------------------------------------

class Mask3DSegmentor(BaseSegmentor):
    """Run Mask3D (cvg fork) inference on an aligned Open3D mesh.

    Parameters
    ----------
    checkpoint_path:
        Absolute path to the ScanNet200 .ckpt file.  The filename must start
        with ``scannet200`` (e.g. ``scannet200_benchmark.ckpt``) so that
        ``get_model()`` auto-selects the correct Hydra config.
    device:
        Torch device string.  Defaults to ``"cuda"`` if available.
    confidence_threshold:
        Minimum per-instance confidence accepted from the model.
        Lower values = more predictions but noisier labels.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | None = None,
        confidence_threshold: float = 0.5,
        protection_config: StructuralProtectionConfig | None = None,
        graph_cleanup_config: GraphCleanupConfig | None = None,
    ) -> None:
        self.checkpoint_path      = Path(checkpoint_path)
        self.device               = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.confidence_threshold = confidence_threshold
        self.protection_config    = protection_config or StructuralProtectionConfig()
        self.graph_cleanup_config = graph_cleanup_config or GraphCleanupConfig()
        self._model               = None  # lazy-loaded

    @property
    def name(self) -> str:
        return "mask3d"

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from mask3d import get_model  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "The `mask3d` package is not installed.  Clone https://github.com/cvg/Mask3D "
                "into your environment and run `pip install .` from the repo root.  "
                "Original error: " + str(exc)
            ) from exc

        log.info("Loading Mask3D model from %s …", self.checkpoint_path)
        model = get_model(checkpoint_path=str(self.checkpoint_path))
        model.eval()
        model.to(self.device)
        self._model = model
        log.info("Mask3D model ready on %s", self.device)
        return model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        mesh: o3d.geometry.TriangleMesh,
        source_path: Path,
    ) -> SegmentorResult:
        """Segment *mesh* and return a SegmentorResult.

        The aligned Open3D mesh is passed through Mask3D's prepare_data(), and
        the per-vertex labels are backprojected from the instance predictions.
        """
        try:
            from mask3d import prepare_data, map_output_to_pointcloud  # type: ignore[import]
        except ImportError as exc:
            raise ImportError("mask3d package not installed: " + str(exc)) from exc

        model = self._load_model()
        mesh.compute_vertex_normals()
        n_total = len(np.asarray(mesh.vertices))

        log.info("Mask3D inference on %d vertices …", n_total)
        data, points, colors, features, unique_map, inverse_map = prepare_data(
            mesh, torch.device(self.device)
        )

        with torch.no_grad():
            outputs = model(data, raw_coordinates=features)

        # labels_mapped: (N, 1) integer array; label = class_index + 2, or 1 for background
        labels_mapped = map_output_to_pointcloud(mesh, outputs, inverse_map,
                                                  confidence_threshold=self.confidence_threshold)
        labels_mapped = np.asarray(labels_mapped).ravel()   # (N,)

        # Convert integer labels → category strings
        vertex_labels: list[str] = [_label_id_to_str(int(lid)) for lid in labels_mapped]
        protection = apply_structural_protection(
            mesh,
            vertex_labels,
            self.protection_config,
        )
        vertex_labels = protection.vertex_labels
        if protection.n_restored:
            log.info(
                "Mask3D: structural prior restored %d protected shell vertices",
                protection.n_restored,
            )

        graph_cleanup = apply_graph_cleanup(
            mesh,
            vertex_labels,
            protection.prior,
            self.graph_cleanup_config,
        )
        vertex_labels = graph_cleanup.vertex_labels
        if graph_cleanup.n_graph_restored or graph_cleanup.n_graph_pruned:
            log.info(
                "Mask3D: graph cleanup restored %d and pruned %d vertices",
                graph_cleanup.n_graph_restored,
                graph_cleanup.n_graph_pruned,
            )

        flattened_mesh, flatten_stats = flatten_structural_labels_to_room_shell(
            mesh,
            vertex_labels,
            protection.prior,
        )

        # Build outputs
        label_colors = np.array(
            [label_to_color(lbl) for lbl in vertex_labels], dtype=np.float32
        )
        label_map: dict[str, list[int]] = {}
        for vi, lbl in enumerate(vertex_labels):
            label_map.setdefault(lbl, []).append(vi)

        n_removed = sum(1 for lbl in vertex_labels if lbl not in PRESERVED_LABELS)
        log.info(
            "Mask3D: %d / %d vertices non-preserved (%.1f%%) — removing",
            n_removed, n_total, 100.0 * n_removed / max(1, n_total),
        )

        structural_mesh, virtual_shell_stats = append_virtual_room_shell(
            strip_non_structural(flattened_mesh, vertex_labels),
            protection.prior,
        )

        return SegmentorResult(
            structural_mesh=structural_mesh,
            vertex_labels=vertex_labels,
            label_colors=label_colors,
            label_map=label_map,
            n_removed=n_removed,
            backend=self.name,
            extra={
                "structural_protection": protection.to_json(),
                "graph_cleanup": graph_cleanup.to_json(),
                "structural_flattening": flatten_stats,
                "virtual_room_shell": virtual_shell_stats,
            },
        )

