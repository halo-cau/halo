"""Master CV orchestration: clean → segment mesh → voxelize → stamp → pad.

There are TWO distinct "labeling" steps in this pipeline. They operate on
different domains and have different sources, and the function names below
make that explicit:

    ┌──────────────────────────────┬───────────────┬────────────────────┐
    │ Stage                        │ Domain        │ Label source       │
    ├──────────────────────────────┼───────────────┼────────────────────┤
    │ 2 — segmentor.run(mesh)      │ mesh VERTICES │ Mask3D inference   │
    │ 3 — voxelize_and_stamp_meta… │ voxel GRID    │ user ScanMetadata  │
    └──────────────────────────────┴───────────────┴────────────────────┘

Full sequence:

1. **Clean and align** (``cleaner.clean_and_align_meshes``) — SOR denoise,
   RANSAC floor alignment, Manhattan rectification.
2. **Mesh semantic segmentation** (``segmentor.run``) — Mask3D assigns a
   class to each mesh VERTEX (wall / floor / ceiling / server_rack /
   box_clutter / ac_unit). Vertices labeled as movable clutter are removed
   from the mesh, leaving the immovable room shell plus fixed infrastructure.
   Skipped only if no segmentor is configured; a warning is logged.
3. **Voxelize + stamp metadata** (``voxelize_and_stamp_metadata``) — surface
   voxelize the segmented mesh, morphologically close wall holes, then stamp
   semantic IDs onto VOXELS using user-supplied ``ScanMetadata`` (AC vents,
   legacy heat sources, racks, workspaces). Voxel labels here come from the
   user's tags, not from any AI model.
4. **Pad** (``pad_to_fixed_shape``) — zero-pad to the RL observation shape.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import open3d as o3d

from engine.core.data_types import ComponentInstance, PipelineResult, ScanMetadata
from engine.vision.cleaner import clean_and_align_meshes
from engine.vision.segmentor_base import BaseSegmentor, SegmentorResult
from engine.vision.segmentor_factory import get_default_segmentor
from engine.vision.structural_priors import (
    append_virtual_room_shell,
    estimate_room_shell_prior,
    trim_outside_shell,
)
from engine.vision.voxelizer import pad_to_fixed_shape, voxelize_and_stamp_metadata

logger = logging.getLogger(__name__)

# Sentinel used to distinguish "caller didn't pass a segmentor — use the
# configured default" from "caller explicitly passed None — skip segmentation".
_USE_DEFAULT = object()


def run_pipeline(
    scan_path: Path,
    metadata: ScanMetadata,
    segmentor: BaseSegmentor | None = _USE_DEFAULT,  # type: ignore[assignment]
) -> PipelineResult:
    """Execute the full CV pipeline and return the padded voxel grid.

    Parameters
    ----------
    scan_path
        Filesystem path to the uploaded scan (.obj/.ply/.las/.laz).
    metadata
        User-supplied equipment annotations stamped after voxelization.
    segmentor
        Override the 3D segmentation backend. By default the factory-selected
        segmentor is used (or ``None`` if no weights are configured, in which
        case voxelization runs on the cleaned but unsegmented mesh).
    """
    if segmentor is _USE_DEFAULT:
        segmentor = get_default_segmentor()

    _, cleaned_mesh = clean_and_align_meshes(scan_path)

    seg_result: SegmentorResult | None = None
    if segmentor is not None:
        try:
            seg_result = segmentor.run(cleaned_mesh, scan_path)
            structural_mesh = seg_result.structural_mesh
            logger.info(
                "Segmentation [%s]: removed %d/%d vertices (%.1f%%) — keeping "
                "structural + fixed infrastructure",
                seg_result.backend,
                seg_result.n_removed,
                seg_result.n_total,
                100.0 * seg_result.removal_fraction,
            )
        except Exception as exc:  # noqa: BLE001 — degrade rather than crash
            logger.warning(
                "Segmentation failed (%s); falling back to cleaned mesh.", exc,
            )
            structural_mesh = cleaned_mesh
            seg_result = None
    else:
        structural_mesh = cleaned_mesh

    # Estimate the room cuboid from the structural mesh (ceiling-projection
    # rectangle when ceiling vertices are detectable, percentile fallback
    # otherwise), trim vertices outside that cuboid so aisle / window-side
    # outliers don't bleed into the voxel grid, then append the explicit
    # 8-vertex / 12-triangle envelope so the voxelizer sees a closed shell.
    shell_prior = estimate_room_shell_prior(structural_mesh)
    structural_mesh, trim_stats = trim_outside_shell(structural_mesh, shell_prior)
    if trim_stats.get("n_dropped", 0):
        logger.info(
            "Shell trim: dropped %d/%d vertices outside cuboid (margin=%.2f m)",
            trim_stats["n_dropped"], trim_stats["n_in"], trim_stats["margin_m"],
        )
    if shell_prior.extra.get("bounds", {}).get("height_disagreement_flagged"):
        logger.warning(
            "Shell prior: ceiling-derived floor disagrees with domain-prior "
            "fallback by %.2f m — review the scene height.",
            shell_prior.extra["bounds"]["height_disagreement_m"],
        )
    structural_mesh, _ = append_virtual_room_shell(structural_mesh, shell_prior)

    # Project per-vertex segmentor labels (server rack, ac_unit, …) onto the
    # voxel grid so detected infrastructure shows up with the correct voxel
    # ids instead of collapsing into OBSTACLE_WALL. Skip when the segmentor
    # produced no labels (None backend or fallback path).
    segmentor_labels: tuple[np.ndarray, list[str]] | None = None
    if seg_result is not None and seg_result.vertex_labels:
        verts = np.asarray(cleaned_mesh.vertices, dtype=np.float64)
        n = min(len(verts), len(seg_result.vertex_labels))
        segmentor_labels = (verts[:n], list(seg_result.vertex_labels[:n]))

    grid, layout_grid, origin = voxelize_and_stamp_metadata(
        structural_mesh, metadata, segmentor_labels=segmentor_labels,
    )
    padded_grid, grid_offset = pad_to_fixed_shape(grid)
    padded_layout_grid, _ = pad_to_fixed_shape(layout_grid)

    components = _components_from_segmentor(seg_result)
    backend_name = seg_result.backend if seg_result is not None else None

    return PipelineResult(
        grid=grid,
        padded_grid=padded_grid,
        layout_grid=layout_grid,
        padded_layout_grid=padded_layout_grid,
        origin=origin,
        grid_offset=grid_offset,
        components=components,
        backend=backend_name,
    )


def _components_from_segmentor(
    result: SegmentorResult | None,
) -> tuple[ComponentInstance, ...]:
    """Lift the segmentor's component dicts into typed instances.

    The geometric segmentor stores its components under ``extra["components"]``
    as a list of dicts with ``id``, ``label``, ``center``, ``bounds_min``,
    ``bounds_max``, ``n_points``. Backends without a component list return
    nothing here, which is fine: the frontend can still render the voxel
    grid; per-instance edit affordances are simply unavailable.
    """
    if result is None:
        return ()
    raw = result.extra.get("components") if isinstance(result.extra, dict) else None
    if not raw:
        return ()
    out: list[ComponentInstance] = []
    for c in raw:
        try:
            out.append(
                ComponentInstance(
                    id=int(c["id"]),
                    label=str(c["label"]),
                    center=tuple(float(v) for v in c["center"]),
                    bounds_min=tuple(float(v) for v in c["bounds_min"]),
                    bounds_max=tuple(float(v) for v in c["bounds_max"]),
                    n_points=int(c.get("n_points", 0)),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return tuple(out)
