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

from engine.core.data_types import PipelineResult, ScanMetadata
from engine.vision.cleaner import clean_and_align_meshes
from engine.vision.segmentor_base import BaseSegmentor
from engine.vision.segmentor_factory import get_default_segmentor
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

    if segmentor is not None:
        try:
            result = segmentor.run(cleaned_mesh, scan_path)
            structural_mesh = result.structural_mesh
            logger.info(
                "Segmentation [%s]: removed %d/%d vertices (%.1f%%) — keeping "
                "structural + fixed infrastructure",
                result.backend,
                result.n_removed,
                result.n_total,
                100.0 * result.removal_fraction,
            )
        except Exception as exc:  # noqa: BLE001 — degrade rather than crash
            logger.warning(
                "Segmentation failed (%s); falling back to cleaned mesh.", exc,
            )
            structural_mesh = cleaned_mesh
    else:
        structural_mesh = cleaned_mesh

    grid, origin = voxelize_and_stamp_metadata(structural_mesh, metadata)
    padded_grid, grid_offset = pad_to_fixed_shape(grid)

    return PipelineResult(
        grid=grid,
        padded_grid=padded_grid,
        origin=origin,
        grid_offset=grid_offset,
    )
