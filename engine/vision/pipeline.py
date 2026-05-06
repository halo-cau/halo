"""Master CV orchestration: clean → voxelize → label → pad."""

import os
from pathlib import Path

import numpy as np

from engine.core.data_types import PipelineResult, ScanMetadata
from engine.vision.cleaner import clean_and_align
from engine.vision.voxelizer import pad_to_fixed_shape, voxelize_and_label


def run_pipeline(obj_path: Path, metadata: ScanMetadata) -> PipelineResult:
    """Execute the full CV pipeline.

    Returns a PipelineResult containing both the room-sized grid
    and the fixed-shape padded grid for the RL environment.
    Intermediate temp files (cleaned PLY) are always deleted.
    """
    ply_path: Path | None = None
    try:
        ply_path = clean_and_align(obj_path)
        grid, origin = voxelize_and_label(ply_path, metadata)
        padded_grid, grid_offset = pad_to_fixed_shape(grid)
        return PipelineResult(
            grid=grid,
            padded_grid=padded_grid,
            origin=origin,
            grid_offset=grid_offset,
        )
    finally:
        if ply_path is not None and ply_path.exists():
            os.unlink(ply_path)
