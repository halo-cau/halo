"""Master CV orchestration: clean → voxelize → label."""

import os
from pathlib import Path

import numpy as np

from engine.core.data_types import ScanMetadata
from engine.vision.cleaner import clean_and_align
from engine.vision.voxelizer import voxelize_and_label


def run_pipeline(obj_path: Path, metadata: ScanMetadata) -> np.ndarray:
    """Execute the full CV pipeline and return the semantic int8 grid.

    Intermediate temp files (cleaned PLY) are always deleted.
    """
    ply_path: Path | None = None
    try:
        ply_path = clean_and_align(obj_path)
        grid = voxelize_and_label(ply_path, metadata)
        return grid
    finally:
        if ply_path is not None and ply_path.exists():
            os.unlink(ply_path)
