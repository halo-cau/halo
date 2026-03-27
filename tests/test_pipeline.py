"""Integration tests for the full engine pipeline (clean → voxelize → label)."""

from pathlib import Path

import numpy as np
import pytest

from engine.core.config import (
    OBSTACLE_WALL,
    SPACE_EMPTY,
)
from engine.core.data_types import Coordinate, ScanMetadata
from engine.core.exceptions import MeshProcessingError
from engine.vision.pipeline import run_pipeline


class TestRunPipeline:
    """End-to-end tests for the CV pipeline orchestrator."""

    def test_produces_valid_grid(self, sample_obj_path: Path) -> None:
        """Pipeline should return a 3D int8 grid with expected labels."""
        meta = ScanMetadata(
            ac_vents=[Coordinate(1.0, 1.0, 1.0)],
            legacy_servers=[Coordinate(1.5, 1.0, 0.5)],
            human_workspaces=[Coordinate(2.0, 1.5, 0.1)],
        )
        grid = run_pipeline(sample_obj_path, meta)

        assert isinstance(grid, np.ndarray)
        assert grid.dtype == np.int8
        assert grid.ndim == 3
        assert SPACE_EMPTY in np.unique(grid)
        assert OBSTACLE_WALL in np.unique(grid)

    def test_empty_metadata_ok(self, sample_obj_path: Path) -> None:
        """Pipeline should work with no semantic annotations."""
        grid = run_pipeline(sample_obj_path, ScanMetadata())

        assert grid.dtype == np.int8
        unique = set(np.unique(grid))
        # Only empty and wall when no metadata
        assert unique <= {SPACE_EMPTY, OBSTACLE_WALL}

    def test_empty_mesh_raises(self, empty_obj_path: Path) -> None:
        """An empty .obj must raise MeshProcessingError."""
        with pytest.raises(MeshProcessingError):
            run_pipeline(empty_obj_path, ScanMetadata())

    def test_temp_files_cleaned_up(self, sample_obj_path: Path, tmp_path: Path) -> None:
        """No temp .ply files should linger after the pipeline completes."""
        import glob

        before = set(glob.glob("/tmp/*.ply"))
        run_pipeline(sample_obj_path, ScanMetadata())
        after = set(glob.glob("/tmp/*.ply"))

        leaked = after - before
        assert len(leaked) == 0, f"Temp files leaked: {leaked}"
