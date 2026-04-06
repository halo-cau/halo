"""Integration tests for the full engine pipeline (clean → voxelize → label)."""

from pathlib import Path

import numpy as np
import pytest

from engine.core.config import (
    GRID_SHAPE,
    OBSTACLE_WALL,
    SPACE_EMPTY,
)
from engine.core.data_types import Coordinate, CoolingUnit, PipelineResult, ScanMetadata
from engine.core.exceptions import MeshProcessingError
from engine.vision.pipeline import run_pipeline


class TestRunPipeline:
    """End-to-end tests for the CV pipeline orchestrator."""

    def test_produces_valid_grid(self, sample_obj_path: Path) -> None:
        """Pipeline should return a PipelineResult with valid grids."""
        meta = ScanMetadata(
            cooling_units=[CoolingUnit(Coordinate(1.0, 1.0, 1.0))],
            legacy_servers=[Coordinate(1.5, 1.0, 0.5)],
            human_workspaces=[Coordinate(2.0, 1.5, 0.1)],
        )
        result = run_pipeline(sample_obj_path, meta)

        assert isinstance(result, PipelineResult)
        assert isinstance(result.grid, np.ndarray)
        assert result.grid.dtype == np.int8
        assert result.grid.ndim == 3
        assert SPACE_EMPTY in np.unique(result.grid)
        assert OBSTACLE_WALL in np.unique(result.grid)

        # Padded grid has fixed shape
        assert result.padded_grid.shape == GRID_SHAPE
        assert result.padded_grid.dtype == np.int8

    def test_empty_metadata_ok(self, sample_obj_path: Path) -> None:
        """Pipeline should work with no semantic annotations."""
        result = run_pipeline(sample_obj_path, ScanMetadata())

        assert result.grid.dtype == np.int8
        unique = set(np.unique(result.grid))
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
