"""Unit tests for engine.vision.voxelizer — voxelization, closing, semantic fusion."""

from pathlib import Path

import numpy as np
import pytest

from engine.core.config import (
    COOLING_AC_VENT,
    HEAT_LEGACY_SERVER,
    HUMAN_WORKSPACE,
    OBSTACLE_WALL,
    SPACE_EMPTY,
)
from engine.core.data_types import Coordinate, ScanMetadata
from engine.core.exceptions import RoomTooLargeError
from engine.vision.cleaner import clean_and_align
from engine.vision.voxelizer import (
    _morphological_close,
    _surface_voxelize,
    _world_to_index,
    fuse_semantics,
    voxelize_and_label,
)


class TestSurfaceVoxelize:
    """Tests for Trimesh surface voxelization (Step 2)."""

    def test_grid_dtype_and_labels(self, sample_obj_path: Path) -> None:
        """Voxelized grid should be int8 with only 0s and 1s."""
        ply_path = clean_and_align(sample_obj_path)
        try:
            import trimesh

            mesh = trimesh.load(str(ply_path), force="mesh")
            grid, origin = _surface_voxelize(mesh)

            assert grid.dtype == np.int8
            unique = set(np.unique(grid))
            assert unique <= {SPACE_EMPTY, OBSTACLE_WALL}
            assert grid.ndim == 3
        finally:
            ply_path.unlink(missing_ok=True)


class TestMorphologicalClose:
    """Tests for binary closing (Step 3)."""

    def test_fills_small_gap(self) -> None:
        """A small hole in a wall slab should be sealed."""
        grid = np.ones((10, 10, 10), dtype=np.int8)
        grid[5, 5, 5] = SPACE_EMPTY  # punch a 1-voxel hole

        closed = _morphological_close(grid)
        assert closed[5, 5, 5] == OBSTACLE_WALL

    def test_preserves_large_empty_region(self) -> None:
        """Large empty interior must NOT be filled."""
        grid = np.zeros((20, 20, 20), dtype=np.int8)
        # Walls only on the shell
        grid[0, :, :] = OBSTACLE_WALL
        grid[-1, :, :] = OBSTACLE_WALL
        grid[:, 0, :] = OBSTACLE_WALL
        grid[:, -1, :] = OBSTACLE_WALL
        grid[:, :, 0] = OBSTACLE_WALL
        grid[:, :, -1] = OBSTACLE_WALL

        closed = _morphological_close(grid)
        # Interior should still be mostly empty
        interior = closed[5:15, 5:15, 5:15]
        assert (interior == SPACE_EMPTY).sum() > 0


class TestWorldToIndex:
    """Tests for coordinate → index conversion."""

    def test_basic_conversion(self) -> None:
        origin = np.array([0.0, 0.0, 0.0])
        ix, iy, iz = _world_to_index(1.05, 2.15, 0.35, origin)
        # floor(1.05/0.1)=10, floor(2.15/0.1)=21, floor(0.35/0.1)=3
        assert (ix, iy, iz) == (10, 21, 3)

    def test_with_offset_origin(self) -> None:
        origin = np.array([1.0, 1.0, 0.0])
        ix, iy, iz = _world_to_index(1.5, 1.5, 0.5, origin)
        assert (ix, iy, iz) == (5, 5, 5)


class TestFuseSemantics:
    """Tests for semantic label injection (Step 4)."""

    def test_ac_vent_stamped(self) -> None:
        grid = np.zeros((20, 20, 20), dtype=np.int8)
        origin = np.array([0.0, 0.0, 0.0])
        meta = ScanMetadata(ac_vents=[Coordinate(0.5, 0.5, 0.5)])

        result = fuse_semantics(grid, meta, origin)
        assert result[5, 5, 5] == COOLING_AC_VENT

    def test_human_workspace_stamped(self) -> None:
        grid = np.zeros((20, 20, 20), dtype=np.int8)
        origin = np.array([0.0, 0.0, 0.0])
        meta = ScanMetadata(human_workspaces=[Coordinate(1.0, 1.0, 0.0)])

        result = fuse_semantics(grid, meta, origin)
        assert result[10, 10, 0] == HUMAN_WORKSPACE

    def test_heat_injection_spreads(self) -> None:
        grid = np.zeros((30, 30, 30), dtype=np.int8)
        origin = np.array([0.0, 0.0, 0.0])
        meta = ScanMetadata(legacy_servers=[Coordinate(1.5, 1.5, 1.5)])

        result = fuse_semantics(grid, meta, origin)
        heat_count = (result == HEAT_LEGACY_SERVER).sum()
        # Gaussian should spread to more than a single voxel
        assert heat_count > 1

    def test_out_of_bounds_ignored(self) -> None:
        """A coordinate outside the grid should not crash."""
        grid = np.zeros((10, 10, 10), dtype=np.int8)
        origin = np.array([0.0, 0.0, 0.0])
        meta = ScanMetadata(ac_vents=[Coordinate(99.0, 99.0, 99.0)])

        result = fuse_semantics(grid, meta, origin)
        # Grid should be unchanged
        assert (result == SPACE_EMPTY).all()


class TestVoxelizeAndLabel:
    """Integration test for the full Steps 2-4."""

    def test_end_to_end(self, sample_obj_path: Path) -> None:
        """Full voxelization pipeline should produce a labeled int8 grid."""
        ply_path = clean_and_align(sample_obj_path)
        try:
            meta = ScanMetadata(
                ac_vents=[Coordinate(1.0, 1.0, 1.0)],
                legacy_servers=[Coordinate(1.5, 1.0, 0.5)],
                human_workspaces=[Coordinate(2.0, 1.5, 0.1)],
            )
            grid = voxelize_and_label(ply_path, meta)

            assert grid.dtype == np.int8
            assert grid.ndim == 3
            unique = set(np.unique(grid))
            # Must contain at least empty and wall
            assert SPACE_EMPTY in unique
            assert OBSTACLE_WALL in unique
        finally:
            ply_path.unlink(missing_ok=True)

    def test_oversized_room_raises(self, oversized_obj_path: Path) -> None:
        """A room exceeding MAX_ROOM_DIMENSIONS must raise RoomTooLargeError."""
        ply_path = clean_and_align(oversized_obj_path)
        try:
            meta = ScanMetadata()
            with pytest.raises(RoomTooLargeError):
                voxelize_and_label(ply_path, meta)
        finally:
            ply_path.unlink(missing_ok=True)
