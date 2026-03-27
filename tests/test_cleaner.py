"""Unit tests for engine.vision.cleaner — SOR + RANSAC floor alignment."""

from pathlib import Path

import numpy as np
import pytest

from engine.core.exceptions import MeshProcessingError
from engine.vision.cleaner import clean_and_align


class TestCleanAndAlign:
    """Tests for the Open3D ingestion & cleanup step."""

    def test_returns_ply_file(self, sample_obj_path: Path) -> None:
        """clean_and_align should return a valid .ply temp file."""
        ply_path = clean_and_align(sample_obj_path)
        try:
            assert ply_path.exists()
            assert ply_path.suffix == ".ply"
            assert ply_path.stat().st_size > 0
        finally:
            ply_path.unlink(missing_ok=True)

    def test_floor_aligned_near_z0(self, sample_obj_path: Path) -> None:
        """After alignment the median vertex Z should be near 0."""
        import open3d as o3d

        ply_path = clean_and_align(sample_obj_path)
        try:
            mesh = o3d.io.read_triangle_mesh(str(ply_path))
            vertices = np.asarray(mesh.vertices)
            z_min = vertices[:, 2].min()
            # Floor should be within ±0.5 m of Z=0
            assert abs(z_min) < 0.5, f"Floor Z min = {z_min}, expected near 0"
        finally:
            ply_path.unlink(missing_ok=True)

    def test_empty_mesh_raises(self, empty_obj_path: Path) -> None:
        """An empty .obj must raise MeshProcessingError."""
        with pytest.raises(MeshProcessingError):
            clean_and_align(empty_obj_path)
