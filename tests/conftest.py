"""Shared test fixtures — synthetic .obj mesh and metadata."""

import sys
from pathlib import Path

import open3d as o3d
import pytest

# Ensure both engine/ and backend/ are importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))


@pytest.fixture()
def sample_obj_path(tmp_path: Path) -> Path:
    """Generate a simple box .obj file (~2.5m × 2m × 1.5m room).

    Small enough that even worst-case RANSAC rotation stays within
    MAX_ROOM_DIMENSIONS (space diagonal ≈ 3.54 m < 4.0 m Z limit).
    """
    mesh = o3d.geometry.TriangleMesh.create_box(width=2.5, height=2.0, depth=1.5)
    mesh = mesh.subdivide_midpoint(number_of_iterations=3)
    mesh.compute_vertex_normals()
    obj_file = tmp_path / "test_room.obj"
    o3d.io.write_triangle_mesh(str(obj_file), mesh)
    return obj_file


@pytest.fixture()
def oversized_obj_path(tmp_path: Path) -> Path:
    """Generate an .obj that exceeds MAX_ROOM_DIMENSIONS (25m × 25m × 5m)."""
    mesh = o3d.geometry.TriangleMesh.create_box(width=25.0, height=25.0, depth=5.0)
    mesh.compute_vertex_normals()
    obj_file = tmp_path / "huge_room.obj"
    o3d.io.write_triangle_mesh(str(obj_file), mesh)
    return obj_file


@pytest.fixture()
def empty_obj_path(tmp_path: Path) -> Path:
    """Generate a valid .obj file with no geometry."""
    obj_file = tmp_path / "empty.obj"
    obj_file.write_text("# empty OBJ\n")
    return obj_file
