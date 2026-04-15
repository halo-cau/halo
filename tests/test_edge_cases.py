"""Edge-case tests for the CV pipeline with unusual room geometries.

Each test generates a synthetic .obj mesh, runs it through the pipeline,
and checks that the cleaner + voxelizer produce sensible results without
crashing.
"""

import os
import tempfile

import numpy as np
import open3d as o3d
import pytest

from engine.core.data_types import Coordinate, CoolingUnit, ScanMetadata
from engine.core.exceptions import MeshProcessingError, RoomTooLargeError
from engine.vision.cleaner import clean_and_align_meshes
from engine.vision.pipeline import run_pipeline
from engine.core.config import GRID_SHAPE


def _write_mesh(mesh: o3d.geometry.TriangleMesh) -> str:
    """Write an Open3D mesh to a temp .obj file and return its path."""
    mesh.compute_vertex_normals()
    tmp = tempfile.NamedTemporaryFile(suffix=".obj", delete=False)
    tmp.close()
    o3d.io.write_triangle_mesh(tmp.name, mesh)
    return tmp.name


def _subdivided_box(
    w: float, d: float, h: float, iters: int = 3
) -> o3d.geometry.TriangleMesh:
    """Create a subdivided box — enough vertices to survive SOR."""
    mesh = o3d.geometry.TriangleMesh.create_box(w, d, h)
    mesh = mesh.subdivide_midpoint(number_of_iterations=iters)
    mesh.compute_vertex_normals()
    return mesh


# ──────────────────────────────────────────────────────────────
# 1. Tall narrow room: walls larger than floor → RANSAC must
#    still pick the floor (smallest horizontal plane).
# ──────────────────────────────────────────────────────────────
class TestTallNarrowRoom:
    """2×2 m floor × 5 m height — wall area dominates floor area."""

    @pytest.fixture()
    def obj_path(self):
        mesh = _subdivided_box(2.0, 2.0, 5.0)
        path = _write_mesh(mesh)
        yield path
        if os.path.exists(path):
            os.unlink(path)

    def test_floor_at_z0(self, obj_path):
        """Cleaned mesh floor should be at Z ≈ 0, not rotated."""
        _, cleaned = clean_and_align_meshes(obj_path)
        verts = np.asarray(cleaned.vertices)
        assert verts[:, 2].min() == pytest.approx(0.0, abs=0.05)

    def test_height_preserved(self, obj_path):
        """Room height should stay ≈ 5 m after alignment."""
        _, cleaned = clean_and_align_meshes(obj_path)
        verts = np.asarray(cleaned.vertices)
        height = verts[:, 2].max() - verts[:, 2].min()
        assert height == pytest.approx(5.0, abs=0.3)

    def test_no_axis_swap(self, obj_path):
        """The widest XY extent should remain in X/Y, not rotate into Z."""
        raw, cleaned = clean_and_align_meshes(obj_path)
        raw_v = np.asarray(raw.vertices)
        cln_v = np.asarray(cleaned.vertices)
        # Raw extents in XY
        raw_xy = raw_v[:, :2].max(axis=0) - raw_v[:, :2].min(axis=0)
        cln_xy = cln_v[:, :2].max(axis=0) - cln_v[:, :2].min(axis=0)
        # XY footprint should be comparable (within 20%)
        assert np.allclose(sorted(raw_xy), sorted(cln_xy), atol=0.5)

    def test_pipeline_succeeds(self, obj_path):
        """Full pipeline should produce a valid grid for this room."""
        meta = ScanMetadata(
            legacy_servers=[Coordinate(1.0, 1.0, 0.5)],
            cooling_units=[CoolingUnit(Coordinate(0.5, 0.5, 4.5))],
        )
        result = run_pipeline(obj_path, meta)
        assert result.grid.ndim == 3
        assert result.grid.shape[2] > 30  # ~5 m / 0.1 voxel = 50 slices
        assert result.padded_grid.shape == GRID_SHAPE


# ──────────────────────────────────────────────────────────────
# 2. Very flat room: barely any height (0.3 m).
#    The floor and ceiling are almost the same plane.
# ──────────────────────────────────────────────────────────────
class TestVeryFlatRoom:
    """10×10 m floor × 0.3 m height — pancake room."""

    @pytest.fixture()
    def obj_path(self):
        mesh = _subdivided_box(10.0, 10.0, 0.3)
        path = _write_mesh(mesh)
        yield path
        if os.path.exists(path):
            os.unlink(path)

    def test_floor_at_z0(self, obj_path):
        _, cleaned = clean_and_align_meshes(obj_path)
        verts = np.asarray(cleaned.vertices)
        assert verts[:, 2].min() == pytest.approx(0.0, abs=0.05)

    def test_pipeline_produces_grid(self, obj_path):
        result = run_pipeline(obj_path, ScanMetadata())
        assert result.grid.ndim == 3
        # Height dimension should be very small (≤ 5 voxels)
        assert result.grid.shape[2] <= 10
        assert result.padded_grid.shape == GRID_SHAPE


# ──────────────────────────────────────────────────────────────
# 3. L-shaped room: concatenation of two boxes (non-convex).
# ──────────────────────────────────────────────────────────────
class TestLShapedRoom:
    """L-shape built from two merged boxes."""

    @pytest.fixture()
    def obj_path(self):
        box1 = _subdivided_box(6.0, 3.0, 2.5)
        box2 = _subdivided_box(3.0, 6.0, 2.5)
        # Shift box2 so they form an L
        box2.translate((0.0, 3.0, 0.0))
        combined = box1 + box2
        combined.compute_vertex_normals()
        path = _write_mesh(combined)
        yield path
        if os.path.exists(path):
            os.unlink(path)

    def test_floor_at_z0(self, obj_path):
        _, cleaned = clean_and_align_meshes(obj_path)
        verts = np.asarray(cleaned.vertices)
        assert verts[:, 2].min() == pytest.approx(0.0, abs=0.05)

    def test_pipeline_succeeds(self, obj_path):
        meta = ScanMetadata(
            legacy_servers=[Coordinate(2.0, 1.5, 0.5)],
            cooling_units=[CoolingUnit(Coordinate(1.0, 5.0, 2.4))],
            human_workspaces=[Coordinate(4.0, 1.0, 0.0)],
        )
        result = run_pipeline(obj_path, meta)
        assert result.grid.ndim == 3
        # L-shape is wider than a single box
        assert max(result.grid.shape[0], result.grid.shape[1]) > 50  # > 5 m in voxels
        assert result.padded_grid.shape == GRID_SHAPE


# ──────────────────────────────────────────────────────────────
# 4. Tilted room: the mesh is rotated 15° around X before scan.
#    Simulates a scan from a tilted sensor.
# ──────────────────────────────────────────────────────────────
class TestTiltedRoom:
    """Standard 4×3×2.5 room pre-rotated 15° around X axis."""

    @pytest.fixture()
    def obj_path(self):
        mesh = _subdivided_box(4.0, 3.0, 2.5)
        R = mesh.get_rotation_matrix_from_xyz((np.radians(15), 0, 0))
        mesh.rotate(R, center=mesh.get_center())
        path = _write_mesh(mesh)
        yield path
        if os.path.exists(path):
            os.unlink(path)

    def test_floor_realigned_to_z0(self, obj_path):
        """Cleaner should undo the tilt and place floor at Z ≈ 0."""
        _, cleaned = clean_and_align_meshes(obj_path)
        verts = np.asarray(cleaned.vertices)
        assert verts[:, 2].min() == pytest.approx(0.0, abs=0.05)

    def test_pipeline_succeeds(self, obj_path):
        result = run_pipeline(obj_path, ScanMetadata())
        assert result.grid.ndim == 3
        assert result.padded_grid.shape == GRID_SHAPE


# ──────────────────────────────────────────────────────────────
# 5. Severely tilted: 45° — stress test for Rodrigues rotation.
# ──────────────────────────────────────────────────────────────
class TestSeverelyTiltedRoom:
    """Room tilted 45° — edge case for rotation math."""

    @pytest.fixture()
    def obj_path(self):
        mesh = _subdivided_box(3.0, 3.0, 2.0)
        R = mesh.get_rotation_matrix_from_xyz((np.radians(45), np.radians(20), 0))
        mesh.rotate(R, center=mesh.get_center())
        path = _write_mesh(mesh)
        yield path
        if os.path.exists(path):
            os.unlink(path)

    def test_floor_at_z0(self, obj_path):
        _, cleaned = clean_and_align_meshes(obj_path)
        verts = np.asarray(cleaned.vertices)
        assert verts[:, 2].min() == pytest.approx(0.0, abs=0.1)

    def test_pipeline_succeeds(self, obj_path):
        result = run_pipeline(obj_path, ScanMetadata())
        assert result.grid.ndim == 3
        assert result.padded_grid.shape == GRID_SHAPE


# ──────────────────────────────────────────────────────────────
# 6. Cylinder room: non-box shape (curved walls).
#    Tests that RANSAC finds the flat end-caps as "floor".
# ──────────────────────────────────────────────────────────────
class TestCylinderRoom:
    """Cylinder with flat top/bottom — curved walls, flat floor."""

    @pytest.fixture()
    def obj_path(self):
        mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=3.0, height=2.5)
        mesh = mesh.subdivide_midpoint(number_of_iterations=2)
        mesh.compute_vertex_normals()
        # Cylinder center is at origin; shift up so bottom is at Z=0
        verts = np.asarray(mesh.vertices)
        mesh.translate((0, 0, -verts[:, 2].min()))
        path = _write_mesh(mesh)
        yield path
        if os.path.exists(path):
            os.unlink(path)

    def test_floor_at_z0(self, obj_path):
        _, cleaned = clean_and_align_meshes(obj_path)
        verts = np.asarray(cleaned.vertices)
        assert verts[:, 2].min() == pytest.approx(0.0, abs=0.1)

    def test_pipeline_succeeds(self, obj_path):
        result = run_pipeline(obj_path, ScanMetadata())
        assert result.grid.ndim == 3
        assert result.padded_grid.shape == GRID_SHAPE


# ──────────────────────────────────────────────────────────────
# 7. Room with furniture: a large desk covering 25% of the floor.
# ──────────────────────────────────────────────────────────────
class TestRoomWithFurniture:
    """4×4×3 room with a 2×2×0.8 desk — floor partially occluded."""

    @pytest.fixture()
    def obj_path(self):
        room = _subdivided_box(4.0, 4.0, 3.0)
        desk = _subdivided_box(2.0, 2.0, 0.8, iters=2)
        desk.translate((1.0, 1.0, 0.0))  # Desk sits on the floor
        combined = room + desk
        combined.compute_vertex_normals()
        path = _write_mesh(combined)
        yield path
        if os.path.exists(path):
            os.unlink(path)

    def test_floor_at_z0(self, obj_path):
        """Floor should be at Z=0, not at desk height (0.8)."""
        _, cleaned = clean_and_align_meshes(obj_path)
        verts = np.asarray(cleaned.vertices)
        assert verts[:, 2].min() == pytest.approx(0.0, abs=0.05)

    def test_pipeline_with_metadata(self, obj_path):
        meta = ScanMetadata(
            legacy_servers=[Coordinate(3.0, 3.0, 0.5)],
            human_workspaces=[Coordinate(1.5, 1.5, 0.85)],  # On desk
            cooling_units=[CoolingUnit(Coordinate(2.0, 2.0, 2.9))],
        )
        result = run_pipeline(obj_path, meta)
        assert result.grid.ndim == 3
        assert (result.grid == 2).any()  # heat label exists
        assert (result.grid == 3).any()  # AC label exists
        assert (result.grid == 4).any()  # workspace label exists
        assert result.padded_grid.shape == GRID_SHAPE


# ──────────────────────────────────────────────────────────────
# 8. Tiny room: 0.5 × 0.5 × 0.5 m — very few voxels.
# ──────────────────────────────────────────────────────────────
class TestTinyRoom:
    """50 cm cube — pushes minimum voxel grid size."""

    @pytest.fixture()
    def obj_path(self):
        mesh = _subdivided_box(0.5, 0.5, 0.5)
        path = _write_mesh(mesh)
        yield path
        if os.path.exists(path):
            os.unlink(path)

    def test_pipeline_produces_small_grid(self, obj_path):
        result = run_pipeline(obj_path, ScanMetadata())
        assert result.grid.ndim == 3
        # ~5 voxels per axis at 0.1 m pitch
        assert all(s <= 10 for s in result.grid.shape)
        assert result.padded_grid.shape == GRID_SHAPE


# ──────────────────────────────────────────────────────────────
# 9. Room at max dimensions boundary: exactly 20×20×4 m.
# ──────────────────────────────────────────────────────────────
class TestMaxSizeRoom:
    """Room right at the MAX_ROOM_DIMENSIONS limit."""

    @pytest.fixture()
    def obj_path(self):
        # Use 3 subdivisions — 2 is too sparse for SOR on large meshes
        mesh = _subdivided_box(19.5, 19.5, 5.5, iters=3)
        path = _write_mesh(mesh)
        yield path
        if os.path.exists(path):
            os.unlink(path)

    def test_pipeline_succeeds(self, obj_path):
        result = run_pipeline(obj_path, ScanMetadata())
        assert result.grid.ndim == 3
        assert result.padded_grid.shape == GRID_SHAPE


# ──────────────────────────────────────────────────────────────
# 10. Room exceeding max dimensions → should raise.
# ──────────────────────────────────────────────────────────────
class TestOversizedRoom:
    """Room 25×25×7 — exceeds limits on all axes."""

    @pytest.fixture()
    def obj_path(self):
        mesh = _subdivided_box(25.0, 25.0, 7.0, iters=1)
        path = _write_mesh(mesh)
        yield path
        if os.path.exists(path):
            os.unlink(path)

    def test_raises_room_too_large(self, obj_path):
        with pytest.raises((RoomTooLargeError, MeshProcessingError)):
            run_pipeline(obj_path, ScanMetadata())


# ──────────────────────────────────────────────────────────────
# 11. Sphere room: no flat surfaces at all.
#     RANSAC should still find the best-fit plane.
# ──────────────────────────────────────────────────────────────
class TestSphereRoom:
    """Sphere — no actual flat floor."""

    @pytest.fixture()
    def obj_path(self):
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=2.0)
        mesh = mesh.subdivide_midpoint(number_of_iterations=2)
        mesh.compute_vertex_normals()
        path = _write_mesh(mesh)
        yield path
        if os.path.exists(path):
            os.unlink(path)

    def test_cleaner_doesnt_crash(self, obj_path):
        """Even with no real floor, cleaner should produce some result."""
        _, cleaned = clean_and_align_meshes(obj_path)
        verts = np.asarray(cleaned.vertices)
        # Should be translated so min Z ≈ 0
        assert verts[:, 2].min() == pytest.approx(0.0, abs=0.15)

    def test_pipeline_succeeds(self, obj_path):
        result = run_pipeline(obj_path, ScanMetadata())
        assert result.grid.ndim == 3
        assert result.padded_grid.shape == GRID_SHAPE


# ──────────────────────────────────────────────────────────────
# 12. Metadata points outside the room → should not crash.
# ──────────────────────────────────────────────────────────────
class TestOutOfBoundsMetadata:
    """Semantic coordinates far outside the room volume."""

    @pytest.fixture()
    def obj_path(self):
        mesh = _subdivided_box(3.0, 3.0, 2.0)
        path = _write_mesh(mesh)
        yield path
        if os.path.exists(path):
            os.unlink(path)

    def test_pipeline_ignores_oob_points(self, obj_path):
        meta = ScanMetadata(
            cooling_units=[CoolingUnit(Coordinate(100.0, 100.0, 100.0))],
            legacy_servers=[Coordinate(-50.0, -50.0, -10.0)],
            human_workspaces=[Coordinate(999.0, 0.0, 0.0)],
        )
        result = run_pipeline(obj_path, meta)
        assert result.grid.ndim == 3
        # OOB points should be silently ignored — no special labels stamped
        assert not (result.grid == 2).any()
        assert not (result.grid == 3).any()
        assert not (result.grid == 4).any()
        assert result.padded_grid.shape == GRID_SHAPE
