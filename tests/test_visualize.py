"""Tests for the /api/v1/visualize endpoint and GLB export utilities."""

import base64
import json
from pathlib import Path

import open3d as o3d
import pytest
from fastapi.testclient import TestClient

from app.main import app
from engine.core.data_types import Coordinate, ScanMetadata
from engine.vision.cleaner import clean_and_align_meshes
from engine.vision.exporter import o3d_to_glb, paint_semantic_colors


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture()
def valid_obj_bytes(tmp_path: Path) -> bytes:
    mesh = o3d.geometry.TriangleMesh.create_box(width=2.5, height=2.0, depth=1.5)
    mesh = mesh.subdivide_midpoint(number_of_iterations=3)
    mesh.compute_vertex_normals()
    obj_file = tmp_path / "room.obj"
    o3d.io.write_triangle_mesh(str(obj_file), mesh)
    return obj_file.read_bytes()


class TestO3dToGlb:
    """Tests for Open3D → GLB conversion."""

    def test_produces_valid_glb(self) -> None:
        mesh = o3d.geometry.TriangleMesh.create_box()
        mesh.compute_vertex_normals()
        glb = o3d_to_glb(mesh)
        # GLB magic bytes: "glTF"
        assert glb[:4] == b"glTF"
        assert len(glb) > 100

    def test_preserves_vertex_colors(self) -> None:
        import numpy as np

        mesh = o3d.geometry.TriangleMesh.create_box()
        mesh.compute_vertex_normals()
        colors = np.full((len(mesh.vertices), 3), [1.0, 0.0, 0.0])
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        glb = o3d_to_glb(mesh)
        assert glb[:4] == b"glTF"


class TestPaintSemanticColors:
    """Tests for semantic vertex coloring."""

    def test_colors_vertices(self, sample_obj_path: Path) -> None:
        _, cleaned = clean_and_align_meshes(sample_obj_path)
        meta = ScanMetadata(
            ac_vents=[Coordinate(1.0, 1.0, 0.5)],
        )
        colored = paint_semantic_colors(cleaned, meta)
        assert colored.has_vertex_colors()

    def test_no_metadata_still_colors(self, sample_obj_path: Path) -> None:
        _, cleaned = clean_and_align_meshes(sample_obj_path)
        colored = paint_semantic_colors(cleaned, ScanMetadata())
        assert colored.has_vertex_colors()


class TestCleanAndAlignMeshes:
    """Tests for the dual-mesh return function."""

    def test_returns_two_meshes(self, sample_obj_path: Path) -> None:
        raw, cleaned = clean_and_align_meshes(sample_obj_path)
        assert not raw.is_empty()
        assert not cleaned.is_empty()

    def test_raw_differs_from_cleaned(self, sample_obj_path: Path) -> None:
        import numpy as np

        raw, cleaned = clean_and_align_meshes(sample_obj_path)
        raw_verts = np.asarray(raw.vertices)
        cleaned_verts = np.asarray(cleaned.vertices)
        # The cleaned mesh may have fewer vertices or different positions
        assert raw_verts.shape[0] >= cleaned_verts.shape[0]


class TestVisualizeEndpoint:
    """Tests for POST /api/v1/visualize."""

    def test_returns_glb_meshes(
        self, client: TestClient, valid_obj_bytes: bytes
    ) -> None:
        meta = json.dumps(
            {
                "ac_vents": [{"x": 1.0, "y": 1.0, "z": 1.0}],
                "legacy_servers": [{"x": 1.5, "y": 1.0, "z": 0.5}],
                "human_workspaces": [{"x": 2.0, "y": 1.5, "z": 0.1}],
            }
        )
        resp = client.post(
            "/api/v1/visualize",
            files={"file": ("room.obj", valid_obj_bytes, "application/octet-stream")},
            data={"metadata": meta},
        )
        assert resp.status_code == 200
        body = resp.json()

        # All three GLBs should be present
        assert "raw_glb" in body
        assert "cleaned_glb" in body
        assert "semantic_glb" in body

        # Decode and check GLB magic bytes
        for key in ("raw_glb", "cleaned_glb", "semantic_glb"):
            glb_bytes = base64.b64decode(body[key])
            assert glb_bytes[:4] == b"glTF", f"{key} is not valid GLB"

    def test_no_metadata_skips_semantic(
        self, client: TestClient, valid_obj_bytes: bytes
    ) -> None:
        resp = client.post(
            "/api/v1/visualize",
            files={"file": ("room.obj", valid_obj_bytes, "application/octet-stream")},
            data={"metadata": "{}"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["raw_glb"] is not None
        assert body["cleaned_glb"] is not None
        assert body["semantic_glb"] is None

    def test_wrong_extension_rejected(
        self, client: TestClient, valid_obj_bytes: bytes
    ) -> None:
        resp = client.post(
            "/api/v1/visualize",
            files={"file": ("room.stl", valid_obj_bytes, "application/octet-stream")},
            data={"metadata": "{}"},
        )
        assert resp.status_code == 400

    def test_empty_mesh_returns_400(self, client: TestClient) -> None:
        empty_obj = b"# empty OBJ\n"
        resp = client.post(
            "/api/v1/visualize",
            files={"file": ("room.obj", empty_obj, "application/octet-stream")},
            data={"metadata": "{}"},
        )
        assert resp.status_code == 400
