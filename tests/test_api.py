"""Integration tests for the FastAPI /api/v1/process-scan endpoint."""

import json
from pathlib import Path

import open3d as o3d
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture()
def valid_obj_bytes(tmp_path: Path) -> bytes:
    """Generate a small box .obj and return its raw bytes."""
    mesh = o3d.geometry.TriangleMesh.create_box(width=2.5, height=2.0, depth=1.5)
    mesh = mesh.subdivide_midpoint(number_of_iterations=3)
    mesh.compute_vertex_normals()
    obj_file = tmp_path / "room.obj"
    o3d.io.write_triangle_mesh(str(obj_file), mesh)
    return obj_file.read_bytes()


@pytest.fixture()
def valid_metadata() -> str:
    return json.dumps(
        {
            "ac_vents": [{"x": 1.0, "y": 1.0, "z": 1.0}],
            "legacy_servers": [{"x": 1.5, "y": 1.0, "z": 0.5}],
            "human_workspaces": [{"x": 2.0, "y": 1.5, "z": 0.1}],
        }
    )


class TestProcessScanEndpoint:
    """Tests for POST /api/v1/process-scan."""

    def test_success(
        self, client: TestClient, valid_obj_bytes: bytes, valid_metadata: str
    ) -> None:
        resp = client.post(
            "/api/v1/process-scan",
            files={"file": ("room.obj", valid_obj_bytes, "application/octet-stream")},
            data={"metadata": valid_metadata},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "shape" in body
        assert len(body["shape"]) == 3
        assert "label_counts" in body
        assert any(lc["label"] == "wall" for lc in body["label_counts"])

    def test_wrong_extension_rejected(
        self, client: TestClient, valid_obj_bytes: bytes, valid_metadata: str
    ) -> None:
        resp = client.post(
            "/api/v1/process-scan",
            files={"file": ("room.stl", valid_obj_bytes, "application/octet-stream")},
            data={"metadata": valid_metadata},
        )
        assert resp.status_code == 400
        assert "Invalid file type" in resp.json()["detail"]

    def test_oversized_file_rejected(
        self, client: TestClient, valid_metadata: str
    ) -> None:
        # 51 MB of zeros — exceeds the 50 MB limit
        huge = b"\x00" * (51 * 1024 * 1024)
        resp = client.post(
            "/api/v1/process-scan",
            files={"file": ("room.obj", huge, "application/octet-stream")},
            data={"metadata": valid_metadata},
        )
        assert resp.status_code == 400
        assert "50 MB" in resp.json()["detail"]

    def test_invalid_metadata_rejected(
        self, client: TestClient, valid_obj_bytes: bytes
    ) -> None:
        bad_meta = json.dumps({"ac_vents": [{"x": "not_a_number"}]})
        resp = client.post(
            "/api/v1/process-scan",
            files={"file": ("room.obj", valid_obj_bytes, "application/octet-stream")},
            data={"metadata": bad_meta},
        )
        assert resp.status_code == 422

    def test_empty_mesh_returns_400(
        self, client: TestClient, valid_metadata: str
    ) -> None:
        empty_obj = b"# empty OBJ\n"
        resp = client.post(
            "/api/v1/process-scan",
            files={"file": ("empty.obj", empty_obj, "application/octet-stream")},
            data={"metadata": valid_metadata},
        )
        assert resp.status_code == 400

    def test_empty_metadata_ok(
        self, client: TestClient, valid_obj_bytes: bytes
    ) -> None:
        meta = json.dumps(
            {"ac_vents": [], "legacy_servers": [], "human_workspaces": []}
        )
        resp = client.post(
            "/api/v1/process-scan",
            files={"file": ("room.obj", valid_obj_bytes, "application/octet-stream")},
            data={"metadata": meta},
        )
        assert resp.status_code == 200
