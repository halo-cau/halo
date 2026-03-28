"""Step 1: Mesh ingestion, SOR cleanup, and RANSAC floor alignment (Open3D)."""

import tempfile
from pathlib import Path

import numpy as np
import open3d as o3d

from engine.core.config import (
    RANSAC_DISTANCE_THRESHOLD,
    RANSAC_NUM_ITERATIONS,
    RANSAC_NUM_POINTS,
    SOR_NB_NEIGHBORS,
    SOR_STD_RATIO,
)
from engine.core.exceptions import MeshProcessingError


def _align_floor_to_z0(
    mesh: o3d.geometry.TriangleMesh,
    plane_model: np.ndarray,
) -> o3d.geometry.TriangleMesh:
    """Rotate + translate the mesh so the detected floor sits at Z=0, normal +Z."""
    a, b, c, d = plane_model
    normal = np.array([a, b, c], dtype=np.float64)
    normal /= np.linalg.norm(normal)

    # Ensure the normal points toward +Z (flip if pointing down)
    if normal[2] < 0:
        normal = -normal
        d = -d

    target = np.array([0.0, 0.0, 1.0])

    # Rotation from current normal → +Z
    v = np.cross(normal, target)
    cos_angle = float(np.dot(normal, target))

    if np.linalg.norm(v) < 1e-6:
        # Already aligned (or exactly opposite)
        rotation = np.eye(3) if cos_angle > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        skew = np.array(
            [[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]],
            dtype=np.float64,
        )
        rotation = np.eye(3) + skew + skew @ skew * (1.0 / (1.0 + cos_angle))

    mesh.rotate(rotation, center=(0.0, 0.0, 0.0))

    # Translate so the floor plane sits at Z = 0
    vertices = np.asarray(mesh.vertices)
    z_offset = float(np.min(vertices[:, 2]))
    mesh.translate((0.0, 0.0, -z_offset))

    return mesh


def clean_and_align(obj_path: Path) -> Path:
    """Load an .obj mesh, run SOR + RANSAC floor alignment, return cleaned PLY path.

    The returned path is a temporary .ply file consumable by Trimesh.
    The caller is responsible for deleting it when done.
    """
    _, cleaned = clean_and_align_meshes(obj_path)

    # --- Export to a temp PLY for Trimesh ---
    tmp = tempfile.NamedTemporaryFile(suffix=".ply", delete=False)
    tmp.close()
    o3d.io.write_triangle_mesh(tmp.name, cleaned)

    return Path(tmp.name)


def clean_and_align_meshes(
    obj_path: Path,
) -> tuple[o3d.geometry.TriangleMesh, o3d.geometry.TriangleMesh]:
    """Load an .obj mesh, return (raw_mesh, cleaned_aligned_mesh).

    Both are Open3D TriangleMesh objects with computed vertex normals.
    The raw mesh is the original geometry before any processing.
    """
    mesh = o3d.io.read_triangle_mesh(str(obj_path))
    if mesh.is_empty():
        raise MeshProcessingError("The uploaded .obj file contains no geometry.")

    mesh.compute_vertex_normals()
    raw_mesh = o3d.geometry.TriangleMesh(mesh)  # deep copy before cleanup

    # --- Statistical Outlier Removal on the vertex cloud ---
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    pcd.normals = mesh.vertex_normals

    _, inlier_idx = pcd.remove_statistical_outlier(
        nb_neighbors=SOR_NB_NEIGHBORS,
        std_ratio=SOR_STD_RATIO,
    )

    mesh = mesh.select_by_index(inlier_idx)
    if mesh.is_empty():
        raise MeshProcessingError(
            "Mesh is empty after outlier removal. The scan may be too noisy."
        )

    mesh.compute_vertex_normals()

    # --- RANSAC floor detection ---
    # segment_plane finds the largest plane, which may be a wall.
    # Iterate to find the most floor-like plane (normal closest to ±Z).
    pcd_clean = o3d.geometry.PointCloud()
    pcd_clean.points = mesh.vertices
    pcd_clean.normals = mesh.vertex_normals

    best_plane = None
    best_z_component = 0.0
    remaining = pcd_clean
    max_attempts = 5

    for _ in range(max_attempts):
        if len(remaining.points) < RANSAC_NUM_POINTS:
            break

        plane_model, inlier_idx = remaining.segment_plane(
            distance_threshold=RANSAC_DISTANCE_THRESHOLD,
            ransac_n=RANSAC_NUM_POINTS,
            num_iterations=RANSAC_NUM_ITERATIONS,
        )

        if plane_model is None or len(inlier_idx) == 0:
            break

        normal = np.array(plane_model[:3])
        normal /= np.linalg.norm(normal)
        z_component = abs(normal[2])

        if z_component > best_z_component:
            best_z_component = z_component
            best_plane = np.asarray(plane_model)

        # If this plane is clearly a floor (normal almost vertical), stop early
        if z_component > 0.9:
            break

        # Remove inliers and try again to find a more horizontal plane
        remaining = remaining.select_by_index(inlier_idx, invert=True)

    if best_plane is None:
        raise MeshProcessingError("RANSAC could not detect a floor plane in the scan.")

    mesh = _align_floor_to_z0(mesh, best_plane)

    return raw_mesh, mesh
