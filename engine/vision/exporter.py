"""GLB export utilities for visualization (Open3D mesh → GLB bytes via Trimesh)."""

import io

import numpy as np
import open3d as o3d
import trimesh

from engine.core.config import (
    COOLING_AC_VENT,
    HEAT_LEGACY_SERVER,
    HUMAN_WORKSPACE,
    OBSTACLE_WALL,
    SPACE_EMPTY,
    VOXEL_SIZE,
)
from engine.core.data_types import ScanMetadata

# Semantic label → RGBA vertex color (0-255)
SEMANTIC_COLORS: dict[int, tuple[int, int, int, int]] = {
    SPACE_EMPTY: (200, 200, 200, 60),  # transparent light gray
    OBSTACLE_WALL: (180, 180, 180, 255),  # opaque gray
    HEAT_LEGACY_SERVER: (230, 60, 60, 220),  # red
    COOLING_AC_VENT: (60, 140, 230, 220),  # blue
    HUMAN_WORKSPACE: (60, 200, 100, 220),  # green
}


def o3d_to_glb(mesh: o3d.geometry.TriangleMesh) -> bytes:
    """Convert an Open3D TriangleMesh to GLB binary bytes."""
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)

    if mesh.has_vertex_colors():
        colors = (np.asarray(mesh.vertex_colors) * 255).astype(np.uint8)
        tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors)
    else:
        tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    buf = io.BytesIO()
    tri_mesh.export(buf, file_type="glb")
    return buf.getvalue()


def paint_semantic_colors(
    mesh: o3d.geometry.TriangleMesh,
    metadata: ScanMetadata,
) -> o3d.geometry.TriangleMesh:
    """Paint vertex colors on a cleaned mesh based on semantic proximity.

    Each vertex is colored by the nearest semantic label using the voxel grid
    lookup. Vertices that don't map to a special label get the wall/empty color.
    """
    mesh = o3d.geometry.TriangleMesh(mesh)  # deep copy
    mesh.compute_vertex_normals()

    vertices = np.asarray(mesh.vertices)
    n_verts = len(vertices)

    # Build a quick spatial lookup from metadata coordinates
    # Map each semantic point to a world coordinate + label
    semantic_points: list[tuple[np.ndarray, int]] = []
    for unit in metadata.cooling_units:
        semantic_points.append((np.array([unit.position.x, unit.position.y, unit.position.z]), COOLING_AC_VENT))
    for ws in metadata.human_workspaces:
        semantic_points.append((np.array([ws.x, ws.y, ws.z]), HUMAN_WORKSPACE))
    for srv in metadata.legacy_servers:
        semantic_points.append((np.array([srv.x, srv.y, srv.z]), HEAT_LEGACY_SERVER))

    # Default: wall color for all vertices
    colors = np.full((n_verts, 3), SEMANTIC_COLORS[OBSTACLE_WALL][:3], dtype=np.float64)
    colors = colors / 255.0

    # Paint vertices near semantic points within a radius
    radius = VOXEL_SIZE * 6  # 60 cm influence radius for visualization
    for point, label in semantic_points:
        dists = np.linalg.norm(vertices - point, axis=1)
        mask = dists < radius
        r, g, b, _ = SEMANTIC_COLORS[label]
        colors[mask] = np.array([r, g, b], dtype=np.float64) / 255.0

    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    return mesh
