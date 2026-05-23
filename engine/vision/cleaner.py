"""Step 1: scan ingestion, SOR cleanup, RANSAC floor alignment, and Manhattan rectification (Open3D).

The cleaner accepts both meshed scans (OBJ/PLY) and point-cloud scans
(LAS/LAZ).  Downstream code still receives an Open3D ``TriangleMesh`` object;
for point clouds the mesh simply has vertices/colours and no triangle faces.
"""

import tempfile
from pathlib import Path

import numpy as np
import open3d as o3d

from engine.core.config import (
    MANHATTAN_MAX_PLANES,
    MANHATTAN_MIN_PLANE_INLIER_FRAC,
    MANHATTAN_NORMAL_TOL_DEG,
    MANHATTAN_PLANE_INLIER_DIST_M,
    RANSAC_DISTANCE_THRESHOLD,
    RANSAC_NUM_ITERATIONS,
    RANSAC_NUM_POINTS,
    SOR_NB_NEIGHBORS,
    SOR_STD_RATIO,
)
from engine.core.exceptions import MeshProcessingError

POINT_CLOUD_SUFFIXES = {".las", ".laz"}


def _align_floor_to_z0(
    mesh: o3d.geometry.TriangleMesh,
    plane_model: np.ndarray,
    up_axis: int | None = None,
) -> o3d.geometry.TriangleMesh:
    """Rotate + translate the mesh so the detected floor sits at Z=0, normal +Z."""
    a, b, c, d = plane_model
    normal = np.array([a, b, c], dtype=np.float64)
    normal /= np.linalg.norm(normal)

    # Ensure the normal points toward the inferred room-up axis before mapping
    # it to +Z.  Phone scans are not guaranteed to use raw Z as vertical; for
    # this scan raw Z is the entrance/depth direction, so choosing the most
    # raw-Z-like plane incorrectly turns a wall into the floor.
    if up_axis is not None and normal[up_axis] < 0:
        normal = -normal
        d = -d
    elif up_axis is None and normal[2] < 0:
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

    # Translate the fitted floor plane itself to Z=0.  Using min(z) can be
    # pulled down by scan outliers or by geometry below a raised threshold.
    mesh.translate((0.0, 0.0, float(d)))

    return mesh


def _candidate_alignment_extent(
    verts: np.ndarray,
    plane_model: np.ndarray,
    up_axis: int,
) -> np.ndarray:
    """Return bbox extents after aligning a floor-plane candidate to +Z."""
    normal = np.asarray(plane_model[:3], dtype=np.float64)
    normal /= np.linalg.norm(normal)
    if normal[up_axis] < 0:
        normal = -normal

    target = np.array([0.0, 0.0, 1.0])
    v = np.cross(normal, target)
    cos_angle = float(np.dot(normal, target))
    if np.linalg.norm(v) < 1e-6:
        rotation = np.eye(3) if cos_angle > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        skew = np.array(
            [[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]],
            dtype=np.float64,
        )
        rotation = np.eye(3) + skew + skew @ skew * (1.0 / (1.0 + cos_angle))
    aligned = verts @ rotation.T
    return np.ptp(aligned, axis=0)


def _find_floor_plane(
    mesh: o3d.geometry.TriangleMesh,
    max_attempts: int = 12,
) -> tuple[np.ndarray, int]:
    """Find the lower room floor even when raw Z is not the vertical axis.

    The old heuristic selected the plane whose normal was closest to raw Z.
    That fails for phone/photogrammetry scans where raw Z can be the entrance
    direction.  Instead, infer the room-up axis as the shortest raw bbox axis,
    then choose a large plane near the lower side of that axis.
    """
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    bbox_min = verts.min(axis=0)
    bbox_extent = np.ptp(verts, axis=0)
    up_axis = int(np.argmin(bbox_extent))
    room_height_guess = float(bbox_extent[up_axis])

    pcd_clean = o3d.geometry.PointCloud()
    pcd_clean.points = mesh.vertices
    if _has_vertex_normals(mesh):
        pcd_clean.normals = mesh.vertex_normals

    remaining = pcd_clean
    candidates: list[tuple[float, np.ndarray, int, dict[str, float]]] = []
    total_vertices = max(1, len(verts))

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

        plane = np.asarray(plane_model, dtype=np.float64)
        normal = plane[:3].copy()
        normal /= np.linalg.norm(normal)
        axis_alignment = float(abs(normal[up_axis]))

        inlier_pts = np.asarray(remaining.points)[inlier_idx]
        plane_coord = float(np.median(inlier_pts[:, up_axis]))
        lower_fraction = float((plane_coord - bbox_min[up_axis]) / max(room_height_guess, 1e-6))
        aligned_extent = _candidate_alignment_extent(verts, plane, up_axis)
        aligned_height = float(aligned_extent[2])
        inlier_fraction = float(len(inlier_idx) / total_vertices)

        # Floor is a broad plane whose normal follows the inferred up axis and
        # whose coordinate is on the lower side of the room.  Penalize choosing
        # a side wall by preferring candidates that yield a plausible room
        # height after alignment.
        score = 0.0
        score += 4.0 * axis_alignment
        score += min(2.0, 50.0 * inlier_fraction)
        score += max(0.0, 2.0 * (0.45 - lower_fraction))
        if 1.8 <= aligned_height <= 3.4:
            score += 2.0
        else:
            score -= abs(aligned_height - 2.6)

        if axis_alignment > 0.75 and lower_fraction < 0.55:
            candidates.append((score, plane, len(inlier_idx), {
                "axis_alignment": axis_alignment,
                "lower_fraction": lower_fraction,
                "aligned_height": aligned_height,
            }))

        remaining = remaining.select_by_index(inlier_idx, invert=True)

    if not candidates:
        raise MeshProcessingError("RANSAC could not detect a lower floor plane in the scan.")

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1], up_axis


def clean_and_align(obj_path: Path) -> Path:
    """Load a scan, run SOR + RANSAC floor alignment, return cleaned PLY path.

    Supported inputs are OBJ/PLY triangle meshes and LAS/LAZ point clouds.

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
    """Load a scan, return (raw_geometry, cleaned_aligned_geometry).

    Both are Open3D TriangleMesh objects.  LAS/LAZ inputs are represented as
    vertex-only meshes (points + optional RGB colours, no triangle faces).
    """
    staged = clean_and_align_meshes_staged(obj_path)
    return staged["raw"], staged["manhattan"]


def clean_and_align_meshes_staged(
    obj_path: Path,
) -> dict[str, o3d.geometry.TriangleMesh]:
    """Load a scan and return every geometry stage in one consistent frame.

    Returned keys:
    - ``raw``: original loaded mesh, with OBJ textures baked to vertex colors.
    - ``sor``: statistical-outlier-cleaned mesh, before rotation.
    - ``aligned``: SOR mesh after floor/axis alignment, before Manhattan snap.
    - ``manhattan``: final cleaned mesh used by segmentation and voxelization.

    Export scripts and segmentation scripts should use this single staged
    function so every viewer asset is generated from the same room instance.
    """
    # Keep Open3D RANSAC deterministic across viewer export and segmentation.
    o3d.utility.random.seed(42)

    mesh = _load_scan_as_vertex_mesh(obj_path)
    raw_mesh = o3d.geometry.TriangleMesh(mesh)  # deep copy before cleanup

    # --- Statistical Outlier Removal on the vertex cloud ---
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    if _has_vertex_normals(mesh):
        pcd.normals = mesh.vertex_normals

    _, inlier_idx = pcd.remove_statistical_outlier(
        nb_neighbors=SOR_NB_NEIGHBORS,
        std_ratio=SOR_STD_RATIO,
    )

    mesh = _select_vertices(mesh, inlier_idx)
    if len(np.asarray(mesh.vertices)) == 0:
        raise MeshProcessingError(
            "Mesh is empty after outlier removal. The scan may be too noisy."
        )

    _ensure_vertex_normals(mesh)
    sor_mesh = o3d.geometry.TriangleMesh(mesh)

    # --- RANSAC floor detection ---
    best_plane, up_axis = _find_floor_plane(mesh)
    mesh = _align_floor_to_z0(mesh, best_plane, up_axis=up_axis)
    _ensure_vertex_normals(mesh)
    aligned_mesh = o3d.geometry.TriangleMesh(mesh)

    # --- Manhattan World Rectification ---
    mesh = _manhattan_rectify(mesh)

    return {
        "raw": raw_mesh,
        "sor": sor_mesh,
        "aligned": aligned_mesh,
        "manhattan": mesh,
    }


def _load_scan_as_vertex_mesh(scan_path: Path) -> o3d.geometry.TriangleMesh:
    """Load OBJ/PLY mesh or LAS/LAZ point cloud into a TriangleMesh container."""
    scan_path = Path(scan_path)
    suffix = scan_path.suffix.lower()
    if suffix in POINT_CLOUD_SUFFIXES:
        mesh = _load_las_as_vertex_mesh(scan_path)
    else:
        mesh = o3d.io.read_triangle_mesh(str(scan_path), enable_post_processing=True)
        if len(np.asarray(mesh.vertices)) == 0:
            raise MeshProcessingError("The uploaded mesh file contains no geometry.")
        _bake_texture_to_vertex_colors(mesh)

    _ensure_vertex_normals(mesh)
    return mesh


def _load_las_as_vertex_mesh(las_path: Path) -> o3d.geometry.TriangleMesh:
    """Read a LAS/LAZ point cloud while preserving RGB attributes when present."""
    try:
        import laspy  # type: ignore[import]
    except ImportError as exc:
        raise MeshProcessingError(
            "LAS/LAZ support requires laspy. Install with: pip install laspy lazrs"
        ) from exc

    las = laspy.read(str(las_path))
    points = np.column_stack((np.asarray(las.x), np.asarray(las.y), np.asarray(las.z))).astype(np.float64)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if len(points) == 0:
        raise MeshProcessingError("The LAS/LAZ file contains no finite point coordinates.")

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)

    dim_names = set(las.point_format.dimension_names)
    if {"red", "green", "blue"}.issubset(dim_names):
        rgb = np.column_stack((
            np.asarray(las.red),
            np.asarray(las.green),
            np.asarray(las.blue),
        )).astype(np.float64)[finite]
        divisor = 65535.0 if rgb.max(initial=0.0) > 255.0 else 255.0
        colors = np.clip(rgb / divisor, 0.0, 1.0)
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    elif "intensity" in dim_names:
        intensity = np.asarray(las.intensity, dtype=np.float64)[finite]
        max_i = float(intensity.max(initial=0.0))
        gray = (intensity / max_i) if max_i > 0 else np.full(len(points), 0.65)
        colors = np.repeat(gray[:, None], 3, axis=1)
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))

    return mesh


def _has_triangles(mesh: o3d.geometry.TriangleMesh) -> bool:
    return len(np.asarray(mesh.triangles)) > 0


def _has_vertex_normals(mesh: o3d.geometry.TriangleMesh) -> bool:
    return len(np.asarray(mesh.vertex_normals)) == len(np.asarray(mesh.vertices))


def _ensure_vertex_normals(mesh: o3d.geometry.TriangleMesh) -> None:
    """Compute normals for true meshes; leave point clouds colour-driven."""
    if _has_triangles(mesh):
        mesh.compute_vertex_normals()


def _select_vertices(
    mesh: o3d.geometry.TriangleMesh,
    indices: list[int] | np.ndarray,
) -> o3d.geometry.TriangleMesh:
    """Select vertices while preserving point-cloud colours/normals."""
    if _has_triangles(mesh):
        return mesh.select_by_index(indices)

    idx = np.asarray(indices, dtype=np.int64)
    result = o3d.geometry.TriangleMesh()
    result.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices)[idx])
    if mesh.has_vertex_colors():
        result.vertex_colors = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_colors)[idx])
    if _has_vertex_normals(mesh):
        result.vertex_normals = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_normals)[idx])
    return result


def _bake_texture_to_vertex_colors(mesh: o3d.geometry.TriangleMesh) -> None:
    """Bake OBJ texture UVs into per-vertex colors in-place.

    Open3D preserves OBJ textures as triangle UVs + image textures, but our
    segmentation renderer consumes vertex colors after SOR / alignment.  Phone
    scans commonly have no vertex colors, so without this bake DINO sees a flat
    grey mesh and cannot detect movable objects.
    """
    if mesh.has_vertex_colors() or not mesh.has_textures() or len(mesh.triangle_uvs) == 0:
        return

    triangles = np.asarray(mesh.triangles)
    triangle_uvs = np.asarray(mesh.triangle_uvs, dtype=np.float64)
    if len(triangles) == 0 or len(triangle_uvs) != len(triangles) * 3:
        return

    texture = np.asarray(mesh.textures[0])
    if texture.size == 0:
        return
    if texture.ndim == 2:
        texture = np.repeat(texture[:, :, None], 3, axis=2)
    texture = texture[:, :, :3].astype(np.float64) / 255.0
    height, width = texture.shape[:2]

    color_sum = np.zeros((len(mesh.vertices), 3), dtype=np.float64)
    color_count = np.zeros(len(mesh.vertices), dtype=np.float64)

    for tri_idx, tri in enumerate(triangles):
        for corner, vertex_idx in enumerate(tri):
            u, v = triangle_uvs[tri_idx * 3 + corner]
            x = int(np.clip(round(u * (width - 1)), 0, width - 1))
            # OBJ UV origin is bottom-left; image origin is top-left.
            y = int(np.clip(round((1.0 - v) * (height - 1)), 0, height - 1))
            color_sum[vertex_idx] += texture[y, x]
            color_count[vertex_idx] += 1.0

    valid = color_count > 0
    colors = np.full((len(mesh.vertices), 3), 0.6, dtype=np.float64)
    colors[valid] = color_sum[valid] / color_count[valid, None]
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))


def _manhattan_rectify(
    mesh: o3d.geometry.TriangleMesh,
    normal_tol_deg: float = MANHATTAN_NORMAL_TOL_DEG,
    max_planes: int = MANHATTAN_MAX_PLANES,
    plane_inlier_dist_m: float = MANHATTAN_PLANE_INLIER_DIST_M,
    min_plane_inlier_frac: float = MANHATTAN_MIN_PLANE_INLIER_FRAC,
) -> o3d.geometry.TriangleMesh:
    """Snap structural surface vertices to exact axis-aligned planes (Manhattan World).

    Unlike a normal-histogram approach, this uses iterative RANSAC to extract
    dominant planes one at a time.  Only geometric inliers of each plane
    (vertices within `plane_inlier_dist_m`, typically 3 cm) are snapped.
    Objects that protrude from walls (AC units, server racks, door frames,
    window curtains) are NOT inliers of the wall plane at 3 cm tolerance and
    therefore keep their original geometry.

    Algorithm
    ---------
    1. Build a point cloud from the mesh vertices.
    2. Iteratively call segment_plane() up to `max_planes` times.
    3. For each found plane: if it is axis-aligned (dominant normal within
       `normal_tol_deg` of ±X/Y/Z) AND has at least `min_plane_inlier_frac`
       of total vertices, snap all inliers to the median plane coordinate
       along the dominant axis.
    4. Remove inlier indices from the pool after each iteration so the same
       surface is not re-detected.
    """
    _ensure_vertex_normals(mesh)
    verts   = np.asarray(mesh.vertices).copy()
    normals = np.asarray(mesh.vertex_normals)
    n_total = len(verts)
    min_inliers = max(100, int(min_plane_inlier_frac * n_total))
    cos_tol = float(np.cos(np.radians(normal_tol_deg)))

    # Build a PointCloud for RANSAC plane fitting
    pcd = o3d.geometry.PointCloud()
    pcd.points  = o3d.utility.Vector3dVector(verts)
    if len(normals) == len(verts):
        pcd.normals = o3d.utility.Vector3dVector(normals)

    # Track which vertices are still available for plane extraction
    remaining_mask = np.ones(n_total, dtype=bool)

    for _ in range(max_planes):
        remaining_idx = np.where(remaining_mask)[0]
        if len(remaining_idx) < min_inliers * 3:
            break

        sub_pcd = pcd.select_by_index(remaining_idx.tolist())
        plane_model, inlier_local = sub_pcd.segment_plane(
            distance_threshold=plane_inlier_dist_m,
            ransac_n=3,
            num_iterations=500,
        )

        if plane_model is None or len(inlier_local) == 0:
            break

        local_arr  = np.asarray(inlier_local)
        global_idx = remaining_idx[local_arr]

        # Always evict these vertices so the same plane is not re-detected
        remaining_mask[global_idx] = False

        if len(global_idx) < min_inliers:
            continue  # too few — likely an object face, not a structural wall

        # Check axis alignment
        a, b, c, d = plane_model
        pn = np.array([a, b, c], dtype=np.float64)
        pn /= np.linalg.norm(pn)
        dominant_axis = int(np.argmax(np.abs(pn)))

        if np.abs(pn[dominant_axis]) < cos_tol:
            continue  # oblique surface — not a Manhattan plane, skip snapping

        # Snap all inliers to the median coordinate along the dominant axis.
        # The median is a robust estimator of the true plane position and avoids
        # the bias introduced by RANSAC's loose distance threshold.
        plane_coord = float(np.median(verts[global_idx, dominant_axis]))
        verts[global_idx, dominant_axis] = plane_coord

    result = o3d.geometry.TriangleMesh()
    result.vertices  = o3d.utility.Vector3dVector(verts)
    result.triangles = mesh.triangles
    if mesh.has_vertex_colors():
        result.vertex_colors = mesh.vertex_colors
    _ensure_vertex_normals(result)
    return result
