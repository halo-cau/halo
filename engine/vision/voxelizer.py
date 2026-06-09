"""Voxelization and post-voxel metadata stamping.

This module is the *grid-side* of the CV pipeline. It runs AFTER mesh
semantic segmentation (Mask3D, in ``segmentor_*.py``) has already removed
movable objects from the input mesh.

Two distinct kinds of "label" exist in HALO; this file owns the second one:

* **Mesh vertex labels** — assigned per-vertex by Mask3D (wall, floor, server
  rack, box clutter, …) and used to strip movable geometry. Lives in the
  ``segmentor_*`` modules.
* **Voxel labels** — assigned per voxel by *stamping user-supplied metadata*
  (AC vents, legacy heat sources, racks, workspaces) onto the integer voxel
  grid. Lives in this module via :func:`voxelize_and_stamp_metadata`.

Calling these "labels" means two different things, so the public function is
named to make the source explicit: voxel labels are stamped from
``ScanMetadata``, not inferred."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Union

import numpy as np
import trimesh
from scipy.ndimage import binary_closing, gaussian_filter

if TYPE_CHECKING:  # pragma: no cover — typing-only import
    import open3d as o3d

VoxelizerInput = Union[Path, str, trimesh.Trimesh, "o3d.geometry.TriangleMesh"]

from engine.core.config import (
    AC_UNIT_DIMENSIONS,
    CLOSING_ITERATIONS,
    COOLING_AC_VENT,
    DEFAULT_RACK_TYPE,
    EXHAUST_DEPTH_VOXELS,
    GRID_SHAPE,
    HEAT_LEGACY_SERVER,
    HEAT_RADIUS_VOXELS,
    HEAT_SIGMA,
    HUMAN_WORKSPACE,
    INTAKE_DEPTH_VOXELS,
    MAX_ROOM_DIMENSIONS,
    OBSTACLE_WALL,
    RACK_BODY,
    RACK_DIMENSIONS,
    RACK_EXHAUST,
    RACK_INTAKE,
    SPACE_EMPTY,
    STAMP_RADIUS_VOXELS,
    VOXEL_SIZE,
    WORKSPACE_DIMENSIONS,
)
from engine.core.data_types import RackFacing, RackPlacement, ScanMetadata
from engine.core.exceptions import MeshProcessingError, RoomTooLargeError


def _check_bounds(mesh: trimesh.Trimesh) -> None:
    """Raise if the mesh bounding box exceeds MAX_ROOM_DIMENSIONS.

    Compares sorted extents against sorted limits so axis ordering
    doesn't matter (the cleaner may rotate the mesh).
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if len(vertices) == 0:
        raise MeshProcessingError("Mesh contains no vertices for voxelization.")

    raw_extents = np.ptp(vertices, axis=0)
    extents = sorted(raw_extents)
    max_dims = sorted(MAX_ROOM_DIMENSIONS)
    if any(e > m for e, m in zip(extents, max_dims, strict=True)):
        raise RoomTooLargeError(
            dimensions=(
                round(float(raw_extents[0]), 2),
                round(float(raw_extents[1]), 2),
                round(float(raw_extents[2]), 2),
            ),
            max_dims=MAX_ROOM_DIMENSIONS,
        )


def _surface_voxelize(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    """Step 2: Convert mesh surfaces to a 3D int8 grid (0=air, 1=wall).

    Returns (grid, origin) where origin is the world-space offset of index [0,0,0].
    """
    if len(mesh.faces) == 0:
        return _point_cloud_voxelize(mesh)

    voxel_grid = mesh.voxelized(pitch=VOXEL_SIZE)
    matrix = voxel_grid.matrix.astype(np.int8)
    origin = voxel_grid.transform[:3, 3]
    return matrix, origin


def _point_cloud_voxelize(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    """Voxelize vertex-only scans such as LAS/LAZ point clouds."""
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if len(vertices) == 0:
        raise MeshProcessingError("Point cloud contains no geometry for voxelization.")

    origin = np.floor(vertices.min(axis=0) / VOXEL_SIZE) * VOXEL_SIZE
    indices = np.floor((vertices - origin) / VOXEL_SIZE).astype(int)
    shape = tuple(indices.max(axis=0) + 1)
    grid = np.full(shape, SPACE_EMPTY, dtype=np.int8)
    grid[indices[:, 0], indices[:, 1], indices[:, 2]] = OBSTACLE_WALL
    return grid, origin


def _morphological_close(grid: np.ndarray) -> np.ndarray:
    """Step 3: Seal small holes in walls via binary closing."""
    wall_mask = grid == OBSTACLE_WALL
    closed = binary_closing(wall_mask, iterations=CLOSING_ITERATIONS)
    grid = grid.copy()
    grid[closed & (grid == SPACE_EMPTY)] = OBSTACLE_WALL
    return grid


def _world_to_index(
    x: float, y: float, z: float, origin: np.ndarray
) -> tuple[int, int, int]:
    """Convert real-world coordinates to voxel matrix indices."""
    idx = np.floor((np.array([x, y, z]) - origin) / VOXEL_SIZE).astype(int)
    return int(idx[0]), int(idx[1]), int(idx[2])


def _stamp_point(
    grid: np.ndarray,
    ix: int,
    iy: int,
    iz: int,
    label: int,
) -> None:
    """Write a single semantic label if the index is within bounds."""
    if 0 <= ix < grid.shape[0] and 0 <= iy < grid.shape[1] and 0 <= iz < grid.shape[2]:
        grid[ix, iy, iz] = label


def _stamp_sphere(
    grid: np.ndarray,
    cx: int,
    cy: int,
    cz: int,
    label: int,
    radius: int = STAMP_RADIUS_VOXELS,
) -> None:
    """Stamp a solid sphere of *label* centred at (cx, cy, cz)."""
    sx, sy, sz = grid.shape
    r = radius
    x0, x1 = max(cx - r, 0), min(cx + r + 1, sx)
    y0, y1 = max(cy - r, 0), min(cy + r + 1, sy)
    z0, z1 = max(cz - r, 0), min(cz + r + 1, sz)
    if x0 >= x1 or y0 >= y1 or z0 >= z1:
        return
    for x in range(x0, x1):
        for y in range(y0, y1):
            for z in range(z0, z1):
                if (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= r * r:
                    grid[x, y, z] = label


def _stamp_box(
    grid: np.ndarray,
    cx: int,
    cy: int,
    cz: int,
    label: int,
    dims_m: tuple[float, float, float],
) -> None:
    """Stamp a solid axis-aligned box of *label* centred at (cx, cy) on the floor (cz).

    dims_m is (width_x, depth_y, height_z) in metres.
    """
    vw = max(1, round(dims_m[0] / VOXEL_SIZE))
    vd = max(1, round(dims_m[1] / VOXEL_SIZE))
    vh = max(1, round(dims_m[2] / VOXEL_SIZE))
    sx, sy, sz = grid.shape
    x0 = max(cx - vw // 2, 0)
    x1 = min(x0 + vw, sx)
    y0 = max(cy - vd // 2, 0)
    y1 = min(y0 + vd, sy)
    z0 = max(cz, 0)
    z1 = min(cz + vh, sz)
    if x0 >= x1 or y0 >= y1 or z0 >= z1:
        return
    grid[x0:x1, y0:y1, z0:z1] = label


def _inject_heat(
    grid: np.ndarray,
    cx: int,
    cy: int,
    cz: int,
) -> None:
    """Stamp a 3D Gaussian heat region around a legacy server center."""
    r = HEAT_RADIUS_VOXELS
    sx, sy, sz = grid.shape

    x0, x1 = max(cx - r, 0), min(cx + r + 1, sx)
    y0, y1 = max(cy - r, 0), min(cy + r + 1, sy)
    z0, z1 = max(cz - r, 0), min(cz + r + 1, sz)

    # Center is completely outside the grid — nothing to stamp
    if x0 >= x1 or y0 >= y1 or z0 >= z1:
        return

    sub = np.zeros((x1 - x0, y1 - y0, z1 - z0), dtype=np.float64)
    local_cx, local_cy, local_cz = cx - x0, cy - y0, cz - z0
    sub[local_cx, local_cy, local_cz] = 1.0
    blurred = gaussian_filter(sub, sigma=HEAT_SIGMA)

    threshold = blurred.max() * 0.10
    heat_mask = blurred >= threshold

    region = grid[x0:x1, y0:y1, z0:z1]
    overwrite = heat_mask & ((region == SPACE_EMPTY) | (region == OBSTACLE_WALL))
    region[overwrite] = HEAT_LEGACY_SERVER


# Map per-vertex semantic labels (taxonomy from segmentor_base) to voxel ids.
# Wall / floor / ceiling map back to OBSTACLE_WALL so the projection only changes
# voxels for non-structural infrastructure (racks, AC, workspaces).
_VERTEX_LABEL_TO_VOXEL: dict[str, int] = {
    "server rack": RACK_BODY,
    "server cabinet": RACK_BODY,
    "rack cabinet": RACK_BODY,
    "network rack": RACK_BODY,
    "equipment rack": RACK_BODY,
    "cabinet": RACK_BODY,
    "ac_unit": COOLING_AC_VENT,
    "air conditioning unit": COOLING_AC_VENT,
    "AC unit": COOLING_AC_VENT,
    "human_workspace": HUMAN_WORKSPACE,
    "workspace": HUMAN_WORKSPACE,
    "desk": HUMAN_WORKSPACE,
}


def project_vertex_labels_to_voxels(
    grid: np.ndarray,
    origin: np.ndarray,
    vertices: np.ndarray,
    vertex_labels: list[str],
    *,
    radius_voxels: int = 2,
) -> np.ndarray:
    """Re-label OBSTACLE_WALL voxels using the nearest mesh vertex's label.

    Used to project the segmentor's per-vertex semantic labels onto the voxel
    grid so racks/AC/workspaces detected from the scan come out with their
    proper voxel ids (RACK_BODY / COOLING_AC_VENT / HUMAN_WORKSPACE) rather
    than collapsing into OBSTACLE_WALL. Structural labels (wall/floor/ceiling)
    have no entry in the lookup, so those voxels stay as walls.

    The function only touches voxels currently marked ``OBSTACLE_WALL`` — it
    never overwrites metadata-stamped voxels. Callers stamp user-supplied
    metadata after projection so operator overrides remain the final word.
    """
    if len(vertices) == 0 or len(vertex_labels) == 0:
        return grid

    wall_mask = grid == OBSTACLE_WALL
    if not wall_mask.any():
        return grid

    relevant_mask = np.array(
        [lbl in _VERTEX_LABEL_TO_VOXEL for lbl in vertex_labels],
        dtype=bool,
    )
    if not relevant_mask.any():
        return grid

    n = min(len(vertices), len(vertex_labels))
    kept_idx = np.where(relevant_mask[:n])[0]
    kept_verts = vertices[kept_idx]
    voxel_ids = np.fromiter(
        (_VERTEX_LABEL_TO_VOXEL[vertex_labels[i]] for i in kept_idx),
        dtype=np.int8,
        count=len(kept_idx),
    )

    from scipy.spatial import cKDTree

    tree = cKDTree(kept_verts)
    wall_idx = np.argwhere(wall_mask)
    wall_centers = (wall_idx + 0.5) * VOXEL_SIZE + origin
    radius_m = float(radius_voxels) * VOXEL_SIZE
    dist, nearest = tree.query(wall_centers, k=1, distance_upper_bound=radius_m)

    hit = np.isfinite(dist)
    if not hit.any():
        return grid

    grid = grid.copy()
    updates = wall_idx[hit]
    grid[updates[:, 0], updates[:, 1], updates[:, 2]] = voxel_ids[nearest[hit]]
    return grid


# Canonical priors per detected label: (voxel_id, base dims in metres W×D×H).
# Detected racks are forced to 42U so their voxel representation is a solid
# block instead of a hollow point-cloud shell. ACs use a wall-mounted slab.
_LABEL_TO_PRIOR: dict[str, tuple[int, tuple[float, float, float]]] = {
    "server rack":           (RACK_BODY,        RACK_DIMENSIONS[DEFAULT_RACK_TYPE]),
    "server cabinet":        (RACK_BODY,        RACK_DIMENSIONS[DEFAULT_RACK_TYPE]),
    "rack cabinet":          (RACK_BODY,        RACK_DIMENSIONS[DEFAULT_RACK_TYPE]),
    "network rack":          (RACK_BODY,        RACK_DIMENSIONS[DEFAULT_RACK_TYPE]),
    "equipment rack":        (RACK_BODY,        RACK_DIMENSIONS[DEFAULT_RACK_TYPE]),
    "cabinet":               (RACK_BODY,        RACK_DIMENSIONS[DEFAULT_RACK_TYPE]),
    "ac_unit":               (COOLING_AC_VENT,  AC_UNIT_DIMENSIONS),
    "air conditioning unit": (COOLING_AC_VENT,  AC_UNIT_DIMENSIONS),
    "AC unit":               (COOLING_AC_VENT,  AC_UNIT_DIMENSIONS),
    "human_workspace":       (HUMAN_WORKSPACE,  WORKSPACE_DIMENSIONS),
    "workspace":             (HUMAN_WORKSPACE,  WORKSPACE_DIMENSIONS),
    "desk":                  (HUMAN_WORKSPACE,  WORKSPACE_DIMENSIONS),
}


def _cluster_detected_instances(
    vertices: np.ndarray,
    vertex_labels: list[str],
    *,
    eps_m: float = 0.30,
    min_points: int = 30,
) -> list[dict]:
    """Group vertices by label and DBSCAN-cluster each group into instances.

    Returns one dict per cluster with ``label``, ``bounds_min``, ``bounds_max``,
    ``n_points``. Only labels in ``_LABEL_TO_PRIOR`` are considered.
    """
    import open3d as o3d

    arr_labels = np.asarray(vertex_labels, dtype=object)
    instances: list[dict] = []
    for label in _LABEL_TO_PRIOR:
        mask = arr_labels == label
        if int(mask.sum()) < min_points:
            continue
        pts = vertices[mask]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        cluster_ids = np.asarray(
            pcd.cluster_dbscan(eps=eps_m, min_points=min_points, print_progress=False),
            dtype=np.int32,
        )
        for cid in sorted(int(c) for c in np.unique(cluster_ids) if c >= 0):
            member = pts[cluster_ids == cid]
            if len(member) < min_points:
                continue
            mn = member.min(axis=0)
            mx = member.max(axis=0)
            instances.append({
                "label": label,
                "bounds_min": mn.tolist(),
                "bounds_max": mx.tolist(),
                "n_points": int(len(member)),
            })
    return instances


def _stamp_detected_priors(
    grid: np.ndarray,
    origin: np.ndarray,
    instances: list[dict],
) -> np.ndarray:
    """Stamp each detected instance as a solid canonical-size box.

    Force-applies the per-label prior (42U racks, fixed AC slab, workspace dims)
    so detected objects are filled volumes, not hollow surface shells. The box
    is positioned at the instance's XY centre and sits with its base at the
    instance's min-Z (so racks remain on the floor and AC units stay at scan
    height). For rack-like and AC-like labels we orient depth along the
    detected longer horizontal extent.
    """
    grid = grid.copy()
    for inst in instances:
        label = inst.get("label", "")
        prior = _LABEL_TO_PRIOR.get(label)
        if prior is None:
            continue
        voxel_id, base_dims = prior

        mn = np.asarray(inst["bounds_min"], dtype=np.float64)
        mx = np.asarray(inst["bounds_max"], dtype=np.float64)
        cx_w, cy_w = (mn[0] + mx[0]) * 0.5, (mn[1] + mx[1]) * 0.5
        z_bottom_w = float(mn[2])

        ix, iy, _ = _world_to_index(cx_w, cy_w, 0.0, origin)
        _, _, iz = _world_to_index(0.0, 0.0, z_bottom_w, origin)

        # Orient depth along the detected longer horizontal extent.
        ext = mx - mn
        if ext[0] > ext[1]:
            dims = (base_dims[1], base_dims[0], base_dims[2])
        else:
            dims = base_dims

        _stamp_box(grid, ix, iy, iz, voxel_id, dims)
    return grid


def _stamp_metadata_labels(
    grid: np.ndarray,
    metadata: ScanMetadata,
    origin: np.ndarray,
) -> np.ndarray:
    """Stamp voxel labels from user-supplied ``ScanMetadata``.

    This is the voxel-level labeling pass. It runs AFTER voxelization and
    assigns semantic IDs (AC vent, legacy heat source, rack body/intake/
    exhaust, human workspace) to voxels based on real-world coordinates the
    user supplied — not based on any AI inference.
    """
    grid = grid.copy()

    for unit in metadata.cooling_units:
        ix, iy, iz = _world_to_index(unit.position.x, unit.position.y, unit.position.z, origin)
        _stamp_sphere(grid, ix, iy, iz, COOLING_AC_VENT)

    for ws in metadata.human_workspaces:
        ix, iy, iz = _world_to_index(ws.x, ws.y, ws.z, origin)
        _stamp_box(grid, ix, iy, iz, HUMAN_WORKSPACE, WORKSPACE_DIMENSIONS)

    for server in metadata.legacy_servers:
        ix, iy, iz = _world_to_index(server.x, server.y, server.z, origin)
        _inject_heat(grid, ix, iy, iz)

    for rack in metadata.racks:
        _stamp_rack(grid, rack, origin)

    return grid


def _stamp_rack(
    grid: np.ndarray,
    rack: RackPlacement,
    origin: np.ndarray,
) -> None:
    """Stamp a full server rack into the voxel grid.

    The rack is placed so that ``rack.position`` is the front-bottom-center.
    The front face (intake) is a 1-voxel slab facing ``rack.facing``.
    The rear face (exhaust) is the opposite slab.  Everything between
    is ``RACK_BODY``.

    ASHRAE rack dimensions are looked up from RACK_DIMENSIONS.
    """
    dims = RACK_DIMENSIONS.get(rack.rack_type)
    if dims is None:
        return
    rack_w, rack_d, rack_h = dims  # width, depth, height in metres

    # Convert rack dimensions to voxel counts
    vw = max(1, round(rack_w / VOXEL_SIZE))  # width in voxels
    vd = max(1, round(rack_d / VOXEL_SIZE))  # depth in voxels
    vh = max(1, round(rack_h / VOXEL_SIZE))  # height in voxels

    # Position is front-bottom-center in world space.
    pos = rack.position
    cx, cy, cz = _world_to_index(pos.x, pos.y, pos.z, origin)

    # Build axis-aligned bounding box depending on facing direction.
    # Facing = direction the front (intake) looks towards.
    # The rack body extends *behind* the front face.
    facing = rack.facing
    half_w = vw // 2

    if facing == RackFacing.PLUS_X:
        # Front at max-X, body extends toward -X
        x0, x1 = cx - vd + 1, cx + 1
        y0, y1 = cy - half_w, cy - half_w + vw
        intake_axis, intake_side = 0, "max"  # X-axis, high side
    elif facing == RackFacing.MINUS_X:
        x0, x1 = cx, cx + vd
        y0, y1 = cy - half_w, cy - half_w + vw
        intake_axis, intake_side = 0, "min"
    elif facing == RackFacing.PLUS_Y:
        x0, x1 = cx - half_w, cx - half_w + vw
        y0, y1 = cy - vd + 1, cy + 1
        intake_axis, intake_side = 1, "max"
    elif facing == RackFacing.MINUS_Y:
        x0, x1 = cx - half_w, cx - half_w + vw
        y0, y1 = cy, cy + vd
        intake_axis, intake_side = 1, "min"
    else:
        return

    z0, z1 = cz, cz + vh

    # Clamp to grid bounds
    sx, sy, sz = grid.shape
    x0c, x1c = max(x0, 0), min(x1, sx)
    y0c, y1c = max(y0, 0), min(y1, sy)
    z0c, z1c = max(z0, 0), min(z1, sz)
    if x0c >= x1c or y0c >= y1c or z0c >= z1c:
        return

    # Fill entire rack volume with RACK_BODY
    grid[x0c:x1c, y0c:y1c, z0c:z1c] = RACK_BODY

    # Stamp INTAKE slab (front face)
    id_ = INTAKE_DEPTH_VOXELS
    if intake_axis == 0:
        if intake_side == "max":
            si = max(x1c - id_, x0c)
            grid[si:x1c, y0c:y1c, z0c:z1c] = RACK_INTAKE
        else:
            ei = min(x0c + id_, x1c)
            grid[x0c:ei, y0c:y1c, z0c:z1c] = RACK_INTAKE
    else:
        if intake_side == "max":
            si = max(y1c - id_, y0c)
            grid[x0c:x1c, si:y1c, z0c:z1c] = RACK_INTAKE
        else:
            ei = min(y0c + id_, y1c)
            grid[x0c:x1c, y0c:ei, z0c:z1c] = RACK_INTAKE

    # Stamp EXHAUST slab (rear face — opposite side)
    ed = EXHAUST_DEPTH_VOXELS
    if intake_axis == 0:
        if intake_side == "max":  # exhaust at min-X
            ei = min(x0c + ed, x1c)
            grid[x0c:ei, y0c:y1c, z0c:z1c] = RACK_EXHAUST
        else:  # exhaust at max-X
            si = max(x1c - ed, x0c)
            grid[si:x1c, y0c:y1c, z0c:z1c] = RACK_EXHAUST
    else:
        if intake_side == "max":  # exhaust at min-Y
            ei = min(y0c + ed, y1c)
            grid[x0c:x1c, y0c:ei, z0c:z1c] = RACK_EXHAUST
        else:  # exhaust at max-Y
            si = max(y1c - ed, y0c)
            grid[x0c:x1c, si:y1c, z0c:z1c] = RACK_EXHAUST


def _to_trimesh(source: VoxelizerInput) -> trimesh.Trimesh:
    """Coerce any supported input into a trimesh.Trimesh.

    Supports filesystem paths (anything trimesh.load handles), trimesh.Trimesh
    instances (returned as-is), and Open3D TriangleMesh objects (converted
    directly via vertex / triangle arrays — no disk IO).
    """
    if isinstance(source, trimesh.Trimesh):
        return source
    if isinstance(source, (str, Path)):
        return trimesh.load(str(source), force="mesh")
    # Treat anything else as an Open3D mesh (duck-typed to avoid a hard
    # dependency on open3d at import time).
    vertices = getattr(source, "vertices", None)
    triangles = getattr(source, "triangles", None)
    if vertices is None or triangles is None:
        raise TypeError(
            f"Unsupported voxelizer input type: {type(source).__name__}"
        )
    return trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(triangles, dtype=np.int64),
        process=False,
    )


def _build_layout_grid(shape: tuple[int, int, int], dtype) -> np.ndarray:
    """Return a perfect shell-only cuboid: 1-voxel walls on all six faces, air inside.

    Used to expose the room layout (floor + walls + ceiling) with every interior
    voxel labeled ``SPACE_EMPTY``. Shares its shape and origin with the main
    grid so the two can be rendered in the same world frame.
    """
    layout = np.full(shape, SPACE_EMPTY, dtype=dtype)
    layout[ 0, :, :] = OBSTACLE_WALL
    layout[-1, :, :] = OBSTACLE_WALL
    layout[:,  0, :] = OBSTACLE_WALL
    layout[:, -1, :] = OBSTACLE_WALL
    layout[:, :,  0] = OBSTACLE_WALL  # floor
    layout[:, :, -1] = OBSTACLE_WALL  # ceiling
    return layout


def voxelize_and_stamp_metadata(
    source: VoxelizerInput,
    metadata: ScanMetadata,
    *,
    segmentor_labels: tuple[np.ndarray, list[str]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Voxelize an (already-segmented) mesh and stamp metadata labels onto it.

    Pipeline contract:
      1. Surface-voxelize the input mesh into an int8 grid (wall / empty).
      2. Morphologically close the wall mask to seal small scan holes.
      3. Optionally cluster per-vertex segmentor labels into per-object
         instances and stamp each as a *solid* canonical-size box (42U
         racks, fixed AC slab, workspace dims) via
         :func:`_stamp_detected_priors`. This fills the interior of every
         detected object so a sparse point-cloud scan never produces hollow
         rack/AC voxels.
      4. Stamp voxel labels from ``ScanMetadata`` (AC vents, legacy heat,
         racks, workspaces). User-supplied metadata is the final word so
         operator edits always override projected segmentor labels.

    The input mesh is expected to have movable objects already removed by
    the upstream segmentor. If you pass an unsegmented mesh, those movables
    will end up baked into the wall mask.

    Accepts a filesystem path, a ``trimesh.Trimesh``, or an
    ``open3d.geometry.TriangleMesh``. Path inputs are loaded with trimesh;
    Open3D meshes are converted directly without round-tripping through disk.

    ``segmentor_labels`` is an optional ``(vertices, labels)`` pair where
    ``vertices`` is an ``(N, 3)`` world-coords array and ``labels`` is a
    list of canonical segmentor labels for each vertex.

    Returns ``(grid, layout_grid, origin)``:
      * ``grid`` — final int8 ndarray with stamped infrastructure.
      * ``layout_grid`` — same shape/origin, shell-only cuboid (walls + air).
      * ``origin`` — world-space offset of index ``[0, 0, 0]``.
    """
    mesh = _to_trimesh(source)
    if mesh.is_empty:
        raise MeshProcessingError("Cleaned mesh contains no geometry for voxelization.")

    _check_bounds(mesh)

    grid, origin = _surface_voxelize(mesh)
    grid = _morphological_close(grid)
    layout_grid = _build_layout_grid(grid.shape, grid.dtype)
    if segmentor_labels is not None:
        vertices, labels = segmentor_labels
        instances = _cluster_detected_instances(vertices, labels)
        grid = _stamp_detected_priors(grid, origin, instances)
    grid = _stamp_metadata_labels(grid, metadata, origin)

    return grid, layout_grid, origin


def stamp_rack_on_grid(
    grid: np.ndarray,
    rack: RackPlacement,
    origin: np.ndarray,
) -> None:
    """Public alias for :func:`_stamp_rack` — in-place rack stamping.

    Use this from the optimize endpoint (and any external caller) to overlay
    RL-chosen rack placements onto a cached scan grid before invoking the
    thermal solver. The function modifies ``grid`` in place.
    """
    _stamp_rack(grid, rack, origin)


def pad_to_fixed_shape(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Zero-pad a voxel grid to GRID_SHAPE, centering the room.

    Returns (padded_grid, offset) where offset is the (x, y, z) index
    at which the original grid starts inside the padded grid.
    """
    gx, gy, gz = grid.shape
    fx, fy, fz = GRID_SHAPE

    if gx > fx or gy > fy or gz > fz:
        raise MeshProcessingError(
            f"Grid shape {grid.shape} exceeds fixed shape {GRID_SHAPE}. "
            "Room too large for the RL observation space."
        )

    # Center the room in the padded grid
    ox = (fx - gx) // 2
    oy = (fy - gy) // 2
    oz = 0  # Keep floor at the bottom of the Z axis

    padded = np.full(GRID_SHAPE, SPACE_EMPTY, dtype=grid.dtype)
    padded[ox : ox + gx, oy : oy + gy, oz : oz + gz] = grid

    return padded, (ox, oy, oz)
