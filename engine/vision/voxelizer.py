"""Step 2-4: Surface voxelization, morphological closing, and semantic fusion."""

from pathlib import Path

import numpy as np
import trimesh
from scipy.ndimage import binary_closing, gaussian_filter

from engine.core.config import (
    CLOSING_ITERATIONS,
    COOLING_AC_VENT,
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
)
from engine.core.data_types import RackFacing, RackPlacement, ScanMetadata
from engine.core.exceptions import MeshProcessingError, RoomTooLargeError


def _check_bounds(mesh: trimesh.Trimesh) -> None:
    """Raise if the mesh bounding box exceeds MAX_ROOM_DIMENSIONS.

    Compares sorted extents against sorted limits so axis ordering
    doesn't matter (the cleaner may rotate the mesh).
    """
    extents = sorted(mesh.bounding_box.extents)
    max_dims = sorted(MAX_ROOM_DIMENSIONS)
    if any(e > m for e, m in zip(extents, max_dims, strict=True)):
        raise RoomTooLargeError(
            dimensions=(
                round(mesh.bounding_box.extents[0], 2),
                round(mesh.bounding_box.extents[1], 2),
                round(mesh.bounding_box.extents[2], 2),
            ),
            max_dims=MAX_ROOM_DIMENSIONS,
        )


def _surface_voxelize(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    """Step 2: Convert mesh surfaces to a 3D int8 grid (0=air, 1=wall).

    Returns (grid, origin) where origin is the world-space offset of index [0,0,0].
    """
    voxel_grid = mesh.voxelized(pitch=VOXEL_SIZE)
    matrix = voxel_grid.matrix.astype(np.int8)
    origin = voxel_grid.transform[:3, 3]
    return matrix, origin


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


def fuse_semantics(
    grid: np.ndarray,
    metadata: ScanMetadata,
    origin: np.ndarray,
) -> np.ndarray:
    """Step 4: Stamp semantic labels onto the voxel grid from metadata."""
    grid = grid.copy()

    for unit in metadata.cooling_units:
        ix, iy, iz = _world_to_index(unit.position.x, unit.position.y, unit.position.z, origin)
        _stamp_sphere(grid, ix, iy, iz, COOLING_AC_VENT)

    for ws in metadata.human_workspaces:
        ix, iy, iz = _world_to_index(ws.x, ws.y, ws.z, origin)
        _stamp_sphere(grid, ix, iy, iz, HUMAN_WORKSPACE)

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


def voxelize_and_label(
    ply_path: Path,
    metadata: ScanMetadata,
) -> tuple[np.ndarray, np.ndarray]:
    """Run Steps 2-4: voxelization → closing → semantic fusion.

    Returns (grid, origin) where grid is the final int8 ndarray
    (shape matches the actual room) and origin is the world-space
    offset of index [0,0,0].
    """
    mesh = trimesh.load(str(ply_path), force="mesh")
    if mesh.is_empty:
        raise MeshProcessingError("Cleaned mesh contains no geometry for voxelization.")

    _check_bounds(mesh)

    # Step 2
    grid, origin = _surface_voxelize(mesh)

    # Step 3
    grid = _morphological_close(grid)

    # Step 4
    grid = fuse_semantics(grid, metadata, origin)

    return grid, origin


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
