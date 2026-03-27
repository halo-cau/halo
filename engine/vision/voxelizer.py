"""Step 2-4: Surface voxelization, morphological closing, and semantic fusion."""

from pathlib import Path

import numpy as np
import trimesh
from scipy.ndimage import binary_closing, gaussian_filter

from engine.core.config import (
    CLOSING_ITERATIONS,
    COOLING_AC_VENT,
    HEAT_LEGACY_SERVER,
    HEAT_RADIUS_VOXELS,
    HEAT_SIGMA,
    HUMAN_WORKSPACE,
    MAX_ROOM_DIMENSIONS,
    OBSTACLE_WALL,
    SPACE_EMPTY,
    VOXEL_SIZE,
)
from engine.core.data_types import ScanMetadata
from engine.core.exceptions import MeshProcessingError, RoomTooLargeError


def _check_bounds(mesh: trimesh.Trimesh) -> None:
    """Raise if the mesh bounding box exceeds MAX_ROOM_DIMENSIONS."""
    extents = mesh.bounding_box.extents
    max_dims = MAX_ROOM_DIMENSIONS
    if extents[0] > max_dims[0] or extents[1] > max_dims[1] or extents[2] > max_dims[2]:
        raise RoomTooLargeError(
            dimensions=(
                round(extents[0], 2),
                round(extents[1], 2),
                round(extents[2], 2),
            ),
            max_dims=max_dims,
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

    for vent in metadata.ac_vents:
        ix, iy, iz = _world_to_index(vent.x, vent.y, vent.z, origin)
        _stamp_point(grid, ix, iy, iz, COOLING_AC_VENT)

    for ws in metadata.human_workspaces:
        ix, iy, iz = _world_to_index(ws.x, ws.y, ws.z, origin)
        _stamp_point(grid, ix, iy, iz, HUMAN_WORKSPACE)

    for server in metadata.legacy_servers:
        ix, iy, iz = _world_to_index(server.x, server.y, server.z, origin)
        _inject_heat(grid, ix, iy, iz)

    return grid


def voxelize_and_label(
    ply_path: Path,
    metadata: ScanMetadata,
) -> np.ndarray:
    """Run Steps 2-4: voxelization → closing → semantic fusion.

    Returns the final int8 ndarray (shape: [X, Y, Z]).
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

    return grid
