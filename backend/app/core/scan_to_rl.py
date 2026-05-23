"""Adapter between scan-domain voxel grids and the RL env's 2-D grid.

The CV pipeline produces a 3-D semantic grid at VOXEL_SIZE = 0.1 m with shape
up to (100, 100, 50). The RL env operates on a 2-D 50×50 grid at CELL_M =
0.2 m. These three helpers bridge the gap:

    voxel_grid_to_rl_obstacle      3-D semantic grid -> 50×50 obstacle map
    metadata_to_rl_cooling_pos     CoolingUnit list  -> (N, 2) cell indices
    rl_actions_to_rack_placements  RL [(x, y, dir)]  -> [RackPlacement] (world)
"""

from __future__ import annotations

import numpy as np

from engine.core.config import (
    COOLING_AC_VENT,
    DEFAULT_RACK_AIRFLOW_CFM,
    DEFAULT_RACK_POWER_KW,
    HEAT_LEGACY_SERVER,
    HUMAN_WORKSPACE,
    OBSTACLE_WALL,
    RACK_BODY,
    RACK_INTAKE,
    SPACE_EMPTY,
)
from engine.core.data_types import (
    Coordinate,
    CoolingUnit,
    RackPlacement,
    ScanMetadata,
)
from engine.rl.thermal_bridge import CELL_M, _DIR_TO_FACING

# A 2x2 block of CV voxels covers one RL cell (CELL_M / VOXEL_SIZE = 0.2/0.1 = 2).
_VOXELS_PER_CELL: int = int(round(CELL_M / 0.1))

# Voxel labels that block rack placement. RACK_INTAKE / RACK_EXHAUST / RACK_BODY
# are *not* in this set on the user-upload path — those are stamped later by the
# RL agent's chosen placements. AC vents are not obstacles either (they're on the
# ceiling). Workspaces and legacy servers ARE obstacles because the agent must
# route around them.
_OBSTACLE_LABELS: tuple[int, ...] = (
    OBSTACLE_WALL,
    HUMAN_WORKSPACE,
    HEAT_LEGACY_SERVER,
)


def voxel_grid_to_rl_obstacle(grid: np.ndarray, rl_grid_size: int = 50) -> np.ndarray:
    """Project a 3-D semantic voxel grid down to a 2-D 50×50 obstacle map.

    Every CELL_M × CELL_M floor patch is marked as an obstacle if *any*
    voxel in its 2×2 footprint (across all Z levels) carries a label in
    ``_OBSTACLE_LABELS``. This is the conservative behaviour the RL env
    expects: a column with a wall, desk, or legacy server anywhere along
    its height is impassable.

    Parameters
    ----------
    grid : np.ndarray
        Semantic voxel grid, shape (nx, ny, nz) with nx, ny up to 100.
    rl_grid_size : int, default 50
        Side length of the RL grid in cells. The output is padded /
        cropped to this size if the input grid does not fill it.

    Returns
    -------
    np.ndarray, shape (rl_grid_size, rl_grid_size), dtype float32
        Binary obstacle map (0 = free, 1 = blocked).
    """
    nx, ny, _ = grid.shape

    # Collapse over Z: a column is obstructed if it carries any obstacle label.
    blocked_xy = np.zeros((nx, ny), dtype=bool)
    for label in _OBSTACLE_LABELS:
        blocked_xy |= np.any(grid == label, axis=2)

    # Downsample 2:1 (voxel pitch 0.1 m -> RL cell pitch 0.2 m).
    nx_cells = nx // _VOXELS_PER_CELL
    ny_cells = ny // _VOXELS_PER_CELL
    trimmed = blocked_xy[: nx_cells * _VOXELS_PER_CELL, : ny_cells * _VOXELS_PER_CELL]
    blocks = trimmed.reshape(nx_cells, _VOXELS_PER_CELL, ny_cells, _VOXELS_PER_CELL)
    cells = blocks.any(axis=(1, 3))

    # Pad / crop to (rl_grid_size, rl_grid_size).
    out = np.zeros((rl_grid_size, rl_grid_size), dtype=np.float32)
    cx = min(nx_cells, rl_grid_size)
    cy = min(ny_cells, rl_grid_size)
    out[:cx, :cy] = cells[:cx, :cy].astype(np.float32)
    return out


def metadata_to_rl_cooling_pos(
    metadata: ScanMetadata,
    origin: np.ndarray,
    rl_grid_size: int = 50,
) -> np.ndarray:
    """Convert world-coord CoolingUnit positions to RL grid indices.

    Parameters
    ----------
    metadata : ScanMetadata
        Holds the user-supplied cooling unit list.
    origin : np.ndarray
        World-space offset of the voxel grid's [0, 0, 0] corner, in metres.
    rl_grid_size : int, default 50
        Side length of the RL grid; positions are clamped into this range.

    Returns
    -------
    np.ndarray, shape (N, 2), dtype int64
        (i, j) RL cell index per cooling unit. Returns ``(0, 2)`` if there
        are no cooling units (callers should treat this as an error).
    """
    if not metadata.cooling_units:
        return np.zeros((0, 2), dtype=np.int64)

    ox, oy = float(origin[0]), float(origin[1])
    positions = []
    for unit in metadata.cooling_units:
        gx = int(np.clip(round((unit.position.x - ox) / CELL_M), 0, rl_grid_size - 1))
        gy = int(np.clip(round((unit.position.y - oy) / CELL_M), 0, rl_grid_size - 1))
        positions.append((gx, gy))
    return np.array(positions, dtype=np.int64)


def rl_actions_to_rack_placements(
    actions: list[dict],
    origin: np.ndarray,
    rack_type: str = "42U",
    power_kw: float = DEFAULT_RACK_POWER_KW,
    airflow_cfm: float = DEFAULT_RACK_AIRFLOW_CFM,
) -> list[RackPlacement]:
    """Convert RL grid actions to world-coord RackPlacement objects.

    Mirrors ``ThermalBridge._to_rack_placements`` but takes the action dicts
    returned by ``RLService.optimize`` directly. The RL ``dir`` field is the
    exhaust direction; we translate to a solver ``RackFacing`` (intake side)
    via ``engine.rl.thermal_bridge._DIR_TO_FACING``.

    Parameters
    ----------
    actions : list[dict]
        Each entry must have keys ``x``, ``y`` (RL grid indices) and
        ``dir`` (RL exhaust direction 0–3).
    origin : np.ndarray
        World-space offset added to grid-cell-centre coords so the returned
        placements live in the same frame as the user's scan.
    rack_type, power_kw, airflow_cfm
        Defaults applied to every placed rack — the RL policy does not (yet)
        emit per-rack power, so we use the engine config defaults.

    Returns
    -------
    list[RackPlacement]
    """
    ox, oy, oz = float(origin[0]), float(origin[1]), float(origin[2])
    placements: list[RackPlacement] = []
    for action in actions:
        gx = int(action["x"])
        gy = int(action["y"])
        d = int(action["dir"])
        facing = _DIR_TO_FACING[d]
        wx = ox + (gx + 0.5) * CELL_M
        wy = oy + (gy + 0.5) * CELL_M
        placements.append(
            RackPlacement(
                position=Coordinate(x=wx, y=wy, z=oz),
                facing=facing,
                rack_type=rack_type,
                power_kw=power_kw,
                airflow_cfm=airflow_cfm,
                fixed=True,
            )
        )
    return placements
