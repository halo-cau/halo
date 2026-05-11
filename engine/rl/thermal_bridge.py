"""Adapter between the RL environment's 2-D grid and the 3-D thermal engine.

Why this exists
---------------
DataCenterEnv uses a compact 2-D 50×50 grid with a lightweight thermal
model (Gaussian jets + bilinear advection) for training speed.  That model
diverges from the physics-based 3-D solver in ``engine.thermal.solver`` in
dimensionality, temperature scale, advection scheme, and source/sink models.

This module bridges the gap: it translates RL environment state into the
inputs expected by ``compute_thermal_field`` and ``compute_metrics``, runs
the authoritative 3-D solver, and returns an ``MetricsResult`` that the
reward function can trust.

Physical mapping
----------------
Each RL grid cell corresponds to 0.4 m × 0.4 m of floor space (CELL_M).
A 50×50 grid therefore covers a 20 m × 20 m room — identical to the
voxel engine's default GRID_SHAPE of 200×200 at VOXEL_SIZE=0.1 m.

Ceiling height defaults to 3.0 m (30 voxels).  Each RL cooling position
maps to a ceiling-mounted CRAC blowing straight down.

Direction mapping (RL exhaust direction → RackFacing intake direction)
----------------------------------------------------------------------
RL dirs: 0=(+x exhaust), 1=(−x exhaust), 2=(+y exhaust), 3=(−y exhaust)
Solver RackFacing is the *intake* direction (opposite of exhaust):
    RL dir 0 → RackFacing.MINUS_X
    RL dir 1 → RackFacing.PLUS_X
    RL dir 2 → RackFacing.MINUS_Y
    RL dir 3 → RackFacing.PLUS_Y
"""

from __future__ import annotations

import numpy as np

from engine.core.config import (
    COOLING_AC_VENT,
    DEFAULT_AC_AIRFLOW_CFM,
    DEFAULT_AC_CAPACITY_KW,
    DEFAULT_AC_SUPPLY_TEMP_C,
    DEFAULT_RACK_AIRFLOW_CFM,
    DEFAULT_RACK_POWER_KW,
    OBSTACLE_WALL,
    RACK_BODY,
    RACK_DIMENSIONS,
    RACK_EXHAUST,
    RACK_INTAKE,
    SPACE_EMPTY,
    VOXEL_SIZE,
)
from engine.core.data_types import Coordinate, CoolingUnit, RackFacing, RackPlacement
from engine.thermal.metrics import MetricsResult, compute_metrics
from engine.thermal.solver import compute_thermal_field

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
CELL_M: float = 0.4          # metres per RL grid cell (50 cells → 20 m room)
CEILING_M: float = 3.0       # room ceiling height in metres
CEILING_V: int = int(round(CEILING_M / VOXEL_SIZE))   # = 30 voxels

# Voxels per RL cell along each horizontal axis
_CELL_V: int = int(round(CELL_M / VOXEL_SIZE))        # = 4

# Direction → RackFacing (RL exhaust direction → solver intake direction)
_DIR_TO_FACING: dict[int, RackFacing] = {
    0: RackFacing.MINUS_X,   # RL exhaust +x → intake −x
    1: RackFacing.PLUS_X,    # RL exhaust −x → intake +x
    2: RackFacing.MINUS_Y,   # RL exhaust +y → intake −y
    3: RackFacing.PLUS_Y,    # RL exhaust −y → intake +y
}

# Fixed facing → (axis, sign) for exhaust — mirrors solver._facing_to_axis_dir
_FACING_AXIS_SIGN: dict[RackFacing, tuple[int, int]] = {
    RackFacing.PLUS_X:  (0, -1),
    RackFacing.MINUS_X: (0, +1),
    RackFacing.PLUS_Y:  (1, -1),
    RackFacing.MINUS_Y: (1, +1),
}


class ThermalBridge:
    """Translates a DataCenterEnv snapshot into a 3-D thermal solve.

    Parameters
    ----------
    grid_size : int
        Side length of the RL grid (default 50).
    rack_type : str
        Rack model key used for all placed racks (default "42U").
    power_kw : float
        Per-rack thermal dissipation in kW.
    airflow_cfm : float
        Per-rack fan airflow in CFM.
    """

    def __init__(
        self,
        grid_size: int = 50,
        rack_type: str = "42U",
        power_kw: float = DEFAULT_RACK_POWER_KW,
        airflow_cfm: float = DEFAULT_RACK_AIRFLOW_CFM,
    ) -> None:
        self.grid_size = grid_size
        self.rack_type = rack_type
        self.power_kw = power_kw
        self.airflow_cfm = airflow_cfm

        # Pre-compute voxel grid shape: (X, Y, Z)
        self._nx = grid_size * _CELL_V        # 50 * 4 = 200
        self._ny = grid_size * _CELL_V        # 200
        self._nz = CEILING_V                  # 30
        self._origin = np.zeros(3, dtype=np.float64)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve_metrics(
        self,
        rack_map: np.ndarray,
        rack_dir: np.ndarray,
        obstacle: np.ndarray,
        cooling_pos: np.ndarray,
    ) -> MetricsResult | None:
        """Run the 3-D thermal engine and return ASHRAE metrics.

        Returns ``None`` if no racks have been placed yet.

        Parameters
        ----------
        rack_map : np.ndarray (grid_size, grid_size)
            1 where a rack occupies that cell.
        rack_dir : np.ndarray (grid_size, grid_size)
            Direction integer 0-3 for each rack cell.
        obstacle : np.ndarray (grid_size, grid_size)
            1 where the cell is an obstacle.
        cooling_pos : np.ndarray (N, 2)
            Grid indices (i, j) of each cooling unit.
        """
        rack_cells = np.argwhere(rack_map == 1)
        if len(rack_cells) == 0:
            return None

        racks = self._to_rack_placements(rack_cells, rack_dir)
        cooling_units = self._to_cooling_units(cooling_pos)
        semantic_grid = self._build_semantic_grid(obstacle, racks, cooling_pos)

        temp_3d = compute_thermal_field(
            semantic_grid, racks, self._origin, cooling_units,
        )
        return compute_metrics(semantic_grid, temp_3d, racks, self._origin)

    # ------------------------------------------------------------------
    # Internal builders
    # ------------------------------------------------------------------

    def _to_rack_placements(
        self,
        rack_cells: np.ndarray,
        rack_dir: np.ndarray,
    ) -> list[RackPlacement]:
        """Convert RL grid rack positions to RackPlacement objects."""
        placements: list[RackPlacement] = []
        for gx, gy in rack_cells:
            facing = _DIR_TO_FACING[int(rack_dir[gx, gy])]
            # World position = centre of the RL cell, at floor level
            wx = (gx + 0.5) * CELL_M
            wy = (gy + 0.5) * CELL_M
            placements.append(
                RackPlacement(
                    position=Coordinate(x=wx, y=wy, z=0.0),
                    facing=facing,
                    rack_type=self.rack_type,
                    power_kw=self.power_kw,
                    airflow_cfm=self.airflow_cfm,
                    fixed=True,
                )
            )
        return placements

    def _to_cooling_units(self, cooling_pos: np.ndarray) -> list[CoolingUnit]:
        """Map RL cooling positions to ceiling-mounted CoolingUnit objects."""
        units: list[CoolingUnit] = []
        for gx, gy in cooling_pos:
            wx = (float(gx) + 0.5) * CELL_M
            wy = (float(gy) + 0.5) * CELL_M
            units.append(
                CoolingUnit(
                    position=Coordinate(x=wx, y=wy, z=CEILING_M),
                    capacity_kw=DEFAULT_AC_CAPACITY_KW,
                    supply_direction=(0.0, 0.0, -1.0),   # blowing downward
                    supply_temp_c=DEFAULT_AC_SUPPLY_TEMP_C,
                    airflow_cfm=DEFAULT_AC_AIRFLOW_CFM,
                )
            )
        return units

    def _build_semantic_grid(
        self,
        obstacle: np.ndarray,
        racks: list[RackPlacement],
        cooling_pos: np.ndarray,
    ) -> np.ndarray:
        """Build the 3-D semantic voxel grid expected by the solver.

        Labels assigned:
            OBSTACLE_WALL  – perimeter walls + RL obstacle cells
            COOLING_AC_VENT – 1-voxel column at each cooler's floor position
            RACK_INTAKE    – intake face of each rack
            RACK_EXHAUST   – exhaust face of each rack
            RACK_BODY      – interior rack voxels
            SPACE_EMPTY    – everything else (air)

        Perimeter walls are stamped on all four sides so the solver has a
        closed boundary (avoids airflow escaping at grid edges).
        """
        grid = np.full(
            (self._nx, self._ny, self._nz), SPACE_EMPTY, dtype=np.int8,
        )

        # ── Perimeter walls ───────────────────────────────────────────
        grid[0, :, :] = OBSTACLE_WALL
        grid[-1, :, :] = OBSTACLE_WALL
        grid[:, 0, :] = OBSTACLE_WALL
        grid[:, -1, :] = OBSTACLE_WALL

        # ── RL obstacle cells ─────────────────────────────────────────
        obs_cells = np.argwhere(obstacle == 1)
        for gx, gy in obs_cells:
            vx0, vx1 = gx * _CELL_V, (gx + 1) * _CELL_V
            vy0, vy1 = gy * _CELL_V, (gy + 1) * _CELL_V
            grid[vx0:vx1, vy0:vy1, :] = OBSTACLE_WALL

        # ── Cooling AC vent markers ───────────────────────────────────
        for gx, gy in cooling_pos:
            vx = int(round((float(gx) + 0.5) * _CELL_V))
            vy = int(round((float(gy) + 0.5) * _CELL_V))
            vx = int(np.clip(vx, 0, self._nx - 1))
            vy = int(np.clip(vy, 0, self._ny - 1))
            # Mark a 2×2 column at the ceiling as AC vent
            for dx in range(2):
                for dy in range(2):
                    xi, yi = vx + dx, vy + dy
                    if 0 <= xi < self._nx and 0 <= yi < self._ny:
                        grid[xi, yi, self._nz - 1] = COOLING_AC_VENT

        # ── Rack voxels ───────────────────────────────────────────────
        dims = RACK_DIMENSIONS.get(self.rack_type, RACK_DIMENSIONS["42U"])
        rack_w, rack_d, rack_h = dims
        vw = max(1, int(round(rack_w / VOXEL_SIZE)))   # voxels wide
        vd = max(1, int(round(rack_d / VOXEL_SIZE)))   # voxels deep
        vh = max(1, int(round(rack_h / VOXEL_SIZE)))   # voxels tall
        half_w = vw // 2

        for rack in racks:
            cx = int(round(rack.position.x / VOXEL_SIZE))
            cy = int(round(rack.position.y / VOXEL_SIZE))
            cz = int(round(rack.position.z / VOXEL_SIZE))

            facing = rack.facing
            axis, sign = _FACING_AXIS_SIGN[facing]

            # Bounding box identical to solver._rack_bbox
            if facing == RackFacing.PLUS_X:
                x0, x1 = cx - vd + 1, cx + 1
                y0, y1 = cy - half_w, cy - half_w + vw
            elif facing == RackFacing.MINUS_X:
                x0, x1 = cx, cx + vd
                y0, y1 = cy - half_w, cy - half_w + vw
            elif facing == RackFacing.PLUS_Y:
                x0, x1 = cx - half_w, cx - half_w + vw
                y0, y1 = cy - vd + 1, cy + 1
            else:  # MINUS_Y
                x0, x1 = cx - half_w, cx - half_w + vw
                y0, y1 = cy, cy + vd

            z0, z1 = cz, cz + vh

            # Clamp to grid
            x0c, x1c = max(x0, 0), min(x1, self._nx)
            y0c, y1c = max(y0, 0), min(y1, self._ny)
            z0c, z1c = max(z0, 0), min(z1, self._nz)
            if x0c >= x1c or y0c >= y1c or z0c >= z1c:
                continue

            # Body
            grid[x0c:x1c, y0c:y1c, z0c:z1c] = RACK_BODY

            # Intake face (first voxel along the depth axis, at position side)
            if axis == 0:
                grid[x0c:x0c + 1, y0c:y1c, z0c:z1c] = RACK_INTAKE
                grid[x1c - 1:x1c, y0c:y1c, z0c:z1c] = RACK_EXHAUST
            else:
                grid[x0c:x1c, y0c:y0c + 1, z0c:z1c] = RACK_INTAKE
                grid[x0c:x1c, y1c - 1:y1c, z0c:z1c] = RACK_EXHAUST

        return grid
