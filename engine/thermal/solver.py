"""Steady-state thermal field solver for data-center voxel grids.

Physics model (simplified CFD on a voxel grid):

1. **Source terms** — Each rack's exhaust face continuously injects heat.
   The injection temperature scales with rack power (kW).  A directional
   plume extends from the exhaust face along the exhaust axis with Gaussian
   decay, modelling the initial momentum of the hot-air jet.

2. **Buoyancy** — Hot air rises.  Each iteration, a fraction of every air
   voxel's excess heat (above ambient) is advected upward (+Z).  This
   creates realistic hot-air stratification at the ceiling.

3. **Isotropic diffusion** — A 3×3×3 uniform filter models turbulent
   mixing / conduction between neighbouring voxels.

4. **AC sink terms** — AC vent voxels act as heat sinks, pulling nearby
   air toward a supply temperature.  A small influence zone around each
   vent models the cold-air jet from a ceiling diffuser.

5. **Walls / obstacles** — Solid voxels do not participate in diffusion
   and are pinned at ambient.

The solver iterates until the maximum per-voxel change drops below a
tolerance, or a maximum iteration count is reached.

Grid convention (matches the CV pipeline):
    - X, Y: horizontal floor plane
    - Z: vertical (floor = 0)
    - Voxel pitch: config.VOXEL_SIZE (0.1 m)
"""

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter

from engine.core.config import (
    AIR_CP_J_KG_K,
    AIR_DENSITY_KG_M3,
    ASHRAE_RECOMMENDED_INLET,
    CFM_TO_M3_S,
    COOLING_AC_VENT,
    DEFAULT_AC_AIRFLOW_CFM,
    DEFAULT_AC_CAPACITY_KW,
    DEFAULT_RACK_POWER_KW,
    OBSTACLE_WALL,
    RACK_BODY,
    RACK_DIMENSIONS,
    RACK_EXHAUST,
    RACK_INTAKE,
    SPACE_EMPTY,
    VOXEL_SIZE,
)
from engine.core.data_types import (
    CoolingUnit,
    RackFacing,
    RackPlacement,
)

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------
_AMBIENT: float = ASHRAE_RECOMMENDED_INLET          # 22 °C
_PLUME_SIGMA: float = 4.0                            # Gaussian decay width (voxels)
_BUOYANCY_COEFF: float = 0.08                        # fraction of ΔT advected up per iter
_DIFFUSION_ALPHA: float = 0.18                       # blending weight for isotropic mixing
_AC_COOLING_RATE: float = 0.30                       # how strongly AC pulls toward supply
_AC_JET_CROSS_SIGMA: float = 3.0                      # Gaussian spread perpendicular to jet (voxels)
_MAX_ITERS: int = 250                                # iteration budget
_CONVERGENCE_TOL: float = 0.005                      # °C max-change for early stop

# Exhaust plume velocity → plume length scaling
_MIN_PLUME_VOXELS: int = 8
_MAX_PLUME_VOXELS: int = 40
_EXHAUST_OPEN_FRACTION: float = 0.3  # ~30% of rear face is open perforation
_PLUME_VOXELS_PER_MPS: float = 6.0   # voxels of plume reach per m/s exhaust velocity

# AC jet length scaling
_AC_BASE_JET_LENGTH: int = 25       # jet length at DEFAULT_AC_AIRFLOW_CFM
_AC_MIN_JET_LENGTH: int = 8
_AC_MAX_JET_LENGTH: int = 50


def compute_thermal_field(
    grid: np.ndarray,
    racks: list[RackPlacement],
    origin: np.ndarray,
    cooling_units: list[CoolingUnit] | None = None,
) -> np.ndarray:
    """Compute a steady-state temperature field for the room.

    Parameters
    ----------
    grid : np.ndarray (int8, 3-D)
        Semantic voxel grid from the CV pipeline.
    racks : list[RackPlacement]
        Rack placements with power and facing direction.
    origin : np.ndarray (3,)
        World-space offset of voxel index [0, 0, 0].
    cooling_units : list[CoolingUnit] | None
        Cooling units with capacity and supply direction.  If *None*,
        falls back to treating grid-level AC-vent labels as isotropic
        10 kW units (backward compatibility).

    Returns
    -------
    np.ndarray (float32, same shape as *grid*)
        Temperature in °C at every voxel.
    """
    temp = np.full(grid.shape, _AMBIENT, dtype=np.float32)

    # Pre-compute masks
    air_mask = np.isin(grid, [SPACE_EMPTY, RACK_INTAKE, RACK_EXHAUST, COOLING_AC_VENT])
    solid_mask = ~air_mask
    ac_mask = grid == COOLING_AC_VENT

    # Build per-unit directional AC influence fields
    units = cooling_units or []
    units_with_influence: list[tuple[CoolingUnit, np.ndarray]] = []
    ac_influence = np.zeros(grid.shape, dtype=np.float32)
    for unit in units:
        infl = _build_ac_influence_single(
            grid, unit, origin, air_mask, solid_mask,
        )
        units_with_influence.append((unit, infl))
        np.maximum(ac_influence, infl, out=ac_influence)

    # Build persistent exhaust source field (doesn't change per-iteration)
    exhaust_source = _build_exhaust_source(grid, racks, origin)

    # --- Iterative solver ---
    for _ in range(_MAX_ITERS):
        prev = temp.copy()

        # 1. Isotropic diffusion (turbulent mixing)
        smoothed = uniform_filter(temp, size=3, mode="nearest")
        temp[air_mask] += _DIFFUSION_ALPHA * (smoothed[air_mask] - temp[air_mask])

        # 2. Buoyancy: advect excess heat upward (+Z)
        _apply_buoyancy(temp, air_mask, solid_mask)

        # 3. Re-apply exhaust heat source (constant injection)
        heat_mask = exhaust_source > _AMBIENT
        temp[heat_mask] = np.maximum(temp[heat_mask], exhaust_source[heat_mask])

        # 4. AC cooling — per-unit supply temperature along directional jets
        if units_with_influence:
            for unit, infl in units_with_influence:
                mask_vals = infl[air_mask]
                where_active = mask_vals > 0
                if not where_active.any():
                    continue
                cooling = _AC_COOLING_RATE * mask_vals[where_active] * (
                    temp[air_mask][where_active] - unit.supply_temp_c
                )
                temp_air = temp[air_mask]
                temp_air[where_active] -= cooling
                temp[air_mask] = temp_air

        # 5. Pin solid voxels
        temp[solid_mask] = _AMBIENT

        # Convergence check
        max_change = np.max(np.abs(temp - prev))
        if max_change < _CONVERGENCE_TOL:
            break

    return temp


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _facing_to_axis_dir(facing: RackFacing) -> tuple[int, int]:
    """Return (axis_index, sign) for the exhaust direction."""
    return {
        RackFacing.PLUS_X:  (0, -1),
        RackFacing.MINUS_X: (0, +1),
        RackFacing.PLUS_Y:  (1, -1),
        RackFacing.MINUS_Y: (1, +1),
    }[facing]


def _ac_jet_length_for_cfm(cfm: float) -> int:
    """Derive jet reach (voxels) from the unit's airflow volume."""
    ratio = cfm / DEFAULT_AC_AIRFLOW_CFM
    length = int(round(_AC_BASE_JET_LENGTH * ratio))
    return max(_AC_MIN_JET_LENGTH, min(_AC_MAX_JET_LENGTH, length))


def _build_ac_influence_single(
    grid: np.ndarray,
    unit: CoolingUnit,
    origin: np.ndarray,
    air_mask: np.ndarray,
    solid_mask: np.ndarray,
) -> np.ndarray:
    """Build a directional cold-air influence field for a single cooling unit.

    The jet length scales with the unit's *airflow_cfm*; the influence
    strength scales with *capacity_kw*.  Ray-marching projects influence
    along the unit's arbitrary 3D *supply_direction* with Gaussian
    cross-section decay.
    """
    influence = np.zeros(grid.shape, dtype=np.float32)
    sx, sy, sz = grid.shape

    capacity_scale = unit.capacity_kw / DEFAULT_AC_CAPACITY_KW
    jet_length = _ac_jet_length_for_cfm(unit.airflow_cfm)
    jet_sigma = _AC_JET_CROSS_SIGMA

    # Normalise direction vector
    dx, dy, dz = unit.supply_direction
    mag = (dx * dx + dy * dy + dz * dz) ** 0.5
    if mag < 1e-9:
        return influence
    dx, dy, dz = dx / mag, dy / mag, dz / mag

    # Starting voxel index
    cx = (unit.position.x - origin[0]) / VOXEL_SIZE
    cy = (unit.position.y - origin[1]) / VOXEL_SIZE
    cz = (unit.position.z - origin[2]) / VOXEL_SIZE

    spread = int(jet_sigma * 2) + 1

    for step in range(jet_length):
        # Position along the ray (in voxel coordinates)
        px = cx + dx * step
        py = cy + dy * step
        pz = cz + dz * step

        # Integer centre of this step
        ipx, ipy, ipz = int(round(px)), int(round(py)), int(round(pz))

        # Early termination if ray centre is outside grid
        if not (0 <= ipx < sx and 0 <= ipy < sy and 0 <= ipz < sz):
            break

        # Linear decay along jet axis
        decay = 1.0 - step / jet_length

        # Influence nearby voxels within the cross-section spread
        for da in range(-spread, spread + 1):
            for db in range(-spread, spread + 1):
                for dc in range(-spread, spread + 1):
                    vx = ipx + da
                    vy = ipy + db
                    vz = ipz + dc
                    if not (0 <= vx < sx and 0 <= vy < sy and 0 <= vz < sz):
                        continue

                    # Perpendicular distance from this voxel to the ray
                    rx = vx - px
                    ry = vy - py
                    rz = vz - pz
                    dot = rx * dx + ry * dy + rz * dz
                    perp_sq = (rx - dot * dx) ** 2 + (ry - dot * dy) ** 2 + (rz - dot * dz) ** 2

                    cross_decay = np.exp(-0.5 * perp_sq / (jet_sigma ** 2))
                    val = capacity_scale * decay * cross_decay
                    if val > influence[vx, vy, vz]:
                        influence[vx, vy, vz] = val

    # Normalise to [0, 1]
    mx = influence.max()
    if mx > 0:
        influence /= mx

    # Only apply in air
    influence[solid_mask] = 0.0
    return influence


def _exhaust_delta_t(power_kw: float, airflow_cfm: float) -> float:
    """Physics-based exhaust ΔT: P / (V̇ · ρ · Cₚ)."""
    volume_m3s = airflow_cfm * CFM_TO_M3_S
    if volume_m3s < 1e-9:
        return 0.0
    return (power_kw * 1000.0) / (volume_m3s * AIR_DENSITY_KG_M3 * AIR_CP_J_KG_K)


def _exhaust_plume_voxels(airflow_cfm: float, rack_type: str) -> int:
    """Derive plume reach from exhaust velocity = V̇ / A_effective.

    Real server racks only have ~30% of the rear face open for airflow
    (perforated panels).  The effective area is scaled by _EXHAUST_OPEN_FRACTION.
    """
    dims = RACK_DIMENSIONS.get(rack_type, RACK_DIMENSIONS["42U"])
    width, _depth, height = dims  # (width, depth, height) in metres
    face_area_m2 = width * height * _EXHAUST_OPEN_FRACTION
    volume_m3s = airflow_cfm * CFM_TO_M3_S
    velocity_mps = volume_m3s / face_area_m2 if face_area_m2 > 1e-9 else 0.0
    voxels = int(round(velocity_mps * _PLUME_VOXELS_PER_MPS))
    return max(_MIN_PLUME_VOXELS, min(_MAX_PLUME_VOXELS, voxels))


def _build_exhaust_source(
    grid: np.ndarray,
    racks: list[RackPlacement],
    origin: np.ndarray,
) -> np.ndarray:
    """Build a static source-temperature field from all rack exhausts.

    Uses first-principles ΔT = P / (V̇ · ρ · Cₚ) and derives plume reach
    from the exhaust velocity.
    """
    source = np.full(grid.shape, _AMBIENT, dtype=np.float32)
    sx, sy, sz = grid.shape
    exhaust_mask = grid == RACK_EXHAUST

    if not exhaust_mask.any():
        return source

    for rack in racks:
        delta_t = _exhaust_delta_t(rack.power_kw, rack.airflow_cfm)
        plume_depth = _exhaust_plume_voxels(rack.airflow_cfm, rack.rack_type)
        axis, sign = _facing_to_axis_dir(rack.facing)

        ex, ey, ez = np.where(exhaust_mask)

        for step in range(plume_depth):
            # Gaussian decay from exhaust face
            decay = np.exp(-0.5 * (step / _PLUME_SIGMA) ** 2)
            plume_temp = _AMBIENT + delta_t * decay

            offset = sign * (step + 1)

            if axis == 0:
                shifted = ex + offset
                valid = (shifted >= 0) & (shifted < sx)
                coords = (shifted[valid], ey[valid], ez[valid])
            else:
                shifted = ey + offset
                valid = (shifted >= 0) & (shifted < sy)
                coords = (ex[valid], shifted[valid], ez[valid])

            if len(coords[0]) == 0:
                continue

            # Only seed air voxels
            labels = grid[coords]
            air = (labels == SPACE_EMPTY) | (labels == RACK_INTAKE) | (labels == COOLING_AC_VENT)
            if not air.any():
                continue

            fc = (coords[0][air], coords[1][air], coords[2][air])
            source[fc] = np.maximum(source[fc], plume_temp)

    return source


def _apply_buoyancy(
    temp: np.ndarray,
    air_mask: np.ndarray,
    solid_mask: np.ndarray,
) -> None:
    """Advect excess heat upward by one Z-layer per call.

    Each air voxel donates a fraction of its excess heat (above ambient)
    to the voxel directly above, if that voxel is also air.
    """
    sz = temp.shape[2]
    if sz < 2:
        return

    excess = np.maximum(temp - _AMBIENT, 0.0)
    transfer = _BUOYANCY_COEFF * excess

    # Voxels that can receive (z=1..sz-1, air above)
    can_give = air_mask[:, :, :-1]
    can_recv = air_mask[:, :, 1:]
    both = can_give & can_recv

    # Move heat up
    temp[:, :, :-1][both] -= transfer[:, :, :-1][both]
    temp[:, :, 1:][both]  += transfer[:, :, :-1][both]
