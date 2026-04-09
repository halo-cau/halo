"""Steady-state thermal field solver for data-center voxel grids.

Physics model (advection-diffusion on a voxel grid):

1. **Prescribed velocity field** — A static 3-D velocity field is built
   from rack fan specs (intake suction + exhaust jet) and AC unit supply
   jets.  A lightweight Jacobi pressure-projection enforces approximate
   mass conservation.

2. **Upwind advection** — Heat is transported along the velocity field
   (plus a temperature-dependent buoyancy component in +Z) using a
   first-order upwind finite-difference scheme.  This captures directed
   airflow, hot/cold aisle separation, and exhaust-to-intake
   recirculation that isotropic diffusion alone cannot model.

3. **Sub-grid diffusion** — A 3×3×3 uniform filter models residual
   turbulent mixing at scales below the voxel pitch.

4. **Source terms** — Each rack's exhaust face injects heat based on
   ΔT = P / (V̇ · ρ · Cₚ) with Gaussian plume decay.

5. **AC sink terms** — Directional AC influence zones pull nearby air
   toward each unit's supply temperature.  Cold-air transport is now
   primarily handled by the velocity field.

6. **Walls / obstacles** — Solid voxels are excluded from advection and
   diffusion and pinned at ambient.

Grid convention (matches the CV pipeline):
    - X, Y: horizontal floor plane
    - Z: vertical (floor = 0)
    - Voxel pitch: config.VOXEL_SIZE (0.1 m)
"""

import numpy as np
from scipy.ndimage import uniform_filter

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

# Exhaust plume geometry
_PLUME_SIGMA: float = 4.0                            # Gaussian decay width (voxels)
_MIN_PLUME_VOXELS: int = 8
_MAX_PLUME_VOXELS: int = 40
_EXHAUST_OPEN_FRACTION: float = 0.3                  # ~30% rear-face open perforation
_PLUME_VOXELS_PER_MPS: float = 6.0                   # plume reach per m/s exhaust velocity

# AC jet geometry
_AC_BASE_JET_LENGTH: int = 25
_AC_MIN_JET_LENGTH: int = 8
_AC_MAX_JET_LENGTH: int = 50
_AC_JET_CROSS_SIGMA: float = 3.0                     # Gaussian spread ⊥ to jet (voxels)
_AC_COOLING_RATE: float = 0.20                       # reduced — advection now assists transport
_AC_OUTLET_AREA_M2: float = 0.5                      # effective AC outlet cross-section

# Solver iteration
_MAX_ITERS: int = 250
_CONVERGENCE_TOL: float = 0.005                      # °C max-change for early stop

# Advection & velocity field
_ADVECTION_CFL: float = 0.45                         # CFL number for upwind stability
_DIFFUSION_ALPHA: float = 0.05                       # sub-grid mixing (reduced from 0.18)
_BUOYANCY_EXCHANGE: float = 0.12                     # fraction of local instability ΔT exchanged
_BUOYANCY_PLUME: float = 0.03                        # fraction of excess-above-ambient advected up
_INTAKE_REACH_FRACTION: float = 0.5                  # intake zone = fraction of exhaust reach
_MIN_INTAKE_REACH: int = 4
_INTAKE_VELOCITY_SCALE: float = 0.6                  # intake suction vs exhaust jet magnitude
_PRESSURE_ITERS: int = 6                             # Jacobi iterations for div correction


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
        Cooling units with capacity and supply direction.

    Returns
    -------
    np.ndarray (float32, same shape as *grid*)
        Temperature in °C at every voxel.
    """
    temp = np.full(grid.shape, _AMBIENT, dtype=np.float32)

    # Pre-compute masks
    air_mask = np.isin(grid, [SPACE_EMPTY, RACK_INTAKE, RACK_EXHAUST, COOLING_AC_VENT])
    solid_mask = ~air_mask

    # --- Velocity field (built once per layout) ---
    vel_x, vel_y, vel_z = _build_velocity_field(
        grid, racks, cooling_units, origin, air_mask, solid_mask,
    )

    # CFL-limited advection time-step (forced flow only, buoyancy is separate)
    v_max = max(np.abs(vel_x).max(), np.abs(vel_y).max(),
                np.abs(vel_z).max(), 1e-9)
    dt = _ADVECTION_CFL * VOXEL_SIZE / v_max

    # --- AC influence (for supply-temp sink coupling) ---
    units = cooling_units or []
    units_with_influence: list[tuple[CoolingUnit, np.ndarray]] = []
    for unit in units:
        infl = _build_ac_influence_single(
            grid, unit, origin, air_mask, solid_mask,
        )
        units_with_influence.append((unit, infl))

    # --- Static exhaust heat-source field ---
    exhaust_source = _build_exhaust_source(grid, racks, origin)

    # Physical temperature floor: cannot go below the coldest source
    temp_floor = min((u.supply_temp_c for u in units), default=_AMBIENT)

    # --- Iterative solver ---
    for _ in range(_MAX_ITERS):
        prev = temp.copy()

        # 1. Advection: transport heat along forced velocity field
        _apply_advection(temp, vel_x, vel_y, vel_z, air_mask, dt)

        # 2. Buoyancy: direct inter-layer heat exchange (decoupled from CFL)
        #    Two-term model applied per z-layer pair:
        #    a) Instability exchange: when T(z) > T(z+1), heat moves up
        #       proportionally to the local ΔT. Self-limiting.
        #    b) Persistent plume: excess above ambient drives continuous
        #       upward transport, modelling buoyant plumes from heat sources.
        _apply_buoyancy(temp, air_mask)

        # 2. Sub-grid diffusion (turbulent mixing)
        smoothed = uniform_filter(temp, size=3, mode="nearest")
        temp[air_mask] += _DIFFUSION_ALPHA * (smoothed[air_mask] - temp[air_mask])

        # 3. Exhaust heat source (constant injection)
        heat_mask = exhaust_source > _AMBIENT
        temp[heat_mask] = np.maximum(temp[heat_mask], exhaust_source[heat_mask])

        # 4. AC cooling — per-unit supply temperature
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

        # 5. Clamp numerical undershoot
        temp[air_mask] = np.maximum(temp[air_mask], temp_floor)

        # 6. Pin solid voxels at ambient
        temp[solid_mask] = _AMBIENT

        # Convergence check
        max_change = np.max(np.abs(temp - prev))
        if max_change < _CONVERGENCE_TOL:
            break

    return temp


# ---------------------------------------------------------------------------
# Velocity field construction
# ---------------------------------------------------------------------------

def _build_velocity_field(
    grid: np.ndarray,
    racks: list[RackPlacement],
    cooling_units: list[CoolingUnit] | None,
    origin: np.ndarray,
    air_mask: np.ndarray,
    solid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a static 3-D velocity field from rack fans and AC jets.

    Returns (vx, vy, vz) arrays in m/s, same shape as *grid*.
    """
    shape = grid.shape
    vx = np.zeros(shape, dtype=np.float32)
    vy = np.zeros(shape, dtype=np.float32)
    vz = np.zeros(shape, dtype=np.float32)

    for rack in racks:
        _add_rack_velocity(vx, vy, grid, rack, origin, air_mask)

    for unit in (cooling_units or []):
        _add_ac_velocity(vx, vy, vz, unit, origin, air_mask)

    # Zero velocity in solids
    vx[solid_mask] = 0.0
    vy[solid_mask] = 0.0
    vz[solid_mask] = 0.0

    # Pressure projection for approximate mass conservation
    _divergence_correction(vx, vy, vz, air_mask)

    return vx, vy, vz


def _add_rack_velocity(
    vx: np.ndarray,
    vy: np.ndarray,
    grid: np.ndarray,
    rack: RackPlacement,
    origin: np.ndarray,
    air_mask: np.ndarray,
) -> None:
    """Superimpose exhaust-jet and intake-suction velocity for one rack."""
    axis, sign = _facing_to_axis_dir(rack.facing)

    dims = RACK_DIMENSIONS.get(rack.rack_type, RACK_DIMENSIONS["42U"])
    width, depth, height = dims
    vd = int(round(depth / VOXEL_SIZE))
    vw = int(round(width / VOXEL_SIZE))
    vh = int(round(height / VOXEL_SIZE))
    half_w = vw // 2

    face_area = width * height
    vol_m3s = rack.airflow_cfm * CFM_TO_M3_S
    v_face = vol_m3s / face_area              # m/s through the full face

    # Rack position → voxel indices
    cx = int(round((rack.position.x - origin[0]) / VOXEL_SIZE))
    cy = int(round((rack.position.y - origin[1]) / VOXEL_SIZE))
    cz = int(round((rack.position.z - origin[2]) / VOXEL_SIZE))

    sx, sy, sz = grid.shape

    # Intake is at the position coordinate; exhaust at the far end
    intake_pos = cx if axis == 0 else cy
    exhaust_pos = intake_pos + sign * (vd - 1)

    # Cross-section bounds perpendicular to flow axis
    if axis == 0:
        cross_lo = max(0, cy - half_w)
        cross_hi = min(sy, cy - half_w + vw)
    else:
        cross_lo = max(0, cx - half_w)
        cross_hi = min(sx, cx - half_w + vw)
    z_lo = max(0, cz)
    z_hi = min(sz, cz + vh)

    vel_arr = vx if axis == 0 else vy
    grid_len = sx if axis == 0 else sy

    # --- Exhaust jet (extends from exhaust face outward) ---
    exhaust_reach = _exhaust_plume_voxels(rack.airflow_cfm, rack.rack_type)
    exhaust_sigma = max(exhaust_reach / 3.0, 1.0)

    for step in range(exhaust_reach):
        decay = np.exp(-0.5 * (step / exhaust_sigma) ** 2)
        v_mag = sign * v_face * decay

        idx = exhaust_pos + sign * (step + 1)
        if not (0 <= idx < grid_len):
            break

        if axis == 0:
            sl = air_mask[idx, cross_lo:cross_hi, z_lo:z_hi]
            vel_arr[idx, cross_lo:cross_hi, z_lo:z_hi] += np.where(
                sl, v_mag, 0.0,
            ).astype(np.float32)
        else:
            sl = air_mask[cross_lo:cross_hi, idx, z_lo:z_hi]
            vel_arr[cross_lo:cross_hi, idx, z_lo:z_hi] += np.where(
                sl, v_mag, 0.0,
            ).astype(np.float32)

    # --- Intake suction (extends from intake face into the room) ---
    intake_reach = max(
        int(exhaust_reach * _INTAKE_REACH_FRACTION), _MIN_INTAKE_REACH,
    )
    intake_sigma = max(intake_reach / 3.0, 1.0)

    for step in range(intake_reach):
        decay = np.exp(-0.5 * (step / intake_sigma) ** 2)
        v_mag = sign * v_face * _INTAKE_VELOCITY_SCALE * decay

        # Suction extends opposite to the exhaust direction
        idx = intake_pos + (-sign) * (step + 1)
        if not (0 <= idx < grid_len):
            break

        if axis == 0:
            sl = air_mask[idx, cross_lo:cross_hi, z_lo:z_hi]
            vel_arr[idx, cross_lo:cross_hi, z_lo:z_hi] += np.where(
                sl, v_mag, 0.0,
            ).astype(np.float32)
        else:
            sl = air_mask[cross_lo:cross_hi, idx, z_lo:z_hi]
            vel_arr[cross_lo:cross_hi, idx, z_lo:z_hi] += np.where(
                sl, v_mag, 0.0,
            ).astype(np.float32)


def _add_ac_velocity(
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    unit: CoolingUnit,
    origin: np.ndarray,
    air_mask: np.ndarray,
) -> None:
    """Superimpose a directed cold-air jet velocity for one AC unit."""
    jet_length = _ac_jet_length_for_cfm(unit.airflow_cfm)
    vol_m3s = unit.airflow_cfm * CFM_TO_M3_S
    v_jet = vol_m3s / _AC_OUTLET_AREA_M2

    # Normalise direction
    ddx, ddy, ddz = unit.supply_direction
    mag = (ddx * ddx + ddy * ddy + ddz * ddz) ** 0.5
    if mag < 1e-9:
        return
    ddx, ddy, ddz = ddx / mag, ddy / mag, ddz / mag

    # Starting voxel (float)
    cx = (unit.position.x - origin[0]) / VOXEL_SIZE
    cy = (unit.position.y - origin[1]) / VOXEL_SIZE
    cz = (unit.position.z - origin[2]) / VOXEL_SIZE

    sx, sy, sz = vx.shape
    jet_sigma = _AC_JET_CROSS_SIGMA
    spread = int(jet_sigma * 2) + 1

    for step in range(jet_length):
        px = cx + ddx * step
        py = cy + ddy * step
        pz = cz + ddz * step

        ipx, ipy, ipz = int(round(px)), int(round(py)), int(round(pz))
        if not (0 <= ipx < sx and 0 <= ipy < sy and 0 <= ipz < sz):
            break

        decay = 1.0 - step / jet_length
        v_step = v_jet * decay

        for da in range(-spread, spread + 1):
            for db in range(-spread, spread + 1):
                for dc in range(-spread, spread + 1):
                    vi = ipx + da
                    vj = ipy + db
                    vk = ipz + dc
                    if not (0 <= vi < sx and 0 <= vj < sy and 0 <= vk < sz):
                        continue
                    if not air_mask[vi, vj, vk]:
                        continue

                    rx = vi - px
                    ry = vj - py
                    rz = vk - pz
                    dot = rx * ddx + ry * ddy + rz * ddz
                    perp_sq = ((rx - dot * ddx) ** 2
                               + (ry - dot * ddy) ** 2
                               + (rz - dot * ddz) ** 2)
                    cross = float(np.exp(-0.5 * perp_sq / (jet_sigma ** 2)))
                    v_local = v_step * cross

                    vx[vi, vj, vk] += v_local * ddx
                    vy[vi, vj, vk] += v_local * ddy
                    vz[vi, vj, vk] += v_local * ddz


def _divergence_correction(
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    air_mask: np.ndarray,
) -> None:
    """Iterative Jacobi pressure-projection to reduce velocity divergence.

    Solves ∇²p = ∇·v then corrects v ← v − ∇p (in-place).
    """
    dx = VOXEL_SIZE
    p = np.zeros_like(vx)

    for _ in range(_PRESSURE_ITERS):
        # Divergence (central differences)
        div = np.zeros_like(vx)
        div[1:, :, :] += vx[1:, :, :] - vx[:-1, :, :]
        div[:, 1:, :] += vy[:, 1:, :] - vy[:, :-1, :]
        div[:, :, 1:] += vz[:, :, 1:] - vz[:, :, :-1]
        div /= dx

        # Jacobi update: p = (Σ neighbours − dx²·div) / 6
        p_sum = np.zeros_like(p)
        p_sum[:-1] += p[1:];   p_sum[1:] += p[:-1]
        p_sum[:, :-1] += p[:, 1:];  p_sum[:, 1:] += p[:, :-1]
        p_sum[:, :, :-1] += p[:, :, 1:];  p_sum[:, :, 1:] += p[:, :, :-1]

        p[air_mask] = (p_sum[air_mask] - dx * dx * div[air_mask]) / 6.0

    # Correct velocity: v ← v − ∇p  (central gradient)
    gradx = np.zeros_like(vx)
    grady = np.zeros_like(vy)
    gradz = np.zeros_like(vz)
    gradx[1:-1, :, :] = (p[2:, :, :] - p[:-2, :, :]) / (2.0 * dx)
    grady[:, 1:-1, :] = (p[:, 2:, :] - p[:, :-2, :]) / (2.0 * dx)
    gradz[:, :, 1:-1] = (p[:, :, 2:] - p[:, :, :-2]) / (2.0 * dx)

    vx[air_mask] -= gradx[air_mask]
    vy[air_mask] -= grady[air_mask]
    vz[air_mask] -= gradz[air_mask]


# ---------------------------------------------------------------------------
# Advection
# ---------------------------------------------------------------------------

def _apply_advection(
    temp: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    air_mask: np.ndarray,
    dt: float,
) -> None:
    """First-order upwind advection of the temperature field (in-place)."""
    inv_dx = 1.0 / VOXEL_SIZE

    # X-axis gradients
    dT_back_x = np.zeros_like(temp)
    dT_fwd_x = np.zeros_like(temp)
    dT_back_x[1:, :, :] = (temp[1:, :, :] - temp[:-1, :, :]) * inv_dx
    dT_fwd_x[:-1, :, :] = (temp[1:, :, :] - temp[:-1, :, :]) * inv_dx

    # Y-axis gradients
    dT_back_y = np.zeros_like(temp)
    dT_fwd_y = np.zeros_like(temp)
    dT_back_y[:, 1:, :] = (temp[:, 1:, :] - temp[:, :-1, :]) * inv_dx
    dT_fwd_y[:, :-1, :] = (temp[:, 1:, :] - temp[:, :-1, :]) * inv_dx

    # Z-axis gradients
    dT_back_z = np.zeros_like(temp)
    dT_fwd_z = np.zeros_like(temp)
    dT_back_z[:, :, 1:] = (temp[:, :, 1:] - temp[:, :, :-1]) * inv_dx
    dT_fwd_z[:, :, :-1] = (temp[:, :, 1:] - temp[:, :, :-1]) * inv_dx

    # Upwind: backward diff where v > 0, forward diff where v < 0
    advect = (
        np.where(vx >= 0, vx * dT_back_x, vx * dT_fwd_x)
        + np.where(vy >= 0, vy * dT_back_y, vy * dT_fwd_y)
        + np.where(vz >= 0, vz * dT_back_z, vz * dT_fwd_z)
    )

    temp[air_mask] -= dt * advect[air_mask]


# ---------------------------------------------------------------------------
# Internal helpers (unchanged public API for tests)
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
        px = cx + dx * step
        py = cy + dy * step
        pz = cz + dz * step

        ipx, ipy, ipz = int(round(px)), int(round(py)), int(round(pz))
        if not (0 <= ipx < sx and 0 <= ipy < sy and 0 <= ipz < sz):
            break

        decay = 1.0 - step / jet_length

        for da in range(-spread, spread + 1):
            for db in range(-spread, spread + 1):
                for dc in range(-spread, spread + 1):
                    vi = ipx + da
                    vj = ipy + db
                    vk = ipz + dc
                    if not (0 <= vi < sx and 0 <= vj < sy and 0 <= vk < sz):
                        continue

                    rx = vi - px
                    ry = vj - py
                    rz = vk - pz
                    dot = rx * dx + ry * dy + rz * dz
                    perp_sq = ((rx - dot * dx) ** 2
                               + (ry - dot * dy) ** 2
                               + (rz - dot * dz) ** 2)

                    cross_decay = np.exp(-0.5 * perp_sq / (jet_sigma ** 2))
                    val = capacity_scale * decay * cross_decay
                    if val > influence[vi, vj, vk]:
                        influence[vi, vj, vk] = val

    mx = influence.max()
    if mx > 0:
        influence /= mx

    influence[solid_mask] = 0.0
    return influence


def _world_to_index_solver(
    x: float, y: float, z: float, origin: np.ndarray,
) -> tuple[int, int, int]:
    """Convert world coordinates to voxel indices (round, not floor)."""
    return (
        int(round((x - origin[0]) / VOXEL_SIZE)),
        int(round((y - origin[1]) / VOXEL_SIZE)),
        int(round((z - origin[2]) / VOXEL_SIZE)),
    )


def _rack_bbox(
    rack: RackPlacement, origin: np.ndarray, grid_shape: tuple[int, ...],
) -> tuple[int, int, int, int, int, int]:
    """Return clamped (x0, x1, y0, y1, z0, z1) for a rack's full volume."""
    dims = RACK_DIMENSIONS.get(rack.rack_type)
    if dims is None:
        return (0, 0, 0, 0, 0, 0)
    rack_w, rack_d, rack_h = dims
    vw = max(1, round(rack_w / VOXEL_SIZE))
    vd = max(1, round(rack_d / VOXEL_SIZE))
    vh = max(1, round(rack_h / VOXEL_SIZE))

    cx, cy, cz = _world_to_index_solver(
        rack.position.x, rack.position.y, rack.position.z, origin,
    )
    half_w = vw // 2
    facing = rack.facing

    if facing == RackFacing.PLUS_X:
        x0, x1 = cx - vd + 1, cx + 1
        y0, y1 = cy - half_w, cy - half_w + vw
    elif facing == RackFacing.MINUS_X:
        x0, x1 = cx, cx + vd
        y0, y1 = cy - half_w, cy - half_w + vw
    elif facing == RackFacing.PLUS_Y:
        x0, x1 = cx - half_w, cx - half_w + vw
        y0, y1 = cy - vd + 1, cy + 1
    elif facing == RackFacing.MINUS_Y:
        x0, x1 = cx - half_w, cx - half_w + vw
        y0, y1 = cy, cy + vd
    else:
        return (0, 0, 0, 0, 0, 0)

    z0, z1 = cz, cz + vh
    sx, sy, sz = grid_shape
    return (max(x0, 0), min(x1, sx), max(y0, 0), min(y1, sy), max(z0, 0), min(z1, sz))


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
    width, _depth, height = dims
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

        # Scope to this rack's exhaust voxels via bounding box
        x0, x1, y0, y1, z0, z1 = _rack_bbox(rack, origin, grid.shape)
        if x0 >= x1 or y0 >= y1 or z0 >= z1:
            continue
        rack_exhaust = exhaust_mask[x0:x1, y0:y1, z0:z1]
        rex, rey, rez = np.where(rack_exhaust)
        # Shift local indices back to global
        rex = rex + x0
        rey = rey + y0
        rez = rez + z0

        if len(rex) == 0:
            continue

        for step in range(plume_depth):
            decay = np.exp(-0.5 * (step / _PLUME_SIGMA) ** 2)
            plume_temp = _AMBIENT + delta_t * decay

            offset = sign * (step + 1)

            if axis == 0:
                shifted = rex + offset
                valid = (shifted >= 0) & (shifted < sx)
                coords = (shifted[valid], rey[valid], rez[valid])
            else:
                shifted = rey + offset
                valid = (shifted >= 0) & (shifted < sy)
                coords = (rex[valid], shifted[valid], rez[valid])

            if len(coords[0]) == 0:
                continue

            labels = grid[coords]
            air = (labels == SPACE_EMPTY) | (labels == RACK_INTAKE) | (labels == COOLING_AC_VENT)
            if not air.any():
                continue

            fc = (coords[0][air], coords[1][air], coords[2][air])
            source[fc] = np.maximum(source[fc], plume_temp)

    return source


# ---------------------------------------------------------------------------
# Buoyancy (decoupled from advection CFL)
# ---------------------------------------------------------------------------

def _apply_buoyancy(
    temp: np.ndarray,
    air_mask: np.ndarray,
) -> None:
    """Two-term buoyancy: direct inter-layer heat exchange.

    Operates on z-layer pairs independently of the advection time-step,
    so buoyancy reaches steady-state even when forced-flow velocities are
    high and the CFL-limited dt is very small.

    Term 1 — Instability exchange:
        When T(z) > T(z+1), a fraction of the difference is exchanged
        upward.  This rapidly eliminates unstable inversions (hot below
        cold).  Self-limiting: once the column is stably stratified,
        this term is zero.

    Term 2 — Persistent plume:
        Any air voxel warmer than ambient donates a fraction of its
        excess heat to the voxel directly above (if it's air).  This
        models the continuous buoyant plume rising from heat sources
        even when the column is already stably stratified.
    """
    sz = temp.shape[2]
    if sz < 2:
        return

    can_give = air_mask[:, :, :-1]
    can_recv = air_mask[:, :, 1:]
    both = can_give & can_recv

    # Term 1: instability — move heat up when lower voxel is hotter
    dt_below_above = temp[:, :, :-1] - temp[:, :, 1:]
    instability = np.maximum(dt_below_above, 0.0)
    xfer_instability = _BUOYANCY_EXCHANGE * instability

    temp[:, :, :-1][both] -= xfer_instability[both]
    temp[:, :, 1:][both]  += xfer_instability[both]

    # Term 2: persistent plume — excess above ambient drifts upward
    excess = np.maximum(temp[:, :, :-1] - _AMBIENT, 0.0)
    xfer_plume = _BUOYANCY_PLUME * excess

    temp[:, :, :-1][both] -= xfer_plume[both]
    temp[:, :, 1:][both]  += xfer_plume[both]
