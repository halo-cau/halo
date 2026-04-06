"""ASHRAE TC 9.9 compliance metrics computed from the solved thermal field.

All sampling reads directly from the 3-D temperature array produced by
``compute_thermal_field``.  No additional simulation is performed here.

Metrics
-------
Per-rack:
    intake_temp     – mean temperature at RACK_INTAKE voxels
    exhaust_temp    – mean temperature at RACK_EXHAUST voxels
    delta_t         – exhaust_temp − intake_temp
    inlet_compliant – intake_temp within ASHRAE recommended range

Room-level:
    RCI_HI / RCI_LO – Rack Cooling Index (high / low end)
    SHI / RHI       – Supply / Return Heat Index
    RTI             – Return Temperature Index (per cooling unit)
    vertical_profile – mean air temperature at each Z-layer
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from engine.core.config import (
    ASHRAE_INLET_TEMP_RANGE,
    ASHRAE_INLET_ALLOWABLE_RANGE,
    CFM_TO_M3_S,
    AIR_DENSITY_KG_M3,
    AIR_CP_J_KG_K,
    RACK_EXHAUST,
    RACK_INTAKE,
    SPACE_EMPTY,
    COOLING_AC_VENT,
    VOXEL_SIZE,
)
from engine.core.data_types import CoolingUnit, RackPlacement

# ── Result containers ─────────────────────────────────────

@dataclass
class RackMetrics:
    """Per-rack thermal metrics."""
    rack_index: int
    intake_temp: float
    exhaust_temp: float
    delta_t: float
    inlet_compliant: bool           # within ASHRAE *recommended*
    inlet_within_allowable: bool    # within ASHRAE *allowable*


@dataclass
class RoomMetrics:
    """Room-level ASHRAE compliance metrics."""
    rci_hi: float          # Rack Cooling Index – high end [0–100 %]
    rci_lo: float          # Rack Cooling Index – low end  [0–100 %]
    shi: float             # Supply Heat Index  [0–1]
    rhi: float             # Return Heat Index  [0–1]
    mean_intake: float     # average of all rack intake temps
    mean_exhaust: float    # average of all rack exhaust temps
    mean_return: float     # estimated CRAC return-air temp
    vertical_profile: list[float]   # mean air temp at each Z-layer


@dataclass
class MetricsResult:
    """Combined metrics returned to the API."""
    racks: list[RackMetrics]
    room: RoomMetrics


# ── Public API ────────────────────────────────────────────

def compute_metrics(
    grid: np.ndarray,
    temp: np.ndarray,
    racks: list[RackPlacement],
    origin: np.ndarray,
    cooling_units: list[CoolingUnit] | None = None,
) -> MetricsResult:
    """Compute all ASHRAE metrics from a solved thermal field.

    Parameters
    ----------
    grid : int8 3-D array
        Semantic voxel grid.
    temp : float32 3-D array (same shape)
        Solved temperature field from ``compute_thermal_field``.
    racks : list[RackPlacement]
        Rack metadata (position, facing, power, airflow).
    origin : 1-D array (3,)
        World-space origin of voxel [0,0,0].
    cooling_units : list[CoolingUnit] | None
        Cooling unit metadata (used for RTI / return-air estimation).
    """
    rec_lo, rec_hi = ASHRAE_INLET_TEMP_RANGE        # (18, 27)
    allow_lo, allow_hi = ASHRAE_INLET_ALLOWABLE_RANGE  # (15, 35)

    # --- Per-rack metrics ----
    intake_mask = grid == RACK_INTAKE
    exhaust_mask = grid == RACK_EXHAUST

    rack_metrics: list[RackMetrics] = []
    all_intake_temps: list[float] = []
    all_exhaust_temps: list[float] = []

    for idx, rack in enumerate(racks):
        # For now we treat *all* intake/exhaust voxels as belonging to
        # every rack (single-rack demo).  A future improvement would
        # associate voxels to individual racks by proximity.
        i_temps = temp[intake_mask]
        e_temps = temp[exhaust_mask]

        i_mean = float(i_temps.mean()) if len(i_temps) > 0 else float(temp.mean())
        e_mean = float(e_temps.mean()) if len(e_temps) > 0 else i_mean

        dt = e_mean - i_mean
        compliant = rec_lo <= i_mean <= rec_hi
        within_allow = allow_lo <= i_mean <= allow_hi

        rack_metrics.append(RackMetrics(
            rack_index=idx,
            intake_temp=round(i_mean, 2),
            exhaust_temp=round(e_mean, 2),
            delta_t=round(dt, 2),
            inlet_compliant=compliant,
            inlet_within_allowable=within_allow,
        ))

        all_intake_temps.append(i_mean)
        all_exhaust_temps.append(e_mean)

    # --- Aggregates ---
    mean_intake = float(np.mean(all_intake_temps)) if all_intake_temps else 0.0
    mean_exhaust = float(np.mean(all_exhaust_temps)) if all_exhaust_temps else 0.0

    # --- RCI (Rack Cooling Index) ---
    rci_hi = _compute_rci_hi(all_intake_temps, rec_hi, allow_hi)
    rci_lo = _compute_rci_lo(all_intake_temps, rec_lo, allow_lo)

    # --- Return-air temperature estimation ---
    # Approximate: average air temperature in the upper 25% of the room
    # (return plenums are typically near ceiling).
    sz = grid.shape[2]
    upper_start = int(sz * 0.75)
    air_mask_upper = np.isin(grid[:, :, upper_start:],
                             [SPACE_EMPTY, RACK_INTAKE, RACK_EXHAUST, COOLING_AC_VENT])
    upper_temps = temp[:, :, upper_start:][air_mask_upper]
    mean_return = float(upper_temps.mean()) if len(upper_temps) > 0 else mean_exhaust

    # --- SHI / RHI ---
    # SHI = ΔQ_supply / (ΔQ_supply + Q_equipment)
    # where ΔQ_supply captures how much the supply air warms before reaching intakes
    # and Q_equipment is the rack heat.
    supply_temp = _estimate_supply_temp(cooling_units)
    dq_supply = sum(max(t - supply_temp, 0.0) for t in all_intake_temps)
    q_equip = sum(max(e - i, 0.0) for i, e in zip(all_intake_temps, all_exhaust_temps))
    total = dq_supply + q_equip
    shi = dq_supply / total if total > 0 else 0.0
    rhi = 1.0 - shi

    # --- Vertical stratification profile ---
    air_mask_full = np.isin(grid, [SPACE_EMPTY, RACK_INTAKE, RACK_EXHAUST, COOLING_AC_VENT])
    vertical_profile: list[float] = []
    for z in range(sz):
        layer_air = air_mask_full[:, :, z]
        if layer_air.any():
            vertical_profile.append(round(float(temp[:, :, z][layer_air].mean()), 2))
        else:
            vertical_profile.append(round(float(temp[:, :, z].mean()), 2))

    room = RoomMetrics(
        rci_hi=round(rci_hi, 1),
        rci_lo=round(rci_lo, 1),
        shi=round(shi, 4),
        rhi=round(rhi, 4),
        mean_intake=round(mean_intake, 2),
        mean_exhaust=round(mean_exhaust, 2),
        mean_return=round(mean_return, 2),
        vertical_profile=vertical_profile,
    )

    return MetricsResult(racks=rack_metrics, room=room)


# ── Internal helpers ──────────────────────────────────────

def _compute_rci_hi(intake_temps: list[float], t_rec_hi: float, t_allow_hi: float) -> float:
    """RCI_HI = [1 − Σ max(T_in − T_rec_hi, 0) / (n × (T_allow_hi − T_rec_hi))] × 100."""
    n = len(intake_temps)
    if n == 0:
        return 100.0
    denom = n * (t_allow_hi - t_rec_hi)
    if denom == 0:
        return 100.0
    total_excess = sum(max(t - t_rec_hi, 0.0) for t in intake_temps)
    return max(0.0, (1.0 - total_excess / denom) * 100.0)


def _compute_rci_lo(intake_temps: list[float], t_rec_lo: float, t_allow_lo: float) -> float:
    """RCI_LO = [1 − Σ max(T_rec_lo − T_in, 0) / (n × (T_rec_lo − T_allow_lo))] × 100."""
    n = len(intake_temps)
    if n == 0:
        return 100.0
    denom = n * (t_rec_lo - t_allow_lo)
    if denom == 0:
        return 100.0
    total_deficit = sum(max(t_rec_lo - t, 0.0) for t in intake_temps)
    return max(0.0, (1.0 - total_deficit / denom) * 100.0)


def _estimate_supply_temp(cooling_units: list[CoolingUnit] | None) -> float:
    """Weighted average supply temperature across all AC units."""
    if not cooling_units:
        return 14.0  # fallback
    total_cfm = sum(u.airflow_cfm for u in cooling_units)
    if total_cfm == 0:
        return cooling_units[0].supply_temp_c
    weighted = sum(u.supply_temp_c * u.airflow_cfm for u in cooling_units)
    return weighted / total_cfm
