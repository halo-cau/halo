"""Shared data types for the engine pipeline."""

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class RackFacing(Enum):
    """Direction the rack intake (front) faces.

    Convention: the exhaust (rear) is the opposite side.
    Values correspond to voxel-grid axis directions.
    """

    PLUS_X = "+x"
    MINUS_X = "-x"
    PLUS_Y = "+y"
    MINUS_Y = "-y"


@dataclass(frozen=True)
class Coordinate:
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class RackPlacement:
    """A single server rack in the room.

    Attributes:
        position: World-space coordinate of the rack's front-bottom-center.
        facing: Direction the intake (front) faces.
        rack_type: Key into config.RACK_DIMENSIONS.
        power_kw: Thermal dissipation in kilowatts.
        airflow_cfm: Server-fan airflow in cubic feet per minute.  Together
            with *power_kw* this determines the exhaust ΔT via:
            ΔT = P / (V̇ · ρ · Cₚ).
        fixed: If True, this rack is pre-existing and cannot be moved by the RL agent.
    """

    position: Coordinate
    facing: RackFacing
    rack_type: str = "42U"
    power_kw: float = 5.0
    airflow_cfm: float = 800.0
    fixed: bool = True


@dataclass(frozen=True)
class CoolingUnit:
    """A cooling unit (CRAC, in-row cooler, ceiling diffuser, etc.).

    Attributes:
        position: World-space coordinate of the unit.
        capacity_kw: Cooling capacity in kilowatts.
        supply_direction: 3D direction vector (dx, dy, dz) of the cold-air
            jet.  Will be normalised internally by the solver.
        supply_temp_c: Supply air temperature in °C.
        airflow_cfm: Unit airflow volume in cubic feet per minute.
            Determines the jet reach and cooling volume.
    """

    position: Coordinate
    capacity_kw: float = 10.0
    supply_direction: tuple[float, float, float] = (0.0, 0.0, -1.0)
    supply_temp_c: float = 14.0
    airflow_cfm: float = 2000.0


@dataclass(frozen=True)
class ScanMetadata:
    cooling_units: list[CoolingUnit] = field(default_factory=list)
    legacy_servers: list[Coordinate] = field(default_factory=list)
    human_workspaces: list[Coordinate] = field(default_factory=list)
    racks: list[RackPlacement] = field(default_factory=list)


@dataclass(frozen=True)
class PipelineResult:
    """Output of the full CV pipeline.

    Attributes:
        grid: Semantic voxel grid with shape matching the actual room.
        padded_grid: Same data zero-padded to GRID_SHAPE — fixed-size
            observation for the RL environment.
        origin: World-space offset of voxel index [0,0,0] in the
            unpadded grid, in meters.
        grid_offset: Index offset where ``grid`` is placed inside
            ``padded_grid``, i.e. ``padded_grid[ox:ox+gx, ...]``.
    """

    grid: np.ndarray
    padded_grid: np.ndarray
    origin: np.ndarray
    grid_offset: tuple[int, int, int]
