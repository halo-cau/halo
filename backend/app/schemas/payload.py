"""Pydantic models for the /process-scan API layer.

Engine-level data types (Coordinate, ScanMetadata) live in engine.core.data_types.
These schemas handle JSON validation at the HTTP boundary and convert to engine types.
"""

from pydantic import BaseModel

from engine.core.data_types import Coordinate as EngineCoordinate
from engine.core.data_types import CoolingUnit as EngineCoolingUnit
from engine.core.data_types import RackFacing as EngineRackFacing
from engine.core.data_types import RackPlacement as EngineRackPlacement
from engine.core.data_types import ScanMetadata as EngineScanMetadata


class CoordinateSchema(BaseModel):
    x: float
    y: float
    z: float


class RackPlacementSchema(BaseModel):
    position: CoordinateSchema
    facing: str  # "+x", "-x", "+y", "-y"
    rack_type: str = "42U"
    power_kw: float = 5.0
    airflow_cfm: float = 800.0
    fixed: bool = True


class CoolingUnitSchema(BaseModel):
    position: CoordinateSchema
    capacity_kw: float = 10.0
    supply_direction: list[float] = [0.0, 0.0, -1.0]  # 3D direction vector [dx, dy, dz]
    supply_temp_c: float = 14.0
    airflow_cfm: float = 2000.0


class ScanMetadataSchema(BaseModel):
    cooling_units: list[CoolingUnitSchema] = []
    legacy_servers: list[CoordinateSchema] = []
    human_workspaces: list[CoordinateSchema] = []
    racks: list[RackPlacementSchema] = []

    def to_engine(self) -> EngineScanMetadata:
        """Convert validated Pydantic model to the engine's dataclass."""
        return EngineScanMetadata(
            cooling_units=[
                EngineCoolingUnit(
                    position=EngineCoordinate(c.position.x, c.position.y, c.position.z),
                    capacity_kw=c.capacity_kw,
                    supply_direction=tuple(c.supply_direction),
                    supply_temp_c=c.supply_temp_c,
                    airflow_cfm=c.airflow_cfm,
                )
                for c in self.cooling_units
            ],
            legacy_servers=[
                EngineCoordinate(c.x, c.y, c.z) for c in self.legacy_servers
            ],
            human_workspaces=[
                EngineCoordinate(c.x, c.y, c.z) for c in self.human_workspaces
            ],
            racks=[
                EngineRackPlacement(
                    position=EngineCoordinate(r.position.x, r.position.y, r.position.z),
                    facing=EngineRackFacing(r.facing),
                    rack_type=r.rack_type,
                    power_kw=r.power_kw,
                    airflow_cfm=r.airflow_cfm,
                    fixed=r.fixed,
                )
                for r in self.racks
            ],
        )


class LabelCount(BaseModel):
    label: str
    value: int
    count: int


class ProcessScanResponse(BaseModel):
    shape: list[int]
    label_counts: list[LabelCount]
