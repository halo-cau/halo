"""Pydantic models for the /process-scan API layer.

Engine-level data types (Coordinate, ScanMetadata) live in engine.core.data_types.
These schemas handle JSON validation at the HTTP boundary and convert to engine types.
"""

from pydantic import BaseModel

from engine.core.data_types import Coordinate as EngineCoordinate
from engine.core.data_types import ScanMetadata as EngineScanMetadata


class CoordinateSchema(BaseModel):
    x: float
    y: float
    z: float


class ScanMetadataSchema(BaseModel):
    ac_vents: list[CoordinateSchema] = []
    legacy_servers: list[CoordinateSchema] = []
    human_workspaces: list[CoordinateSchema] = []

    def to_engine(self) -> EngineScanMetadata:
        """Convert validated Pydantic model to the engine's dataclass."""
        return EngineScanMetadata(
            ac_vents=[EngineCoordinate(c.x, c.y, c.z) for c in self.ac_vents],
            legacy_servers=[
                EngineCoordinate(c.x, c.y, c.z) for c in self.legacy_servers
            ],
            human_workspaces=[
                EngineCoordinate(c.x, c.y, c.z) for c in self.human_workspaces
            ],
        )


class LabelCount(BaseModel):
    label: str
    value: int
    count: int


class ProcessScanResponse(BaseModel):
    shape: list[int]
    label_counts: list[LabelCount]
