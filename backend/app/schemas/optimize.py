"""Pydantic schemas for the scan-aware /optimize endpoint.

Note: ``app.schemas.datacenter`` holds the *raw* 2-D RL request used by
/inference (caller supplies obstacle + cooling_pos directly). This module
holds the *scan-aware* request used by /optimize (caller supplies a cached
scan_id; backend derives obstacle + cooling_pos from the stored grid).
"""

from pydantic import BaseModel, Field

from app.schemas.visualize import EquipmentItem, MetricsData, ThermalData


class OptimizeScanRequest(BaseModel):
    scan_id: str
    num_racks: int = Field(ge=1, le=100)
    # num_coolers is not accepted — coolers come from the cached scan's
    # metadata (user-tagged at /visualize time).


class PlacementItem(BaseModel):
    """A single RL-chosen rack placement, in both world and grid coords."""

    rack_index: int
    grid_x: int  # RL cell column (0..49)
    grid_y: int  # RL cell row (0..49)
    direction: int  # RL exhaust direction (0=+x, 1=-x, 2=+y, 3=-y)
    facing: str  # solver RackFacing string ("+x" / "-x" / "+y" / "-y")
    position: list[float]  # world-space [x, y, z] in metres
    power_kw: float
    airflow_cfm: float


class OptimizeScanResponse(BaseModel):
    scan_id: str
    placements: list[PlacementItem]
    equipment: list[EquipmentItem]
    thermal: ThermalData | None = None
    metrics: MetricsData | None = None
