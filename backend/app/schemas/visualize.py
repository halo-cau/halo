"""Pydantic models for the /visualize endpoint."""

from pydantic import BaseModel


class VoxelData(BaseModel):
    """Compact representation of the non-empty voxels in the semantic grid."""

    shape: list[int]  # [X, Y, Z] of the unpadded room grid
    voxel_size: float  # metres per voxel
    origin: list[float]  # world-space [x, y, z] of index [0,0,0]
    # Flat lists — parallel arrays; one entry per non-empty voxel.
    positions: list[list[int]]  # [[ix, iy, iz], ...]
    labels: list[int]  # semantic tag for each position


class ThermalData(BaseModel):
    """Compact thermal field — temperatures at non-empty voxel positions."""

    positions: list[list[int]]  # [[ix, iy, iz], ...] (same as VoxelData.positions)
    temperatures: list[float]  # °C at each position
    min_temp: float  # global min for color-map scaling
    max_temp: float  # global max for color-map scaling


class RackMetricsData(BaseModel):
    """Per-rack ASHRAE metrics."""
    rack_index: int
    intake_temp: float
    exhaust_temp: float
    delta_t: float
    inlet_compliant: bool
    inlet_within_allowable: bool


class RoomMetricsData(BaseModel):
    """Room-level ASHRAE metrics."""
    rci_hi: float
    rci_lo: float
    shi: float
    rhi: float
    mean_intake: float
    mean_exhaust: float
    mean_return: float
    vertical_profile: list[float]


class MetricsData(BaseModel):
    """Combined ASHRAE compliance metrics."""
    racks: list[RackMetricsData]
    room: RoomMetricsData


class EquipmentItem(BaseModel):
    """Equipment descriptor for 3D styled rendering in the frontend."""
    id: str
    category: str  # "server_rack", "cooling_unit", "workspace"
    label: str
    position: list[float]  # [x, y, z] center-bottom, Z-up coords
    size: list[float]  # [width, depth, height] in metres
    color: str  # hex color string, e.g. "#37474F"
    heat_output: float  # kW (0 for non-heat-sources)
    facing: str | None = None  # "+x", "-x", "+y", "-y" for racks


class VisualizeResponse(BaseModel):
    raw_glb: str  # base64-encoded GLB (before cleanup)
    cleaned_glb: str  # base64-encoded GLB (after SOR + floor alignment)
    semantic_glb: str | None = None  # base64-encoded GLB with vertex colors (optional)
    voxel_grid: VoxelData | None = None  # compact voxel grid for 3D rendering
    thermal: ThermalData | None = None  # thermal field for heat-map rendering
    metrics: MetricsData | None = None  # ASHRAE compliance metrics
    equipment: list[EquipmentItem] | None = None  # styled equipment for 3D rendering
