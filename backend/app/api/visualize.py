"""POST /api/v1/visualize — process a scan and cache it under a scan_id.

Pipeline order (matches the business spec):
  raw mesh ─▶ clean / align ─▶ segment movables out ─▶ voxelize ─▶ stamp metadata

The cached scan_id can then be passed to ``POST /api/v1/optimize`` so the RL
policy and thermal solver re-use the same voxel grid without re-doing CV.
"""

import base64
import json
import logging
import os
import tempfile
import uuid
from pathlib import Path

import numpy as np
import open3d as o3d
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pydantic import ValidationError

from app.core.config import MAX_UPLOAD_SIZE_BYTES, TEMP_SCAN_DIR
from app.core.exceptions import ALLOWED_SCAN_EXTENSIONS, InvalidFileTypeError, MeshTooLargeError
from app.core.scan_cache import CachedScan, scan_cache
from app.schemas.payload import ScanMetadataSchema
from app.schemas.visualize import (
    EquipmentItem,
    MetricsData,
    RackMetricsData,
    RoomMetricsData,
    ThermalData,
    VisualizeResponse,
    VoxelData,
)
from engine.core.config import (
    ASHRAE_RECOMMENDED_INLET,
    MAX_ROOM_DIMENSIONS,
    RACK_DIMENSIONS,
    VOXEL_SIZE,
)
from engine.core.data_types import (
    Coordinate,
    CoolingUnit,
    RackFacing,
    RackPlacement,
    ScanMetadata,
)
from engine.core.exceptions import EngineError
from engine.thermal.metrics import compute_metrics
from engine.thermal.solver import compute_thermal_field
from engine.vision.cleaner import clean_and_align_meshes
from engine.vision.exporter import o3d_to_glb, paint_semantic_colors
from engine.vision.segmentor_factory import get_default_segmentor
from engine.vision.voxelizer import pad_to_fixed_shape, voxelize_and_stamp_metadata

logger = logging.getLogger(__name__)

router = APIRouter()


def _build_equipment_list(metadata: ScanMetadata) -> list[EquipmentItem]:
    """Convert engine metadata into frontend-friendly equipment descriptors.

    Positions are centre-bottom in Z-up world coords.
    """
    items: list[EquipmentItem] = []

    # Racks
    for i, rack in enumerate(metadata.racks):
        w, d, h = RACK_DIMENSIONS.get(rack.rack_type, (0.6, 1.0, 2.0))
        cx, cy, cz = rack.position.x, rack.position.y, rack.position.z
        facing = rack.facing.value
        if facing == "+x":
            cx -= d / 2
        elif facing == "-x":
            cx += d / 2
        elif facing == "+y":
            cy -= d / 2
        elif facing == "-y":
            cy += d / 2

        items.append(EquipmentItem(
            id=f"rack_{i:02d}",
            category="server_rack",
            label=f"Rack {i + 1}",
            position=[cx, cy, cz],
            size=[w, d, h],
            color="#37474F",
            heat_output=rack.power_kw,
            facing=facing,
        ))

    for i, cu in enumerate(metadata.cooling_units):
        items.append(EquipmentItem(
            id=f"crac_{i:02d}",
            category="cooling_unit",
            label=f"CRAC {i + 1}",
            position=[cu.position.x, cu.position.y, cu.position.z],
            size=[0.6, 0.6, 1.5],
            color="#004d40",
            heat_output=0.0,
        ))

    for i, ws in enumerate(metadata.human_workspaces):
        items.append(EquipmentItem(
            id=f"desk_{i:02d}",
            category="workspace",
            label=f"Desk {i + 1}",
            position=[ws.x, ws.y, ws.z],
            size=[1.2, 0.6, 0.75],
            color="#3e2723",
            heat_output=0.0,
        ))

    for i, ls in enumerate(metadata.legacy_servers):
        items.append(EquipmentItem(
            id=f"legacy_{i:02d}",
            category="legacy_server",
            label=f"Legacy {i + 1}",
            position=[ls.x, ls.y, ls.z],
            size=[0.4, 0.4, 0.8],
            color="#ff9800",
            heat_output=0.5,
        ))

    return items


def _voxel_data_from_grid(grid: np.ndarray, origin: np.ndarray) -> VoxelData:
    """Compact non-empty voxels for transport to the frontend."""
    nz = np.argwhere(grid != 0)
    labels = grid[nz[:, 0], nz[:, 1], nz[:, 2]].tolist() if len(nz) > 0 else []
    return VoxelData(
        shape=list(grid.shape),
        voxel_size=VOXEL_SIZE,
        origin=origin.tolist(),
        positions=nz.tolist(),
        labels=labels,
    )


def _compute_thermal(
    grid: np.ndarray,
    racks: list[RackPlacement],
    origin: np.ndarray,
    cooling_units: list[CoolingUnit] | None = None,
) -> tuple[ThermalData | None, MetricsData | None]:
    """Run the thermal solver and return compact thermal data + metrics."""
    if not racks:
        return None, None
    temp = compute_thermal_field(grid, racks, origin, cooling_units=cooling_units)

    ambient = ASHRAE_RECOMMENDED_INLET
    threshold = 1.0
    interesting = np.abs(temp - ambient) > threshold
    nz = np.argwhere(interesting)
    if len(nz) == 0:
        return None, None

    keep = (nz[:, 0] % 2 == 0) & (nz[:, 1] % 2 == 0)
    nz = nz[keep]
    if len(nz) == 0:
        return None, None

    temps = temp[nz[:, 0], nz[:, 1], nz[:, 2]].tolist()
    thermal = ThermalData(
        positions=nz.tolist(),
        temperatures=temps,
        min_temp=float(np.min(temps)),
        max_temp=float(np.max(temps)),
    )

    mr = compute_metrics(grid, temp, racks, origin, cooling_units)
    metrics = MetricsData(
        racks=[
            RackMetricsData(
                rack_index=r.rack_index,
                intake_temp=r.intake_temp,
                exhaust_temp=r.exhaust_temp,
                delta_t=r.delta_t,
                inlet_compliant=r.inlet_compliant,
                inlet_within_allowable=r.inlet_within_allowable,
            )
            for r in mr.racks
        ],
        room=RoomMetricsData(
            rci_hi=mr.room.rci_hi,
            rci_lo=mr.room.rci_lo,
            shi=mr.room.shi,
            rhi=mr.room.rhi,
            mean_intake=mr.room.mean_intake,
            mean_exhaust=mr.room.mean_exhaust,
            mean_return=mr.room.mean_return,
            vertical_profile=mr.room.vertical_profile,
        ),
    )
    return thermal, metrics


def _run_canonical_pipeline(
    cleaned_mesh: o3d.geometry.TriangleMesh,
    source_path: Path,
    metadata: ScanMetadata,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int, int]]:
    """Apply segmentation (if available) then voxelize.

    Returns (grid, padded_grid, origin, grid_offset).
    """
    segmentor = get_default_segmentor()
    structural_mesh: o3d.geometry.TriangleMesh = cleaned_mesh
    if segmentor is not None:
        try:
            result = segmentor.run(cleaned_mesh, source_path)
            structural_mesh = result.structural_mesh
            logger.info(
                "Segmentation [%s]: removed %d/%d vertices (%.1f%%)",
                result.backend,
                result.n_removed,
                result.n_total,
                100.0 * result.removal_fraction,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Segmentation failed (%s); voxelizing cleaned mesh as-is.", exc,
            )

    grid, origin = voxelize_and_stamp_metadata(structural_mesh, metadata)
    padded_grid, grid_offset = pad_to_fixed_shape(grid)
    return grid, padded_grid, origin, grid_offset


def _build_response_and_cache(
    raw_mesh: o3d.geometry.TriangleMesh,
    cleaned_mesh: o3d.geometry.TriangleMesh,
    source_path: Path,
    metadata: ScanMetadata,
) -> VisualizeResponse:
    """Shared body of POST /visualize and GET /visualize/demo."""
    raw_glb = base64.b64encode(o3d_to_glb(raw_mesh)).decode("ascii")
    cleaned_glb = base64.b64encode(o3d_to_glb(cleaned_mesh)).decode("ascii")

    grid, padded_grid, origin, grid_offset = _run_canonical_pipeline(
        cleaned_mesh, source_path, metadata,
    )

    voxel_data = _voxel_data_from_grid(grid, origin)

    semantic_glb: str | None = None
    has_semantics = (
        metadata.cooling_units
        or metadata.legacy_servers
        or metadata.human_workspaces
        or metadata.racks
    )
    if has_semantics:
        colored_mesh = paint_semantic_colors(cleaned_mesh, metadata)
        semantic_glb = base64.b64encode(o3d_to_glb(colored_mesh)).decode("ascii")

    thermal, metrics = _compute_thermal(
        grid, metadata.racks, origin, metadata.cooling_units,
    )

    # Real ceiling height from the voxel grid (Z dimension × voxel pitch).
    ceiling_m = float(grid.shape[2]) * VOXEL_SIZE
    ceiling_m = min(ceiling_m, MAX_ROOM_DIMENSIONS[2])

    scan_id = uuid.uuid4().hex
    scan_cache.put(
        scan_id,
        CachedScan(
            grid=grid,
            padded_grid=padded_grid,
            origin=origin,
            grid_offset=grid_offset,
            metadata=metadata,
            ceiling_m=ceiling_m,
            voxel_data=voxel_data,
        ),
    )

    return VisualizeResponse(
        scan_id=scan_id,
        raw_glb=raw_glb,
        cleaned_glb=cleaned_glb,
        semantic_glb=semantic_glb,
        voxel_grid=voxel_data,
        thermal=thermal,
        metrics=metrics,
        equipment=_build_equipment_list(metadata) or None,
    )


@router.post("/visualize", response_model=VisualizeResponse)
async def visualize(
    file: UploadFile = File(...),
    metadata: str = Form("{}"),
) -> VisualizeResponse:
    """Process an uploaded scan and return cached GLB / voxel / thermal data."""
    filename = file.filename or ""
    if not filename.lower().endswith(ALLOWED_SCAN_EXTENSIONS):
        raise InvalidFileTypeError(filename)

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_SIZE_BYTES:
        raise MeshTooLargeError(len(contents))

    try:
        parsed = ScanMetadataSchema.model_validate(json.loads(metadata))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    engine_metadata = parsed.to_engine()

    TEMP_SCAN_DIR.mkdir(parents=True, exist_ok=True)
    ext = Path(filename).suffix.lower()
    scan_path = TEMP_SCAN_DIR / f"{uuid.uuid4()}{ext}"
    try:
        scan_path.write_bytes(contents)
        raw_mesh, cleaned_mesh = clean_and_align_meshes(scan_path)
        return _build_response_and_cache(
            raw_mesh, cleaned_mesh, scan_path, engine_metadata,
        )
    except EngineError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    finally:
        if scan_path.exists():
            os.unlink(scan_path)


@router.get("/visualize/demo", response_model=VisualizeResponse)
async def visualize_demo() -> VisualizeResponse:
    """Generate a demo room mesh on-the-fly — no upload required.

    Room dimensions are kept at or below MAX_ROOM_DIMENSIONS (10×10×5 m).
    The layout is two hot-/cold-aisle rack rows with workspaces along the
    left wall and CRAC diffusers at the corners.
    """
    mesh = o3d.geometry.TriangleMesh.create_box(9.8, 9.8, 2.5)
    mesh.compute_vertex_normals()
    mesh = mesh.subdivide_midpoint(number_of_iterations=3)

    with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as tmp:
        obj_path = Path(tmp.name)
    try:
        o3d.io.write_triangle_mesh(str(obj_path), mesh)
        raw_mesh, cleaned_mesh = clean_and_align_meshes(obj_path)

        demo_metadata = ScanMetadata(
            human_workspaces=[
                Coordinate(1.0, 1.5, 0.0),
                Coordinate(1.0, 3.0, 0.0),
                Coordinate(1.0, 4.5, 0.0),
                Coordinate(1.0, 6.0, 0.0),
                Coordinate(1.0, 7.5, 0.0),
                Coordinate(1.0, 9.0, 0.0),
            ],
            legacy_servers=[Coordinate(9.0, 0.5, 0.0)],
            cooling_units=[
                CoolingUnit(Coordinate(5.0, 2.0, 2.4), capacity_kw=12.0,
                            supply_direction=(0, 0, -1), supply_temp_c=14.0, airflow_cfm=2000.0),
                CoolingUnit(Coordinate(8.0, 2.0, 2.4), capacity_kw=12.0,
                            supply_direction=(0, 0, -1), supply_temp_c=14.0, airflow_cfm=2000.0),
                CoolingUnit(Coordinate(5.0, 8.0, 2.4), capacity_kw=12.0,
                            supply_direction=(0, 0, -1), supply_temp_c=14.0, airflow_cfm=2000.0),
                CoolingUnit(Coordinate(8.0, 8.0, 2.4), capacity_kw=12.0,
                            supply_direction=(0, 0, -1), supply_temp_c=14.0, airflow_cfm=2000.0),
                CoolingUnit(Coordinate(6.5, 9.8, 2.0), capacity_kw=8.0,
                            supply_direction=(0, -1, -1), supply_temp_c=16.0, airflow_cfm=1200.0),
            ],
            racks=[
                RackPlacement(Coordinate(4.0, 3.0, 0.0), RackFacing.MINUS_Y,
                              power_kw=5.0, airflow_cfm=800.0),
                RackPlacement(Coordinate(5.5, 3.0, 0.0), RackFacing.MINUS_Y,
                              power_kw=8.0, airflow_cfm=500.0),
                RackPlacement(Coordinate(7.0, 3.0, 0.0), RackFacing.MINUS_Y,
                              power_kw=5.0, airflow_cfm=800.0),
                RackPlacement(Coordinate(8.5, 3.0, 0.0), RackFacing.MINUS_Y,
                              power_kw=6.0, airflow_cfm=700.0),
                RackPlacement(Coordinate(4.0, 7.0, 0.0), RackFacing.PLUS_Y,
                              power_kw=5.0, airflow_cfm=800.0),
                RackPlacement(Coordinate(5.5, 7.0, 0.0), RackFacing.PLUS_Y,
                              power_kw=7.0, airflow_cfm=600.0),
                RackPlacement(Coordinate(7.0, 7.0, 0.0), RackFacing.PLUS_Y,
                              power_kw=5.0, airflow_cfm=800.0),
                RackPlacement(Coordinate(8.5, 7.0, 0.0), RackFacing.PLUS_Y,
                              power_kw=4.0, airflow_cfm=900.0),
            ],
        )
        return _build_response_and_cache(
            raw_mesh, cleaned_mesh, obj_path, demo_metadata,
        )
    finally:
        if obj_path.exists():
            os.unlink(obj_path)
