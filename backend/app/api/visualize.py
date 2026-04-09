"""POST /api/v1/visualize — return GLB meshes for 3D visualization."""

import base64
import json
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
from app.schemas.payload import ScanMetadataSchema
from app.schemas.visualize import EquipmentItem, MetricsData, RackMetricsData, RoomMetricsData, ThermalData, VisualizeResponse, VoxelData
from engine.core.config import ASHRAE_RECOMMENDED_INLET, RACK_DIMENSIONS, VOXEL_SIZE
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
from engine.vision.voxelizer import voxelize_and_label

router = APIRouter()


def _build_equipment_list(metadata: ScanMetadata) -> list[EquipmentItem]:
    """Convert engine metadata into frontend-friendly equipment descriptors.

    Positions are centre-bottom in Z-up world coords.
    """
    items: list[EquipmentItem] = []

    # Racks
    for i, rack in enumerate(metadata.racks):
        w, d, h = RACK_DIMENSIONS.get(rack.rack_type, (0.6, 1.0, 2.0))
        # Shift from front-bottom-centre to body-centre-bottom
        cx, cy, cz = rack.position.x, rack.position.y, rack.position.z
        facing = rack.facing.value  # "+x", "-x", "+y", "-y"
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

    # Cooling units
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

    # Workspaces (desk boxes)
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

    # Legacy servers
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


def _voxelize_mesh(
    cleaned_mesh: o3d.geometry.TriangleMesh, metadata: ScanMetadata,
) -> tuple[VoxelData, np.ndarray, np.ndarray]:
    """Save a cleaned Open3D mesh to a temp PLY, voxelize, and return compact data.

    Returns (VoxelData, grid, origin) — the raw grid/origin are needed for the
    thermal solver.
    """
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp:
        ply_path = tmp.name
    try:
        o3d.io.write_triangle_mesh(ply_path, cleaned_mesh)
        grid, origin = voxelize_and_label(ply_path, metadata)
    finally:
        if os.path.exists(ply_path):
            os.unlink(ply_path)

    # Extract non-empty voxels as compact parallel arrays
    nz = np.argwhere(grid != 0)  # [[ix, iy, iz], ...]
    labels = grid[nz[:, 0], nz[:, 1], nz[:, 2]].tolist() if len(nz) > 0 else []

    voxel_data = VoxelData(
        shape=list(grid.shape),
        voxel_size=VOXEL_SIZE,
        origin=origin.tolist(),
        positions=nz.tolist(),
        labels=labels,
    )
    return voxel_data, grid, origin


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

    # Include voxels whose temperature deviates noticeably from ambient.
    ambient = ASHRAE_RECOMMENDED_INLET
    threshold = 1.0  # °C deviation to include
    interesting = np.abs(temp - ambient) > threshold
    nz = np.argwhere(interesting)
    if len(nz) == 0:
        return None, None

    # Spatial down-sample on X/Y only (keep full Z height resolution)
    # to reduce instance count ~4× while preserving vertical detail.
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

    # Compute ASHRAE metrics
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


@router.post("/visualize", response_model=VisualizeResponse)
async def visualize(
    file: UploadFile = File(...),
    metadata: str = Form("{}"),
) -> VisualizeResponse:
    """Return base64-encoded GLB meshes for raw, cleaned, and semantic stages."""

    # --- Validate file extension ---
    filename = file.filename or ""
    if not filename.lower().endswith(ALLOWED_SCAN_EXTENSIONS):
        raise InvalidFileTypeError(filename)

    # --- Validate file size ---
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_SIZE_BYTES:
        raise MeshTooLargeError(len(contents))

    # --- Parse metadata (optional for visualization) ---
    try:
        parsed = ScanMetadataSchema.model_validate(json.loads(metadata))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    engine_metadata = parsed.to_engine()

    # --- Save temp .obj ---
    TEMP_SCAN_DIR.mkdir(parents=True, exist_ok=True)
    ext = Path(filename).suffix.lower()
    scan_path = TEMP_SCAN_DIR / f"{uuid.uuid4()}{ext}"
    try:
        scan_path.write_bytes(contents)

        # --- Run cleaner to get both meshes ---
        raw_mesh, cleaned_mesh = clean_and_align_meshes(scan_path)
    except EngineError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    finally:
        if scan_path.exists():
            os.unlink(scan_path)

    # --- Convert to GLB ---
    raw_glb = base64.b64encode(o3d_to_glb(raw_mesh)).decode("ascii")
    cleaned_glb = base64.b64encode(o3d_to_glb(cleaned_mesh)).decode("ascii")

    # --- Semantic coloring (only if metadata has any points) ---
    semantic_glb: str | None = None
    has_semantics = (
        engine_metadata.cooling_units
        or engine_metadata.legacy_servers
        or engine_metadata.human_workspaces
    )
    if has_semantics:
        colored_mesh = paint_semantic_colors(cleaned_mesh, engine_metadata)
        semantic_glb = base64.b64encode(o3d_to_glb(colored_mesh)).decode("ascii")

    # --- Voxel grid + thermal ---
    voxel_grid, grid, origin = _voxelize_mesh(cleaned_mesh, engine_metadata)
    thermal, metrics = _compute_thermal(
        grid, engine_metadata.racks, origin, engine_metadata.cooling_units,
    )

    return VisualizeResponse(
        raw_glb=raw_glb,
        cleaned_glb=cleaned_glb,
        semantic_glb=semantic_glb,
        voxel_grid=voxel_grid,
        thermal=thermal,
        metrics=metrics,
        equipment=_build_equipment_list(engine_metadata) or None,
    )


@router.get("/visualize/demo", response_model=VisualizeResponse)
async def visualize_demo() -> VisualizeResponse:
    """Generate a demo room mesh on-the-fly — no upload required."""

    # Build a simple box room mesh (12×10×2.5 m)
    mesh = o3d.geometry.TriangleMesh.create_box(12.0, 10.0, 2.5)
    mesh.compute_vertex_normals()
    mesh = mesh.subdivide_midpoint(number_of_iterations=3)

    # Write to temp .obj so cleaner can process it
    with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as tmp:
        obj_path = tmp.name
    try:
        o3d.io.write_triangle_mesh(obj_path, mesh)
        raw_mesh, cleaned_mesh = clean_and_align_meshes(obj_path)
    finally:
        if os.path.exists(obj_path):
            os.unlink(obj_path)

    raw_glb = base64.b64encode(o3d_to_glb(raw_mesh)).decode("ascii")
    cleaned_glb = base64.b64encode(o3d_to_glb(cleaned_mesh)).decode("ascii")

    # Demo metadata — realistic server room layout:
    #   Room: 12 m (X) × 10 m (Y) × 2.5 m (Z)
    #   Racks shifted right (X≈5–11) to leave workspace area on the left
    #   Two rows of racks forming hot-aisle / cold-aisle with ~4 m gap
    #   Row A (Y≈3.0) exhausts toward +Y, Row B (Y≈7.0) exhausts toward −Y
    #   Hot aisle between Y≈4.0 and Y≈6.0
    #   Workspaces (desks) along left wall X≈1.0
    #   Legacy server in far corner
    demo_metadata = ScanMetadata(
        human_workspaces=[
            # Workspaces (desks) along left wall, spaced along Y
            Coordinate(1.0, 1.5, 0.0),
            Coordinate(1.0, 3.0, 0.0),
            Coordinate(1.0, 4.5, 0.0),
            Coordinate(1.0, 6.0, 0.0),
            Coordinate(1.0, 7.5, 0.0),
            Coordinate(1.0, 9.0, 0.0),
        ],
        legacy_servers=[
            # Old tower server in the far corner
            Coordinate(11.0, 0.5, 0.0),
        ],
        cooling_units=[
            # Ceiling CRAC diffusers — spread across rack area
            CoolingUnit(Coordinate(6.0, 2.0, 2.4), capacity_kw=12.0,
                        supply_direction=(0, 0, -1), supply_temp_c=14.0, airflow_cfm=2000.0),
            CoolingUnit(Coordinate(10.0, 2.0, 2.4), capacity_kw=12.0,
                        supply_direction=(0, 0, -1), supply_temp_c=14.0, airflow_cfm=2000.0),
            CoolingUnit(Coordinate(6.0, 8.0, 2.4), capacity_kw=12.0,
                        supply_direction=(0, 0, -1), supply_temp_c=14.0, airflow_cfm=2000.0),
            CoolingUnit(Coordinate(10.0, 8.0, 2.4), capacity_kw=12.0,
                        supply_direction=(0, 0, -1), supply_temp_c=14.0, airflow_cfm=2000.0),
            # Wall-mounted split unit on Y=10 wall
            CoolingUnit(Coordinate(8.0, 9.8, 2.0), capacity_kw=8.0,
                        supply_direction=(0, -1, -1), supply_temp_c=16.0, airflow_cfm=1200.0),
        ],
        racks=[
            # --- Row A (Y≈3.0): intake facing −Y (cold aisle), exhaust toward +Y (hot aisle) ---
            RackPlacement(Coordinate(5.0, 3.0, 0.0), RackFacing.MINUS_Y,
                          power_kw=5.0, airflow_cfm=800.0),
            RackPlacement(Coordinate(7.0, 3.0, 0.0), RackFacing.MINUS_Y,
                          power_kw=8.0, airflow_cfm=500.0),   # dense blade chassis
            RackPlacement(Coordinate(9.0, 3.0, 0.0), RackFacing.MINUS_Y,
                          power_kw=5.0, airflow_cfm=800.0),
            RackPlacement(Coordinate(11.0, 3.0, 0.0), RackFacing.MINUS_Y,
                          power_kw=6.0, airflow_cfm=700.0),
            # --- Row B (Y≈7.0): intake facing +Y (cold aisle), exhaust toward −Y (hot aisle) ---
            RackPlacement(Coordinate(5.0, 7.0, 0.0), RackFacing.PLUS_Y,
                          power_kw=5.0, airflow_cfm=800.0),
            RackPlacement(Coordinate(7.0, 7.0, 0.0), RackFacing.PLUS_Y,
                          power_kw=7.0, airflow_cfm=600.0),
            RackPlacement(Coordinate(9.0, 7.0, 0.0), RackFacing.PLUS_Y,
                          power_kw=5.0, airflow_cfm=800.0),
            RackPlacement(Coordinate(11.0, 7.0, 0.0), RackFacing.PLUS_Y,
                          power_kw=4.0, airflow_cfm=900.0),   # lightly loaded
        ],
    )
    colored_mesh = paint_semantic_colors(cleaned_mesh, demo_metadata)
    semantic_glb = base64.b64encode(o3d_to_glb(colored_mesh)).decode("ascii")

    # --- Voxel grid + thermal ---
    voxel_grid, grid, origin = _voxelize_mesh(cleaned_mesh, demo_metadata)
    thermal, metrics = _compute_thermal(
        grid, demo_metadata.racks, origin, demo_metadata.cooling_units,
    )

    return VisualizeResponse(
        raw_glb=raw_glb,
        cleaned_glb=cleaned_glb,
        semantic_glb=semantic_glb,
        voxel_grid=voxel_grid,
        thermal=thermal,
        metrics=metrics,
        equipment=_build_equipment_list(demo_metadata) or None,
    )
