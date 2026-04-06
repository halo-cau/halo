"""POST /api/v1/visualize — return GLB meshes for 3D visualization."""

import base64
import json
import os
import tempfile
import uuid

import numpy as np
import open3d as o3d
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pydantic import ValidationError

from app.core.config import MAX_UPLOAD_SIZE_BYTES, TEMP_SCAN_DIR
from app.core.exceptions import InvalidFileTypeError, MeshTooLargeError
from app.schemas.payload import ScanMetadataSchema
from app.schemas.visualize import MetricsData, RackMetricsData, RoomMetricsData, ThermalData, VisualizeResponse, VoxelData
from engine.core.config import VOXEL_SIZE
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
    # Return temperatures only at non-empty voxel positions
    nz = np.argwhere(grid != 0)
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
    if not filename.lower().endswith(".obj"):
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
    obj_path = TEMP_SCAN_DIR / f"{uuid.uuid4()}.obj"
    try:
        obj_path.write_bytes(contents)

        # --- Run cleaner to get both meshes ---
        raw_mesh, cleaned_mesh = clean_and_align_meshes(obj_path)
    except EngineError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    finally:
        if obj_path.exists():
            os.unlink(obj_path)

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
    )


@router.get("/visualize/demo", response_model=VisualizeResponse)
async def visualize_demo() -> VisualizeResponse:
    """Generate a demo room mesh on-the-fly — no upload required."""

    # Build a simple box room mesh (4×3×2.5 m)
    mesh = o3d.geometry.TriangleMesh.create_box(4.0, 3.0, 2.5)
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
    #   Room: 4 m (X) × 3 m (Y) × 2.5 m (Z)
    #   Y ≈ 0 wall: full row of human workstations (desks at floor level)
    #   Y ≈ 3 wall: ASHRAE racks facing +Y (intake toward aisle, exhaust toward wall)
    #   Ceiling: AC vents spread across the room
    demo_metadata = ScanMetadata(
        human_workspaces=[
            Coordinate(0.5, 0.3, 0.0),
            Coordinate(1.0, 0.3, 0.0),
            Coordinate(1.5, 0.3, 0.0),
            Coordinate(2.0, 0.3, 0.0),
            Coordinate(2.5, 0.3, 0.0),
            Coordinate(3.0, 0.3, 0.0),
            Coordinate(3.5, 0.3, 0.0),
        ],
        legacy_servers=[],
        cooling_units=[
            # Two ceiling CRAC diffusers blowing straight down
            CoolingUnit(Coordinate(1.0, 1.0, 2.4), capacity_kw=12.0,
                        supply_direction=(0, 0, -1), supply_temp_c=14.0, airflow_cfm=2000.0),
            CoolingUnit(Coordinate(3.0, 1.0, 2.4), capacity_kw=12.0,
                        supply_direction=(0, 0, -1), supply_temp_c=14.0, airflow_cfm=2000.0),
            # Wall-mounted split unit, louvers angled 45° downward into room
            CoolingUnit(Coordinate(2.0, 2.8, 2.0), capacity_kw=8.0,
                        supply_direction=(0, -1, -1), supply_temp_c=16.0, airflow_cfm=1200.0),
        ],
        racks=[
            # Mixed 1U servers — moderate airflow
            RackPlacement(Coordinate(1.0, 2.0, 0.0), RackFacing.PLUS_Y,
                          power_kw=5.0, airflow_cfm=800.0),
            # Dense blade chassis — high power, lower airflow → hotter exhaust
            RackPlacement(Coordinate(2.0, 2.0, 0.0), RackFacing.PLUS_Y,
                          power_kw=8.0, airflow_cfm=500.0),
            # Mixed 1U servers
            RackPlacement(Coordinate(3.0, 2.0, 0.0), RackFacing.PLUS_Y,
                          power_kw=5.0, airflow_cfm=800.0),
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
    )
