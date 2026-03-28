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
from app.schemas.visualize import VisualizeResponse
from engine.core.data_types import Coordinate, ScanMetadata
from engine.core.exceptions import EngineError
from engine.vision.cleaner import clean_and_align_meshes
from engine.vision.exporter import o3d_to_glb, paint_semantic_colors

router = APIRouter()


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
        engine_metadata.ac_vents
        or engine_metadata.legacy_servers
        or engine_metadata.human_workspaces
    )
    if has_semantics:
        colored_mesh = paint_semantic_colors(cleaned_mesh, engine_metadata)
        semantic_glb = base64.b64encode(o3d_to_glb(colored_mesh)).decode("ascii")

    return VisualizeResponse(
        raw_glb=raw_glb,
        cleaned_glb=cleaned_glb,
        semantic_glb=semantic_glb,
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

    # Demo metadata with sample points inside the box
    demo_metadata = ScanMetadata(
        ac_vents=[Coordinate(1.0, 1.0, 2.4)],
        legacy_servers=[Coordinate(3.0, 2.0, 0.5)],
        human_workspaces=[Coordinate(2.0, 1.5, 0.0)],
    )
    colored_mesh = paint_semantic_colors(cleaned_mesh, demo_metadata)
    semantic_glb = base64.b64encode(o3d_to_glb(colored_mesh)).decode("ascii")

    return VisualizeResponse(
        raw_glb=raw_glb,
        cleaned_glb=cleaned_glb,
        semantic_glb=semantic_glb,
    )
