"""POST /api/v1/process-scan endpoint."""

import json
import os
import uuid
from pathlib import Path

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pydantic import ValidationError

from app.core.config import MAX_UPLOAD_SIZE_BYTES, TEMP_SCAN_DIR
from app.core.exceptions import ALLOWED_SCAN_EXTENSIONS, InvalidFileTypeError, MeshTooLargeError
from app.schemas.payload import LabelCount, ProcessScanResponse, ScanMetadataSchema
from engine.core.config import SEMANTIC_LABELS
from engine.core.exceptions import EngineError
from engine.vision.pipeline import run_pipeline

router = APIRouter()


@router.post("/process-scan", response_model=ProcessScanResponse)
async def process_scan(
    file: UploadFile = File(...),
    metadata: str = Form(...),
) -> ProcessScanResponse:
    """Ingest an .obj mesh and metadata, return a semantic grid summary."""

    # --- Validate file extension ---
    filename = file.filename or ""
    if not filename.lower().endswith(ALLOWED_SCAN_EXTENSIONS):
        raise InvalidFileTypeError(filename)

    # --- Validate file size ---
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_SIZE_BYTES:
        raise MeshTooLargeError(len(contents))

    # --- Parse & validate metadata JSON → convert to engine dataclass ---
    try:
        parsed = ScanMetadataSchema.model_validate(json.loads(metadata))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    engine_metadata = parsed.to_engine()

    # --- Save to temp directory ---
    TEMP_SCAN_DIR.mkdir(parents=True, exist_ok=True)
    ext = Path(filename).suffix.lower()
    scan_path = TEMP_SCAN_DIR / f"{uuid.uuid4()}{ext}"
    try:
        scan_path.write_bytes(contents)

        # --- Run CV pipeline (engine layer) ---
        result = run_pipeline(scan_path, engine_metadata)
        grid = result.grid
    except EngineError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    finally:
        if scan_path.exists():
            os.unlink(scan_path)

    # --- Build response ---
    unique, counts = np.unique(grid, return_counts=True)
    label_counts = [
        LabelCount(
            label=SEMANTIC_LABELS.get(int(v), f"unknown_{v}"),
            value=int(v),
            count=int(c),
        )
        for v, c in zip(unique, counts, strict=True)
    ]

    return ProcessScanResponse(
        shape=list(grid.shape),
        label_counts=label_counts,
    )
