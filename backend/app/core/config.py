"""Backend-specific configuration (API / upload concerns only).

Engine-level constants (VOXEL_SIZE, semantic tags, CV params) live in
engine.core.config — this file only holds web/upload settings.
"""

from pathlib import Path

# --- File Upload ---
MAX_UPLOAD_SIZE_BYTES: int = 50 * 1024 * 1024  # 50 MB

# --- Temporary Storage ---
TEMP_SCAN_DIR: Path = Path(__file__).resolve().parents[2] / "data" / "temp_scans"
