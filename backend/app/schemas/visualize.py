"""Pydantic models for the /visualize endpoint."""

from pydantic import BaseModel


class VisualizeResponse(BaseModel):
    raw_glb: str  # base64-encoded GLB (before cleanup)
    cleaned_glb: str  # base64-encoded GLB (after SOR + floor alignment)
    semantic_glb: str | None = None  # base64-encoded GLB with vertex colors (optional)
