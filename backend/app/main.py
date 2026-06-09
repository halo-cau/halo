"""HALO Backend — FastAPI application entry point."""

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.endpoints import router
from app.api.twin import router as twin_router
from app.api.visualize import router as visualize_router

app = FastAPI(
    title="HALO Backend",
    description="Thermodynamic server room optimization — 3D scan processing API",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _no_cache(request: Request, call_next):
    """Always revalidate so a frontend rebuild or a re-voxelized artifact is never served stale."""
    resp = await call_next(request)
    resp.headers["Cache-Control"] = "no-cache, must-revalidate"
    return resp


app.include_router(router, prefix="/api/v1")
app.include_router(visualize_router, prefix="/api/v1")
app.include_router(twin_router, prefix="/api/v1")

# Vendored three.js for the self-contained room editor (frontend/dist/editor.html imports
# /vendor/three.module.js). Mounted before the "/" catch-all so it is not shadowed.
_VENDOR = Path(__file__).resolve().parents[2] / "tools" / "vendor"
if _VENDOR.exists():
    app.mount("/vendor", StaticFiles(directory=str(_VENDOR)), name="vendor")

# Serve the built dashboard from the same origin as the API (so /api/v1/twin is same-origin). Mounted
# LAST so the /api/v1 routes above take precedence over the static catch-all.
_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if _DIST.exists():
    app.mount("/", StaticFiles(directory=str(_DIST), html=True), name="frontend")
