"""HALO Backend — FastAPI application entry point."""

from fastapi import FastAPI

from app.api.endpoints import router

app = FastAPI(
    title="HALO Backend",
    description="Thermodynamic server room optimization — 3D scan processing API",
    version="0.1.0",
)

app.include_router(router, prefix="/api/v1")
