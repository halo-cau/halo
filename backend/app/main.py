"""HALO Backend — FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.endpoints import router
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

app.include_router(router, prefix="/api/v1")
app.include_router(visualize_router, prefix="/api/v1")
