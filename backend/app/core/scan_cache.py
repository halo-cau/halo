"""In-process LRU cache for processed scans.

The CV pipeline (clean → segment → voxelize) is the slow part of the
/visualize flow. We stash the result under an opaque scan_id so /optimize
can re-run RL on different rack counts without re-doing CV.

Cache is in-memory only — restart drops all entries. Callers that miss the
cache get a 404 and must re-upload via /visualize.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from app.core.config import SCAN_CACHE_MAX_ENTRIES
from app.schemas.visualize import VoxelData
from engine.core.data_types import ScanMetadata


@dataclass
class CachedScan:
    """Everything /optimize needs from a prior /visualize call.

    Attributes
    ----------
    grid : np.ndarray
        Room-sized int8 semantic voxel grid (unpadded). Shape (nx, ny, nz)
        where each axis is in VOXEL_SIZE = 0.1 m voxels.
    padded_grid : np.ndarray
        Zero-padded to GRID_SHAPE = (100, 100, 50) for fixed-shape
        downstream consumers (the RL bridge and obstacle projector).
    origin : np.ndarray
        World-space offset of voxel index [0, 0, 0] in the unpadded grid.
    grid_offset : tuple[int, int, int]
        Index offset where ``grid`` is placed inside ``padded_grid``.
    metadata : ScanMetadata
        User-supplied equipment annotations (cooling units, legacy servers,
        workspaces, fixed racks).
    ceiling_m : float
        Real ceiling height in metres, used by the 3-D thermal solver.
    voxel_data : VoxelData
        Pre-computed compact voxel representation for the frontend.
    """

    grid: np.ndarray
    padded_grid: np.ndarray
    origin: np.ndarray
    grid_offset: tuple[int, int, int]
    metadata: ScanMetadata
    ceiling_m: float
    voxel_data: VoxelData


class ScanCache:
    """Thread-safe LRU dict keyed by scan_id."""

    def __init__(self, max_entries: int = SCAN_CACHE_MAX_ENTRIES) -> None:
        self._store: OrderedDict[str, CachedScan] = OrderedDict()
        self._max_entries = max_entries
        self._lock = threading.Lock()

    def put(self, scan_id: str, scan: CachedScan) -> None:
        with self._lock:
            self._store[scan_id] = scan
            self._store.move_to_end(scan_id)
            while len(self._store) > self._max_entries:
                self._store.popitem(last=False)

    def get(self, scan_id: str) -> CachedScan | None:
        with self._lock:
            scan = self._store.get(scan_id)
            if scan is not None:
                self._store.move_to_end(scan_id)
            return scan

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


scan_cache = ScanCache()
