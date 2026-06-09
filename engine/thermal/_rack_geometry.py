"""Rack -> voxel index geometry shared by the thermal heat-source builder and the metric readout.

The floor rounding and bounding box here are IDENTICAL to engine/vision/voxelizer.py::_stamp_rack, so
both the exhaust heat source (``solver._build_exhaust_source``) and the intake/exhaust readout
(``metrics._sample_rack_temps``) select exactly the voxels the stamper laid on each rack -- for every
facing. Keeping a single definition removes the prior duplication, where the solver and metrics each
carried their own copy with different rounding that only happened to agree.

Numpy + engine.core only, so the thermal engine stays import-safe (no CV deps).
"""

from __future__ import annotations

import numpy as np

from engine.core.config import RACK_DIMENSIONS, VOXEL_SIZE
from engine.core.data_types import RackFacing, RackPlacement


def world_to_index(
    x: float, y: float, z: float, origin: np.ndarray
) -> tuple[int, int, int]:
    """World metres -> voxel indices via floor, the same convention the stamper uses."""
    idx = np.floor((np.array([x, y, z], dtype=float) - origin) / VOXEL_SIZE).astype(int)
    return int(idx[0]), int(idx[1]), int(idx[2])


def rack_bbox(
    rack: RackPlacement,
    origin: np.ndarray,
    grid_shape: tuple[int, ...],
) -> tuple[int, int, int, int, int, int]:
    """Clamped ``(x0, x1, y0, y1, z0, z1)`` of a rack's stamped volume.

    This is the exact range ``_stamp_rack`` fills, so it always contains that rack's RACK_INTAKE /
    RACK_EXHAUST voxels and never reaches a neighbour's. ``rack.position`` is the front-bottom-center
    (the intake face); the body extends ``depth`` behind it along ``facing``.
    """
    dims = RACK_DIMENSIONS.get(rack.rack_type)
    if dims is None:
        return (0, 0, 0, 0, 0, 0)
    rack_w, rack_d, rack_h = dims
    vw = max(1, round(rack_w / VOXEL_SIZE))
    vd = max(1, round(rack_d / VOXEL_SIZE))
    vh = max(1, round(rack_h / VOXEL_SIZE))
    cx, cy, cz = world_to_index(
        rack.position.x, rack.position.y, rack.position.z, origin
    )
    half_w = vw // 2

    if rack.facing == RackFacing.PLUS_X:
        x0, x1, y0, y1 = cx - vd + 1, cx + 1, cy - half_w, cy - half_w + vw
    elif rack.facing == RackFacing.MINUS_X:
        x0, x1, y0, y1 = cx, cx + vd, cy - half_w, cy - half_w + vw
    elif rack.facing == RackFacing.PLUS_Y:
        x0, x1, y0, y1 = cx - half_w, cx - half_w + vw, cy - vd + 1, cy + 1
    elif rack.facing == RackFacing.MINUS_Y:
        x0, x1, y0, y1 = cx - half_w, cx - half_w + vw, cy, cy + vd
    else:
        return (0, 0, 0, 0, 0, 0)

    z0, z1 = cz, cz + vh
    sx, sy, sz = grid_shape
    return (max(x0, 0), min(x1, sx), max(y0, 0), min(y1, sy), max(z0, 0), min(z1, sz))
