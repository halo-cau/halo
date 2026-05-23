"""POST /api/v1/optimize — RL placement on a previously /visualize'd scan.

The flow:
    1. Look up the cached scan by scan_id (404 if expired or never uploaded).
    2. Project the 3-D voxel grid down to a 50x50 obstacle map and convert
       cooling units to RL cell indices.
    3. Run the trained MaskablePPO policy via ``rl_service.optimize`` (503 if
       the checkpoint isn't present on the host).
    4. Translate the RL grid actions to world-coord ``RackPlacement`` objects
       and stamp them into a copy of the cached padded grid so the thermal
       solver sees the new layout.
    5. Run ``compute_thermal_field`` + ``compute_metrics`` and return the
       placements together with ASHRAE compliance data.

This endpoint never re-runs the heavy CV pipeline — re-optimizing a scan
with a different ``num_racks`` is fast.
"""

from __future__ import annotations

import logging

import numpy as np
from fastapi import APIRouter, HTTPException, status

from app.core.exceptions import ModelNotAvailableError, ScanNotFoundError
from app.core.rl_service import rl_service
from app.core.scan_cache import scan_cache
from app.core.scan_to_rl import (
    metadata_to_rl_cooling_pos,
    rl_actions_to_rack_placements,
    voxel_grid_to_rl_obstacle,
)
from app.schemas.optimize import OptimizeScanRequest, OptimizeScanResponse, PlacementItem
from app.schemas.visualize import EquipmentItem
from engine.core.config import RACK_DIMENSIONS
from engine.vision.voxelizer import stamp_rack_on_grid

# Reuse the existing thermal helper from visualize.py for response shaping.
from app.api.visualize import _compute_thermal  # noqa: E402

logger = logging.getLogger(__name__)

router = APIRouter()


def _placements_to_equipment(placements) -> list[EquipmentItem]:
    """Render RL placements as EquipmentItem entries for the frontend."""
    items: list[EquipmentItem] = []
    for i, rack in enumerate(placements):
        w, d, h = RACK_DIMENSIONS.get(rack.rack_type, (0.6, 1.0, 2.0))
        cx, cy, cz = rack.position.x, rack.position.y, rack.position.z
        facing = rack.facing.value
        # Shift centre to body-centre-bottom (front face is at position).
        if facing == "+x":
            cx -= d / 2
        elif facing == "-x":
            cx += d / 2
        elif facing == "+y":
            cy -= d / 2
        elif facing == "-y":
            cy += d / 2
        items.append(EquipmentItem(
            id=f"rl_rack_{i:02d}",
            category="server_rack",
            label=f"RL Rack {i + 1}",
            position=[cx, cy, cz],
            size=[w, d, h],
            color="#26a69a",
            heat_output=rack.power_kw,
            facing=facing,
        ))
    return items


@router.post("/optimize", response_model=OptimizeScanResponse)
def optimize(data: OptimizeScanRequest) -> OptimizeScanResponse:
    cached = scan_cache.get(data.scan_id)
    if cached is None:
        raise ScanNotFoundError(data.scan_id)

    obstacle = voxel_grid_to_rl_obstacle(cached.padded_grid)
    cooling_pos = metadata_to_rl_cooling_pos(cached.metadata, cached.origin)
    if cooling_pos.shape[0] == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Scan metadata has no cooling units — the RL policy requires "
                "at least one cooling position. Re-upload via /visualize with "
                "tagged cooling_units."
            ),
        )

    try:
        actions = rl_service.optimize(
            obstacle=obstacle,
            cooling_pos=cooling_pos,
            rack_num=data.num_racks,
        )
    except ModelNotAvailableError:
        raise
    except Exception as exc:  # noqa: BLE001 — surface as 500
        logger.exception("rl_service.optimize failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RL inference failed: {exc}",
        ) from exc

    placements = rl_actions_to_rack_placements(
        actions, origin=cached.origin,
    )

    # Stamp RL placements into a copy of the padded grid so the thermal
    # solver sees the new layout. The cached grid stays unmodified.
    grid_with_racks = cached.padded_grid.copy()
    for rack in placements:
        stamp_rack_on_grid(grid_with_racks, rack, cached.origin)

    # Combine user-tagged fixed racks (already in the grid) with new RL racks
    # for the solver's `racks` argument.
    all_racks = list(cached.metadata.racks) + placements

    thermal, metrics = _compute_thermal(
        grid_with_racks, all_racks, cached.origin, cached.metadata.cooling_units,
    )

    # Build the per-placement response items. rack_index is offset by the
    # number of pre-existing user-tagged racks so frontend can correlate.
    base_idx = len(cached.metadata.racks)
    placement_items = [
        PlacementItem(
            rack_index=base_idx + i,
            grid_x=int(actions[i]["x"]),
            grid_y=int(actions[i]["y"]),
            direction=int(actions[i]["dir"]),
            facing=rack.facing.value,
            position=[rack.position.x, rack.position.y, rack.position.z],
            power_kw=rack.power_kw,
            airflow_cfm=rack.airflow_cfm,
        )
        for i, rack in enumerate(placements)
    ]

    return OptimizeScanResponse(
        scan_id=data.scan_id,
        placements=placement_items,
        equipment=_placements_to_equipment(placements),
        thermal=thermal,
        metrics=metrics,
    )
