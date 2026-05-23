"""POST /api/v1/inference — raw 2D RL policy interface.

This endpoint exposes the underlying ``DataCenterEnv``-compatible interface
directly: caller provides a 50x50 obstacle map and a list of cooling cell
indices, gets back the policy's chosen rack placements as RL grid actions.

For scan-driven optimization that handles voxel-grid → 2D adapters, use
``POST /api/v1/optimize`` instead.
"""

from fastapi import APIRouter

from app.core.exceptions import ModelNotAvailableError
from app.core.rl_service import rl_service
from app.schemas.datacenter import OptimizeRequest, OptimizeResponse

router = APIRouter()


@router.post("/inference", response_model=OptimizeResponse)
def inference(data: OptimizeRequest) -> dict:
    # ModelNotAvailableError is a 503 HTTPException — FastAPI propagates it.
    try:
        result = rl_service.optimize(
            obstacle=data.obstacle,
            cooling_pos=data.cooling_pos,
            rack_num=data.rack_num,
        )
    except ModelNotAvailableError:
        raise
    return {"result": result}
