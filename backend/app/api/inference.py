from fastapi import APIRouter, HTTPException
from app.core.rl_service import rl_service
from app.schemas.datacenter import OptimizeRequest, OptimizeResponse

router = APIRouter()


@router.post("/inference", response_model=OptimizeResponse)
async def optimize(data: OptimizeRequest):

    try:
        result = rl_service.optimize(
            obstacle=data.obstacle, cooling_pos=data.cooling_pos, rack_num=data.rack_num
        )

        return {
            "total_energy": result["total_energy"],
            "max_temp": result["max_temp"],
            "data": result["data"],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
