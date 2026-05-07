from pydantic import BaseModel


class OptimizeRequest(BaseModel):
    obstacle: list[list[int]]
    cooling_pos: list[list[int]]
    rack_num: int


class OptimizeResponse(BaseModel):
    result: list[dict]
