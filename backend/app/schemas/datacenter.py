from pydantic import BaseModel


class OptimizeRequest(BaseModel):
    obstacle: list[list[int]]
    # obstacle(wall) layout. shape of room is rectangle. If the room size is 13 * 22,
    # then obstacle[:37, :] = 1 and obstacle[:, 28:] = 1. (Total grid size = 50 * 50)
    # Or other format(ex. [1:38, :] = 1, [:, 27:50] = 1).

    cooling_pos: list[list[int]]
    # fixed position of cooler.
    # ex.[[3, 4], [10, 13], [24, 41]].

    rack_num: int


class Rack(BaseModel):
    x: int  # x pos
    y: int  # y pos
    dir: int  # rack direction


class OptimizeResponse(BaseModel):
    total_energy: float  # total cooling energy
    max_temp: list[list[float]]  # max temperature of each grid
    data: list[Rack]  # rack position and direction list
