#!/usr/bin/env python3
"""Pass the CV-reconstructed, user-edited room into the RL env.

Reads the movables-removed empty room (``voxel_empty_grid.npy``) + the manifest (``placements.json``)
from a run, builds the RL env's 2-D obstacle map (the room footprint at the env's 0.4 m cell scale),
takes cooling positions from the AC and the rack count from the manifest, resets ``DataCenterEnv``
with them, and (optionally) runs the trained policy to get rack placements.

Usage (halo env):
    python scripts/recon/pass_to_rl.py --run tools/recon_web/runs/pi3_chest32_final [--infer]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from engine.core.config import SPACE_EMPTY
from engine.rl.datacenter import DataCenterEnv

RL_SIZE = 50          # env grid (matches the trained model)
CELL_M = 0.4          # thermal_bridge: 0.4 m per RL cell -> 50x50 covers 20x20 m


def cv_to_rl(run: Path):
    """CV empty-room grid + manifest -> (obstacle[50,50], cooling[N,2], rack_num, (cx,cy))."""
    empty = np.load(run / "voxel_empty_grid.npy")
    man = json.loads((run / "placements.json").read_text())
    vsz = float(man["voxel_size"]); origin = np.array(man["origin"], float)
    f = max(1, int(round(CELL_M / vsz)))                       # CV voxels per RL cell (4)

    # 2-D obstacle footprint: any occupied voxel in the interior band (exclude floor + ceiling planes)
    interior = (empty[:, :, 1:-1] != SPACE_EMPTY).any(axis=2)
    nx, ny = interior.shape
    cx, cy = min(nx // f, RL_SIZE), min(ny // f, RL_SIZE)
    obstacle = np.ones((RL_SIZE, RL_SIZE), dtype=np.float32)   # everything outside the room = obstacle
    for i in range(cx):
        for j in range(cy):
            obstacle[i, j] = 1.0 if interior[i * f:(i + 1) * f, j * f:(j + 1) * f].any() else 0.0

    free = np.argwhere(obstacle == 0)
    def snap(cell):                                           # move a cell onto the nearest free cell
        if obstacle[cell[0], cell[1]] == 0 or len(free) == 0:
            return cell
        return list(free[np.argmin(np.abs(free - cell).sum(1))])

    cooling = [snap([int((p["center"][0] - origin[0]) / CELL_M),
                     int((p["center"][1] - origin[1]) / CELL_M)])
               for p in man["instances"] if p["name"].startswith("ac_unit")]
    if not cooling:
        cooling = [list(free[len(free) // 2])] if len(free) else [[cx // 2, cy // 2]]
    rack_num = max(1, sum(1 for p in man["instances"] if p.get("kind") == "rack"))
    return obstacle, np.array(cooling), rack_num, (cx, cy)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="run dir (voxel_empty_grid.npy + placements.json)")
    ap.add_argument("--infer", action="store_true", help="also run the trained policy for a layout")
    args = ap.parse_args()

    obstacle, cooling, rack_num, (cx, cy) = cv_to_rl(args.run)
    free = int((obstacle == 0).sum())
    print(f"CV room -> RL: {cx}x{cy}-cell footprint in a {RL_SIZE}x{RL_SIZE} grid @ {CELL_M} m/cell; "
          f"{free} free cells; {rack_num} racks to place; cooling at {cooling.tolist()}")

    env = DataCenterEnv(grid_size=RL_SIZE, rack_num=rack_num)
    obs, _ = env.reset(options={"obstacle": obstacle, "cooling_pos": cooling, "rack_num": rack_num})
    assert np.array_equal(env.obstacle, obstacle), "env did NOT take the CV obstacle"
    print(f"env.reset OK -> obstacle loaded ({int((env.obstacle == 0).sum())} free), "
          f"rack_num={env.rack_num}, obs{tuple(obs.shape)}")

    fx, fy = np.argwhere(env.obstacle == 0)[0]                # one valid step on a free cell
    env.step((int(fx) * RL_SIZE + int(fy)) * 4 + 2)
    print(f"env.step OK -> placed a rack at ({fx},{fy}); racks now {int(env.rack_map.sum())}")

    if args.infer:
        try:
            from sb3_contrib import MaskablePPO
            from engine.rl import inference as inf
            model = MaskablePPO.load(str(REPO / "engine" / "rl" / "model.zip"))
            res = inf.run_inference(model, {"obstacle": obstacle.tolist(),
                                            "cooling_pos": cooling.tolist(), "rack_num": rack_num})
            print(f"policy placed {len(res)} racks; first few: {res[:5]}")
        except Exception as e:                                # noqa: BLE001
            print(f"(policy inference skipped: {type(e).__name__}: {e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
