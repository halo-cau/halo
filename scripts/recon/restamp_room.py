#!/usr/bin/env python3
"""Re-stamp a room's voxel grid from its (edited) ``placements.json`` manifest.

The per-instance editor lets the user move / remove instances; it writes the edited manifest and
calls this to rebuild ``voxel_grid.npy`` + ``voxel.ply`` (and the movables-removed empty room).
Shares ``apply_placements`` with the voxelizer, so an unedited manifest reproduces the original
grid bit-for-bit.

Usage (halo env):
    python scripts/recon/restamp_room.py --run tools/recon_web/runs/pi3_chest32_final
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from engine.core.config import (
    COOLING_AC_VENT, RACK_BODY, RACK_EXHAUST, RACK_INTAKE, SPACE_EMPTY, VOXEL_SIZE,
)
from engine.vision.voxelizer import _build_layout_grid
from scripts.recon.voxelize_labeled_cloud import VOX_COLOR, apply_placements

MOVABLE_VOX = (RACK_BODY, RACK_INTAKE, RACK_EXHAUST, COOLING_AC_VENT)   # stripped for the empty room


def _export_ply(grid: np.ndarray, origin: np.ndarray, path: Path) -> int:
    occ = np.argwhere(grid != SPACE_EMPTY)
    rgb = np.array([VOX_COLOR.get(int(grid[x, y, z]), (0.4, 0.4, 0.42)) for x, y, z in occ])
    trimesh.PointCloud(vertices=((occ + 0.5) * VOXEL_SIZE + origin).astype(np.float32),
                       colors=(rgb * 255).astype(np.uint8)).export(path)
    return len(occ)


def restamp(run: Path) -> dict:
    manifest = json.loads((run / "placements.json").read_text())
    shape = tuple(manifest["shape"])
    origin = np.array(manifest["origin"], dtype=float)

    grid = _build_layout_grid(shape, np.int8)
    apply_placements(grid, manifest["instances"], origin)
    np.save(run / "voxel_grid.npy", grid)
    n_occ = _export_ply(grid, origin, run / "voxel.ply")

    empty = grid.copy()
    empty[np.isin(empty, MOVABLE_VOX)] = SPACE_EMPTY
    np.save(run / "voxel_empty_grid.npy", empty)
    n_empty = _export_ply(empty, origin, run / "voxel_empty.ply")

    return {"instances": len(manifest["instances"]), "voxels": int(n_occ), "empty_voxels": int(n_empty)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="run dir holding placements.json")
    args = ap.parse_args()
    res = restamp(args.run)
    print(f"re-stamped {res['instances']} instances -> {res['voxels']} voxels "
          f"({res['empty_voxels']} in the movables-removed empty room)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
