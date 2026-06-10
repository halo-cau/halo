"""CV digital twin -> RL DataCenterEnv input (the inverse of ``thermal_bridge``).

``thermal_bridge`` maps an RL state (rack_map / obstacle / cooling_pos) FORWARD to a 3-D voxel grid for
the thermal solve.  This module closes the loop the other way: it takes the SCANNED room's *empty-room*
voxel grid (walls + fixed infrastructure, 0.1 m voxels) plus its ``placements.json`` manifest and produces
the ``(obstacle, cooling_pos, rack_num, ceiling_m)`` that ``DataCenterEnv.reset(options=...)`` expects, so
the trained policy can optimise the layout of the *actual scanned room* rather than a synthetic one.

Grid conventions (must stay consistent with ``thermal_bridge``):
  * CV voxelizer : VOXEL_SIZE = 0.1 m, GRID_SHAPE = (100, 100, 50)  -> 10 x 10 x 5 m max room.
  * RL env       : CELL_M = 0.2 m, 50 x 50 cells                    -> 10 x 10 m max room.
  * _CELL_V = CELL_M / VOXEL_SIZE = 2 voxels per RL cell, so a 2x2 voxel block downsamples to one cell.

DESIGN NOTES / DECISIONS:
  1. Floor & ceiling are full-footprint slabs (z=0 and z=top) in the empty grid; projecting them naively
     would make EVERY cell an obstacle.  We project a band from ``obstacle_floor_m`` (default 0.5 m) up to
     just below the ceiling, so walls, standing infrastructure AND short objects (down to 0.5 m) become
     obstacles while the two slabs are excluded.  Lowering the plane to 0.5 m catches the short cabinets a
     ~1 m plane would miss.
  2. ``rack_num`` and the cooling layout are real DESIGN TARGETS (intended): the user may want a different
     number of racks / AC units than the previous scan.  ``rack_num`` defaults to the twin's stamped count
     but is meant to be set explicitly; ``cooling_pos`` overrides the scanned AC cells verbatim.
  3. The AC is *movable*, so it is stripped from the empty grid; its position is read from the manifest and
     mapped to ``cooling_pos`` as FREE cells (matching how the policy trained, see
     ``DataCenterEnv._generate_cooling``).  Pass ``ac_as_obstacle=True`` to forbid racks on the AC footprint.
  4. The room must fit the fixed 50 x 50 RL grid (<= ~10 m per side); a larger room raises ValueError.
  5. The CV twin is 100x100 voxels (0.1 m); the RL policy is 50x50 cells (0.2 m).  This downsamples 2:1
     (a 2x2 voxel block -> one cell), the exact inverse of ``thermal_bridge``'s 1->2 upsample, so the model
     (which is, and always was, 50x50) is unaffected -- the two grids are the matched 0.1 m / 0.2 m pair.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from engine.core.config import COOLING_AC_VENT, GRID_SHAPE, VOXEL_SIZE

CELL_M: float = 0.2                              # metres per RL cell (matches thermal_bridge.CELL_M)
_CELL_V: int = int(round(CELL_M / VOXEL_SIZE))   # = 2 voxels per RL cell
RL_GRID: int = GRID_SHAPE[0] // _CELL_V          # = 50 cells (the trained policy's fixed grid_size)


def twin_to_rl_input(empty_grid, manifest, rack_num=None, cooling_pos=None,
                     ac_as_obstacle=False, max_coolers=6, obstacle_floor_m=0.5):
    """Map a CV twin to ``DataCenterEnv.reset`` options. Returns a dict with keys ``obstacle`` (RL_GRID x
    RL_GRID int8, 1=blocked), ``cooling_pos`` ((N,2) int cells), ``rack_num`` (int), ``ceiling_m`` (float).

    ``empty_grid`` is ``voxel_empty_grid.npy`` (room shell + fixed infra, movables removed); ``manifest``
    is the parsed ``placements.json``. See the module docstring for the deliberate limitations.

    The whole dict can be passed straight to ``DataCenterEnv.reset(options=out)``: reset now reads
    ``ceiling_m`` and rebuilds the thermal bridge so the 3-D solve uses the scanned room height.
    """
    g = np.asarray(empty_grid)
    if g.ndim != 3:
        raise ValueError(f"empty_grid must be a 3-D voxel grid, got shape {g.shape}")
    vx, vy, vz = g.shape
    rl = RL_GRID
    rcx, rcy = vx // _CELL_V, vy // _CELL_V          # room size in RL cells
    if rcx > rl or rcy > rl:
        raise ValueError(
            f"scanned room is {rcx} x {rcy} RL cells ({rcx * CELL_M:.1f} x {rcy * CELL_M:.1f} m), which "
            f"exceeds the fixed {rl} x {rl} RL grid (~10 m). The trained policy cannot represent it.")

    # 1. OBSTACLE — project a band from ``obstacle_floor_m`` (default 0.5 m) up to just below the ceiling
    #    slab, then downsample 2:1 to RL cells. Starting at 0.5 m (not ~1 m) keeps SHORT objects that sit
    #    below the 1 m plane (low cabinets, etc.); the z=0 floor and z=top ceiling full-footprint slabs are
    #    excluded (they would otherwise block every cell). Cells outside the room footprint stay obstacle
    #    (margins), exactly the form DataCenterEnv._generate_layout produces.
    z0 = min(int(round(obstacle_floor_m / VOXEL_SIZE)), max(1, vz - 2))
    z1 = max(z0 + 1, vz - 1)                          # exclude the ceiling slab at z = vz-1
    occ2d = (g[:, :, z0:z1] != 0).any(axis=2)         # (vx, vy) wall/infra/short-object columns
    obstacle = np.ones((rl, rl), np.int8)
    blocks = occ2d[: rcx * _CELL_V, : rcy * _CELL_V].reshape(rcx, _CELL_V, rcy, _CELL_V)
    obstacle[:rcx, :rcy] = blocks.any(axis=(1, 3)).astype(np.int8)

    # 2. COOLING — a real DESIGN TARGET: the user may want a different number / placement of AC units than
    #    the scan. If ``cooling_pos`` is given it is used verbatim (the design); otherwise the scanned AC
    #    instances are mapped to their FREE RL cells, subsampled to <= max_coolers (trained on 2-10).
    if cooling_pos is not None and len(cooling_pos):
        cool = np.asarray(cooling_pos, dtype=int).reshape(-1, 2)
        if ac_as_obstacle:
            for gx, gy in cool:
                if 0 <= gx < rl and 0 <= gy < rl:
                    obstacle[gx, gy] = 1
    else:
        cells: list[tuple[int, int]] = []
        for p in manifest.get("instances", []):
            is_ac = int(p.get("vox_id", -1)) == int(COOLING_AC_VENT) or str(p.get("name", "")).startswith("ac_unit")
            if not is_ac:
                continue
            cx, cy, _ = p["center"]
            w, d, _ = p["dims"]
            gx0, gx1 = int((cx - w / 2) / CELL_M), int((cx + w / 2) / CELL_M)
            gy0, gy1 = int((cy - d / 2) / CELL_M), int((cy + d / 2) / CELL_M)
            for gx in range(gx0, gx1 + 1):
                for gy in range(gy0, gy1 + 1):
                    if 0 <= gx < rl and 0 <= gy < rl:
                        if ac_as_obstacle:
                            obstacle[gx, gy] = 1
                        if obstacle[gx, gy] == 0:
                            cells.append((gx, gy))
        cells = sorted(set(cells))
        if not cells:                                # no AC labelled -> one cell on a free wall
            free = np.argwhere(obstacle == 0)
            cells = [tuple(int(c) for c in free[len(free) // 3])] if len(free) else [(rl // 2, rl // 2)]
        if len(cells) > max_coolers:                 # even subsample to stay in the trained range
            cells = [cells[i] for i in np.linspace(0, len(cells) - 1, max_coolers).round().astype(int)]
        cool = np.array(cells, dtype=int)

    # 3. RACK_NUM (design target; fall back to the twin's stamped count) + 4. CEILING_M (room height).
    rack_n = _twin_rack_count(manifest) if rack_num is None else max(1, int(rack_num))
    return {"obstacle": obstacle, "cooling_pos": cool,
            "rack_num": rack_n, "ceiling_m": float(vz * VOXEL_SIZE)}


def _twin_rack_count(manifest) -> int:
    return max(1, sum(1 for p in manifest.get("instances", [])
                      if p.get("kind") == "rack" or str(p.get("name", "")).startswith("server rack")))


def twin_dir_to_rl_input(run_dir, rack_num=None, **kw):
    """Convenience: load ``voxel_empty_grid.npy`` + ``placements.json`` from a twin run dir and convert."""
    run_dir = Path(run_dir)
    empty_grid = np.load(run_dir / "voxel_empty_grid.npy")
    manifest = json.loads((run_dir / "placements.json").read_text())
    return twin_to_rl_input(empty_grid, manifest, rack_num=rack_num, **kw)
