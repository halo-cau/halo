"""Serving for the imitation-learned layout policy (macro placement).

Serves a layout policy cloned from expert demonstrations (``engine/rl/layout_model.zip`` plus a sidecar
recording the room geometry the model expects). It runs the policy on a room and returns the layout it
proposes -- the aligned server rows, the cooling unit, and the network rack -- decoded to RL cells. The
``/optimize`` route prefers this policy for rooms compatible with the model's geometry, so the dashboard's
"RL optimize" button reflects the imitation-trained layout; otherwise it uses the generic ``rl_service``
single-rack policy.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
MODEL_PATH = Path(os.environ.get("HALO_LAYOUT_MODEL", str(_REPO / "engine" / "rl" / "layout_model.zip")))
META_PATH = _REPO / "engine" / "rl" / "layout_model.meta.json"
# Movable equipment the policy itself places -> stripped to recover the empty room shell.
_MOVABLE = (3, 5, 6, 7, 8, 12)

_model = None


def _meta() -> dict:
    try:
        return json.loads(META_PATH.read_text())
    except Exception:  # noqa: BLE001 -- no/!invalid sidecar just means the layout policy is unavailable
        return {}


def available_for(run_dir) -> bool:
    """True when a cloned layout model is loaded and compatible with the room's geometry.

    Compatibility is a voxel-grid shape check against the geometry the model expects (matching on shape
    rather than a job id, so freshly created rooms of that geometry are covered too)."""
    if not MODEL_PATH.exists():
        return False
    shape = _meta().get("shape")
    if not shape:
        return False
    gp = Path(run_dir) / "voxel_grid.npy"
    if not gp.exists():
        return False
    try:
        return list(np.load(gp, mmap_mode="r").shape) == list(shape)
    except Exception:  # noqa: BLE001 -- unreadable grid -> not applicable
        return False


def _load():
    global _model
    if _model is None:
        from sb3_contrib import MaskablePPO
        _model = MaskablePPO.load(str(MODEL_PATH), device="cpu")
    return _model


_CELL_M = 0.2  # metres per RL cell (matches twin_bridge.CELL_M)


def rollout(run_dir) -> dict:
    """Run the macro policy on the room and decode the layout it proposes.

    Returns ``racks`` as ``[{x, y, dir}]`` (the schema ``/optimize`` already decodes from rl_service) and
    ``fixed`` -- the AC and network rack as world-frame instances at the positions the POLICY chose, not the
    room's current positions, so the proposal shows the equipment relocated. Equipment dimensions are taken
    from the room's own manifest so the rendered boxes match.
    """
    from engine.rl.macro_env import MacroPlacementEnv
    from engine.rl.twin_bridge import twin_to_rl_input

    run_dir = Path(run_dir)
    grid = np.load(run_dir / "voxel_grid.npy")
    manifest = json.loads((run_dir / "placements.json").read_text())
    empty = grid.copy()
    empty[np.isin(empty, _MOVABLE)] = 0
    rl = twin_to_rl_input(empty, manifest, rack_num=12)
    obstacle, ceiling = rl["obstacle"], rl["ceiling_m"]

    rpr = int(_meta().get("racks_per_row", 6))
    env = MacroPlacementEnv(grid_size=obstacle.shape[0], racks_per_row=rpr, ceiling_m=ceiling)
    obs, _ = env.reset(options={"obstacle": obstacle, "rack_num": rpr * 2, "ceiling_m": ceiling})
    model = _load()
    for _ in range(env.n_actions):
        a, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
        obs, _, _done, _, _info = env.step(int(a))

    racks = [{"x": int(x), "y": int(y), "dir": int(env.rack_dir[x, y])}
             for x, y in np.argwhere(env.rack_map == 1)]

    def _dims(pred, default):
        src = next((p for p in manifest.get("instances", []) if pred(p)), None)
        return src["dims"] if src and "dims" in src else default

    ac_dims = _dims(lambda p: str(p.get("name", "")).startswith("ac_unit") or p.get("vox_id") == 3,
                    [0.5, 1.5, 2.0])
    net_dims = _dims(lambda p: p.get("name") == "network rack" or p.get("vox_id") == 8,
                     [0.6, 0.8, 2.2])

    fixed = []
    for gx, gy in env.cooling_pos.tolist():
        fixed.append({"name": "ac_unit 1", "kind": "box", "vox_id": 3, "movable": True,
                      "center": [round(gx * _CELL_M, 3), round(gy * _CELL_M, 3), round(ac_dims[2] / 2, 3)],
                      "dims": ac_dims})
    for gx, gy, _d in env.netrack:
        fixed.append({"name": "network rack", "kind": "box", "vox_id": 8, "movable": True,
                      "center": [round(gx * _CELL_M, 3), round(gy * _CELL_M, 3), round(net_dims[2] / 2, 3)],
                      "dims": net_dims})
    return {"racks": racks, "fixed": fixed}
