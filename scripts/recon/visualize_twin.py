#!/usr/bin/env python3
"""Render the digital-twin result as the two-panel figure the project standardises on:

  LEFT  — top-down instance plan (each placement as a labelled rectangle + the room outline)
  RIGHT — the 3D SEMANTIC voxel grid (the actual voxel_grid.npy), coloured by voxel id:
          green = rack body, blue = intake, orange = exhaust, magenta = AC, yellow = UPS,
          olive = power cabinet. The room shell (walls/floor/ceiling) is omitted so the
          equipment reads clearly, exactly like the reference figure.

Reads ``<run>/placements.json`` + ``<run>/voxel_grid.npy`` and writes ``<run>/twin_view.png``.
Runs in the halo env (matplotlib). Called by pipeline_web.py after voxelization so every run
emits the figure.

Usage:  python scripts/recon/visualize_twin.py --run tools/recon_web/runs/<job>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from engine.core.config import OBSTACLE_WALL, SPACE_EMPTY, VOXEL_SIZE
from scripts.recon.voxelize_labeled_cloud import VOX_COLOR

# top-down plan colours (match the semantic palette)
PLAN_COL = {"server rack": "#2ecc40", "ac_unit": "#e83be8", "fire hose": "#9aa0b0",
            "power cabinet": "#6a6e2a", "ups": "#e8c30a"}


def _kind(name: str) -> str:
    return name.rsplit(" ", 1)[0] if name and name[-1].isdigit() else name


def render(run: Path, out: Path | None = None) -> Path:
    out = out or (run / "twin_view.png")
    manifest = json.loads((run / "placements.json").read_text())
    ext = manifest["ext"]
    grid = np.load(run / "voxel_grid.npy")
    insts = manifest["instances"]

    fig = plt.figure(figsize=(16, 6.5))

    # ── LEFT: top-down plan ──────────────────────────────────────────────
    ax = fig.add_subplot(1, 2, 1)
    ex, ey = ext[0], ext[1]
    ax.add_patch(Rectangle((0, 0), ex, ey, fill=False, ec="#666", lw=1.5))
    for i in insts:
        k = _kind(i["name"]); cx, cy, _ = i["center"]; w, d, _ = i["dims"]
        ax.add_patch(Rectangle((cx - w / 2, cy - d / 2), w, d, fc=PLAN_COL.get(k, "#888"),
                               ec="#222", lw=0.6, alpha=0.92))
        lab = (i["name"].split()[-1] if k == "server rack"
               else {"ac_unit": "AC", "ups": "UPS"}.get(k, i["name"]))
        ax.annotate(lab, (cx, cy), ha="center", va="center", fontsize=8, color="#111")
    n = lambda key: sum(1 for i in insts if _kind(i["name"]) == key)  # noqa: E731
    nr, nac, nups = n("server rack"), n("ac_unit"), n("ups")
    ax.set_xlim(-0.5, ex + 1.0); ax.set_ylim(-0.5, ey + 0.6); ax.set_aspect("equal")
    ax.set_title(f"Top-down plan  ·  {nr} racks ({nr // 2}+{nr - nr // 2}), {nac} AC, {nups} UPS"
                 f"  ·  room {ex:.1f}×{ey:.1f} m")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.grid(True, lw=0.3, alpha=0.3)

    # ── RIGHT: 3D semantic voxel grid (equipment only; shell omitted) ─────
    ax3 = fig.add_subplot(1, 2, 2, projection="3d")
    occ = (grid != SPACE_EMPTY) & (grid != OBSTACLE_WALL)        # drop shell + wall boxes
    colors = np.zeros(grid.shape + (4,), dtype=float)
    for vid, rgb in VOX_COLOR.items():
        if vid == OBSTACLE_WALL:
            continue
        colors[grid == vid] = (*rgb, 1.0)
    if occ.any():
        ax3.voxels(occ, facecolors=colors, edgecolor=None, shade=True)
    ax3.set_box_aspect(grid.shape)
    ax3.view_init(elev=22, azim=-62)
    ax3.set_xlabel("X"); ax3.set_ylabel("Y"); ax3.set_zlabel("Z")
    ax3.set_title("Semantic voxel grid  ·  green=rack body, blue=intake, orange=exhaust, "
                  "magenta=AC, yellow=UPS")

    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="run dir (placements.json + voxel_grid.npy)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    p = render(args.run, args.out)
    print(f"twin view -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
