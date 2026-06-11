#!/usr/bin/env python3
"""Per-rack facing (intake/exhaust direction) from SAM3 front-door + geometric fallback.

The thermal model needs each rack's FRONT (cold-aisle intake) vs BACK (hot-aisle
exhaust). We get the front from SAM3 — feasibility-tested: "perforated front door
of a server rack" fires reliably on the captured (aisle-side) faces. Pipeline:

  1. SAM3 front-door mask per photo -> lift to 3D via the per-view pointmaps
     (same gravity frame as the labeled cloud) -> front-door 3D points.
  2. For each rack instance (from the label sidecar): its captured points form a
     thin vertical slab (the front face); the depth axis is that slab's normal.
     Project nearby front-door points onto the depth axis -> the side they're on
     is the FRONT -> facing = that direction, snapped to a world axis (RackFacing).
  3. Fallback (no SAM3 evidence): front faces the room centre (cold-aisle layout).

Outputs ``rack_facing.json`` in the labeled run. GPU step (SAM3) — caller should
check VRAM first.

Usage (halo env):
    python scripts/recon/rack_facing.py \
        --recon-run tools/recon_web/runs/pi3_chest32 \
        --labeled-run tools/recon_web/runs/pi3_chest32_final
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "recon"))

from _recon_io import gravity_R
from engine.vision.segmentor_dino_sam import (
    _clone_sam3_image_state, _load_sam3, _sam3_masks_to_numpy, _sam3_scores_to_numpy,
)

FRONT_PROMPT = "perforated front door of a server rack"
AXES = {"PLUS_X": np.array([1.0, 0.0]), "MINUS_X": np.array([-1.0, 0.0]),
        "PLUS_Y": np.array([0.0, 1.0]), "MINUS_Y": np.array([0.0, -1.0])}


def _resize(mask, hw):
    H, W = hw
    if mask.shape == (H, W):
        return mask
    return np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((W, H), Image.NEAREST)) > 127


def lift_front_door(views, up_npy, proc, conf_thr, thr):
    R = gravity_R(np.load(str(up_npy)))
    pts = []
    for vi, vf in enumerate(views):
        d = np.load(vf, allow_pickle=True)
        pts3d, conf = d["pts3d"], d["conf"]; H, W = conf.shape
        rgb = np.asarray(Image.open(str(d["img_path"])).convert("RGB"), np.uint8)
        base = proc.set_image(Image.fromarray(rgb))
        st = _clone_sam3_image_state(base)
        try:
            st = proc.set_text_prompt(FRONT_PROMPT, st)
        except Exception:  # noqa: BLE001
            continue
        m = _sam3_masks_to_numpy(st, rgb.shape[:2])
        if len(m) == 0:
            continue
        sc = _sam3_scores_to_numpy(st, len(m))
        keep = m[sc >= thr] if (sc >= thr).any() else m[:0]
        if len(keep) == 0:
            continue
        sel = _resize(np.any(keep, axis=0), (H, W)) & (conf > conf_thr)
        if sel.any():
            pts.append((pts3d[sel].reshape(-1, 3)) @ R.T)
    return np.concatenate(pts) if pts else np.empty((0, 3))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon-run", type=Path, required=True, help="run with recon_raw.views + recon_raw.up.npy")
    ap.add_argument("--labeled-run", type=Path, required=True, help="run with recon.ply + point_labels.npz")
    ap.add_argument("--conf", type=float, default=1.5)
    ap.add_argument("--thr", type=float, default=0.45)
    ap.add_argument("--near-radius", type=float, default=0.25, help="front-door proximity to a rack (cloud units)")
    ap.add_argument("--skip-sam3", action="store_true",
                    help="skip the SAM3 front-door confirmation pass (direction is geometric); CPU-only")
    args = ap.parse_args()

    P = np.asarray(trimesh.load(args.labeled_run / "recon.ply", process=False).vertices, np.float64)
    names = np.load(args.labeled_run / "point_labels.npz", allow_pickle=False)["names"].astype(str)
    racks = {n: P[names == n] for n in np.unique(names) if n.startswith("server rack")}
    room_c = P[names == "floor"][:, :2].mean(0) if (names == "floor").any() else P[:, :2].mean(0)
    print(f"{len(racks)} rack instances; room centre={np.round(room_c, 2)}")

    # rows: cluster rack centroids along the perpendicular of the row axis
    rnames = sorted(racks)
    cents = np.array([racks[n][:, :2].mean(0) for n in rnames])
    _, _, gvt = np.linalg.svd(cents - cents.mean(0), full_matrices=False)
    side = (cents - cents.mean(0)) @ gvt[1]
    row_id = {rnames[i]: int(side[i] > 0) for i in range(len(rnames))}
    row_cent = {r: cents[[i for i in range(len(rnames)) if int(side[i] > 0) == r]].mean(0) for r in (0, 1)}
    print(f"rows at {np.round(row_cent[0], 3)} / {np.round(row_cent[1], 3)}")

    # optional SAM3 confirmation that the captured faces are FRONTS (so toward-aisle = front)
    fdtree = None
    if not args.skip_sam3:
        proc = _load_sam3(None, "cuda", args.thr)
        views = sorted((args.recon_run / "recon_raw.views").glob("view_*.npz"))
        fd = lift_front_door(views, args.recon_run / "recon_raw.up.npy", proc, args.conf, args.thr)
        print(f"SAM3 front-door 3D points: {len(fd):,}")
        fdtree = cKDTree(fd[:, :2]) if len(fd) else None

    # facing = front-face SURFACE NORMAL oriented toward the AISLE (the other row).
    # The captured face is the front (camera was in the central cold aisle; SAM3-confirmed),
    # so its outward normal points across the aisle -> intake; the body extrudes the OTHER
    # way (toward the wall) -> never fills the aisle.
    out = {}
    for n in rnames:
        Q = racks[n]; c = Q[:, :2].mean(0)
        _, _, vt = np.linalg.svd(Q - Q.mean(0), full_matrices=False)
        normal = vt[2][:2]                              # slab normal (horizontal) = front-back axis
        normal = normal / (np.linalg.norm(normal) + 1e-9)
        if normal @ (row_cent[1 - row_id[n]] - c) < 0:  # orient toward the other row / aisle = FRONT
            normal = -normal
        sam3_conf = fdtree is not None and len(fdtree.query_ball_point(c, r=args.near_radius)) >= 20
        facing = max(AXES, key=lambda k: float(normal @ AXES[k]))
        out[n] = {"facing": facing, "source": ("normal+sam3" if sam3_conf else "normal"),
                  "center": [round(float(x), 4) for x in Q.mean(0)]}

    # row-consistency: facing is a row property (uniform racks all face the same way).
    # Cluster rack centroids into rows (along the perpendicular of the global row axis),
    # then assign every rack in a row the row's MAJORITY facing — fixes end-rack outliers.
    from collections import Counter
    names_r = list(out)
    cents = np.array([out[n]["center"][:2] for n in names_r])
    _, _, gvt = np.linalg.svd(cents - cents.mean(0), full_matrices=False)
    perp = gvt[1]                                          # across-rows axis
    side = (cents - cents.mean(0)) @ perp
    row_id = (side > 0).astype(int)                        # 2 rows
    for r in (0, 1):
        members = [names_r[i] for i in range(len(names_r)) if row_id[i] == r]
        if not members:
            continue
        maj = Counter(out[m]["facing"] for m in members).most_common(1)[0][0]
        for m in members:
            if out[m]["facing"] != maj:
                out[m]["facing_raw"] = out[m]["facing"]
                out[m]["source"] += "+rowfix"
            out[m]["facing"] = maj
            out[m]["row"] = r

    for n in sorted(out):
        fx = f"  was {out[n]['facing_raw']}" if "facing_raw" in out[n] else ""
        print(f"  {n:14s} row{out[n].get('row','?')} facing={out[n]['facing']:8s} ({out[n]['source']}){fx}")
    (args.labeled_run / "rack_facing.json").write_text(json.dumps(out, indent=2))
    nf = sum("rowfix" in v["source"] for v in out.values())
    rows = {r: Counter(v["facing"] for v in out.values() if v.get("row") == r).most_common(1)[0][0]
            for r in (0, 1)}
    print(f"\nwrote {args.labeled_run/'rack_facing.json'}; row facings={rows}; end-rack fixes={nf}/{len(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
