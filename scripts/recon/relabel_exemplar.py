#!/usr/bin/env python3
"""Carve a named instance out of the labeled cloud with a SAM3 BOX EXEMPLAR.

When two objects share a visual class — e.g. a network / switch rack versus the server
racks — a SAM3 *text* concept cannot separate them (it fires on every rack). Instead the
operator boxes the target in a few views; this prompts SAM3 with each box, lifts the masks
to 3D via the per-view pointmaps, and RELABELS the labeled cloud's points in that blob to a
given canonical, overriding their class. Generalizable: any room's operator designates the
network rack the same way (no incidental anchors).

Usage (halo env):
  python scripts/recon/relabel_exemplar.py --run <run> --views <run>/recon_raw.views \
    --up <run>/recon_raw.up.npy --label ups \
    --exemplar IMG_3173:0.50,0.05,0.78,1.0 --exemplar IMG_3151:0.62,0,1,1 \
    --exemplar IMG_3150:0.70,0,1,1 --exemplar IMG_3149:0.62,0,1,1
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _recon_io import gravity_R
from engine.vision.segmentor_dino_sam import _load_sam3, _predict_masks_sam3


def _resize(mask: np.ndarray, hw: tuple) -> np.ndarray:
    H, W = hw
    if mask.shape == (H, W):
        return mask
    return np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((W, H), Image.NEAREST)) > 127


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--views", type=Path, required=True)
    ap.add_argument("--up", type=Path, required=True)
    ap.add_argument("--label", default="ups", help="canonical to assign the boxed instance")
    ap.add_argument("--exemplar", action="append", required=True,
                    help="IMG_TAG:x0,y0,x1,y1 (normalized box); repeatable across views")
    ap.add_argument("--conf", type=float, default=1.0, help="per-view confidence gate on the pointmap")
    ap.add_argument("--dist", type=float, default=0.03, help="relabel radius in cloud units")
    ap.add_argument("--apply", action="store_true", help="write the relabeled cloud (else dry-run report)")
    args = ap.parse_args()

    R = gravity_R(np.load(str(args.up)))
    views = {Path(str(np.load(v, allow_pickle=True)["img_path"])).name: v
             for v in sorted(args.views.glob("view_*.npz"))}

    proc = _load_sam3(None, "cuda", 0.4)
    blob = []
    for ex in args.exemplar:
        tag, box = ex.split(":"); x0, y0, x1, y1 = (float(t) for t in box.split(","))
        vf = next((p for n, p in views.items() if tag in n), None)
        if vf is None:
            print(f"  !! no view matches {tag}"); continue
        d = np.load(vf, allow_pickle=True)
        pts3d, conf = d["pts3d"], d["conf"]
        rgb = np.asarray(Image.open(str(d["img_path"])).convert("RGB"), np.uint8)
        Hi, Wi = rgb.shape[:2]
        box_px = np.array([[x0 * Wi, y0 * Hi, x1 * Wi, y1 * Hi]])
        mask = _predict_masks_sam3(proc, rgb, box_px, [args.label])[0]      # (Hi,Wi) bool
        sel = _resize(mask, conf.shape) & (conf > args.conf)
        p = pts3d[sel].reshape(-1, 3)
        blob.append(p)
        print(f"  {tag}: mask {int(mask.sum())} px -> {len(p)} lifted points")
    B = (np.concatenate(blob) @ R.T) if blob else np.empty((0, 3))
    if not len(B):
        print("no exemplar points lifted"); return 1
    c = B.mean(0); ext = B.max(0) - B.min(0)
    print(f"BLOB: {len(B)} pts  centroid=({c[0]:.2f},{c[1]:.2f},{c[2]:.2f})  ext={ext.round(2)}")

    pc = trimesh.load(args.run / "labeled.ply", process=False)
    V = np.asarray(pc.vertices, float)
    names = np.load(args.run / "point_labels.npz", allow_pickle=False)["names"].astype("<U24")
    near = cKDTree(B).query(V, k=1, workers=-1)[0] < args.dist
    from collections import Counter
    print(f"recon points within {args.dist} of blob: {int(near.sum())}  "
          f"(currently: {dict(Counter(names[near]).most_common(5))})")
    if not args.apply:
        print("dry-run (pass --apply to write)"); return 0

    if args.label == "ups":           # the exemplar IS the true UPS/network rack; drop SAM3's stray
        names[np.char.startswith(names.astype(str), "ups")] = "unknown"   # 'ups' mislabels (incl. the
        #                                            fire-hose cabinet) so they don't collide / inflate it
    names[near] = f"{args.label} 1"
    np.savez(args.run / "point_labels.npz", names=names)
    # recolor the relabeled points so labeled.ply matches (bright yellow for ups)
    col = np.asarray(pc.colors)[:, :3] if pc.colors is not None and len(pc.colors) == len(V) \
        else np.full((len(V), 3), 120, np.uint8)
    col = np.array(col, dtype=np.uint8)
    col[near] = (235, 195, 25)
    trimesh.PointCloud(vertices=V.astype(np.float32), colors=col).export(args.run / "labeled.ply")
    print(f"relabeled {int(near.sum())} points -> '{args.label} 1'; rewrote labeled.ply + point_labels.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
