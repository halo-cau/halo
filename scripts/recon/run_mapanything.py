#!/usr/bin/env python3
"""MapAnything (Meta, 2025) reconstruction -> colored PLY.

Unified feed-forward metric 3D: one forward pass over all views, no global
alignment / pairing graph -> no "entangled slices" failure mode.  Run with the
ISOLATED `mapany` conda env's python (it pins uniception/torch that would clash
with halo); the pipeline shells out to it.  Memory-efficient inference with
minibatch_size=1 keeps 8 GB viable (~0.07 GB/view + model).

Usage (isolated env):
    conda run -n mapany python scripts/recon/run_mapanything.py IMG_DIR OUT.ply \
        [--frames 0] [--conf-pct 10] [--apache]
"""
import argparse
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_OFFLINE", "0")  # weights fetched once, then cached

import glob
from pathlib import Path

import numpy as np
import torch

from _recon_io import even_subsample, clean_and_write_ply, save_cam_up

APACHE = "facebook/map-anything-apache"   # Apache-2.0 weights
FULL = "facebook/map-anything"            # CC-BY-NC weights (higher quality)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("img_dir", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--frames", type=int, default=0, help="subsample to N frames (0 = all)")
    ap.add_argument("--conf-pct", type=int, default=10,
                    help="drop this bottom-percentile of confidence pixels")
    ap.add_argument("--apache", action="store_true",
                    help="use Apache-2.0 weights (default uses the full CC-BY-NC model)")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from mapanything.models import MapAnything
    from mapanything.utils.image import load_images

    files = sorted(glob.glob(os.path.join(str(args.img_dir), "*.jpg")))
    if args.frames:
        files = even_subsample(files, args.frames)
    print(f"MapAnything: {len(files)} imgs, weights={APACHE if args.apache else FULL}")

    model = MapAnything.from_pretrained(APACHE if args.apache else FULL).to(device).eval()
    views = load_images(files)

    with torch.no_grad():
        preds = model.infer(
            views,
            memory_efficient_inference=True,
            minibatch_size=1,              # smallest 8 GB footprint
            use_amp=True, amp_dtype="bf16",
            apply_mask=True, mask_edges=True,
            apply_confidence_mask=True, confidence_percentile=args.conf_pct,
        )

    P, C, rots = [], [], []
    for pred in preds:
        pts = pred["pts3d"].reshape(-1, 3).float().cpu().numpy()
        rgb = pred["img_no_norm"].reshape(-1, 3).float().cpu().numpy()
        m = pred["mask"].reshape(-1).bool().cpu().numpy()
        P.append(pts[m]); C.append(rgb[m])
        rots.append(pred["camera_poses"][0, :3, :3].float().cpu().numpy())

    P = np.concatenate(P); C = np.concatenate(C)
    if C.max() > 1.5:                      # img_no_norm came back as 0..255
        C = C / 255.0
    print(f"  masked cloud: {len(P):,} points")
    save_cam_up(np.stack(rots), str(args.out))
    clean_and_write_ply(str(args.out), P, C)
    if device == "cuda":
        print(f"peak VRAM {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
