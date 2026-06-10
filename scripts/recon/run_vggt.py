#!/usr/bin/env python3
"""VGGT-1B feed-forward reconstruction -> colored PLY.

Cross-frame global attention means VRAM scales with frame count; on an 8 GB
card ~20 frames at 518px is the safe ceiling, so we evenly subsample.

Usage:
    python scripts/recon/run_vggt.py IMG_DIR OUT.ply [--frames 20] [--conf-pct 30]
"""
import argparse
import glob
import os
from pathlib import Path

import numpy as np
import torch

from _recon_io import add_vendor_paths, even_subsample, clean_and_write_ply, save_cam_up

add_vendor_paths("vggt")
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("img_dir", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--frames", type=int, default=20, help="evenly-subsampled frame count")
    ap.add_argument("--conf-pct", type=float, default=30.0,
                    help="drop points below this depth-confidence percentile")
    args = ap.parse_args()

    device = "cuda"
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    files = sorted(glob.glob(os.path.join(args.img_dir, "*.jpg")))
    files = even_subsample(files, args.frames)
    print(f"VGGT: {len(files)} frames (of subsample target {args.frames})")

    # bf16 weights (not just bf16 autocast) halve the 1B-param footprint
    # ~4GB->2GB, which is what makes this fit in 8GB VRAM.
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).to(dtype).eval()
    images = load_and_preprocess_images(files).to(device)  # (N,3,H,W) in [0,1]
    print(f"  input {tuple(images.shape)}, weights={dtype}, "
          f"free VRAM {torch.cuda.mem_get_info()[0]/1e9:.1f} GB")

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
        batch = images[None]  # (1,N,3,H,W)
        tokens, ps_idx = model.aggregator(batch)
        pose_enc = model.camera_head(tokens)[-1]
        extr, intr = pose_encoding_to_extri_intri(pose_enc, batch.shape[-2:])
        torch.cuda.empty_cache()
        depth_map, depth_conf = model.depth_head(tokens, batch, ps_idx)

    # depth_map (1,N,H,W,1) -> world points (N,H,W,3)
    # gravity-up from camera orientations (extr is world->cam; transpose -> cam->world)
    e = extr.squeeze(0).detach().cpu().numpy()
    save_cam_up(np.transpose(e[:, :3, :3], (0, 2, 1)), str(args.out))

    pts = unproject_depth_map_to_point_map(
        depth_map.squeeze(0), extr.squeeze(0), intr.squeeze(0))
    conf = depth_conf.squeeze(0).float().cpu().numpy()              # (N,H,W)
    cols = images.permute(0, 2, 3, 1).float().cpu().numpy()         # (N,H,W,3) in [0,1]

    pts = pts.reshape(-1, 3)
    cols = cols.reshape(-1, 3)
    conf = conf.reshape(-1)
    thr = np.percentile(conf, args.conf_pct)
    keep = conf >= thr
    print(f"  conf>={thr:.3f} keeps {keep.sum():,}/{keep.size:,} points")

    clean_and_write_ply(args.out, pts[keep], cols[keep])
    print(f"peak VRAM {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
