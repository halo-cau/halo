#!/usr/bin/env python3
"""Diagnose a capture's multi-view geometry from MASt3R camera poses.

Reports, per consecutive view pair, the translation (baseline), the rotation
angle, and the resulting parallax angle relative to scene depth.  Low parallax
(near-pure rotation) is what makes triangulated depth noisy — this quantifies it.

Usage:
    python scripts/recon/analyze_capture.py IMG_DIR [--frames 20] [--winsize 3]
"""
import argparse
import glob
import os
import tempfile
from pathlib import Path

import numpy as np
import torch

from _recon_io import add_vendor_paths, even_subsample

add_vendor_paths("mast3r")
import mast3r.utils.path_to_dust3r  # noqa: F401
from mast3r.model import AsymmetricMASt3R
from mast3r.image_pairs import make_pairs
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from dust3r.utils.image import load_images

WEIGHTS = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("img_dir", type=Path)
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--winsize", type=int, default=3)
    args = ap.parse_args()
    device = "cuda"

    files = even_subsample(sorted(glob.glob(os.path.join(args.img_dir, "*.jpg"))), args.frames)
    model = AsymmetricMASt3R.from_pretrained(WEIGHTS).to(device).eval()
    imgs = load_images(files, size=512, verbose=False)
    pairs = make_pairs(imgs, scene_graph=f"swin-{args.winsize}", prefilter=None, symmetrize=True)
    with tempfile.TemporaryDirectory() as cache:
        scene = sparse_global_alignment(files, pairs, cache, model, lr1=0.07, niter1=200,
                                        lr2=0.014, niter2=200, device=device,
                                        opt_depth=True, shared_intrinsics=True, matching_conf_thr=5.0)
        poses = scene.get_im_poses().detach().cpu().numpy()           # (N,4,4) cam2world
        pts3d, _, _ = scene.get_dense_pts3d(clean_depth=True)
        pts = np.concatenate([p.reshape(-1, 3).detach().cpu().numpy() for p in pts3d])

    centers = poses[:, :3, 3]
    fwd = poses[:, :3, 2]                                            # camera forward (+Z) in world
    depth = float(np.median(np.linalg.norm(pts - centers.mean(0), axis=1)))

    base, rot, par = [], [], []
    for i in range(len(centers) - 1):
        b = float(np.linalg.norm(centers[i + 1] - centers[i]))
        cang = float(np.clip(np.dot(fwd[i], fwd[i + 1]), -1, 1))
        base.append(b)
        rot.append(np.degrees(np.arccos(cang)))
        par.append(np.degrees(np.arctan2(b, depth)))                # parallax vs scene depth

    span = np.ptp(centers, axis=0)
    print(f"frames={len(files)}  median scene depth≈{depth:.2f} (recon units)")
    print(f"camera path: total {sum(base):.2f}, bbox span {np.round(span,2)}")
    print(f"per-step baseline  median {np.median(base):.3f}  (min {min(base):.3f} max {max(base):.3f})")
    print(f"per-step rotation  median {np.median(rot):.1f} deg")
    print(f"per-step PARALLAX  median {np.median(par):.2f} deg  (min {min(par):.2f})")
    good = np.mean(np.array(par) >= 2.0) * 100
    print(f"steps with parallax >= 2deg: {good:.0f}%")
    print("VERDICT:", "LOW parallax -> noisy depth (move more between shots)" if np.median(par) < 2.0
          else "parallax OK -> noise is from texture/overlap/meshing, not baseline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
