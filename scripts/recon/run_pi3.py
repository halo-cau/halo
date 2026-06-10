#!/usr/bin/env python3
"""Pi3 (π³, ICCV 2025) reconstruction -> colored PLY (+ optional per-view dump).

Permutation-equivariant feed-forward geometry: all views attend jointly in one
pass, no reference frame, no pairing graph -> robust, and (like MapAnything) it
can't hit the SfM "entangled slices" failure.  Run with the ISOLATED `pi3` env
(pins torch 2.5.1, clashes with halo/mapany); the pipeline shells out to it.

For the real-photo SAM3 labeler, --dump-views writes per-view pts3d + a
validity-encoded conf to <recon>.views/: Pi3's confidence is a sigmoid prob on a
different scale than MASt3R's, so we store conf=2.0 where Pi3's mask
(sigmoid(conf)>0.1 AND non-edge) holds and 0 elsewhere -> the labeler's
conf>1.5 gate keeps exactly Pi3's valid pixels, no special-casing.

Usage (isolated env):
    conda run -n pi3 python scripts/recon/run_pi3.py IMG_DIR OUT.ply [--frames 0] [--dump-views]
"""
import argparse
import glob
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

from _recon_io import even_subsample, clean_and_write_ply, save_cam_up


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("img_dir", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--frames", type=int, default=0, help="subsample to N frames (0 = all)")
    ap.add_argument("--pixel-limit", type=int, default=0,
                    help="per-view pixel budget for resize (0 = Pi3 default 255000). Peak VRAM "
                         "tracks total tokens = frames * (pixel_limit/196); lower this to fit "
                         "more frames in the same memory (coverage up, per-view detail down).")
    ap.add_argument("--dump-views", action="store_true",
                    help="save per-view pts3d/conf for real-photo SAM3 lifting")
    ap.add_argument("--scout-out", type=Path, default=None,
                    help="frame-selection scout: write {poses,conf_frac,files} npz and skip the "
                         "heavy PLY/dump-views. Run at low --pixel-limit over ALL frames to get "
                         "camera geometry + per-view validity cheaply, then select a subset.")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from pi3.models.pi3 import Pi3
    from pi3.utils.basic import load_images_as_tensor
    from pi3.utils.geometry import depth_normal_edge

    files = sorted(glob.glob(os.path.join(str(args.img_dir), "*.jpg")))
    li_kw = {"PIXEL_LIMIT": args.pixel_limit} if args.pixel_limit else {}
    imgs = load_images_as_tensor(str(args.img_dir), interval=1, **li_kw)  # (N,3,H,W) in [0,1]
    if args.frames and args.frames < len(files):
        idx = np.linspace(0, len(files) - 1, args.frames).round().astype(int)
        idx = sorted(set(idx.tolist()))
        imgs = imgs[idx]; files = [files[i] for i in idx]
    imgs = imgs.to(device)
    print(f"Pi3: {len(files)} imgs, shape {tuple(imgs.shape)}")

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        res = model_infer(Pi3, imgs)

    pts = res["points"][0]                                             # (N,H,W,3) world
    conf = torch.sigmoid(res["conf"][0, ..., 0])                       # (N,H,W) prob
    valid = conf > 0.1
    non_edge = ~depth_normal_edge(res["local_points"], rtol=0.03, mask=(conf > 0.1)[None])[0]
    valid = (valid & non_edge)                                         # (N,H,W)
    rgb = imgs.permute(0, 2, 3, 1)                                     # (N,H,W,3) [0,1]

    if args.scout_out is not None:
        poses = res["camera_poses"][0].float().cpu().numpy()           # (N,4,4) cam->world
        conf_frac = valid.float().mean(dim=(1, 2)).cpu().numpy()       # per-view valid fraction
        args.scout_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.scout_out, poses=poses.astype(np.float32),
                            conf_frac=conf_frac.astype(np.float32),
                            files=np.array([os.path.basename(f) for f in files]))
        print(f"  scout -> {args.scout_out} ({len(files)} frames, "
              f"mean valid {conf_frac.mean():.3f})")
        if device == "cuda":
            print(f"peak VRAM {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
        return 0

    if args.dump_views:
        vdir = Path(str(args.out).replace(".ply", ".views"))
        vdir.mkdir(parents=True, exist_ok=True)
        for i in range(len(files)):
            np.savez_compressed(
                vdir / f"view_{i:03d}.npz",
                pts3d=pts[i].float().cpu().numpy().astype(np.float32),
                conf=np.where(valid[i].cpu().numpy(), 2.0, 0.0).astype(np.float32),
                img_path=str(files[i]))
        print(f"  dumped {len(files)} view geometries -> {vdir}")

    save_cam_up(res["camera_poses"][0, :, :3, :3].float().cpu().numpy(), str(args.out))
    P = pts[valid].float().cpu().numpy()
    C = rgb[valid].float().cpu().numpy()
    print(f"  valid cloud: {len(P):,} points")
    clean_and_write_ply(str(args.out), P, C)
    if device == "cuda":
        print(f"peak VRAM {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    return 0


def model_infer(Pi3, imgs):
    model = Pi3.from_pretrained("yyfz233/Pi3").to(imgs.device).eval()
    return model(imgs[None])                                          # add batch dim


if __name__ == "__main__":
    raise SystemExit(main())
