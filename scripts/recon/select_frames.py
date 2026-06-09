#!/usr/bin/env python3
"""Pose-aware frame selection for Pi3 reconstruction.

When the capture pool exceeds Pi3's 8 GB token budget (~37.5k tokens =
frames * pixel_limit/196), *which* frames you keep matters more than how many:
keep maximal unique coverage per token, drop redundant viewpoints and bad
frames. Pipeline:

  0. convert  any images -> upright RGB jpgs
  1. quality gate  drop motion-blur (low Laplacian variance) + blown/dark
                   (clipped-histogram) frames -- Pi3's conf mask culls them anyway
  2. scout     low-res Pi3 over the survivors -> per-frame camera pose + valid
               fraction (cheap: fits all frames at a small pixel_limit)
  3. select    drop low-confidence frames, then farthest-point sample in pose
               space (camera position + viewing direction) to the budget N ->
               guarantees angular/positional spread, keeps the lone top-down /
               gap-filler views (they are far in pose space), drops near-dupes

Outputs the chosen jpgs into --out plus selection_report.json. Feed --out to
pipeline_web.py --model pi3 for the full-res run.

Usage (halo env; shells to the isolated `pi3` env for the scout):
    python scripts/recon/select_frames.py --images DIR --out DIR \
        [--target-frames 50] [--scout-pixel-limit 50000] \
        [--sharp-pct 15] [--conf-gate-pct 10]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pillow_heif
from PIL import Image, ImageOps
from scipy.ndimage import laplace

pillow_heif.register_heif_opener()

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CONDA = os.path.expanduser("~/ENTER/bin/conda")
IMG_EXT = {".heic", ".heif", ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def convert(images: Path, jpg: Path) -> list[str]:
    """any images -> upright RGB jpgs (EXIF-transposed, <=1600px). Returns stems."""
    jpg.mkdir(parents=True, exist_ok=True)
    srcs = sorted(p for p in images.iterdir() if p.suffix.lower() in IMG_EXT)
    out = []
    for i, s in enumerate(srcs):
        im = ImageOps.exif_transpose(Image.open(s)).convert("RGB")
        if max(im.size) > 1600:
            im.thumbnail((1600, 1600), Image.LANCZOS)
        name = f"{i:03d}_{s.stem}.jpg"
        im.save(jpg / name, "JPEG", quality=95)
        out.append(name)
    return out


def quality_metrics(jpg_path: Path) -> tuple[float, float]:
    """(sharpness, clip_frac): Laplacian variance (focus) and fraction of
    near-saturated/near-black pixels (blown highlights / crushed shadows)."""
    g = np.asarray(Image.open(jpg_path).convert("L"), dtype=np.float32)
    g /= 255.0
    sharp = float(laplace(g).var())
    clip = float(((g > 0.98) | (g < 0.02)).mean())
    return sharp, clip


def fps_select(feat: np.ndarray, n: int, seed: int) -> list[int]:
    """Farthest-point sampling: greedily add the frame maximizing min-distance
    to the already-selected set, starting from ``seed``."""
    sel = [seed]
    d = np.linalg.norm(feat - feat[seed], axis=1)
    while len(sel) < n:
        i = int(d.argmax())
        if i in sel:
            break
        sel.append(i)
        d = np.minimum(d, np.linalg.norm(feat - feat[i], axis=1))
    return sel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True, help="source image pool (any format)")
    ap.add_argument("--out", type=Path, required=True, help="dir to write the selected jpgs")
    ap.add_argument("--target-frames", type=int, default=50, help="budget N to select")
    ap.add_argument("--scout-pixel-limit", type=int, default=50000,
                    help="low res for the all-frames scout pass (fits many frames cheaply)")
    ap.add_argument("--sharp-pct", type=float, default=15.0,
                    help="drop the blurriest this %% of frames (Laplacian variance)")
    ap.add_argument("--clip-max", type=float, default=0.35,
                    help="drop frames with more than this fraction of clipped pixels")
    ap.add_argument("--conf-gate-pct", type=float, default=10.0,
                    help="drop the lowest this %% of frames by Pi3 scout valid-fraction")
    ap.add_argument("--dir-weight", type=float, default=1.0,
                    help="weight of viewing-direction vs normalized position in pose-space FPS")
    args = ap.parse_args()

    work = args.out.parent / (args.out.name + "_work")
    jpg = work / "jpg"
    print(f"[0] converting {args.images} -> {jpg}")
    names = convert(args.images, jpg)
    print(f"    {len(names)} images")

    # 1. quality gate ------------------------------------------------------
    metrics = {n: quality_metrics(jpg / n) for n in names}
    sharps = np.array([metrics[n][0] for n in names])
    thr = np.percentile(sharps, args.sharp_pct)
    blurry = [n for n in names if metrics[n][0] < thr]
    blown = [n for n in names if metrics[n][1] > args.clip_max]
    gated = [n for n in names if n not in set(blurry) | set(blown)]
    print(f"[1] quality gate: drop {len(blurry)} blurry (sharp<{thr:.4f}) + "
          f"{len(set(blown) - set(blurry))} blown -> {len(gated)} survive")

    gdir = work / "gated"
    gdir.mkdir(parents=True, exist_ok=True)
    for n in gated:
        shutil.copyfile(jpg / n, gdir / n)

    if len(gated) <= args.target_frames:
        print(f"[2-3] {len(gated)} <= target {args.target_frames}: keeping all survivors")
        selected = gated
        scout_info = "skipped (within budget)"
    else:
        # 2. scout (isolated pi3 env) -------------------------------------
        scout = work / "scout.npz"
        print(f"[2] Pi3 scout over {len(gated)} frames @ {args.scout_pixel_limit}px ...")
        env = dict(os.environ, HF_HUB_OFFLINE="0",
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        r = subprocess.run(
            [CONDA, "run", "--no-capture-output", "-n", "pi3", "python",
             "scripts/recon/run_pi3.py", str(gdir), str(work / "scout.ply"),
             "--pixel-limit", str(args.scout_pixel_limit), "--scout-out", str(scout)],
            cwd=str(REPO), env=env)
        if r.returncode != 0 or not scout.exists():
            print("ERROR: scout failed", file=sys.stderr)
            return 1
        z = np.load(scout, allow_pickle=True)
        poses, conf_frac = z["poses"], z["conf_frac"]
        sfiles = [str(x) for x in z["files"]]

        # 3. select: conf-gate then farthest-point in pose space ----------
        keep = conf_frac >= np.percentile(conf_frac, args.conf_gate_pct)
        kidx = np.where(keep)[0]
        C = poses[kidx, :3, 3]                                   # camera centers
        d = poses[kidx, :3, 2]                                   # viewing directions
        d = d / np.clip(np.linalg.norm(d, axis=1, keepdims=True), 1e-9, None)
        Cn = (C - C.mean(0)) / (C.std() + 1e-9)                  # isotropic normalize
        feat = np.concatenate([Cn, args.dir_weight * d], axis=1)
        seed = int(kidx[np.argmax(conf_frac[kidx])])             # start from the best frame
        seed_local = int(np.where(kidx == seed)[0][0])
        chosen_local = fps_select(feat, min(args.target_frames, len(kidx)), seed_local)
        selected = [sfiles[kidx[i]] for i in chosen_local]
        scout_info = {"scouted": len(gated), "conf_gated_out": int((~keep).sum()),
                      "selected": len(selected), "mean_conf": float(conf_frac.mean())}
        print(f"[3] conf-gate drop {int((~keep).sum())}, FPS select {len(selected)} "
              f"of {len(kidx)} in pose space")

    # write selected jpgs + report ----------------------------------------
    args.out.mkdir(parents=True, exist_ok=True)
    for n in selected:
        shutil.copyfile(jpg / n, args.out / n)
    report = {
        "n_source": len(names), "n_selected": len(selected),
        "target_frames": args.target_frames,
        "dropped_blurry": sorted(blurry), "dropped_blown": sorted(set(blown) - set(blurry)),
        "scout": scout_info, "selected": sorted(selected),
    }
    (args.out / "selection_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nselected {len(selected)} frames -> {args.out}")
    print(f"report -> {args.out / 'selection_report.json'}")
    print(f"next: pipeline_web.py --images {args.out} --out <run> --model pi3 "
          f"--pi3-frames {len(selected)} --pi3-pixel-limit <fit>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
