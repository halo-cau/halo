"""Predict server-room semantics with a finetuned Sonata head; export a viewer PLY.

Works on the las6 .npz (reports per-class IoU on the val split) OR on any point
cloud — e.g. the MASt3R/Point-SAM MVS cloud — to test cross-domain transfer
(LiDAR-trained → photo-reconstructed). Outputs a class-colored PLY + legend for
tools/ply_viewer.html.

Run in the `sonata` env:
    conda run -n sonata python scripts/sonata/predict_sonata_las6.py --ckpt data/sonata_las6/sonata_head_las6.pth
    # cross-domain test on the MVS cloud:
    conda run -n sonata python scripts/sonata/predict_sonata_las6.py --ckpt ... \
        --input server_room_phone/my_room_images/server_room6_rgb.ply
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from _sonata_common import (  # noqa: E402  (same dir)
    CLASS_NAMES,
    load_encoder,
    per_class_iou,
    predict_labels,
    save_labeled_ply,
)
from finetune_sonata_las6 import SegHead  # reuse head definition

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_INPUT = _ROOT / "data" / "sonata_las6" / "las6.npz"


def _load_cloud(path: Path) -> tuple[dict, np.ndarray | None, np.ndarray | None]:
    """Return ({coord,color(0-255),normal}, gt_or_None, split_or_None)."""
    import open3d as o3d

    if path.suffix == ".npz":
        d = np.load(path)
        keys = set(d.keys())
        if "coord" in keys:  # our las6 format (color already 0-255)
            coord, color, normal = d["coord"], d["color"], d["normal"]
            gt = d["segment"] if "segment" in keys else None
            split = d["split"] if "split" in keys else None
        elif "pts" in keys:  # Point-SAM export (rgb 0-1)
            coord, color = d["pts"], d["rgb"] * 255.0
            normal, gt, split = None, None, None
        else:
            raise SystemExit(f"Unrecognized npz keys: {sorted(keys)}")
    else:  # .ply / .pcd
        pcd = o3d.io.read_point_cloud(str(path))
        coord = np.asarray(pcd.points, dtype=np.float32)
        color = (np.asarray(pcd.colors, dtype=np.float32) * 255.0) if pcd.has_colors() \
            else np.full((len(coord), 3), 128.0, dtype=np.float32)
        normal, gt, split = None, None, None

    coord = coord.astype(np.float32)
    if normal is None:  # Sonata needs real normals; compute if the cloud lacks them
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coord.astype(np.float64))
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=32))
        pcd.orient_normals_towards_camera_location(coord.mean(0).astype(np.float64))
        normal = np.asarray(pcd.normals, dtype=np.float32)
    raw = {"coord": coord, "color": color.astype(np.float32), "normal": normal.astype(np.float32)}
    return raw, (gt if gt is None else gt.astype(np.int64)), split


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--input", type=Path, default=_DEFAULT_INPUT)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--patch", type=int, default=1024)
    ap.add_argument("--enable-flash", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    enc = ckpt.get("encoder", {})
    model = load_encoder(enable_flash=args.enable_flash or enc.get("enable_flash", False),
                         patch=enc.get("patch", args.patch), device=args.device)
    head = SegHead(**ckpt["config"]).to(args.device)
    head.load_state_dict(ckpt["state_dict"])

    raw, gt, split = _load_cloud(args.input)
    print(f"Loaded {len(raw['coord']):,} pts from {args.input.name}")
    pred = predict_labels(model, head, raw, args.device)

    counts = {CLASS_NAMES[c]: int((pred == c).sum()) for c in range(len(CLASS_NAMES))}
    total = len(pred)
    print("pred per class:", {k: f"{v} ({100*v/total:.0f}%)" for k, v in counts.items() if v})
    if gt is not None:
        mask = (split == 1) if split is not None else np.ones(len(gt), bool)
        scope = "val split" if split is not None else "all labeled"
        iou = per_class_iou(pred[mask], gt[mask])
        print(f"[{scope}] mIoU={iou['mIoU']}  " + " ".join(f"{n}={iou[n]}" for n in CLASS_NAMES))

    out = args.out or (_ROOT / "data" / "sonata_las6" / f"pred_{args.input.stem}.ply")
    save_labeled_ply(raw["coord"], pred, out)
    rel = out.relative_to(_ROOT) if out.is_relative_to(_ROOT) else out
    print(f"\nWrote {out}")
    print(f"View:\n  python -m http.server 8011    # from {_ROOT}\n"
          f"  http://localhost:8011/tools/ply_viewer.html?ply=/{rel}")


if __name__ == "__main__":
    main()
