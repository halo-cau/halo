"""Name class-agnostic Point-SAM instances with the geometry-priors namer.

Loads a ``*_instances.npz`` (keys: ``pts`` (N,3), ``labels`` (N,) instance ids,
optional ``rgb``), runs :func:`engine.vision.instance_namer.name_instances`,
prints a per-instance report, and writes:

  * ``<stem>_named.ply``        — point cloud recolored by semantic class
  * ``<stem>_named.legend.json`` — per-instance labels + features + the palette

Then view it (matches the "serve PLY + ?ply= URL" workflow)::

    python -m http.server 8000          # from the repo root
    # open tools/ply_viewer.html?ply=/<path-to>/<stem>_named.ply

Usage::

    conda run -n halo python scripts/name_pointsam_instances.py \\
        --npz server_room_phone/my_room_images/pointsam_instances.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.vision.instance_namer import NamerConfig, name_instances  # noqa: E402
from engine.vision.segmentor_base import LABEL_PALETTE  # noqa: E402

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_NPZ = _PROJECT_ROOT / "server_room_phone" / "my_room_images" / "pointsam_instances.npz"


def _load_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    keys = set(data.keys())
    pts_key = next((k for k in ("pts", "points", "xyz", "coords") if k in keys), None)
    lab_key = next((k for k in ("labels", "instance", "instances", "instance_ids", "seg") if k in keys), None)
    if pts_key is None or lab_key is None:
        raise SystemExit(f"{path} must contain point + instance arrays; found keys {sorted(keys)}")
    pts = np.asarray(data[pts_key], dtype=np.float64)
    labels = np.asarray(data[lab_key]).ravel().astype(np.int64)
    return pts, labels


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", type=Path, default=_DEFAULT_NPZ, help="Instances .npz (pts + labels).")
    ap.add_argument("--out", type=Path, default=None, help="Output named .ply (default: <stem>_named.ply).")
    ap.add_argument("--up-axis", type=int, default=2, choices=[0, 1, 2],
                    help="Vertical axis index (default 2 = Z). Pass -1 to auto-infer.")
    args = ap.parse_args()

    npz_path = args.npz.expanduser().resolve()
    if not npz_path.exists():
        raise SystemExit(f"Not found: {npz_path}")
    pts, labels = _load_npz(npz_path)
    up = None if args.up_axis < 0 else args.up_axis

    print(f"Loaded {len(pts):,} points, {len(np.unique(labels))} instances from {npz_path.name}")
    result = name_instances(pts, labels, cfg=NamerConfig(), up_axis=up)
    fr = result.frame
    print(f"Room frame: up=axis{fr.up_axis}  floor={fr.floor:.2f}  ceiling={fr.ceiling:.2f}  "
          f"height={fr.height:.2f}m  footprint≈{fr.hi[fr.lateral_axes[0]]-fr.lo[fr.lateral_axes[0]]:.1f}×"
          f"{fr.hi[fr.lateral_axes[1]]-fr.lo[fr.lateral_axes[1]]:.1f}m\n")

    print(f"{'inst':>4} {'label':<12} {'conf':>5} {'pts':>7}   reason")
    print("-" * 100)
    for ni in result.instances:
        print(f"{ni.instance_id:>4} {ni.label:<12} {ni.confidence:>5.2f} {ni.features.n_points:>7}   {ni.reason}")

    print("\nInstances per class:", result.label_counts())
    pcounts = result.point_label_counts()
    total = sum(pcounts.values())
    print("Points per class:    ", {k: f"{v} ({100*v/total:.0f}%)" for k, v in
                                    sorted(pcounts.items(), key=lambda kv: -kv[1])})

    # --- write class-colored PLY ---
    out_ply = (args.out or npz_path.with_name(npz_path.stem.replace("_instances", "") + "_named.ply")).resolve()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(result.point_colors)
    o3d.io.write_point_cloud(str(out_ply), pcd)
    print(f"\nWrote named cloud → {out_ply}")

    # --- write legend JSON (per-instance + palette used) ---
    legend = result.to_json()
    legend["palette"] = {lbl: [round(c, 4) for c in LABEL_PALETTE.get(lbl, LABEL_PALETTE["unknown"])]
                         for lbl in result.label_counts()}
    legend["source_npz"] = str(npz_path)
    out_json = out_ply.with_suffix(".legend.json")
    out_json.write_text(json.dumps(legend, indent=2))
    print(f"Wrote legend      → {out_json}")

    rel = out_ply.relative_to(_PROJECT_ROOT) if out_ply.is_relative_to(_PROJECT_ROOT) else out_ply
    print(f"\nView it (port 8011 avoids the FastAPI backend on 8000):\n"
          f"  python -m http.server 8011     # from {_PROJECT_ROOT}\n"
          f"  open  http://localhost:8011/tools/ply_viewer.html?ply=/{rel}")


if __name__ == "__main__":
    main()
