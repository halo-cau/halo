"""Apply a standard 42U rack geometric prior to the labeled point cloud.

A server rack cannot physically be shaped differently from the EIA-310 /
ASHRAE 42U standard: 60 cm wide × 100 cm deep × 200 cm tall. Rather than
trusting scan-noise-bleed in the current rack labels (wall vertices behind
racks, floor vertices in front, etc.), this script:

1. Re-detects rack rows via K-means on the X coordinate of currently
   rack-labeled points (so X is shared across racks in the same row,
   removing per-rack centroid bias from scanner viewing angles).
2. For each rack instance, computes its Y centroid from the current
   labels and builds a synthetic 42U bbox at ``(row_X, Y_centroid)``.
3. Re-labels vertices by bbox containment:
   - Inside any bbox → ``server_rack`` with that bbox's instance ID.
   - Currently rack but outside all bboxes → ``ignore``.
   - ``(class, instance)`` pairs listed via ``--preserve`` are skipped
     (default: ``5:1`` so the on-rack box cluster stays as box_clutter).

The result: clean 42U rack instances aligned to the standard footprint,
free of wall/floor label noise.

Usage::

    python scripts/force_rack_prior.py --scene las6_corrected

Override defaults if your demo room has wider racks::

    python scripts/force_rack_prior.py --scene las6_corrected \\
        --rack-dims 0.60 1.20 2.00   # 1200 mm deep
"""

from __future__ import annotations

import argparse
import colorsys
from pathlib import Path

import numpy as np
import open3d as o3d
from sklearn.cluster import KMeans

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "mask3d_server_room"

_GOLDEN_RATIO_CONJ = 0.6180339887498949


def _color_for(sid: int, iid: int) -> tuple[int, int, int]:
    base = (sid * 0.137) % 1.0
    h = (base + iid * _GOLDEN_RATIO_CONJ) % 1.0
    sat = 0.62 + 0.10 * ((iid * 3) % 5) / 4.0
    val = 0.88 + 0.07 * ((iid * 5) % 3) / 2.0
    r, g, b = colorsys.hsv_to_rgb(h, sat, val)
    return int(r * 255), int(g * 255), int(b * 255)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", required=True)
    p.add_argument("--split", default="train", choices=("train", "validation"))
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--rack-class-id", type=int, default=4)
    p.add_argument("--n-rows", type=int, default=2)
    p.add_argument("--row-axis", choices=("x", "y", "z"), default="x")
    p.add_argument("--width-axis", choices=("x", "y", "z"), default="y",
                   help="World axis that the rack's 60 cm width is aligned with.")
    p.add_argument(
        "--rack-dims",
        type=float,
        nargs=3,
        default=(0.60, 1.00, 2.00),
        metavar=("WIDTH", "DEPTH", "HEIGHT"),
        help="Rack width / depth / height in metres (default = 42U standard).",
    )
    p.add_argument(
        "--preserve",
        nargs="+",
        default=["5:1"],
        metavar="CLASS:INST",
        help="(class_id, instance_id) pairs to leave untouched even when inside "
             "a rack bbox. Default preserves box_clutter inst1 (boxes on top of "
             "racks).",
    )
    args = p.parse_args()

    row_axis = {"x": 0, "y": 1, "z": 2}[args.row_axis]
    width_axis = {"x": 0, "y": 1, "z": 2}[args.width_axis]
    height_axis = 2
    # The remaining axis is depth.
    depth_axis = next(a for a in (0, 1, 2) if a not in (row_axis, width_axis))
    if row_axis != depth_axis:
        # Standard layout: row_axis IS the depth_axis (rows extend along the
        # width direction; each rack is depth-aligned with the row-separating axis).
        # We expect row_axis == depth_axis. Warn if not.
        print(
            f"WARNING: row-axis={args.row_axis} differs from inferred depth-axis="
            f"{'xyz'[depth_axis]}. The standard layout has rack depth perpendicular "
            f"to the rack-row line."
        )

    width, depth, height = args.rack_dims
    preserve_set: set[tuple[int, int]] = set()
    for spec in args.preserve:
        cls_str, inst_str = spec.split(":")
        preserve_set.add((int(cls_str), int(inst_str)))

    npy_path = args.data_root / args.split / f"{args.scene}.npy"
    gt_path = args.data_root / "instance_gt" / args.split / f"{args.scene}.txt"
    preview = args.data_root / "previews" / f"{args.scene}_labels_preview.ply"

    arr = np.load(npy_path)
    sem = arr[:, 10].astype(np.int32)
    inst = arr[:, 11].astype(np.int32)
    xyz = arr[:, :3]
    print(f"Loaded {npy_path}  N={len(arr):,}")

    # --- Step 1: K-means on the row axis using current rack points to find rows.
    rack_mask = sem == args.rack_class_id
    rack_xyz = xyz[rack_mask]
    if len(rack_xyz) == 0:
        raise SystemExit("No points labelled as the rack class — nothing to do.")
    km = KMeans(n_clusters=args.n_rows, n_init=10, random_state=42).fit(rack_xyz[:, row_axis:row_axis+1])
    row_centroids = sorted(km.cluster_centers_.ravel().tolist())
    print(f"Row centroids on axis {args.row_axis}: {[round(c, 3) for c in row_centroids]}")

    # --- Step 2: per-instance synthetic bbox.
    # Use row centroid for the row axis (averaged across racks in the row),
    # and the instance's data centroid for the width axis.
    rack_instances = sorted(np.unique(inst[rack_mask]).tolist())
    bboxes: list[tuple[int, np.ndarray, np.ndarray]] = []
    for inst_id in rack_instances:
        pts = xyz[rack_mask & (inst == inst_id)]
        if len(pts) == 0:
            continue
        # Pick the closest row centroid for this instance.
        cx = pts[:, row_axis].mean()
        row_x = min(row_centroids, key=lambda c: abs(c - cx))
        wy = float(np.median(pts[:, width_axis]))
        bbox_min = np.zeros(3, dtype=np.float64)
        bbox_max = np.zeros(3, dtype=np.float64)
        bbox_min[row_axis]    = row_x - depth / 2
        bbox_max[row_axis]    = row_x + depth / 2
        bbox_min[width_axis]  = wy - width / 2
        bbox_max[width_axis]  = wy + width / 2
        bbox_min[height_axis] = 0.0
        bbox_max[height_axis] = height
        bboxes.append((inst_id, bbox_min, bbox_max))
        print(
            f"  inst{inst_id:>2}  row_{args.row_axis}={row_x:>6.2f}  "
            f"width_{args.width_axis}={wy:>6.2f}  "
            f"bbox=[{bbox_min.round(2).tolist()} .. {bbox_max.round(2).tolist()}]"
        )

    # --- Step 3: vectorised inside-bbox check + relabel.
    N = len(xyz)
    K = len(bboxes)
    inside = np.zeros((N, K), dtype=bool)
    for k, (_, bmin, bmax) in enumerate(bboxes):
        inside[:, k] = (
            (xyz[:, 0] >= bmin[0]) & (xyz[:, 0] <= bmax[0]) &
            (xyz[:, 1] >= bmin[1]) & (xyz[:, 1] <= bmax[1]) &
            (xyz[:, 2] >= bmin[2]) & (xyz[:, 2] <= bmax[2])
        )
    inside_any = inside.any(axis=1)
    first_bbox_col = np.argmax(inside, axis=1)
    inst_ids_arr = np.array([b[0] for b in bboxes], dtype=np.int32)

    preserve_mask = np.zeros(N, dtype=bool)
    for (cls, ins) in preserve_set:
        preserve_mask |= (sem == cls) & (inst == ins)

    # Promote inside-bbox vertices to server_rack with the bbox's instance.
    to_promote = inside_any & ~preserve_mask
    promoted_class_change = to_promote & (sem != args.rack_class_id)
    promoted_instance_change = to_promote & (sem == args.rack_class_id) & (inst != inst_ids_arr[first_bbox_col])
    sem[to_promote] = args.rack_class_id
    inst[to_promote] = inst_ids_arr[first_bbox_col[to_promote]]

    # Demote stray rack labels outside all bboxes to ignore.
    to_demote = (~inside_any) & (sem == args.rack_class_id) & (~preserve_mask)
    sem[to_demote] = 255
    inst[to_demote] = -1

    print()
    print(f"  Promoted to rack (was other class) : {int(promoted_class_change.sum()):>7,}")
    print(f"  Reassigned to a different rack inst: {int(promoted_instance_change.sum()):>7,}")
    print(f"  Demoted stray rack -> ignore       : {int(to_demote.sum()):>7,}")
    print(f"  Preserved (class:inst pairs)       : {int(preserve_mask.sum()):>7,}")

    # --- Step 4: save .npy + regenerate GT + preview.
    arr[:, 10] = sem.astype(arr.dtype)
    arr[:, 11] = inst.astype(arr.dtype)
    np.save(npy_path, arr)
    print(f"Saved {npy_path}")

    gt = np.zeros(N, dtype=np.int32)
    valid = sem != 255
    gt[valid] = sem[valid] * 1000 + inst[valid] + 1
    np.savetxt(gt_path, gt, fmt="%d")
    print(f"Saved {gt_path}")

    colors = np.full((N, 3), 100, dtype=np.uint8)
    for sid in np.unique(sem):
        if sid == 255:
            continue
        sid_mask = sem == sid
        for iid in np.unique(inst[sid_mask]):
            pair = sid_mask & (inst == iid)
            colors[pair] = _color_for(int(sid), int(iid))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
    preview.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(preview), pcd, write_ascii=False)
    print(f"Saved {preview}")

    print()
    print("=== Final label distribution ===")
    names = {1: "wall", 2: "floor", 3: "ceiling", 4: "server_rack", 5: "box_clutter", 6: "ac_unit", 255: "ignore"}
    for sid in sorted(np.unique(sem)):
        m = sem == sid
        insts = sorted(np.unique(inst[m]).tolist())
        print(f"  {names.get(int(sid), '?'):<12} sem={int(sid):>3}  n={int(m.sum()):>7,}  instances={insts}")


if __name__ == "__main__":
    main()
