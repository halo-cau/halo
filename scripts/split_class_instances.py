"""Split a single per-class instance into N instances by spatial partitioning.

The Mask3D labeling workflow's first pass (``apply_mask3d_label_overrides.py``)
collapses each class to a single instance because the user typically applies
labels by class, not by individual object. This script re-splits chosen
classes into N spatially-separated instances so Mask3D learns to predict
per-object masks (one mask per rack, one per AC unit, etc.).

Algorithm per class:

1. Extract all points with that semantic ID.
2. Try DBSCAN with the given ``--eps``. If it returns exactly the target
   cluster count, accept those clusters.
3. Otherwise, fall back to equal-extent partitioning along the class's
   *principal* axis (PCA-derived) — robust against tightly-packed racks
   that have no inter-object gap.

Updates:
* ``data/mask3d_server_room/train/<scene>.npy`` (column 11 in place).
* ``data/mask3d_server_room/instance_gt/train/<scene>.txt`` regenerated as
  ``semantic_id * 1000 + instance_id + 1`` (0 for ignore).
* ``data/mask3d_server_room/previews/<scene>_labels_preview.ply`` recolored
  by (semantic, instance) pair so each instance gets a distinct shade.

Usage::

    python scripts/split_class_instances.py \\
        --scene las6_corrected \\
        --counts server_rack=7 ac_unit=1 box_clutter=1
"""

from __future__ import annotations

import argparse
import colorsys
from pathlib import Path

import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN

CLASS_NAME_TO_ID = {
    "wall": 1, "floor": 2, "ceiling": 3,
    "server_rack": 4, "box_clutter": 5, "ac_unit": 6,
    "ignore": 255,
}
CLASS_ID_TO_NAME = {v: k for k, v in CLASS_NAME_TO_ID.items()}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "mask3d_server_room"


def _principal_axis(xyz: np.ndarray) -> tuple[int, np.ndarray]:
    """Return ``(best_axis_index, axis_unit_vector)`` from PCA's largest eigvec.

    For an axis-aligned cuboid bank of racks, this should agree with the
    longest world-axis. The PCA fallback handles slightly-rotated rooms too.
    """
    centred = xyz - xyz.mean(axis=0)
    cov = np.cov(centred.T)
    eig_w, eig_v = np.linalg.eigh(cov)
    principal = eig_v[:, -1]  # largest eigenvalue → principal axis
    # Snap to nearest world axis for ergonomics.
    best_axis = int(np.argmax(np.abs(principal)))
    return best_axis, principal


def _split_equal_extent(xyz: np.ndarray, n: int, trim_percentile: float = 1.0) -> np.ndarray:
    """Partition ``xyz`` into ``n`` equal-extent slabs along its principal axis.

    Uses the inner ``[trim_percentile, 100-trim_percentile]`` band of the
    principal-axis coordinate to compute slab edges, then clamps outliers
    to the nearest end slab. This prevents a few stray points at the tail
    of the distribution from stretching the range and leaving most slabs
    empty (a real failure mode on phone scans of tightly-packed rack rows).
    """
    axis_idx, _ = _principal_axis(xyz)
    coords = xyz[:, axis_idx]
    p = float(np.clip(trim_percentile, 0.0, 10.0))
    lo = float(np.percentile(coords, p))
    hi = float(np.percentile(coords, 100.0 - p))
    edges = np.linspace(lo, hi, n + 1)
    # digitize returns 0..n (0 for below lo, n for above hi). Clamp to 0..n-1.
    bin_ids = np.digitize(coords, edges[1:-1])
    return np.clip(bin_ids, 0, n - 1).astype(np.int32)


def _split_dbscan(xyz: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(xyz)
    return db.labels_.astype(np.int32)


_GOLDEN_RATIO_CONJ = 0.6180339887498949  # gives maximally-distinct sequential hues


def _color_for(semantic_id: int, instance_id: int) -> np.ndarray:
    """Stable per-(class, instance) color with high sequential contrast.

    Base hue is fixed per semantic class (so all racks live in a recognisable
    hue family); each instance offset rotates the hue by the golden-ratio
    conjugate, which spreads N consecutive instances around the colour wheel
    with the maximum possible minimum separation.
    """
    base_hue = (semantic_id * 0.137) % 1.0
    hue = (base_hue + instance_id * _GOLDEN_RATIO_CONJ) % 1.0
    # Saturation/value vary slightly so the floor and ceiling don't get the
    # same washed-out grey when hues are close to grey-zone reds.
    sat = 0.62 + 0.10 * ((instance_id * 3) % 5) / 4.0
    val = 0.88 + 0.07 * ((instance_id * 5) % 3) / 2.0
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    return np.array([r, g, b], dtype=np.float64)


def _split_rows_along_axis(
    xyz: np.ndarray,
    row_axis: int,
    n_rows: int,
) -> tuple[np.ndarray, list[float]]:
    """K-means the points along ``row_axis`` into ``n_rows`` groups.

    Returns ``(row_label_per_point, row_centroids_sorted_low_to_high)``.
    The label maps each point to its row index after sorting rows by their
    centroid coordinate (row 0 = lowest, row n_rows-1 = highest).
    """
    from sklearn.cluster import KMeans
    coords = xyz[:, row_axis].reshape(-1, 1)
    km = KMeans(n_clusters=n_rows, n_init=10, random_state=42).fit(coords)
    raw_labels = km.labels_
    centroids = km.cluster_centers_.ravel()
    # Reorder so row 0 has the lowest centroid coord.
    order = np.argsort(centroids)
    remap = {old: new for new, old in enumerate(order)}
    sorted_labels = np.array([remap[int(c)] for c in raw_labels], dtype=np.int32)
    sorted_centroids = centroids[order].tolist()
    return sorted_labels, sorted_centroids


def split_class_multi_row(
    xyz: np.ndarray,
    per_row_counts: list[int],
    row_axis: int,
    trim_percentile: float,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Two-stage split: K-means on ``row_axis``, then equal-extent on the long axis.

    ``per_row_counts`` is a list like ``[6, 7]`` meaning 6 instances in row 0
    and 7 in row 1, for 13 total. Instance IDs are allocated sequentially:
    row 0 gets ``[0..5]``, row 1 gets ``[6..12]``.

    Outlier mask flags points outside the per-row long-axis percentile band.
    """
    n_rows = len(per_row_counts)
    row_labels, row_centroids = _split_rows_along_axis(xyz, row_axis, n_rows)

    # Within each row, find the principal axis (NOT the row axis) and equal-split.
    long_axis_candidates = [a for a in range(3) if a != row_axis]
    # Pick the long axis as whichever non-row axis has the larger spread.
    spreads = [(xyz[:, a].max() - xyz[:, a].min()) for a in long_axis_candidates]
    long_axis = long_axis_candidates[int(np.argmax(spreads))]

    out = np.zeros(len(xyz), dtype=np.int32)
    outliers = np.zeros(len(xyz), dtype=bool)
    instance_offset = 0
    descriptions: list[str] = []
    p = float(np.clip(trim_percentile, 0.0, 10.0))
    for row in range(n_rows):
        row_mask = row_labels == row
        row_xyz = xyz[row_mask]
        n_this_row = per_row_counts[row]
        coords = row_xyz[:, long_axis]
        lo = float(np.percentile(coords, p))
        hi = float(np.percentile(coords, 100.0 - p))
        edges = np.linspace(lo, hi, n_this_row + 1)
        bin_ids = np.digitize(coords, edges[1:-1])
        bin_ids = np.clip(bin_ids, 0, n_this_row - 1).astype(np.int32)
        # Write instance IDs (with row offset) back into the global array.
        row_global_idx = np.where(row_mask)[0]
        out[row_global_idx] = bin_ids + instance_offset
        # Outliers within this row, by long-axis percentile.
        row_outlier = (coords < lo) | (coords > hi)
        outliers[row_global_idx[row_outlier]] = True
        descriptions.append(
            f"row{row}@{row_centroids[row]:.2f}: n={n_this_row} (ids {instance_offset}..{instance_offset + n_this_row - 1})"
        )
        instance_offset += n_this_row

    method = f"2-row split  axis_row={'XYZ'[row_axis]}  axis_long={'XYZ'[long_axis]}  trim={p}%  | " + " | ".join(descriptions)
    return out, outliers, method


def split_class(
    xyz: np.ndarray,
    target_count: int,
    eps: float,
    min_samples: int,
    trim_percentile: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Split a class point cloud into ``target_count`` instances.

    Returns ``(instance_ids, outlier_mask, method)`` where ``outlier_mask``
    flags principal-axis outliers that should be relabelled as ``ignore``.
    Outliers are points outside the inner
    ``[trim_percentile, 100 - trim_percentile]`` percentile band of the
    principal-axis coordinate — typically wall/floor vertices that got
    swept up by the geometry-seed labeling and don't actually belong to
    any rack.
    """
    n = len(xyz)
    if target_count <= 1:
        return np.zeros(n, dtype=np.int32), np.zeros(n, dtype=bool), "single-instance"

    axis_idx, _ = _principal_axis(xyz)
    coords = xyz[:, axis_idx]
    p = float(np.clip(trim_percentile, 0.0, 10.0))
    lo = float(np.percentile(coords, p))
    hi = float(np.percentile(coords, 100.0 - p))
    outliers = (coords < lo) | (coords > hi)

    labels = _split_dbscan(xyz, eps=eps, min_samples=min_samples)
    n_found = len(set(labels[labels >= 0]))
    if n_found == target_count:
        centroids: list[tuple[int, float]] = []
        for c in sorted(set(labels[labels >= 0])):
            centroids.append((c, float(xyz[labels == c, axis_idx].mean())))
        order = sorted(centroids, key=lambda t: t[1])
        remap = {old: new for new, (old, _) in enumerate(order)}
        out = np.zeros(n, dtype=np.int32)
        cluster_centres = {new: xyz[labels == old][:, axis_idx].mean() for old, new in remap.items()}
        for i in range(n):
            if labels[i] >= 0:
                out[i] = remap[int(labels[i])]
            else:
                out[i] = min(cluster_centres.items(), key=lambda kv: abs(kv[1] - coords[i]))[0]
        return out, outliers, f"dbscan(eps={eps}, k={n_found}, trim={p}%)"

    # Equal-extent split inside the trimmed band; outliers get the nearest
    # end bin (and will be overridden to ignore by the caller).
    edges = np.linspace(lo, hi, target_count + 1)
    bin_ids = np.digitize(coords, edges[1:-1])
    bin_ids = np.clip(bin_ids, 0, target_count - 1).astype(np.int32)
    return bin_ids, outliers, f"equal-extent-along-axis (dbscan_found={n_found}, target={target_count}, trim={p}%)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", required=True, help="Scene name, e.g. las6_corrected.")
    parser.add_argument(
        "--counts",
        nargs="+",
        required=True,
        metavar="CLASS=N",
        help="Target instance counts. Example: server_rack=7 ac_unit=1 box_clutter=1",
    )
    parser.add_argument("--split", choices=["train", "validation"], default="train")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--eps", type=float, default=0.30,
                        help="DBSCAN epsilon in metres (only used when target_count > 1).")
    parser.add_argument("--min-samples", type=int, default=30,
                        help="DBSCAN min_samples per cluster.")
    parser.add_argument("--trim-percentile", type=float, default=1.0,
                        help="Outer percentile per principal axis to mark as outliers. "
                             "Points outside this band are relabelled as ignore when "
                             "--trim-outliers is set.")
    parser.add_argument("--trim-outliers", action="store_true", default=True,
                        help="Relabel principal-axis outliers as ignore (semantic 255, "
                             "instance -1). On by default — pass --no-trim-outliers to disable.")
    parser.add_argument("--no-trim-outliers", dest="trim_outliers", action="store_false")
    parser.add_argument(
        "--row-axis",
        choices=("x", "y", "z"),
        default="x",
        help="Axis along which to separate rows when using CLASS=N,M,... syntax. "
             "For racks facing each other along Y, the cut between rows runs along X "
             "(the default).",
    )
    args = parser.parse_args()
    row_axis_idx = {"x": 0, "y": 1, "z": 2}[args.row_axis]

    # Parse the count map. Single int -> 1D split. Comma-separated list ->
    # multi-row split, one count per row (rows sorted by row-axis centroid).
    counts: dict[int, list[int]] = {}
    for spec in args.counts:
        if "=" not in spec:
            raise SystemExit(f"Bad --counts entry: {spec!r} (expected CLASS=N or CLASS=N,M,...)")
        name, value = spec.split("=", 1)
        if name not in CLASS_NAME_TO_ID:
            raise SystemExit(f"Unknown class name: {name!r}. Valid: {sorted(CLASS_NAME_TO_ID)}")
        per_row = [int(x) for x in value.split(",")]
        counts[CLASS_NAME_TO_ID[name]] = per_row

    npy_path = args.data_root / args.split / f"{args.scene}.npy"
    gt_path  = args.data_root / "instance_gt" / args.split / f"{args.scene}.txt"
    preview  = args.data_root / "previews" / f"{args.scene}_labels_preview.ply"

    if not npy_path.exists():
        raise SystemExit(f"Training file not found: {npy_path}")

    arr = np.load(npy_path)
    sem = arr[:, 10].astype(np.int32)
    instance = arr[:, 11].astype(np.int32)
    xyz = arr[:, :3]

    print(f"Loaded {npy_path}  shape={arr.shape}")

    for class_id, per_row in counts.items():
        mask = sem == class_id
        n_points = int(mask.sum())
        if n_points == 0:
            print(f"  {CLASS_ID_TO_NAME[class_id]:<12} skipped (no points labeled)")
            continue
        class_xyz = xyz[mask]
        n_total = sum(per_row)
        if len(per_row) > 1:
            instance_ids, outliers, method = split_class_multi_row(
                class_xyz, per_row_counts=per_row,
                row_axis=row_axis_idx,
                trim_percentile=args.trim_percentile,
            )
        else:
            instance_ids, outliers, method = split_class(
                class_xyz, target_count=n_total,
                eps=args.eps, min_samples=args.min_samples,
                trim_percentile=args.trim_percentile,
            )
        instance[mask] = instance_ids
        n_outliers = 0
        if args.trim_outliers and outliers.any():
            # Map class-local outlier mask back to global indices.
            class_idx = np.where(mask)[0]
            global_outlier_idx = class_idx[outliers]
            sem[global_outlier_idx] = 255
            instance[global_outlier_idx] = -1
            n_outliers = int(len(global_outlier_idx))
        uniq, cnts = np.unique(instance_ids[~outliers] if args.trim_outliers else instance_ids,
                               return_counts=True)
        breakdown = ", ".join(f"inst{u}={c}" for u, c in zip(uniq, cnts))
        suffix = f"  (trimmed {n_outliers:,} outliers -> ignore)" if n_outliers else ""
        target_repr = ",".join(str(x) for x in per_row)
        print(f"  {CLASS_ID_TO_NAME[class_id]:<12} n_target={target_repr}  via {method}{suffix}")
        print(f"    -> {breakdown}")

    # Validation per workflow spec
    ignored = sem == 255
    instance[ignored] = -1
    if np.any((sem != 255) & (instance < 0)):
        bad = np.where((sem != 255) & (instance < 0))[0]
        raise SystemExit(
            f"Non-ignored points must have non-negative instance IDs. "
            f"Offending indices (first 5): {bad[:5].tolist()}"
        )

    # Write back both semantic (in case we trimmed outliers to ignore) and instance.
    arr[:, 10] = sem.astype(arr.dtype)
    arr[:, 11] = instance.astype(arr.dtype)
    np.save(npy_path, arr)
    print(f"Saved {npy_path}")

    # Regenerate instance_gt.txt: sem * 1000 + inst + 1, with 0 for ignore
    gt = np.zeros(len(arr), dtype=np.int32)
    valid = ~ignored
    gt[valid] = sem[valid] * 1000 + instance[valid] + 1
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(gt_path, gt, fmt="%d")
    print(f"Saved {gt_path}")

    # Regenerate preview PLY colored by (semantic, instance) — golden-ratio
    # hue stepping per instance for maximum sequential contrast.
    colors = np.full((len(arr), 3), 100, dtype=np.uint8)  # grey for ignore
    for sid in np.unique(sem):
        if sid == 255:
            continue
        sid_mask = sem == sid
        for inst_id in np.unique(instance[sid_mask]):
            pair_mask = sid_mask & (instance == inst_id)
            colors[pair_mask] = (_color_for(int(sid), int(inst_id)) * 255).astype(np.uint8)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
    preview.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(preview), pcd, write_ascii=False)
    print(f"Saved {preview}")


if __name__ == "__main__":
    main()
