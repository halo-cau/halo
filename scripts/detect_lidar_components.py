"""Geometry-first auto-detection prototype for LiDAR point clouds.

This is intentionally not a final semantic model.  It answers the first
question for the LiDAR path: are the room shell and interior equipment
geometrically separable enough that a point-cloud encoder / component-level
classifier can work on top?

Outputs, by default under server_room_phone/pipeline_vis/:
  s3_seg_lidar_geometry_labels.ply  – point cloud coloured by heuristic labels
  s3_seg_lidar_geometry.json        – component statistics and label counts
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.vision.cleaner import clean_and_align_meshes_staged
from engine.vision.segmentor_base import label_to_color

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCAN_PATH = PROJECT_ROOT / "server_room_phone" / "lidar" / "sv_room_lidar_1" / "2026. 5. 7.laz"
DEFAULT_OUT_DIR = PROJECT_ROOT / "server_room_phone" / "pipeline_vis"

SEP = "─" * 60


def _geometry_arrays(mesh: o3d.geometry.TriangleMesh) -> tuple[np.ndarray, np.ndarray | None]:
    points = np.asarray(mesh.vertices, dtype=np.float64)
    colors = np.asarray(mesh.vertex_colors, dtype=np.float64) if mesh.has_vertex_colors() else None
    if colors is not None and len(colors) != len(points):
        colors = None
    return points, colors


def _shell_labels(
    points: np.ndarray,
    *,
    outer_percentile: float,
    floor_band_m: float,
    ceiling_band_m: float,
    wall_band_m: float,
) -> tuple[np.ndarray, dict]:
    p = float(np.clip(outer_percentile, 0.0, 10.0))
    bounds_min = np.percentile(points, p, axis=0)
    bounds_max = np.percentile(points, 100.0 - p, axis=0)

    labels = np.full(len(points), "unknown", dtype=object)
    floor_z = float(bounds_min[2])
    ceil_z = float(bounds_max[2])

    floor = points[:, 2] <= floor_z + floor_band_m
    ceiling = points[:, 2] >= ceil_z - ceiling_band_m
    wall_dist = np.minimum.reduce([
        np.abs(points[:, 0] - bounds_min[0]),
        np.abs(points[:, 0] - bounds_max[0]),
        np.abs(points[:, 1] - bounds_min[1]),
        np.abs(points[:, 1] - bounds_max[1]),
    ])
    wall = (wall_dist <= wall_band_m) & ~floor & ~ceiling

    labels[floor] = "floor"
    labels[ceiling] = "ceiling"
    labels[wall] = "wall"

    stats = {
        "bounds_min": np.round(bounds_min, 4).tolist(),
        "bounds_max": np.round(bounds_max, 4).tolist(),
        "floor_z": round(floor_z, 4),
        "ceiling_z": round(ceil_z, 4),
        "outer_percentile": p,
        "floor_band_m": floor_band_m,
        "ceiling_band_m": ceiling_band_m,
        "wall_band_m": wall_band_m,
    }
    return labels, stats


def _classify_component(
    pts: np.ndarray,
    floor_z: float,
    ceiling_z: float,
) -> str:
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    extent = mx - mn
    height = float(extent[2])
    xy = np.sort(extent[:2])
    footprint_area = float(max(extent[0], 0.02) * max(extent[1], 0.02))
    z_mid = float((mn[2] + mx[2]) * 0.5)
    floor_clearance = float(mn[2] - floor_z)

    # Server-rack rows are tall, vertical, floor-standing components.  This is
    # deliberately permissive: final naming should be done by a learned encoder
    # or human confirmation, but these candidates are what the encoder should
    # inspect first.
    if (
        height >= 1.15
        and mx[2] >= floor_z + 1.35
        and mn[2] <= floor_z + 0.75
        and footprint_area >= 0.18
        and xy[1] >= 0.45
    ):
        return "server rack"

    # Floor-standing low/medium cuboids are more likely boxes than racks.
    # This is intentionally broad so the output is a useful review layer; a
    # learned component classifier should later confirm the actual class.
    if (
        floor_clearance <= 0.45
        and 0.12 <= height <= 1.20
        and footprint_area >= 0.06
        and xy[1] >= 0.25
    ):
        return "cardboard box"

    # Ceiling / upper-wall mechanical boxes.  Useful for future AC/vent review.
    if (
        z_mid >= ceiling_z - 0.75
        and 0.18 <= height <= 1.10
        and footprint_area >= 0.08
    ):
        return "air conditioning unit"

    return "object"


def _cluster_interior_components(
    points: np.ndarray,
    labels: np.ndarray,
    shell_stats: dict,
    *,
    voxel_size_m: float,
    eps_m: float,
    min_points: int,
    assign_radius_m: float,
    min_component_points: int,
) -> tuple[np.ndarray, list[dict]]:
    floor_z = float(shell_stats["floor_z"])
    ceiling_z = float(shell_stats["ceiling_z"])
    interior = (
        (labels == "unknown")
        & (points[:, 2] >= floor_z + 0.20)
        & (points[:, 2] <= ceiling_z - 0.10)
    )
    interior_idx = np.where(interior)[0]
    if len(interior_idx) == 0:
        return labels, []

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[interior_idx])
    pcd_down = pcd.voxel_down_sample(voxel_size_m)
    down_points = np.asarray(pcd_down.points, dtype=np.float64)
    if len(down_points) == 0:
        return labels, []

    cluster_ids = np.asarray(
        pcd_down.cluster_dbscan(eps=eps_m, min_points=min_points, print_progress=False),
        dtype=np.int32,
    )
    valid_down = cluster_ids >= 0
    if not np.any(valid_down):
        return labels, []

    tree = cKDTree(down_points[valid_down])
    valid_cluster_ids = cluster_ids[valid_down]
    dist, nearest = tree.query(points[interior_idx], k=1, distance_upper_bound=assign_radius_m)
    assigned_cluster = np.full(len(interior_idx), -1, dtype=np.int32)
    hit = np.isfinite(dist)
    assigned_cluster[hit] = valid_cluster_ids[nearest[hit]]

    components: list[dict] = []
    next_component_id = 0
    for cluster_id in sorted(int(c) for c in np.unique(assigned_cluster) if c >= 0):
        member_local = np.where(assigned_cluster == cluster_id)[0]
        member_idx = interior_idx[member_local]
        if len(member_idx) < min_component_points:
            continue
        pts = points[member_idx]
        semantic = _classify_component(pts, floor_z, ceiling_z)
        labels[member_idx] = semantic
        mn = pts.min(axis=0)
        mx = pts.max(axis=0)
        extent = mx - mn
        center = (mn + mx) * 0.5
        components.append({
            "id": next_component_id,
            "cluster_id": cluster_id,
            "label": semantic,
            "n_points": int(len(member_idx)),
            "center": np.round(center, 4).tolist(),
            "bounds_min": np.round(mn, 4).tolist(),
            "bounds_max": np.round(mx, 4).tolist(),
            "extent": np.round(extent, 4).tolist(),
        })
        next_component_id += 1

    return labels, components


def _write_label_ply(points: np.ndarray, labels: np.ndarray, path: Path) -> None:
    colors = np.asarray([label_to_color(str(lbl)) for lbl in labels], dtype=np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))
    o3d.io.write_point_cloud(str(path), pcd, write_ascii=False)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Detect room shell + equipment candidates from a LiDAR point cloud.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--scan", default=str(DEFAULT_SCAN_PATH), help="Path to LAS/LAZ/PLY/OBJ scan file.")
    p.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR), help="Directory for viewer artifacts.")
    p.add_argument("--outer-percentile", type=float, default=0.5)
    p.add_argument(
        "--floor-band",
        type=float,
        default=0.28,
        help="Distance above the inferred cuboid floor that is forcibly labelled floor.",
    )
    p.add_argument(
        "--ceiling-band",
        type=float,
        default=0.28,
        help="Distance below the inferred cuboid ceiling that is forcibly labelled ceiling.",
    )
    p.add_argument(
        "--wall-band",
        type=float,
        default=0.42,
        help="Distance from any inferred cuboid wall that is forcibly labelled wall.",
    )
    p.add_argument("--cluster-voxel-size", type=float, default=0.06)
    p.add_argument("--cluster-eps", type=float, default=0.12)
    p.add_argument("--cluster-min-points", type=int, default=10)
    p.add_argument("--assign-radius", type=float, default=0.09)
    p.add_argument("--min-component-points", type=int, default=500)
    return p


def main() -> None:
    args = build_parser().parse_args()
    scan_path = Path(args.scan).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{SEP}\n  LiDAR geometry auto-detection prototype\n{SEP}")
    print(f"  Scan: {scan_path}")
    stages = clean_and_align_meshes_staged(scan_path)
    # Use the aligned, pre-Manhattan cloud so object shapes remain unsnapped.
    points, _ = _geometry_arrays(stages["aligned"])
    print(f"  Points after SOR/alignment: {len(points):,}")

    labels, shell_stats = _shell_labels(
        points,
        outer_percentile=args.outer_percentile,
        floor_band_m=args.floor_band,
        ceiling_band_m=args.ceiling_band,
        wall_band_m=args.wall_band,
    )
    labels, components = _cluster_interior_components(
        points,
        labels,
        shell_stats,
        voxel_size_m=args.cluster_voxel_size,
        eps_m=args.cluster_eps,
        min_points=args.cluster_min_points,
        assign_radius_m=args.assign_radius,
        min_component_points=args.min_component_points,
    )

    label_counts = dict(Counter(str(lbl) for lbl in labels))
    components.sort(key=lambda item: item["n_points"], reverse=True)
    rack_count = sum(1 for c in components if c["label"] == "server rack")
    object_count = sum(1 for c in components if c["label"] == "object")
    box_count = sum(1 for c in components if c["label"] == "cardboard box")
    ac_count = sum(1 for c in components if c["label"] == "air conditioning unit")

    label_path = out_dir / "s3_seg_lidar_geometry_labels.ply"
    json_path = out_dir / "s3_seg_lidar_geometry.json"
    _write_label_ply(points, labels, label_path)
    data = {
        "backend": "lidar_geometry",
        "scan": scan_path.name,
        "n_total": int(len(points)),
        "label_counts": label_counts,
        "components": components,
        "extra": {
            "shell": shell_stats,
            "n_components": len(components),
            "n_server_rack_candidates": rack_count,
            "n_object_candidates": object_count,
            "n_box_candidates": box_count,
            "n_ac_candidates": ac_count,
            "label_policy": "strict_cuboid_shell_first_then_component_clustering",
            "cluster_voxel_size_m": args.cluster_voxel_size,
            "cluster_eps_m": args.cluster_eps,
            "assign_radius_m": args.assign_radius,
            "min_component_points": args.min_component_points,
        },
    }
    json_path.write_text(json.dumps(data, indent=2))

    print(
        f"  Components: {len(components):,}  racks={rack_count:,}  "
        f"boxes={box_count:,}  AC={ac_count:,}  object={object_count:,}"
    )
    print("  Label counts:")
    for label, count in sorted(label_counts.items(), key=lambda item: -item[1]):
        print(f"    {label:<24} {count:>9,} ({100.0 * count / max(1, len(points)):.1f}%)")
    print(f"  → {label_path}")
    print(f"  → {json_path}")
    print(SEP)


if __name__ == "__main__":
    main()