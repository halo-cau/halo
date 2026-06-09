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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.vision.cleaner import clean_and_align_meshes_staged
from engine.vision.segmentor_base import label_to_color
from engine.vision.segmentor_geometric import (
    _classify_component,
    _cluster_interior_components,
    _shell_labels,
)

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