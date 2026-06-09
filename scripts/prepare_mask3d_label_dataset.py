"""Prepare a small server-room dataset for Mask3D fine-tuning.

The script converts an aligned point cloud or mesh into the processed layout
used by the packaged Mask3D dataset loader:

    xyz, rgb, normals, segment_id, semantic_id, instance_id

Labels can be bootstrapped from the room-shell geometry heuristic and then
overridden with manually segmented PLY files.  Each PLY file inside a manual
class folder is treated as one instance and is matched back to the source cloud
with nearest-neighbor lookup.

Example
-------
    python scripts/prepare_mask3d_label_dataset.py \
        --scene las6 \
        --source-ply server_room_phone/pipeline_vis_las6/s3_manhattan.ply \
        --geometry-seed \
        --manual-dir server_room_phone/label_overrides/las6 \
        --output-dir data/mask3d_server_room \
        --split train
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
import yaml
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.detect_lidar_components import (  # noqa: E402
    _cluster_interior_components,
    _shell_labels,
)


IGNORE_LABEL = 255
NO_INSTANCE = -1


@dataclass(frozen=True)
class ClassSpec:
    id: int
    name: str
    color: tuple[int, int, int]
    instance_policy: str


CLASS_SPECS: dict[str, ClassSpec] = {
    "wall": ClassSpec(1, "wall", (55, 138, 221), "stuff"),
    "floor": ClassSpec(2, "floor", (180, 178, 169), "stuff"),
    "ceiling": ClassSpec(3, "ceiling", (232, 229, 221), "stuff"),
    "server_rack": ClassSpec(4, "server_rack", (29, 158, 117), "instance"),
    "box_clutter": ClassSpec(5, "box_clutter", (232, 158, 79), "instance"),
    "ac_unit": ClassSpec(6, "ac_unit", (36, 169, 225), "instance"),
}

PIPELINE_LABEL_TO_CLASS = {
    "wall": "wall",
    "floor": "floor",
    "ceiling": "ceiling",
    "server rack": "server_rack",
    "server cabinet": "server_rack",
    "rack cabinet": "server_rack",
    "network rack": "server_rack",
    "equipment rack": "server_rack",
    "cardboard box": "box_clutter",
    "trash can": "box_clutter",
    "chair": "box_clutter",
    "desk": "box_clutter",
    "table": "box_clutter",
    "air conditioning unit": "ac_unit",
    "AC unit": "ac_unit",
    "ac_unit": "ac_unit",
    "cooling unit": "ac_unit",
}

MANUAL_DIR_ALIASES = {
    "server rack": "server_rack",
    "server-rack": "server_rack",
    "rack": "server_rack",
    "boxes": "box_clutter",
    "box": "box_clutter",
    "clutter": "box_clutter",
    "ac": "ac_unit",
    "ac_unit": "ac_unit",
    "air_conditioning_unit": "ac_unit",
    "cooling_unit": "ac_unit",
    "ignore": "ignore",
    "unknown": "ignore",
}


def _read_geometry(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mesh = o3d.io.read_triangle_mesh(str(path))
    if len(mesh.vertices) > 0:
        mesh.compute_vertex_normals()
        points = np.asarray(mesh.vertices, dtype=np.float64)
        colors = (
            np.asarray(mesh.vertex_colors, dtype=np.float64)
            if mesh.has_vertex_colors() and len(mesh.vertex_colors) == len(mesh.vertices)
            else np.full((len(points), 3), 0.72, dtype=np.float64)
        )
        normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
        if len(normals) != len(points):
            normals = np.zeros((len(points), 3), dtype=np.float64)
        return points, colors, normals

    point_cloud = o3d.io.read_point_cloud(str(path))
    if len(point_cloud.points) == 0:
        raise ValueError(f"No vertices or points found in {path}")
    if not point_cloud.has_normals():
        point_cloud.estimate_normals()
    points = np.asarray(point_cloud.points, dtype=np.float64)
    colors = (
        np.asarray(point_cloud.colors, dtype=np.float64)
        if point_cloud.has_colors() and len(point_cloud.colors) == len(point_cloud.points)
        else np.full((len(points), 3), 0.72, dtype=np.float64)
    )
    normals = np.asarray(point_cloud.normals, dtype=np.float64)
    return points, colors, normals


def _read_selection_points(path: Path) -> np.ndarray:
    mesh = o3d.io.read_triangle_mesh(str(path))
    if len(mesh.vertices) > 0:
        return np.asarray(mesh.vertices, dtype=np.float64)
    point_cloud = o3d.io.read_point_cloud(str(path))
    return np.asarray(point_cloud.points, dtype=np.float64)


def _segment_ids(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if voxel_size <= 0:
        return np.arange(len(points), dtype=np.int32)
    quantized = np.floor((points - points.min(axis=0)) / voxel_size).astype(np.int64)
    _, inverse = np.unique(quantized, axis=0, return_inverse=True)
    return inverse.astype(np.int32)


def _apply_geometry_seed(
    points: np.ndarray,
    semantic_ids: np.ndarray,
    *,
    outer_percentile: float,
    floor_band: float,
    ceiling_band: float,
    wall_band: float,
    cluster_voxel_size: float,
    cluster_eps: float,
    cluster_min_points: int,
    assign_radius: float,
    min_component_points: int,
) -> dict:
    labels, shell_stats = _shell_labels(
        points,
        outer_percentile=outer_percentile,
        floor_band_m=floor_band,
        ceiling_band_m=ceiling_band,
        wall_band_m=wall_band,
    )
    labels, components = _cluster_interior_components(
        points,
        labels,
        shell_stats,
        voxel_size_m=cluster_voxel_size,
        eps_m=cluster_eps,
        min_points=cluster_min_points,
        assign_radius_m=assign_radius,
        min_component_points=min_component_points,
    )

    for pipeline_label, class_key in PIPELINE_LABEL_TO_CLASS.items():
        semantic_ids[labels == pipeline_label] = CLASS_SPECS[class_key].id

    return {
        "shell": shell_stats,
        "components": components,
        "label_counts": dict(Counter(str(label) for label in labels)),
    }


def _apply_label_array_seed(
    labels_path: Path,
    semantic_ids: np.ndarray,
) -> dict:
    loaded = np.load(labels_path, allow_pickle=False)
    if "labels" in loaded:
        labels = loaded["labels"].astype(str)
    elif "vertex_labels" in loaded:
        labels = loaded["vertex_labels"].astype(str)
    else:
        raise ValueError(f"{labels_path} must contain a 'labels' or 'vertex_labels' array")
    if len(labels) != len(semantic_ids):
        raise ValueError(
            f"Label array length mismatch for {labels_path}: "
            f"{len(labels)} labels for {len(semantic_ids)} source points"
        )

    mapped_counts: Counter[str] = Counter()
    for pipeline_label, class_key in PIPELINE_LABEL_TO_CLASS.items():
        label_mask = labels == pipeline_label
        semantic_ids[label_mask] = CLASS_SPECS[class_key].id
        if np.any(label_mask):
            mapped_counts[class_key] += int(label_mask.sum())

    return {
        "path": str(labels_path),
        "raw_label_counts": dict(Counter(str(label) for label in labels)),
        "mapped_counts": dict(mapped_counts),
    }


def _class_key_from_manual_dir(path: Path) -> str | None:
    normalized = path.name.strip().lower().replace(" ", "_").replace("-", "_")
    normalized = MANUAL_DIR_ALIASES.get(normalized, normalized)
    if normalized in CLASS_SPECS or normalized == "ignore":
        return normalized
    return None


def _apply_manual_overrides(
    points: np.ndarray,
    semantic_ids: np.ndarray,
    instance_ids: np.ndarray,
    manual_dir: Path | None,
    match_radius: float,
    next_instance_id: int,
) -> tuple[int, list[dict]]:
    if manual_dir is None or not manual_dir.exists():
        return next_instance_id, []

    source_tree = cKDTree(points)
    applied: list[dict] = []
    for class_dir in sorted(path for path in manual_dir.iterdir() if path.is_dir()):
        class_key = _class_key_from_manual_dir(class_dir)
        if class_key is None:
            continue
        for selection_path in sorted(class_dir.glob("*.ply")):
            selection_points = _read_selection_points(selection_path)
            if len(selection_points) == 0:
                continue
            distances, nearest = source_tree.query(selection_points, k=1, distance_upper_bound=match_radius)
            matched_indices = np.unique(nearest[np.isfinite(distances)])
            matched_indices = matched_indices[matched_indices < len(points)]
            if len(matched_indices) == 0:
                applied.append({
                    "file": str(selection_path),
                    "class": class_key,
                    "matched_points": 0,
                })
                continue

            if class_key == "ignore":
                semantic_ids[matched_indices] = IGNORE_LABEL
                instance_ids[matched_indices] = NO_INSTANCE
                instance_value = NO_INSTANCE
            else:
                class_spec = CLASS_SPECS[class_key]
                semantic_ids[matched_indices] = class_spec.id
                instance_value = next_instance_id
                instance_ids[matched_indices] = instance_value
                next_instance_id += 1

            applied.append({
                "file": str(selection_path),
                "class": class_key,
                "matched_points": int(len(matched_indices)),
                "instance_id": int(instance_value),
            })

    return next_instance_id, applied


def _assign_seed_instances(
    points: np.ndarray,
    semantic_ids: np.ndarray,
    instance_ids: np.ndarray,
    *,
    instance_eps: float,
    instance_min_points: int,
) -> int:
    next_instance_id = 0
    for class_key, class_spec in CLASS_SPECS.items():
        class_indices = np.where(semantic_ids == class_spec.id)[0]
        if len(class_indices) == 0:
            continue
        if class_spec.instance_policy == "stuff":
            instance_ids[class_indices] = next_instance_id
            next_instance_id += 1
            continue

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points[class_indices])
        cluster_ids = np.asarray(
            point_cloud.cluster_dbscan(
                eps=instance_eps,
                min_points=instance_min_points,
                print_progress=False,
            ),
            dtype=np.int32,
        )
        valid_clusters = [int(cluster_id) for cluster_id in np.unique(cluster_ids) if cluster_id >= 0]
        if not valid_clusters:
            instance_ids[class_indices] = next_instance_id
            next_instance_id += 1
            continue
        for cluster_id in valid_clusters:
            cluster_member_indices = class_indices[cluster_ids == cluster_id]
            instance_ids[cluster_member_indices] = next_instance_id
            next_instance_id += 1

    return next_instance_id


def _write_label_database(output_dir: Path) -> None:
    label_database = {
        class_spec.id: {
            "color": list(class_spec.color),
            "name": class_spec.name,
            "validation": True,
        }
        for class_spec in CLASS_SPECS.values()
    }
    with open(output_dir / "label_database.yaml", "w") as file_obj:
        yaml.safe_dump(label_database, file_obj, sort_keys=True)


def _write_color_stats(output_dir: Path, colors: np.ndarray) -> None:
    color_01 = np.clip(colors, 0.0, 1.0)
    stats = {
        "mean": [float(value) for value in color_01.mean(axis=0)],
        "std": [float(max(value, 1e-6)) for value in color_01.std(axis=0)],
    }
    with open(output_dir / "color_mean_std.yaml", "w") as file_obj:
        yaml.safe_dump(stats, file_obj, sort_keys=True)


def _write_database_files(
    output_dir: Path,
    split: str,
    scene: str,
    processed_path: Path,
    instance_gt_path: Path,
    raw_path: Path,
    file_len: int,
) -> None:
    database_entry = {
        "filepath": str(processed_path),
        "scene": scene,
        "raw_filepath": str(raw_path),
        "file_len": int(file_len),
        "instance_gt_filepath": str(instance_gt_path),
    }
    split_database_path = output_dir / f"{split}_database.yaml"
    with open(split_database_path, "w") as file_obj:
        yaml.safe_dump([database_entry], file_obj, sort_keys=False)

    # The packaged loader in this repo currently looks for this capitalized
    # name, so keep it in sync until training config is finalized.
    with open(output_dir / "Validation_database.yaml", "w") as file_obj:
        yaml.safe_dump([database_entry], file_obj, sort_keys=False)


def _write_preview_ply(points: np.ndarray, semantic_ids: np.ndarray, output_path: Path) -> None:
    colors = np.full((len(points), 3), 0.35, dtype=np.float64)
    for class_spec in CLASS_SPECS.values():
        class_mask = semantic_ids == class_spec.id
        colors[class_mask] = np.asarray(class_spec.color, dtype=np.float64) / 255.0
    colors[semantic_ids == IGNORE_LABEL] = np.asarray((96, 96, 96), dtype=np.float64) / 255.0
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(str(output_path), point_cloud, write_ascii=False)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a Mask3D-compatible labeled sample for the server-room capstone dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--scene", required=True, help="Scene id, e.g. las6 or las6_corrected.")
    parser.add_argument("--source-ply", required=True, help="Aligned source PLY; for the current dataset use pipeline_vis_las6/s3_manhattan.ply from room_6.las.")
    parser.add_argument("--output-dir", default="data/mask3d_server_room")
    parser.add_argument("--split", choices=["train", "validation"], default="train")
    parser.add_argument("--manual-dir", default=None, help="Optional manual override folder with class subfolders.")
    parser.add_argument("--geometry-seed", action="store_true", help="Bootstrap labels with the room-shell geometry heuristic.")
    parser.add_argument("--seed-labels-npz", default=None, help="Optional per-vertex label array exported by segment_scan.py or detect_lidar_components.py.")
    parser.add_argument("--match-radius", type=float, default=0.035, help="Nearest-neighbor match radius for manual PLY selections, in metres.")
    parser.add_argument("--segment-voxel-size", type=float, default=0.05, help="Voxel size used to create approximate segment ids.")
    parser.add_argument("--instance-eps", type=float, default=0.18, help="DBSCAN eps for seed object instance splitting.")
    parser.add_argument("--instance-min-points", type=int, default=80)
    parser.add_argument("--outer-percentile", type=float, default=0.5)
    parser.add_argument("--floor-band", type=float, default=0.28)
    parser.add_argument("--ceiling-band", type=float, default=0.28)
    parser.add_argument("--wall-band", type=float, default=0.42)
    parser.add_argument("--cluster-voxel-size", type=float, default=0.06)
    parser.add_argument("--cluster-eps", type=float, default=0.12)
    parser.add_argument("--cluster-min-points", type=int, default=10)
    parser.add_argument("--assign-radius", type=float, default=0.09)
    parser.add_argument("--min-component-points", type=int, default=500)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    source_path = Path(args.source_ply).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    split_dir = output_dir / args.split
    instance_gt_dir = output_dir / "instance_gt" / args.split
    preview_dir = output_dir / "previews"
    split_dir.mkdir(parents=True, exist_ok=True)
    instance_gt_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    points, colors, normals = _read_geometry(source_path)
    semantic_ids = np.full(len(points), IGNORE_LABEL, dtype=np.int32)
    instance_ids = np.full(len(points), NO_INSTANCE, dtype=np.int32)

    seed_report = {}
    if args.geometry_seed:
        seed_report = _apply_geometry_seed(
            points,
            semantic_ids,
            outer_percentile=args.outer_percentile,
            floor_band=args.floor_band,
            ceiling_band=args.ceiling_band,
            wall_band=args.wall_band,
            cluster_voxel_size=args.cluster_voxel_size,
            cluster_eps=args.cluster_eps,
            cluster_min_points=args.cluster_min_points,
            assign_radius=args.assign_radius,
            min_component_points=args.min_component_points,
        )

    label_array_report = {}
    if args.seed_labels_npz:
        label_array_report = _apply_label_array_seed(
            Path(args.seed_labels_npz).expanduser().resolve(),
            semantic_ids,
        )

    next_instance_id = _assign_seed_instances(
        points,
        semantic_ids,
        instance_ids,
        instance_eps=args.instance_eps,
        instance_min_points=args.instance_min_points,
    )
    manual_dir = Path(args.manual_dir).expanduser().resolve() if args.manual_dir else None
    next_instance_id, manual_report = _apply_manual_overrides(
        points,
        semantic_ids,
        instance_ids,
        manual_dir,
        args.match_radius,
        next_instance_id,
    )

    segment_ids = _segment_ids(points, args.segment_voxel_size)
    colors_255 = np.clip(colors * 255.0, 0.0, 255.0)
    processed = np.column_stack(
        (points, colors_255, normals, segment_ids, semantic_ids, instance_ids)
    ).astype(np.float32)

    processed_path = split_dir / f"{args.scene}.npy"
    np.save(processed_path, processed)

    encoded_gt = np.zeros(len(points), dtype=np.int32)
    has_instance = instance_ids >= 0
    encoded_gt[has_instance] = semantic_ids[has_instance] * 1000 + instance_ids[has_instance] + 1
    instance_gt_path = instance_gt_dir / f"{args.scene}.txt"
    np.savetxt(instance_gt_path, encoded_gt, fmt="%d")

    preview_path = preview_dir / f"{args.scene}_labels_preview.ply"
    _write_preview_ply(points, semantic_ids, preview_path)
    _write_label_database(output_dir)
    _write_color_stats(output_dir, colors)
    _write_database_files(output_dir, args.split, args.scene, processed_path, instance_gt_path, source_path, len(points))

    id_to_name = {class_spec.id: class_spec.name for class_spec in CLASS_SPECS.values()}
    semantic_counts = {
        id_to_name.get(int(class_id), "ignore"): int(count)
        for class_id, count in zip(*np.unique(semantic_ids, return_counts=True), strict=False)
    }
    summary = {
        "scene": args.scene,
        "source_ply": str(source_path),
        "processed_path": str(processed_path),
        "instance_gt_path": str(instance_gt_path),
        "preview_path": str(preview_path),
        "n_points": int(len(points)),
        "semantic_counts": semantic_counts,
        "n_instances": int(len(set(int(value) for value in instance_ids if value >= 0))),
        "seed_report": seed_report,
        "label_array_seed": label_array_report,
        "manual_overrides": manual_report,
        "class_ids": {class_key: class_spec.id for class_key, class_spec in CLASS_SPECS.items()},
    }
    summary_path = output_dir / f"{args.scene}_label_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"Prepared scene: {args.scene}")
    print(f"  points          : {len(points):,}")
    print(f"  instances       : {summary['n_instances']:,}")
    print("  semantic counts :")
    for label_name, count in sorted(semantic_counts.items(), key=lambda item: -item[1]):
        print(f"    {label_name:<14} {count:>9,}")
    print(f"  processed       : {processed_path}")
    print(f"  preview         : {preview_path}")
    print(f"  summary         : {summary_path}")


if __name__ == "__main__":
    main()