"""Apply interactive label overrides and write a Mask3D-ready scene.

The browser label tool exports point-index operations. This script combines an
aligned source PLY, optional seed label PLY, and those operations into:

- data/mask3d_server_room/<split>/<scene>.npy
- data/mask3d_server_room/instance_gt/<split>/<scene>.txt
- data/mask3d_server_room/previews/<scene>_labels_preview.ply
- data/mask3d_server_room/<scene>_label_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "mask3d_server_room"

CLASS_IDS: dict[str, int] = {
    "wall": 1,
    "floor": 2,
    "ceiling": 3,
    "server_rack": 4,
    "server rack": 4,
    "server cabinet": 4,
    "rack cabinet": 4,
    "network rack": 4,
    "equipment rack": 4,
    "box_clutter": 5,
    "cardboard box": 5,
    "box": 5,
    "clutter": 5,
    "ac_unit": 6,
    "air conditioning unit": 6,
    "AC unit": 6,
    "cooling unit": 6,
    "ignore": 255,
    "unknown": 255,
    "object": 255,
}

ID_TO_NAME: dict[int, str] = {
    1: "wall",
    2: "floor",
    3: "ceiling",
    4: "server_rack",
    5: "box_clutter",
    6: "ac_unit",
    255: "ignore",
}

ID_TO_COLOR: dict[int, tuple[float, float, float]] = {
    1: (55 / 255, 138 / 255, 221 / 255),
    2: (180 / 255, 178 / 255, 169 / 255),
    3: (232 / 255, 229 / 255, 221 / 255),
    4: (29 / 255, 158 / 255, 117 / 255),
    5: (232 / 255, 158 / 255, 79 / 255),
    6: (36 / 255, 169 / 255, 225 / 255),
    255: (95 / 255, 94 / 255, 90 / 255),
}

SEED_COLORS: list[tuple[np.ndarray, int]] = [
    (np.array([55, 138, 221], dtype=np.float64) / 255.0, 1),
    (np.array([180, 178, 169], dtype=np.float64) / 255.0, 2),
    (np.array([232, 229, 221], dtype=np.float64) / 255.0, 3),
    (np.array([29, 158, 117], dtype=np.float64) / 255.0, 4),
    (np.array([15, 110, 86], dtype=np.float64) / 255.0, 4),
    (np.array([232, 158, 79], dtype=np.float64) / 255.0, 5),
    (np.array([36, 169, 225], dtype=np.float64) / 255.0, 6),
    (np.array([95, 94, 90], dtype=np.float64) / 255.0, 255),
    (np.array([136, 135, 128], dtype=np.float64) / 255.0, 255),
]

DEFAULT_INSTANCE: dict[int, int] = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    255: -1,
}


def _read_geometry(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mesh = o3d.io.read_triangle_mesh(str(path))
    points = np.asarray(mesh.vertices, dtype=np.float32)
    colors = np.asarray(mesh.vertex_colors, dtype=np.float32) if mesh.has_vertex_colors() else np.empty((0, 3))
    normals = np.asarray(mesh.vertex_normals, dtype=np.float32) if mesh.has_vertex_normals() else np.empty((0, 3))

    if len(points) == 0 or len(colors) != len(points):
        pcd = o3d.io.read_point_cloud(str(path))
        if len(points) == 0:
            points = np.asarray(pcd.points, dtype=np.float32)
        if pcd.has_colors() and len(np.asarray(pcd.colors)) == len(points):
            colors = np.asarray(pcd.colors, dtype=np.float32)
        if pcd.has_normals() and len(np.asarray(pcd.normals)) == len(points):
            normals = np.asarray(pcd.normals, dtype=np.float32)

    if len(points) == 0:
        raise ValueError(f"No vertices/points found in {path}")
    if len(colors) != len(points):
        colors = np.full((len(points), 3), 0.72, dtype=np.float32)
    if len(normals) != len(points):
        normals = np.zeros((len(points), 3), dtype=np.float32)
    return points, np.clip(colors, 0.0, 1.0), normals


def _nearest_seed_class(color: np.ndarray) -> int:
    best_id = 255
    best_dist = float("inf")
    for target, class_id in SEED_COLORS:
        dist = float(np.sum((color - target) ** 2))
        if dist < best_dist:
            best_dist = dist
            best_id = class_id
    return best_id if best_dist <= 0.025 else 255


def _seed_labels(seed_path: Path | None, n_points: int) -> tuple[np.ndarray, np.ndarray]:
    semantic = np.full(n_points, 255, dtype=np.int32)
    instance = np.full(n_points, -1, dtype=np.int32)
    if seed_path is None or not seed_path.exists():
        return semantic, instance
    _, colors, _ = _read_geometry(seed_path)
    if len(colors) != n_points:
        raise ValueError(
            f"Seed label PLY point count mismatch: {len(colors):,} != {n_points:,}"
        )
    for idx, color in enumerate(colors):
        class_id = _nearest_seed_class(color)
        semantic[idx] = class_id
        instance[idx] = DEFAULT_INSTANCE[class_id]
    return semantic, instance


def _class_id(label: str | int | None, fallback: int | None = None) -> int:
    if isinstance(label, int):
        return label
    if label is not None:
        key = str(label).strip()
        if key in CLASS_IDS:
            return CLASS_IDS[key]
    if fallback is not None:
        return int(fallback)
    raise ValueError(f"Unknown label {label!r}")


def _operation_indices(op: dict[str, Any], points: np.ndarray) -> np.ndarray:
    if "point_indices" in op:
        indices = np.asarray(op["point_indices"], dtype=np.int64)
        if np.any((indices < 0) | (indices >= len(points))):
            raise ValueError("Override contains point index outside source point range")
        return indices

    box = op.get("box")
    if box:
        xmin, xmax = box["x"]
        ymin, ymax = box["y"]
        zmin, zmax = box["z"]
        mask = (
            (points[:, 0] >= xmin)
            & (points[:, 0] <= xmax)
            & (points[:, 1] >= ymin)
            & (points[:, 1] <= ymax)
            & (points[:, 2] >= zmin)
            & (points[:, 2] <= zmax)
        )
        return np.flatnonzero(mask)

    raise ValueError("Override operation needs point_indices or box")


def _apply_overrides(
    semantic: np.ndarray,
    instance: np.ndarray,
    points: np.ndarray,
    override_paths: list[Path],
) -> list[dict[str, Any]]:
    applied: list[dict[str, Any]] = []
    for path in override_paths:
        with path.open() as f:
            payload = json.load(f)
        operations = payload.get("operations") if isinstance(payload, dict) else payload
        if not isinstance(operations, list):
            raise ValueError(f"{path} does not contain an operations list")
        for op in operations:
            class_id = _class_id(op.get("label"), op.get("semantic_id"))
            indices = _operation_indices(op, points)
            instance_id = -1 if class_id == 255 else int(op.get("instance_id", DEFAULT_INSTANCE[class_id]))
            semantic[indices] = class_id
            instance[indices] = instance_id
            applied.append({
                "source": str(path),
                "label": ID_TO_NAME[class_id],
                "semantic_id": int(class_id),
                "instance_id": int(instance_id),
                "n_points": int(len(indices)),
            })
    return applied


def _segment_ids(points: np.ndarray, voxel_size: float) -> np.ndarray:
    mins = points.min(axis=0)
    quantized = np.floor((points - mins) / voxel_size).astype(np.int64)
    _, inverse = np.unique(quantized, axis=0, return_inverse=True)
    return inverse.astype(np.int32)


def _write_preview(path: Path, points: np.ndarray, semantic: np.ndarray) -> None:
    colors = np.asarray([ID_TO_COLOR[int(class_id)] for class_id in semantic], dtype=np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(path), pcd, write_ascii=False)


def _write_database_entry(database_path: Path, entry: dict[str, Any]) -> None:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    if database_path.exists():
        with database_path.open() as f:
            data = yaml.safe_load(f) or []
    else:
        data = []
    data = [item for item in data if item.get("scene") != entry["scene"]]
    data.append(entry)
    with database_path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply HALO label overrides and export a Mask3D dataset sample.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--scene", required=True, help="Scene name, for example las6_corrected.")
    parser.add_argument("--source-ply", required=True, type=Path, help="Aligned source PLY, usually s3_manhattan.ply.")
    parser.add_argument("--seed-label-ply", type=Path, default=None, help="Optional colored seed label PLY.")
    parser.add_argument("--overrides", type=Path, nargs="*", default=[], help="One or more override JSON files from label_tool.html.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--split", choices=["train", "validation"], default="train")
    parser.add_argument("--segment-voxel-size", type=float, default=0.08, help="Voxel size used to generate superpoint-like segment IDs.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    source_ply = args.source_ply.expanduser().resolve()
    seed_label_ply = args.seed_label_ply.expanduser().resolve() if args.seed_label_ply else None
    output_root = args.output_root.expanduser().resolve()

    points, colors, normals = _read_geometry(source_ply)
    semantic, instance = _seed_labels(seed_label_ply, len(points))
    applied = _apply_overrides(semantic, instance, points, [p.expanduser().resolve() for p in args.overrides])

    ignored = semantic == 255
    instance[ignored] = -1
    if np.any((semantic != 255) & (instance < 0)):
        raise ValueError("Non-ignored points must have non-negative instance IDs")

    segment_ids = _segment_ids(points, args.segment_voxel_size)
    processed = np.column_stack([
        points.astype(np.float32),
        np.clip(colors * 255.0, 0.0, 255.0).astype(np.float32),
        normals.astype(np.float32),
        segment_ids.astype(np.float32),
        semantic.astype(np.float32),
        instance.astype(np.float32),
    ])

    split_dir = output_root / args.split
    gt_dir = output_root / "instance_gt" / args.split
    split_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    processed_path = split_dir / f"{args.scene}.npy"
    gt_path = gt_dir / f"{args.scene}.txt"
    preview_path = output_root / "previews" / f"{args.scene}_labels_preview.ply"
    summary_path = output_root / f"{args.scene}_label_summary.json"

    gt = np.zeros(len(points), dtype=np.int32)
    valid = semantic != 255
    gt[valid] = semantic[valid].astype(np.int32) * 1000 + instance[valid].astype(np.int32) + 1

    np.save(processed_path, processed.astype(np.float32))
    np.savetxt(gt_path, gt, fmt="%d")
    _write_preview(preview_path, points, semantic)

    database_path = output_root / ("train_database.yaml" if args.split == "train" else "Validation_database.yaml")
    entry = {
        "filepath": str(processed_path),
        "scene": args.scene,
        "raw_filepath": str(source_ply),
        "file_len": int(len(points)),
        "instance_gt_filepath": str(gt_path),
    }
    _write_database_entry(database_path, entry)

    counts = Counter(ID_TO_NAME[int(class_id)] for class_id in semantic)
    summary = {
        "scene": args.scene,
        "source_ply": str(source_ply),
        "seed_label_ply": str(seed_label_ply) if seed_label_ply else None,
        "processed_path": str(processed_path),
        "instance_gt_path": str(gt_path),
        "preview_path": str(preview_path),
        "database_path": str(database_path),
        "n_points": int(len(points)),
        "semantic_counts": dict(sorted(counts.items())),
        "n_instances": int(len(set(int(x) for x in instance[instance >= 0]))),
        "manual_overrides": applied,
        "class_ids": {name: class_id for name, class_id in CLASS_IDS.items() if name in {"wall", "floor", "ceiling", "server_rack", "box_clutter", "ac_unit", "ignore"}},
    }
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {processed_path} ({len(points):,} points)")
    print(f"Wrote {gt_path}")
    print(f"Wrote {preview_path}")
    print(f"Updated {database_path}")
    print("Semantic counts:")
    for label, count in sorted(counts.items(), key=lambda item: -item[1]):
        print(f"  {label:<12} {count:>9,}")


if __name__ == "__main__":
    main()