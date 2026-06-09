"""Snap structural vertices in s3_manhattan.ply to exact axis-aligned planes.

The default Manhattan rectification in :func:`engine.vision.cleaner._manhattan_rectify`
uses a 3 cm tolerance, which leaves noticeable wall noise on phone scans. For the
Mask3D labeling workflow this makes lasso selection painful — the user has to
chase noisy point clusters.

This script applies a stronger pass *in place*:

* Loads ``s3_manhattan.ply`` as a point cloud.
* Estimates the Manhattan cuboid shell from outer-percentile bounds.
* Classifies vertices near the shell, with axis-aligned normals, as
  wall / floor / ceiling using :func:`estimate_room_shell_prior`.
* Snaps those vertices to the exact shell coordinates via
  :func:`flatten_structural_labels_to_room_shell`.
* Saves the flattened mesh back to the same path.

Vertex count and ordering are preserved, so existing label override JSONs
(which reference vertex indices) remain valid against the new PLY.

Usage::

    python scripts/force_manhattan_flatten.py \\
        --in  server_room_phone/pipeline_vis_las6/s3_manhattan.ply \\
        --out server_room_phone/pipeline_vis_las6/s3_manhattan.ply \\
        --tolerance 0.30 --normal-cos 0.6

Tune ``--tolerance`` higher (e.g. 0.4) to capture more wall noise, lower
(0.15) if equipment in front of walls is being absorbed into the shell.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def _estimate_normals_for_point_cloud(verts: np.ndarray) -> np.ndarray:
    """Estimate per-vertex normals via Open3D's KNN + tangent-plane fit."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.15, max_nn=30)
    )
    return np.asarray(pcd.normals, dtype=np.float64)


def _snap_axis_to_bound(
    verts: np.ndarray,
    normals: np.ndarray,
    axis: int,
    plane_coord: float,
    tolerance_m: float,
    normal_cos_min: float,
) -> int:
    """Snap vertices near ``plane_coord`` (along ``axis``) to that coordinate.

    A vertex is snapped iff:
      * its current coordinate on ``axis`` is within ``tolerance_m`` of ``plane_coord``
      * its normal's ``axis`` component is at least ``normal_cos_min`` in magnitude

    Returns the number of vertices moved.
    """
    dist = np.abs(verts[:, axis] - plane_coord)
    normal_ok = np.abs(normals[:, axis]) >= normal_cos_min
    snap = (dist <= tolerance_m) & normal_ok & (dist > 0)
    n_snap = int(snap.sum())
    if n_snap:
        verts[snap, axis] = plane_coord
    return n_snap


def force_manhattan_flatten(
    in_path: Path,
    out_path: Path,
    tolerance_m: float,
    normal_cos_min: float,
    outer_percentile: float,
) -> dict:
    """Snap shell vertices to exact planes and write the result.

    Per-axis independent snapping: a vertex can have its X, Y, *and* Z
    coordinates snapped if it's near each axis's outer plane. This produces
    crisp wall-floor and wall-ceiling corners (single-axis snapping leaves
    one coordinate noisy in corner regions, producing jagged seams).

    Operates directly on point-cloud PLYs. Vertex count and ordering are
    preserved so any external label-override JSON that uses point indices
    remains valid against the output PLY.
    """
    mesh = o3d.io.read_triangle_mesh(str(in_path))
    verts = np.asarray(mesh.vertices, dtype=np.float64).copy()
    n_verts = len(verts)
    if n_verts == 0:
        raise ValueError(f"{in_path} contains no vertices")

    normals = _estimate_normals_for_point_cloud(verts)

    p = float(np.clip(outer_percentile, 0.0, 10.0))
    bounds_min = np.percentile(verts, p, axis=0)
    bounds_max = np.percentile(verts, 100.0 - p, axis=0)

    # Treat any room whose floor is within 75 cm of z=0 as standing on z=0.
    floor_z = 0.0 if bounds_min[2] - 0.75 <= 0.0 <= bounds_max[2] + 0.75 else float(bounds_min[2])
    ceiling_z = float(bounds_max[2])

    snap_counts = {
        "wall_min_x": _snap_axis_to_bound(verts, normals, 0, float(bounds_min[0]), tolerance_m, normal_cos_min),
        "wall_max_x": _snap_axis_to_bound(verts, normals, 0, float(bounds_max[0]), tolerance_m, normal_cos_min),
        "wall_min_y": _snap_axis_to_bound(verts, normals, 1, float(bounds_min[1]), tolerance_m, normal_cos_min),
        "wall_max_y": _snap_axis_to_bound(verts, normals, 1, float(bounds_max[1]), tolerance_m, normal_cos_min),
        "floor":     _snap_axis_to_bound(verts, normals, 2, floor_z,                tolerance_m, normal_cos_min),
        "ceiling":   _snap_axis_to_bound(verts, normals, 2, ceiling_z,              tolerance_m, normal_cos_min),
    }

    # Write back — preserve color, drop KNN-estimated normals
    flattened = o3d.geometry.TriangleMesh()
    flattened.vertices = o3d.utility.Vector3dVector(verts)
    if mesh.has_vertex_colors():
        flattened.vertex_colors = mesh.vertex_colors

    o3d.io.write_triangle_mesh(str(out_path), flattened, write_ascii=False)
    return {
        "in_path": str(in_path),
        "out_path": str(out_path),
        "n_vertices": n_verts,
        "tolerance_m": tolerance_m,
        "normal_cos_min": normal_cos_min,
        "outer_percentile": outer_percentile,
        "snap_counts": snap_counts,
        "n_total_snapped": sum(snap_counts.values()),
        "shell_bounds_min": np.round(bounds_min, 4).tolist(),
        "shell_bounds_max": np.round(bounds_max, 4).tolist(),
        "floor_z": round(floor_z, 4),
        "ceiling_z": round(ceiling_z, 4),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="in_path", required=True, help="Input PLY path")
    p.add_argument("--out", dest="out_path", required=True, help="Output PLY path")
    p.add_argument("--tolerance", type=float, default=0.30,
                   help="Snap radius from the estimated shell (m). Higher = more wall noise captured.")
    p.add_argument("--normal-cos", type=float, default=0.60,
                   help="Minimum |normal . axis| for shell classification. Higher = stricter.")
    p.add_argument("--outer-percentile", type=float, default=0.5,
                   help="Robust percentile used for shell bounds.")
    args = p.parse_args()

    result = force_manhattan_flatten(
        in_path=Path(args.in_path).resolve(),
        out_path=Path(args.out_path).resolve(),
        tolerance_m=args.tolerance,
        normal_cos_min=args.normal_cos,
        outer_percentile=args.outer_percentile,
    )

    print(f"\nForced Manhattan flatten complete")
    print(f"  in_path  : {result['in_path']}")
    print(f"  out_path : {result['out_path']}")
    print(f"  n_vertices: {result['n_vertices']:,}")
    print(f"  tolerance: {result['tolerance_m']} m   normal_cos_min: {result['normal_cos_min']}")
    print(f"  shell bounds (min): {result['shell_bounds_min']}")
    print(f"  shell bounds (max): {result['shell_bounds_max']}")
    print(f"  floor_z: {result['floor_z']}   ceiling_z: {result['ceiling_z']}")
    print(f"  per-axis snap counts (vertices moved per plane):")
    for plane, n in result["snap_counts"].items():
        pct = 100.0 * n / max(1, result["n_vertices"])
        print(f"    {plane:<12} {n:>8,}  ({pct:.1f}%)")
    total = result["n_total_snapped"]
    pct_total = 100.0 * total / max(1, result["n_vertices"])
    print(f"  total axis-snaps: {total:,}  ({pct_total:.1f}% of axis-coords)")


if __name__ == "__main__":
    main()
