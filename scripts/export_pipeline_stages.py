"""Export CV pipeline stages for the Three.js viewer.

Supported scan inputs:
  - OBJ / PLY triangle meshes
  - LAS / LAZ point clouds

Outputs (written to server_room_phone/pipeline_vis/ by default):
  s0_raw.ply           – raw scan geometry
  s1_sor.ply           – after statistical outlier removal
  s2_aligned.ply       – after RANSAC floor alignment
  s3_manhattan.ply     – after Manhattan rectification
  s4_voxels.json       – occupied surface/point voxels
  s5_closed.json       – after morphological closing

Usage:
  conda run -n halo python scripts/export_pipeline_stages.py \
      --scan server_room_phone/server_room_0.las \
      --clear-segmentation
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
from scipy.ndimage import binary_closing

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.core.config import CLOSING_ITERATIONS, VOXEL_SIZE
from engine.vision.cleaner import POINT_CLOUD_SUFFIXES, clean_and_align_meshes_staged

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCAN_PATH = PROJECT_ROOT / "server_room_phone" / "textured_output.obj"
DEFAULT_OUT_DIR = DEFAULT_SCAN_PATH.parent / "pipeline_vis"

SEP = "─" * 55


def _has_triangles(mesh: o3d.geometry.TriangleMesh) -> bool:
    return len(np.asarray(mesh.triangles)) > 0


def _count_geometry(mesh: o3d.geometry.TriangleMesh) -> tuple[int, int]:
    return len(np.asarray(mesh.vertices)), len(np.asarray(mesh.triangles))


def _display_kind(mesh: o3d.geometry.TriangleMesh) -> str:
    return "mesh" if _has_triangles(mesh) else "point cloud"


def _fallback_colors(mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    n_v = len(np.asarray(mesh.vertices))
    if mesh.has_vertex_colors():
        return np.clip(np.asarray(mesh.vertex_colors), 0.0, 1.0)
    normals = np.asarray(mesh.vertex_normals)
    if len(normals) == n_v:
        return np.clip(normals * 0.5 + 0.5, 0.0, 1.0)
    return np.full((n_v, 3), 0.62, dtype=np.float64)


def _export_ply(mesh: o3d.geometry.TriangleMesh, path: Path) -> None:
    """Write either a triangle mesh PLY or a vertex-only point-cloud PLY."""
    n_v, n_t = _count_geometry(mesh)
    colors = _fallback_colors(mesh)

    if _has_triangles(mesh):
        exp = o3d.geometry.TriangleMesh()
        exp.vertices = mesh.vertices
        exp.triangles = mesh.triangles
        exp.vertex_normals = mesh.vertex_normals
        exp.vertex_colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
        o3d.io.write_triangle_mesh(str(path), exp, write_ascii=False)
    else:
        exp = o3d.geometry.PointCloud()
        exp.points = mesh.vertices
        exp.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
        normals = np.asarray(mesh.vertex_normals)
        if len(normals) == n_v:
            exp.normals = mesh.vertex_normals
        o3d.io.write_point_cloud(str(path), exp, write_ascii=False)

    sz = path.stat().st_size / 1024
    print(f"  → {path.name}  ({n_v:,} verts, {n_t:,} tris, {_display_kind(mesh)}, {sz:.0f} KB)")


def _export_voxel_json(
    wall_pts: np.ndarray,
    sealed_pts: np.ndarray,
    voxel_size: float,
    path: Path,
    *,
    source: str,
    scan_name: str,
) -> None:
    data = {
        "voxel_size": voxel_size,
        "wall": wall_pts.astype(np.float32).tolist(),
        "sealed": sealed_pts.astype(np.float32).tolist(),
        "source": source,
        "scan": scan_name,
    }
    with open(path, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    sz = path.stat().st_size / 1024
    print(f"  → {path.name}  ({len(wall_pts):,} wall, {len(sealed_pts):,} sealed, {sz:.0f} KB)")


def _voxelize_geometry(
    geometry: o3d.geometry.TriangleMesh,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Voxelize mesh surfaces, or point occupancy for LAS/LAZ point clouds."""
    if _has_triangles(geometry):
        tmp = tempfile.NamedTemporaryFile(suffix=".ply", delete=False)
        tmp.close()
        o3d.io.write_triangle_mesh(tmp.name, geometry)
        try:
            mesh_tri = trimesh.load(tmp.name, force="mesh")
        finally:
            os.unlink(tmp.name)
        voxel_grid = mesh_tri.voxelized(pitch=voxel_size)
        grid = voxel_grid.matrix.astype(bool)
        origin = voxel_grid.transform[:3, 3]
        wall_idx = np.argwhere(grid)
        wall_pts = wall_idx * voxel_size + origin
        return grid, origin, wall_pts

    points = np.asarray(geometry.vertices, dtype=np.float64)
    if len(points) == 0:
        raise ValueError("Cannot voxelize empty point cloud.")
    origin = np.floor(points.min(axis=0) / voxel_size) * voxel_size
    idx = np.floor((points - origin) / voxel_size).astype(np.int32)
    idx = np.unique(idx, axis=0)
    shape = tuple((idx.max(axis=0) + 1).tolist())
    grid = np.zeros(shape, dtype=bool)
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    wall_pts = idx * voxel_size + origin
    return grid, origin, wall_pts


def _voxelize_room_shell_with_interior(
    geometry: o3d.geometry.TriangleMesh,
    voxel_size: float,
    *,
    outer_percentile: float,
    interior_band_m: float,
    face_thickness_voxels: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Build a clean cuboid room shell, then add only interior occupancy.

    This is the presentation/simulation-friendly LiDAR path: walls, floor,
    ceiling, and the open doorway are represented by a complete axis-aligned
    room-shell prior, while noisy points close to those shell faces are ignored.
    Interior points beyond ``interior_band_m`` are retained as rack/object
    occupancy candidates.
    """
    points = np.asarray(geometry.vertices, dtype=np.float64)
    if len(points) == 0:
        raise ValueError("Cannot voxelize empty room shell.")

    p = float(np.clip(outer_percentile, 0.0, 10.0))
    bounds_min = np.percentile(points, p, axis=0)
    bounds_max = np.percentile(points, 100.0 - p, axis=0)
    origin = np.floor(bounds_min / voxel_size) * voxel_size
    max_idx = np.ceil((bounds_max - origin) / voxel_size).astype(np.int32)
    shape = tuple((max_idx + 1).tolist())

    grid = np.zeros(shape, dtype=bool)
    t = max(1, int(face_thickness_voxels))
    grid[:t, :, :] = True
    grid[-t:, :, :] = True
    grid[:, :t, :] = True
    grid[:, -t:, :] = True
    grid[:, :, :t] = True
    grid[:, :, -t:] = True

    inside_bounds = np.all((points >= bounds_min) & (points <= bounds_max), axis=1)
    distance_to_shell = np.minimum.reduce([
        points[:, 0] - bounds_min[0],
        bounds_max[0] - points[:, 0],
        points[:, 1] - bounds_min[1],
        bounds_max[1] - points[:, 1],
        points[:, 2] - bounds_min[2],
        bounds_max[2] - points[:, 2],
    ])
    interior = inside_bounds & (distance_to_shell > interior_band_m)
    if np.any(interior):
        idx = np.floor((points[interior] - origin) / voxel_size).astype(np.int32)
        idx = np.clip(idx, 0, np.asarray(shape, dtype=np.int32) - 1)
        idx = np.unique(idx, axis=0)
        grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True

    wall_idx = np.argwhere(grid)
    wall_pts = wall_idx * voxel_size + origin
    shell_only = np.zeros(shape, dtype=bool)
    shell_only[:t, :, :] = True
    shell_only[-t:, :, :] = True
    shell_only[:, :t, :] = True
    shell_only[:, -t:, :] = True
    shell_only[:, :, :t] = True
    shell_only[:, :, -t:] = True
    stats = {
        "enabled": True,
        "bounds_min": np.round(bounds_min, 4).tolist(),
        "bounds_max": np.round(bounds_max, 4).tolist(),
        "outer_percentile": p,
        "interior_band_m": float(interior_band_m),
        "face_thickness_voxels": int(t),
        "shell_voxels": int(np.count_nonzero(shell_only)),
        "interior_voxels": int(np.count_nonzero(grid & ~shell_only)),
    }
    return grid, shell_only, origin, wall_pts, stats


def _choose_voxel_source(
    scan_path: Path,
    out_dir: Path,
    mesh_manhattan: o3d.geometry.TriangleMesh,
    voxel_source: str,
) -> tuple[o3d.geometry.TriangleMesh, str]:
    if (
        voxel_source == "manhattan"
        or scan_path.suffix.lower() in POINT_CLOUD_SUFFIXES
        or not _has_triangles(mesh_manhattan)
    ):
        return mesh_manhattan, "Manhattan point-cloud/mesh geometry"

    voxel_candidates = [
        (out_dir / "s3_seg_dino_sam2_balanced_graph_structural.ply", "DINO+SAM2 balanced graph non-destructive review mesh"),
        (out_dir / "s3_seg_dino_sam2_balanced_structural.ply", "DINO+SAM2 balanced-protection review mesh"),
        (out_dir / "s3_seg_dino_sam2_structural.ply", "DINO+SAM2 review mesh"),
    ]
    for voxel_source_path, candidate_name in voxel_candidates:
        if not voxel_source_path.exists():
            continue
        candidate_mesh = o3d.io.read_triangle_mesh(str(voxel_source_path))
        if len(np.asarray(candidate_mesh.vertices)) == 0:
            continue
        return candidate_mesh, candidate_name

    return mesh_manhattan, "Manhattan mesh (DINO+SAM2 review mesh not found)"


def _clear_segmentation_outputs(out_dir: Path) -> int:
    n = 0
    for path in out_dir.glob("s3_seg_*"):
        if path.is_file():
            path.unlink()
            n += 1
    return n


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Export staged scan-cleanup artifacts for pipeline_viewer.html.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--scan", default=str(DEFAULT_SCAN_PATH), help="Path to OBJ/PLY/LAS/LAZ scan file.")
    p.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR), help="Directory for viewer artifacts.")
    p.add_argument(
        "--voxel-source",
        choices=["auto", "manhattan", "room-shell"],
        default="auto",
        help="Voxelize prior segmentation output, current Manhattan geometry, or a strong cuboid room-shell prior plus interior occupancy.",
    )
    p.add_argument(
        "--shell-outer-percentile",
        type=float,
        default=0.5,
        help="Robust min/max percentile for the room-shell prior used by --voxel-source room-shell.",
    )
    p.add_argument(
        "--shell-interior-band",
        type=float,
        default=0.22,
        help="Distance from shell faces where raw point occupancy is ignored, preserving clean walls/floor/ceiling.",
    )
    p.add_argument(
        "--shell-face-thickness-voxels",
        type=int,
        default=1,
        help="Voxel thickness of the generated cuboid room shell.",
    )
    p.add_argument(
        "--clear-segmentation",
        action="store_true",
        help="Remove old s3_seg_* outputs so a LAS/LAZ geometry-only run does not show stale labels.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    scan_path = Path(args.scan).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.clear_segmentation:
        removed = _clear_segmentation_outputs(out_dir)
        print(f"  Cleared {removed} stale segmentation artifact(s) from {out_dir}")

    print(f"\n{SEP}\n  Stages 0–3 — shared cleaner lineage\n{SEP}")
    print(f"  Scan: {scan_path}")
    stages = clean_and_align_meshes_staged(scan_path)
    mesh = stages["raw"]
    mesh_sor = stages["sor"]
    mesh_aligned = stages["aligned"]
    mesh_manhattan = stages["manhattan"]

    for label, stage_mesh in [
        ("Raw", mesh),
        ("SOR", mesh_sor),
        ("Aligned", mesh_aligned),
        ("Manhattan", mesh_manhattan),
    ]:
        n_v, n_t = _count_geometry(stage_mesh)
        print(f"  {label:<10}: {n_v:,} verts, {n_t:,} tris ({_display_kind(stage_mesh)})")

    _export_ply(mesh, out_dir / "s0_raw.ply")
    _export_ply(mesh_sor, out_dir / "s1_sor.ply")
    _export_ply(mesh_aligned, out_dir / "s2_aligned.ply")
    _export_ply(mesh_manhattan, out_dir / "s3_manhattan.ply")

    v_aligned = np.asarray(mesh_aligned.vertices)
    v_man = np.asarray(mesh_manhattan.vertices)
    if len(v_aligned) == len(v_man) and len(v_aligned) > 0:
        delta = np.linalg.norm(v_man - v_aligned, axis=1)
        print(f"  Manhattan displacement mean/max: {delta.mean()*100:.1f} / {delta.max()*100:.1f} cm")

    voxel_source_mesh, voxel_source_name = _choose_voxel_source(
        scan_path,
        out_dir,
        mesh_manhattan,
        args.voxel_source,
    )

    # ── Stage 4: Surface / point occupancy voxelization ───────────────────
    print(f"\n{SEP}\n  Stage 4 — Voxelization  ({VOXEL_SIZE} m)\n{SEP}")
    shell_stats = None
    if args.voxel_source == "room-shell":
        voxel_source_name = "strong cuboid room-shell prior + interior point occupancy"
        print(f"  Source: {voxel_source_name}")
        grid_raw, shell_only_grid, origin, wall_pts, shell_stats = _voxelize_room_shell_with_interior(
            mesh_manhattan,
            VOXEL_SIZE,
            outer_percentile=args.shell_outer_percentile,
            interior_band_m=args.shell_interior_band,
            face_thickness_voxels=args.shell_face_thickness_voxels,
        )
        print(
            "  Room shell: "
            f"{shell_stats['shell_voxels']:,} shell voxels + "
            f"{shell_stats['interior_voxels']:,} interior voxels"
        )

        shell_idx = np.argwhere(shell_only_grid)
        shell_pts = shell_idx * VOXEL_SIZE + origin
        empty_shell_stats = dict(shell_stats)
        empty_shell_stats["interior_voxels"] = 0
        s4_empty_data = {
            "voxel_size": VOXEL_SIZE,
            "wall": shell_pts.astype(np.float32).tolist(),
            "sealed": [],
            "source": "empty cuboid room-shell prior (all interior objects removed)",
            "scan": scan_path.name,
            "room_shell_prior": empty_shell_stats,
        }
        s4_empty_path = out_dir / "s4_empty_room_shell.json"
        s4_empty_path.write_text(json.dumps(s4_empty_data, separators=(",", ":")))
        print(
            f"  → {s4_empty_path.name}  "
            f"({len(shell_pts):,} shell-only wall, 0 sealed, {s4_empty_path.stat().st_size / 1024:.0f} KB)"
        )
    else:
        print(f"  Source: {voxel_source_name}")
        grid_raw, origin, wall_pts = _voxelize_geometry(voxel_source_mesh, VOXEL_SIZE)
    print(f"  Grid shape: {grid_raw.shape}  origin: [{origin[0]:.3f},{origin[1]:.3f},{origin[2]:.3f}]")
    s4_data = {
        "voxel_size": VOXEL_SIZE,
        "wall": wall_pts.astype(np.float32).tolist(),
        "sealed": [],
        "source": voxel_source_name,
        "scan": scan_path.name,
    }
    if shell_stats is not None:
        s4_data["room_shell_prior"] = shell_stats
    s4_path = out_dir / "s4_voxels.json"
    s4_path.write_text(json.dumps(s4_data, separators=(",", ":")))
    print(f"  → {s4_path.name}  ({len(wall_pts):,} wall, 0 sealed, {s4_path.stat().st_size / 1024:.0f} KB)")

    # ── Stage 5: Morphological closing ────────────────────────────────────
    print(f"\n{SEP}\n  Stage 5 — Morphological closing  (iter={CLOSING_ITERATIONS})\n{SEP}")
    closed = binary_closing(grid_raw, iterations=CLOSING_ITERATIONS)
    new_mask = closed & ~grid_raw
    new_idx = np.argwhere(new_mask)
    new_pts = new_idx * VOXEL_SIZE + origin
    print(f"  Newly sealed voxels: {len(new_pts):,}")
    s5_data = {
        "voxel_size": VOXEL_SIZE,
        "wall": wall_pts.astype(np.float32).tolist(),
        "sealed": new_pts.astype(np.float32).tolist(),
        "source": voxel_source_name,
        "scan": scan_path.name,
    }
    if shell_stats is not None:
        s5_data["room_shell_prior"] = shell_stats
    s5_path = out_dir / "s5_closed.json"
    s5_path.write_text(json.dumps(s5_data, separators=(",", ":")))
    print(f"  → {s5_path.name}  ({len(wall_pts):,} wall, {len(new_pts):,} sealed, {s5_path.stat().st_size / 1024:.0f} KB)")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  All files written to: {out_dir}")
    print(f"\n  Serve the viewer:")
    print(f"    cd {DEFAULT_SCAN_PATH.parent}")
    print(f"    python -m http.server 8787")
    print(f"    # open http://localhost:8787/pipeline_viewer.html")
    print(SEP)


if __name__ == "__main__":
    main()
