#!/usr/bin/env python3
"""Poisson-mesh a colored point cloud into a TriangleMesh with faces.

The SAM3 / grounding backends render *faces* (render_mesh_views), so a raw
recon point cloud must be surfaced first. Output keeps vertex colors.

Usage:
    python scripts/recon/mesh_cloud.py IN.ply OUT.ply [--depth 9] [--density-q 0.03]
"""
import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("inp", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--depth", type=int, default=9, help="Poisson octree depth")
    ap.add_argument("--density-q", type=float, default=0.03,
                    help="drop vertices below this density quantile (trim bubbles)")
    ap.add_argument("--target-pts", type=int, default=300_000,
                    help="voxel-downsample the cloud to ~this many points first")
    args = ap.parse_args()

    pcd = o3d.io.read_point_cloud(str(args.inp))
    n0 = len(pcd.points)
    diag = float(np.linalg.norm(pcd.get_axis_aligned_bounding_box().get_extent()))
    print(f"loaded {n0:,} pts, bbox diag {diag:.3f}")

    if n0 > args.target_pts:
        voxel = diag / (args.target_pts ** (1 / 3) * 12)
        pcd = pcd.voxel_down_sample(voxel)
        print(f"  voxel {voxel:.4f} -> {len(pcd.points):,} pts")

    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=diag / 200, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(20)

    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=args.depth)
    dens = np.asarray(dens)
    keep = dens >= np.quantile(dens, args.density_q)
    mesh.remove_vertices_by_mask(~keep)
    # clip the Poisson balloon back to the observed extent
    mesh = mesh.crop(pcd.get_axis_aligned_bounding_box().scale(
        1.05, pcd.get_axis_aligned_bounding_box().get_center()))
    mesh.compute_vertex_normals()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(args.out), mesh)
    print(f"  wrote {args.out}  ({len(mesh.vertices):,} verts, {len(mesh.triangles):,} tris)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
