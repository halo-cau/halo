"""Generate a synthetic rectangular cuboid mesh representing the room layout.

The cuboid is the *canonical* room shell — a perfect axis-aligned box derived
from the scan's outer-percentile bounds. It is independent of the noisy scan
point cloud: rather than snapping individual wall points to a plane, we place
a clean box that envelopes the room.

Downstream uses:
  * Label tool overlay: visualize the clean room layout while labeling scan
    points (see ``server_room_phone/label_tool.html`` ``&box=1`` URL param).
  * Voxelizer: feed the cuboid as room geometry; rely on the scan for interior
    obstacles (racks, AC, clutter). The structural_priors module's
    ``append_virtual_room_shell`` already does this for runtime CV.

Vertex count and ordering of the source PLY are NOT touched — this script
writes a SEPARATE output PLY.

Usage::

    python scripts/build_room_cuboid.py \\
        --source server_room_phone/pipeline_vis_las6/s3_manhattan.ply \\
        --out    server_room_phone/pipeline_vis_las6/s3_room_cuboid.ply \\
        --outer-percentile 0.5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


def build_cuboid_mesh(bounds_min: np.ndarray, bounds_max: np.ndarray) -> o3d.geometry.TriangleMesh:
    """Build an axis-aligned cuboid mesh with 8 vertices and 12 triangles."""
    x0, y0, z0 = bounds_min.astype(float).tolist()
    x1, y1, z1 = bounds_max.astype(float).tolist()
    vertices = np.array(
        [
            [x0, y0, z0],  # 0: -X -Y -Z
            [x1, y0, z0],  # 1: +X -Y -Z
            [x1, y1, z0],  # 2: +X +Y -Z
            [x0, y1, z0],  # 3: -X +Y -Z
            [x0, y0, z1],  # 4: -X -Y +Z
            [x1, y0, z1],  # 5: +X -Y +Z
            [x1, y1, z1],  # 6: +X +Y +Z
            [x0, y1, z1],  # 7: -X +Y +Z
        ],
        dtype=np.float64,
    )
    # 12 triangles: outward-facing winding so back-face culling shows the
    # inside of the box when the camera is outside, and the inside-facing
    # back wall when the camera is inside. We render DoubleSide in the label
    # tool anyway so winding doesn't matter for visibility.
    triangles = np.array(
        [
            [0, 2, 1], [0, 3, 2],  # floor   (-Z)
            [4, 5, 6], [4, 6, 7],  # ceiling (+Z)
            [0, 1, 5], [0, 5, 4],  # -Y wall
            [2, 3, 7], [2, 7, 6],  # +Y wall
            [3, 0, 4], [3, 4, 7],  # -X wall
            [1, 2, 6], [1, 6, 5],  # +X wall
        ],
        dtype=np.int32,
    )
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices  = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    return mesh


def _histogram_peak_bound(
    coords: np.ndarray,
    side: str,
    num_bins: int = 200,
    search_fraction: float = 0.40,
    density_threshold_fraction: float = 0.05,
) -> tuple[float, float]:
    """Find the OUTERMOST 1D plane above a density threshold.

    Walls are not necessarily the densest bin: racks pushed against a wall
    occlude scan rays so the wall behind them gets fewer points than the
    rack's front face. We instead walk from the axis extreme inward and
    accept the first bin whose density exceeds a fraction of the global
    histogram peak. This gives us the OUTERMOST dense plane — which is the
    wall — while ignoring sparse outliers (aisle through a door, curtain
    past a window) that don't accumulate enough density.

    Parameters
    ----------
    coords:
        1D axis coordinates of all scan vertices.
    side:
        ``"low"`` walks from ``coords.min()`` inward; ``"high"`` walks from
        ``coords.max()`` inward.
    num_bins:
        Histogram resolution. 200 bins over a 9 m axis ≈ 4.5 cm per bin.
    search_fraction:
        Fraction of axis range to search. 0.40 covers the outer 40% — wider
        than any realistic wall offset, narrower than the room interior.
    density_threshold_fraction:
        A bin must have density ≥ ``this fraction × max(hist)`` to count as
        a real plane. 0.05 ignores curtains/aisles (typically <5% of the
        densest bin in a phone scan) while keeping real walls.

    Returns ``(coord, density)``.
    """
    coords_min = float(np.min(coords))
    coords_max = float(np.max(coords))
    hist, edges = np.histogram(coords, bins=num_bins, range=(coords_min, coords_max))
    threshold = float(hist.max()) * density_threshold_fraction
    n_search = max(1, int(num_bins * search_fraction))

    if side == "low":
        # Walk from the lowest bin upward inside the search window.
        for i in range(n_search):
            if hist[i] >= threshold:
                return 0.5 * (float(edges[i]) + float(edges[i + 1])), float(hist[i])
        # Fallback: densest bin in the search window.
        peak_bin = int(np.argmax(hist[:n_search]))
    elif side == "high":
        # Walk from the highest bin downward inside the search window.
        for i in range(num_bins - 1, num_bins - 1 - n_search, -1):
            if hist[i] >= threshold:
                return 0.5 * (float(edges[i]) + float(edges[i + 1])), float(hist[i])
        peak_bin = num_bins - n_search + int(np.argmax(hist[-n_search:]))
    else:
        raise ValueError(f"side must be 'low' or 'high', got {side!r}")

    return 0.5 * (float(edges[peak_bin]) + float(edges[peak_bin + 1])), float(hist[peak_bin])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, help="Scan PLY to derive bounds from.")
    p.add_argument("--out", required=True, help="Output cuboid PLY path.")
    p.add_argument(
        "--mode",
        choices=("peak", "percentile"),
        default="peak",
        help="Bound estimator. 'peak' uses the densest 1D plane per axis side "
             "(robust against aisles/curtains beyond the real wall). 'percentile' "
             "uses the outer-percentile shortcut (greedy — includes outliers).",
    )
    p.add_argument("--outer-percentile", type=float, default=0.5,
                   help="(percentile mode) Percentile for shell bounds.")
    p.add_argument(
        "--search-fraction",
        type=float,
        default=0.40,
        help="(peak mode) Outer fraction of each axis to search for a wall peak.",
    )
    p.add_argument(
        "--num-bins",
        type=int,
        default=200,
        help="(peak mode) Histogram bin count per axis.",
    )
    p.add_argument(
        "--density-threshold",
        type=float,
        default=0.05,
        help="(peak mode) Min bin density as fraction of histogram peak (0.05=5%%). "
             "Higher rejects more outliers (curtains/aisle) but risks missing thin walls.",
    )
    p.add_argument("--floor-snap-to-zero", action="store_true", default=True,
                   help="If the lower-Z bound is within 75 cm of z=0, snap to z=0.")
    p.add_argument(
        "--manual-bounds",
        type=float,
        nargs=6,
        metavar=("XMIN", "YMIN", "ZMIN", "XMAX", "YMAX", "ZMAX"),
        default=None,
        help="Override estimator with explicit bounds in metres (six numbers).",
    )
    p.add_argument(
        "--target-dims",
        type=float,
        nargs=3,
        metavar=("WIDTH", "DEPTH", "HEIGHT"),
        default=None,
        help="Fix the cuboid to these exact dimensions in metres. Position is "
             "chosen so the cuboid is centred on the densest scan region per "
             "axis (median X, median Y, floor at z=0). Most reliable when you "
             "know the GT room dimensions.",
    )
    args = p.parse_args()

    source_path = Path(args.source).resolve()
    out_path = Path(args.out).resolve()

    mesh_src = o3d.io.read_triangle_mesh(str(source_path))
    verts = np.asarray(mesh_src.vertices, dtype=np.float64)
    if len(verts) == 0:
        raise SystemExit(f"{source_path} has no vertices")

    diagnostics: dict = {}
    if args.manual_bounds is not None:
        b = args.manual_bounds
        bounds_min = np.array([b[0], b[1], b[2]], dtype=np.float64)
        bounds_max = np.array([b[3], b[4], b[5]], dtype=np.float64)
        chosen_mode = "manual"
    elif args.target_dims is not None:
        # Fix dimensions; only choose position.
        w, d, h = args.target_dims
        # Use the MEDIAN per axis as the room centre — robust against aisles
        # and curtains that pull the mean off-centre.
        cx = float(np.median(verts[:, 0]))
        cy = float(np.median(verts[:, 1]))
        bounds_min = np.array([cx - w / 2, cy - d / 2, 0.0], dtype=np.float64)
        bounds_max = np.array([cx + w / 2, cy + d / 2, h],   dtype=np.float64)
        diagnostics["centre_x"] = cx
        diagnostics["centre_y"] = cy
        chosen_mode = f"target-dims W={w} D={d} H={h} (centred on median X={cx:.3f}, Y={cy:.3f})"
    elif args.mode == "peak":
        bounds_min = np.zeros(3, dtype=np.float64)
        bounds_max = np.zeros(3, dtype=np.float64)
        for axis in range(3):
            lo_coord, lo_density = _histogram_peak_bound(
                verts[:, axis], side="low",
                num_bins=args.num_bins,
                search_fraction=args.search_fraction,
                density_threshold_fraction=args.density_threshold,
            )
            hi_coord, hi_density = _histogram_peak_bound(
                verts[:, axis], side="high",
                num_bins=args.num_bins,
                search_fraction=args.search_fraction,
                density_threshold_fraction=args.density_threshold,
            )
            bounds_min[axis] = lo_coord
            bounds_max[axis] = hi_coord
            diagnostics[f"axis{axis}_low_density"] = lo_density
            diagnostics[f"axis{axis}_high_density"] = hi_density
        chosen_mode = "peak"
    else:
        p_lo = float(np.clip(args.outer_percentile, 0.0, 10.0))
        bounds_min = np.percentile(verts, p_lo, axis=0)
        bounds_max = np.percentile(verts, 100.0 - p_lo, axis=0)
        chosen_mode = f"percentile@{p_lo}/{100.0 - p_lo}"

    if args.floor_snap_to_zero and bounds_min[2] - 0.75 <= 0.0 <= bounds_max[2] + 0.75:
        bounds_min[2] = 0.0

    cuboid = build_cuboid_mesh(bounds_min, bounds_max)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(out_path), cuboid, write_ascii=False)

    width  = bounds_max[0] - bounds_min[0]
    depth  = bounds_max[1] - bounds_min[1]
    height = bounds_max[2] - bounds_min[2]
    print(f"\nGenerated room cuboid mesh")
    print(f"  source PLY     : {source_path}")
    print(f"  output PLY     : {out_path}")
    print(f"  mode           : {chosen_mode}")
    if diagnostics:
        print(f"  peak densities : {diagnostics}")
    print(f"  bounds (min)   : {np.round(bounds_min, 4).tolist()}")
    print(f"  bounds (max)   : {np.round(bounds_max, 4).tolist()}")
    print(f"  dimensions WxDxH (m): {width:.3f} x {depth:.3f} x {height:.3f}")
    print(f"  output size: {out_path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
