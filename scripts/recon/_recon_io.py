"""Shared helpers for the reconstruction runners: env paths, frame subsampling,
point-cloud cleaning, and PLY export sized for the browser viewer."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def add_vendor_paths(*names: str) -> None:
    """Put vendored repos (under opt/) on sys.path."""
    for n in names:
        p = REPO / "opt" / n
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def save_cam_up(cam2world_rots, out_ply) -> None:
    """Estimate gravity-up from camera orientations and save <stem>.up.npy.

    People hold phones upright, so each camera's up vector (-Y in the OpenCV
    camera frame) maps to roughly world-up; the mean over views is a robust
    gravity estimate the labeling pipeline uses to orient floor vs ceiling.
    ``cam2world_rots`` is an (N,3,3) array of camera-to-world rotations.
    """
    R = np.asarray(cam2world_rots, dtype=np.float64)
    ups = R @ np.array([0.0, -1.0, 0.0])
    up = ups.mean(axis=0)
    n = float(np.linalg.norm(up))
    if n > 1e-6:
        np.save(str(out_ply).replace(".ply", ".up.npy"), up / n)


def gravity_R(up) -> np.ndarray:
    """Rotation that sends gravity-up vector ``up`` to +Z (Rodrigues).  Shared by
    the recon-cloud orienter and the photo-label lifter so points and labels land
    in the SAME oriented frame."""
    up = np.asarray(up, dtype=np.float64)
    up /= (np.linalg.norm(up) + 1e-9)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(up, z)
    c = float(np.dot(up, z))
    if np.linalg.norm(v) < 1e-6:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    sk = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + sk + sk @ sk * (1.0 / (1.0 + c))


def even_subsample(items: list, k: int) -> list:
    """Pick k evenly-spaced items (keeps first & last)."""
    if k >= len(items) or k <= 0:
        return list(items)
    idx = np.linspace(0, len(items) - 1, k).round().astype(int)
    idx = sorted(set(idx.tolist()))
    return [items[i] for i in idx]


def clean_and_write_ply(path, points, colors, *, max_points=1_200_000,
                        outlier_pct=99.5, min_points=100):
    """Drop non-finite + distance outliers, cap point count, write a PLY.

    points: (N,3) float; colors: (N,3) in [0,1] or uint8.
    """
    import trimesh

    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    colors = np.asarray(colors).reshape(-1, 3)
    if colors.dtype != np.uint8:
        colors = (np.clip(colors, 0, 1) * 255).astype(np.uint8)

    finite = np.isfinite(points).all(axis=1)
    points, colors = points[finite], colors[finite]

    # robust flyer removal: drop points far from the median center
    if len(points) > min_points and outlier_pct < 100:
        center = np.median(points, axis=0)
        d = np.linalg.norm(points - center, axis=1)
        keep = d <= np.percentile(d, outlier_pct)
        points, colors = points[keep], colors[keep]

    n_before = len(points)
    if len(points) > max_points:
        sel = np.random.default_rng(0).choice(len(points), max_points, replace=False)
        points, colors = points[sel], colors[sel]

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pc = trimesh.PointCloud(vertices=points, colors=colors)
    pc.export(path)
    print(f"  wrote {path}  ({len(points):,} pts"
          + (f", downsampled from {n_before:,}" if n_before > len(points) else "") + ")")
    return len(points)
