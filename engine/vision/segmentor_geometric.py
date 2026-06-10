"""Geometric segmentation backend.

Wraps the geometry-first prototype previously living in
``scripts/detect_lidar_components.py`` so the production pipeline can pick it
via the segmentor factory.  The labels it produces are:

- ``wall`` / ``floor`` / ``ceiling`` — from percentile cuboid bounds.
- ``server_rack`` / ``ac_unit`` / ``box_clutter`` / ``object`` — from DBSCAN
  clustering of interior points + simple size/position heuristics.
- ``unknown`` — points that fall outside the shell bands and outside any
  recognised cluster.

The output mesh fed to the voxelizer keeps preserved + ambiguous vertices and
strips explicit movable clutter, identical to the contract documented in
:mod:`engine.vision.segmentor_base`.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from engine.vision.cleaner import clean_and_align_meshes_staged
from engine.vision.segmentor_base import (
    BaseSegmentor,
    SegmentorResult,
    label_to_color,
    strip_non_structural,
)

log = logging.getLogger(__name__)


# Map the geometric prototype's free-form labels to the canonical taxonomy in
# :mod:`engine.vision.segmentor_base` so downstream consumers don't have to
# special-case strings.
_LABEL_CANON: dict[str, str] = {
    "server rack": "server rack",
    "air conditioning unit": "ac_unit",
    "cardboard box": "cardboard box",
    "object": "object",
    "wall": "wall",
    "floor": "floor",
    "ceiling": "ceiling",
    "unknown": "unknown",
}


def _shell_labels(
    points: np.ndarray,
    *,
    outer_percentile: float,
    floor_band_m: float,
    ceiling_band_m: float,
    wall_band_m: float,
) -> tuple[np.ndarray, dict]:
    """Label points near the percentile cuboid as wall/floor/ceiling."""
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
    """Heuristically name a clustered interior component."""
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    extent = mx - mn
    height = float(extent[2])
    xy = np.sort(extent[:2])
    footprint_area = float(max(extent[0], 0.02) * max(extent[1], 0.02))
    z_mid = float((mn[2] + mx[2]) * 0.5)
    floor_clearance = float(mn[2] - floor_z)

    if (
        height >= 1.15
        and mx[2] >= floor_z + 1.35
        and mn[2] <= floor_z + 0.75
        and footprint_area >= 0.18
        and xy[1] >= 0.45
    ):
        return "server rack"

    if (
        floor_clearance <= 0.45
        and 0.12 <= height <= 1.20
        and footprint_area >= 0.06
        and xy[1] >= 0.25
    ):
        return "cardboard box"

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
    """DBSCAN-cluster the unknown interior, classify, write labels in-place."""
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
            "cluster_id": int(cluster_id),
            "label": semantic,
            "n_points": int(len(member_idx)),
            "center": np.round(center, 4).tolist(),
            "bounds_min": np.round(mn, 4).tolist(),
            "bounds_max": np.round(mx, 4).tolist(),
            "extent": np.round(extent, 4).tolist(),
        })
        next_component_id += 1

    return labels, components


class GeometricSegmentor(BaseSegmentor):
    """Geometry-first segmentor: percentile shell + DBSCAN interior clustering.

    Sees the *pre-Manhattan* aligned mesh from
    :func:`engine.vision.cleaner.clean_and_align_meshes_staged` so object
    surfaces are not snapped to axis-aligned planes — racks/AC stay as their
    real shapes for the size heuristics in :func:`_classify_component`.
    """

    def __init__(
        self,
        *,
        outer_percentile: float = 0.5,
        floor_band_m: float = 0.28,
        ceiling_band_m: float = 0.28,
        wall_band_m: float = 0.42,
        cluster_voxel_size_m: float = 0.06,
        cluster_eps_m: float = 0.12,
        cluster_min_points: int = 10,
        assign_radius_m: float = 0.09,
        min_component_points: int = 500,
    ) -> None:
        self.outer_percentile = outer_percentile
        self.floor_band_m = floor_band_m
        self.ceiling_band_m = ceiling_band_m
        self.wall_band_m = wall_band_m
        self.cluster_voxel_size_m = cluster_voxel_size_m
        self.cluster_eps_m = cluster_eps_m
        self.cluster_min_points = cluster_min_points
        self.assign_radius_m = assign_radius_m
        self.min_component_points = min_component_points

    @property
    def name(self) -> str:
        return "geometric"

    def run(
        self,
        mesh: o3d.geometry.TriangleMesh,
        source_path: Path,
    ) -> SegmentorResult:
        # The Manhattan rectification step snaps detected planes to axis-aligned
        # coordinates. That distorts object surfaces (rack faces become walls)
        # and breaks the size heuristics below, so we re-stage from the raw
        # scan and use the pre-Manhattan aligned mesh instead.
        try:
            stages = clean_and_align_meshes_staged(source_path)
            work_mesh = stages["aligned"]
        except Exception as exc:  # noqa: BLE001 — degrade rather than crash
            log.warning(
                "Could not re-stage %s for geometric segmentation (%s); using "
                "the supplied Manhattan mesh instead.",
                source_path, exc,
            )
            work_mesh = mesh

        points = np.asarray(work_mesh.vertices, dtype=np.float64)
        if len(points) == 0:
            return _empty_result(mesh, self.name)

        shell_labels, shell_stats = _shell_labels(
            points,
            outer_percentile=self.outer_percentile,
            floor_band_m=self.floor_band_m,
            ceiling_band_m=self.ceiling_band_m,
            wall_band_m=self.wall_band_m,
        )
        labels_array, components = _cluster_interior_components(
            points,
            shell_labels,
            shell_stats,
            voxel_size_m=self.cluster_voxel_size_m,
            eps_m=self.cluster_eps_m,
            min_points=self.cluster_min_points,
            assign_radius_m=self.assign_radius_m,
            min_component_points=self.min_component_points,
        )

        # Project labels from the aligned-mesh vertex set onto the manhattan
        # mesh vertices via nearest neighbour — both share the same world frame
        # (the cleaner re-uses the same coordinate basis from RANSAC alignment).
        if work_mesh is not mesh:
            target_points = np.asarray(mesh.vertices, dtype=np.float64)
            if len(target_points) == 0:
                return _empty_result(mesh, self.name)
            tree = cKDTree(points)
            _, nearest = tree.query(target_points, k=1)
            canon_labels = [_LABEL_CANON.get(str(labels_array[i]), "unknown") for i in nearest]
        else:
            canon_labels = [_LABEL_CANON.get(str(lbl), "unknown") for lbl in labels_array]

        label_colors = np.array(
            [label_to_color(lbl) for lbl in canon_labels], dtype=np.float32
        )
        label_map: dict[str, list[int]] = {}
        for vi, lbl in enumerate(canon_labels):
            label_map.setdefault(lbl, []).append(vi)

        structural_mesh = strip_non_structural(mesh, canon_labels)
        n_removed = len(canon_labels) - len(np.asarray(structural_mesh.vertices))

        log.info(
            "Geometric segmentor: %d / %d vertices removed (%.1f%%); "
            "components found: %s",
            n_removed,
            len(canon_labels),
            100.0 * n_removed / max(1, len(canon_labels)),
            dict(Counter(c["label"] for c in components)),
        )

        return SegmentorResult(
            structural_mesh=structural_mesh,
            vertex_labels=canon_labels,
            label_colors=label_colors,
            label_map=label_map,
            n_removed=n_removed,
            backend=self.name,
            extra={
                "shell": shell_stats,
                "components": components,
                "label_counts": dict(Counter(canon_labels)),
            },
        )


def _empty_result(mesh: o3d.geometry.TriangleMesh, backend: str) -> SegmentorResult:
    return SegmentorResult(
        structural_mesh=mesh,
        vertex_labels=[],
        label_colors=np.zeros((0, 3), dtype=np.float32),
        label_map={},
        n_removed=0,
        backend=backend,
        extra={},
    )
