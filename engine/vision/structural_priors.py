"""Geometry priors for protecting the room shell during AI cleanup.

The goal is to reduce reliance on LiDAR-quality geometry by using a simple,
explicit assumption: after alignment, the true room envelope is a Manhattan
cuboid made from the outermost floor, ceiling, and wall planes.  AI segmentation
can still remove clutter, but it must overcome this shell prior before deleting
vertices that geometrically look like floor/wall/ceiling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import open3d as o3d

from engine.vision.segmentor_base import AMBIGUOUS_LABELS, STATIC_INFRASTRUCTURE_LABELS, STRUCTURAL_LABELS

ProtectionLevel = Literal["off", "light", "balanced", "strong", "shell"]


@dataclass(frozen=True)
class StructuralProtectionConfig:
    """Controls how aggressively room-shell vertices are protected.

    Levels
    ------
    off:
        No geometric override.
    light:
        Preserve obvious shell vertices unless they receive repeated movable
        votes.  Useful when segmentation is already good.
    balanced:
        Default.  Preserves floor/wall/ceiling priors unless at least three
        movable votes hit the same protected vertex.
    strong:
        Treat the shell prior as authoritative for all detected shell vertices.
    shell:
        Same protection strength as ``strong``; intended for a later full
        cuboid-shell reconstruction mode.
    """

    level: ProtectionLevel = "balanced"
    outer_percentile: float = 0.5
    plane_tolerance_m: float = 0.12
    normal_cos_min: float = 0.35
    protected_min_movable_votes: int | None = None

    def min_movable_votes_to_override(self) -> int:
        """Return votes required before a protected vertex may be deleted."""
        if self.protected_min_movable_votes is not None:
            return self.protected_min_movable_votes
        return {
            "off": 0,
            "light": 2,
            "balanced": 3,
            "strong": 1_000_000,
            "shell": 1_000_000,
        }[self.level]


@dataclass
class RoomShellPrior:
    """Estimated cuboid shell and per-vertex protection masks."""

    bounds_min: np.ndarray
    bounds_max: np.ndarray
    protected_mask: np.ndarray
    protected_labels: list[str]
    distance_to_shell: np.ndarray
    confidence: np.ndarray
    config: StructuralProtectionConfig

    @property
    def n_protected(self) -> int:
        return int(self.protected_mask.sum())

    def to_json(self) -> dict:
        return {
            "level": self.config.level,
            "outer_percentile": self.config.outer_percentile,
            "plane_tolerance_m": self.config.plane_tolerance_m,
            "normal_cos_min": self.config.normal_cos_min,
            "protected_min_movable_votes": self.config.min_movable_votes_to_override(),
            "bounds_min": np.round(self.bounds_min, 4).tolist(),
            "bounds_max": np.round(self.bounds_max, 4).tolist(),
            "n_protected": self.n_protected,
        }


@dataclass
class ProtectedLabelResult:
    """Output of structural-prior label fusion."""

    vertex_labels: list[str]
    prior: RoomShellPrior
    n_restored: int

    def to_json(self) -> dict:
        data = self.prior.to_json()
        data["n_restored"] = self.n_restored
        return data


def estimate_room_shell_prior(
    mesh: o3d.geometry.TriangleMesh,
    config: StructuralProtectionConfig | None = None,
) -> RoomShellPrior:
    """Estimate a protected Manhattan cuboid shell from an aligned mesh.

    The mesh is assumed to be in the cleaner's room frame: floor near ``z=0``
    and walls roughly axis-aligned.  Bounds use percentiles rather than raw
    extrema so a small amount of clutter protruding beyond a wall does not move
    the room shell.
    """
    cfg = config or StructuralProtectionConfig()
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    n = len(verts)

    empty_labels = ["unknown"] * n
    if n == 0 or cfg.level == "off":
        return RoomShellPrior(
            bounds_min=np.zeros(3, dtype=np.float64),
            bounds_max=np.zeros(3, dtype=np.float64),
            protected_mask=np.zeros(n, dtype=bool),
            protected_labels=empty_labels,
            distance_to_shell=np.full(n, np.inf, dtype=np.float64),
            confidence=np.zeros(n, dtype=np.float32),
            config=cfg,
        )

    mesh.compute_vertex_normals()
    normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
    if len(normals) != n:
        normals = np.zeros_like(verts)

    p = float(np.clip(cfg.outer_percentile, 0.0, 10.0))
    bounds_min = np.percentile(verts, p, axis=0)
    bounds_max = np.percentile(verts, 100.0 - p, axis=0)

    protected = np.zeros(n, dtype=bool)
    labels = np.array(empty_labels, dtype=object)
    dist_best = np.full(n, np.inf, dtype=np.float64)
    confidence = np.zeros(n, dtype=np.float32)

    planes: tuple[tuple[int, float, str], ...] = (
        (2, float(bounds_min[2]), "floor"),
        (2, float(bounds_max[2]), "ceiling"),
        (0, float(bounds_min[0]), "wall"),
        (0, float(bounds_max[0]), "wall"),
        (1, float(bounds_min[1]), "wall"),
        (1, float(bounds_max[1]), "wall"),
    )

    for axis, coord, label in planes:
        dist = np.abs(verts[:, axis] - coord)
        normal_ok = np.abs(normals[:, axis]) >= cfg.normal_cos_min
        near = dist <= cfg.plane_tolerance_m
        candidate = near & normal_ok

        update = candidate & (dist < dist_best)
        protected[update] = True
        labels[update] = label
        dist_best[update] = dist[update]

        # Confidence is only a geometric prior, not model certainty.  Vertices
        # exactly on the shell plane and with axis-aligned normals score highest.
        plane_conf = np.clip(1.0 - dist / max(cfg.plane_tolerance_m, 1e-6), 0.0, 1.0)
        plane_conf *= np.clip(np.abs(normals[:, axis]), 0.0, 1.0)
        confidence[update] = np.maximum(confidence[update], plane_conf[update]).astype(np.float32)

    return RoomShellPrior(
        bounds_min=bounds_min.astype(np.float64),
        bounds_max=bounds_max.astype(np.float64),
        protected_mask=protected,
        protected_labels=labels.astype(str).tolist(),
        distance_to_shell=dist_best,
        confidence=confidence,
        config=cfg,
    )


def apply_structural_protection(
    mesh: o3d.geometry.TriangleMesh,
    vertex_labels: list[str],
    config: StructuralProtectionConfig | None = None,
    movable_votes: np.ndarray | None = None,
) -> ProtectedLabelResult:
    """Restore protected floor/wall/ceiling labels after AI segmentation.

    If an ambiguous vertex is near the outer room shell and has a compatible
    surface normal, the geometric prior can relabel it as floor/wall/ceiling.
    Explicit object labels and static infrastructure are not relabelled; the
    separate virtual room shell supplies missing walls behind occluders.
    """
    cfg = config or StructuralProtectionConfig()
    prior = estimate_room_shell_prior(mesh, cfg)
    labels = list(vertex_labels)

    if cfg.level == "off" or prior.n_protected == 0:
        return ProtectedLabelResult(labels, prior, 0)

    votes = np.zeros(len(labels), dtype=np.int32) if movable_votes is None else np.asarray(movable_votes)
    min_votes = cfg.min_movable_votes_to_override()
    n_restored = 0

    for idx, is_protected in enumerate(prior.protected_mask):
        if not is_protected:
            continue
        current = labels[idx]
        protected_label = prior.protected_labels[idx]
        if protected_label not in STRUCTURAL_LABELS:
            continue
        if current in STRUCTURAL_LABELS or current in STATIC_INFRASTRUCTURE_LABELS:
            continue
        if current not in AMBIGUOUS_LABELS:
            continue
        if int(votes[idx]) >= min_votes:
            continue
        labels[idx] = protected_label
        n_restored += 1

    return ProtectedLabelResult(labels, prior, n_restored)


def flatten_structural_labels_to_room_shell(
    mesh: o3d.geometry.TriangleMesh,
    vertex_labels: list[str],
    prior: RoomShellPrior,
    allow_floor_platform: bool = True,
    platform_min_height_m: float = 0.08,
    platform_max_height_m: float = 0.55,
    platform_min_vertices: int = 250,
) -> tuple[o3d.geometry.TriangleMesh, dict]:
    """Snap wall/floor/ceiling vertices to strict flat room-shell planes.

    Fixed infrastructure labels such as server racks and cable trays are left
    untouched.  Floor vertices normally snap to Z=0 (the fitted floor plane from
    the cleaner).  A small raised entrance/platform slab is preserved as a
    second flat floor level when enough floor-labelled vertices support it.
    """
    labels = np.asarray(vertex_labels, dtype=object)
    verts = np.asarray(mesh.vertices, dtype=np.float64).copy()
    if len(verts) == 0:
        return o3d.geometry.TriangleMesh(mesh), {"enabled": True, "n_flattened": 0}

    mn = prior.bounds_min.astype(float)
    mx = prior.bounds_max.astype(float)
    floor_coord = 0.0 if mn[2] - 0.75 <= 0.0 <= mx[2] + 0.75 else float(mn[2])
    ceiling_coord = float(mx[2])
    n_flattened = 0

    floor_idx = np.where(labels == "floor")[0]
    platform_coord: float | None = None
    platform_count = 0
    if len(floor_idx):
        floor_z = verts[floor_idx, 2]
        platform_mask = (
            allow_floor_platform
            & (floor_z >= floor_coord + platform_min_height_m)
            & (floor_z <= floor_coord + platform_max_height_m)
        )
        if int(platform_mask.sum()) >= platform_min_vertices:
            platform_coord = float(np.median(floor_z[platform_mask]))
            platform_vertices = floor_idx[platform_mask]
            base_vertices = floor_idx[~platform_mask]
            verts[platform_vertices, 2] = platform_coord
            verts[base_vertices, 2] = floor_coord
            platform_count = int(len(platform_vertices))
        else:
            verts[floor_idx, 2] = floor_coord
        n_flattened += int(len(floor_idx))

    ceiling_idx = np.where(labels == "ceiling")[0]
    if len(ceiling_idx):
        verts[ceiling_idx, 2] = ceiling_coord
        n_flattened += int(len(ceiling_idx))

    wall_idx = np.where(labels == "wall")[0]
    if len(wall_idx):
        wall_verts = verts[wall_idx]
        distances = np.stack(
            [
                np.abs(wall_verts[:, 0] - mn[0]),
                np.abs(wall_verts[:, 0] - mx[0]),
                np.abs(wall_verts[:, 1] - mn[1]),
                np.abs(wall_verts[:, 1] - mx[1]),
            ],
            axis=1,
        )
        nearest = np.argmin(distances, axis=1)
        for plane_id, axis, coord in [
            (0, 0, mn[0]),
            (1, 0, mx[0]),
            (2, 1, mn[1]),
            (3, 1, mx[1]),
        ]:
            selected = wall_idx[nearest == plane_id]
            if len(selected):
                verts[selected, axis] = float(coord)
        n_flattened += int(len(wall_idx))

    result = o3d.geometry.TriangleMesh()
    result.vertices = o3d.utility.Vector3dVector(verts)
    result.triangles = mesh.triangles
    if mesh.has_vertex_colors():
        result.vertex_colors = mesh.vertex_colors
    result.compute_vertex_normals()

    return result, {
        "enabled": True,
        "n_flattened": n_flattened,
        "floor_coord": round(float(floor_coord), 4),
        "ceiling_coord": round(float(ceiling_coord), 4),
        "wall_bounds_min_xy": np.round(mn[:2], 4).tolist(),
        "wall_bounds_max_xy": np.round(mx[:2], 4).tolist(),
        "platform_coord": None if platform_coord is None else round(platform_coord, 4),
        "platform_vertices": platform_count,
    }


def build_cuboid_shell_mesh(
    prior: RoomShellPrior,
) -> o3d.geometry.TriangleMesh:
    """Build a simple cuboid mesh from the estimated room-shell bounds.

    This is not used as the default cleanup output yet.  It is provided for the
    planned strict ``shell`` mode where simulation prefers a perfect envelope
    over a noisy reconstructed surface.
    """
    mn = prior.bounds_min.astype(float)
    mx = prior.bounds_max.astype(float)
    x0, y0, z0 = mn.tolist()
    x1, y1, z1 = mx.tolist()

    vertices = np.array(
        [
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
        ],
        dtype=np.float64,
    )
    triangles = np.array(
        [
            [0, 1, 2], [0, 2, 3],  # floor
            [4, 6, 5], [4, 7, 6],  # ceiling
            [0, 4, 5], [0, 5, 1],
            [1, 5, 6], [1, 6, 2],
            [2, 6, 7], [2, 7, 3],
            [3, 7, 4], [3, 4, 0],
        ],
        dtype=np.int32,
    )

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    return mesh


def append_virtual_room_shell(
    mesh: o3d.geometry.TriangleMesh,
    prior: RoomShellPrior,
    color: tuple[float, float, float] = (0.18, 0.38, 0.70),
) -> tuple[o3d.geometry.TriangleMesh, dict]:
    """Append a virtual outer room-shell cuboid to an existing mesh.

    The shell is explicit geometry behind occluding racks and objects.  This
    prevents the structural prior from having to reinterpret foreground server
    rack faces as walls just to obtain a closed room envelope.
    """
    base_vertices = np.asarray(mesh.vertices, dtype=np.float64)
    base_triangles = np.asarray(mesh.triangles, dtype=np.int32)
    shell = build_cuboid_shell_mesh(prior)
    shell_vertices = np.asarray(shell.vertices, dtype=np.float64)
    shell_triangles = np.asarray(shell.triangles, dtype=np.int32)

    result = o3d.geometry.TriangleMesh()
    if len(base_vertices) == 0:
        result.vertices = shell.vertices
        result.triangles = shell.triangles
    else:
        result.vertices = o3d.utility.Vector3dVector(np.vstack([base_vertices, shell_vertices]))
        result.triangles = o3d.utility.Vector3iVector(
            np.vstack([base_triangles, shell_triangles + len(base_vertices)])
        )

    if mesh.has_vertex_colors():
        base_colors = np.asarray(mesh.vertex_colors, dtype=np.float64)
        if len(base_colors) == len(base_vertices):
            shell_colors = np.tile(np.asarray(color, dtype=np.float64), (len(shell_vertices), 1))
            result.vertex_colors = o3d.utility.Vector3dVector(np.vstack([base_colors, shell_colors]))

    result.compute_vertex_normals()
    return result, {
        "enabled": True,
        "vertices_added": int(len(shell_vertices)),
        "triangles_added": int(len(shell_triangles)),
        "bounds_min": np.round(prior.bounds_min, 4).tolist(),
        "bounds_max": np.round(prior.bounds_max, 4).tolist(),
    }
