"""Geometry priors for protecting the room shell during AI cleanup.

The goal is to reduce reliance on LiDAR-quality geometry by using a simple,
explicit assumption: after alignment, the true room envelope is a Manhattan
cuboid made from the outermost floor, ceiling, and wall planes.  AI segmentation
can still remove clutter, but it must overcome this shell prior before deleting
vertices that geometrically look like floor/wall/ceiling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import open3d as o3d
from scipy.ndimage import label as ndi_label

from engine.vision.segmentor_base import AMBIGUOUS_LABELS, STATIC_INFRASTRUCTURE_LABELS, STRUCTURAL_LABELS

ProtectionLevel = Literal["off", "light", "balanced", "strong", "shell"]

# Domain prior for typical server-room slab heights (ceiling to floor).
DEFAULT_ROOM_HEIGHT_M: float = 2.7
# If ceiling- and floor-derived heights disagree by more than this, flag it.
HEIGHT_DISAGREEMENT_FLAG_M: float = 0.30


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
    extra: dict = field(default_factory=dict)

    @property
    def n_protected(self) -> int:
        return int(self.protected_mask.sum())

    def to_json(self) -> dict:
        data = {
            "level": self.config.level,
            "outer_percentile": self.config.outer_percentile,
            "plane_tolerance_m": self.config.plane_tolerance_m,
            "normal_cos_min": self.config.normal_cos_min,
            "protected_min_movable_votes": self.config.min_movable_votes_to_override(),
            "bounds_min": np.round(self.bounds_min, 4).tolist(),
            "bounds_max": np.round(self.bounds_max, 4).tolist(),
            "n_protected": self.n_protected,
        }
        if self.extra:
            data["extra"] = self.extra
        return data


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


def _detect_ceiling_vertices_geometric(
    verts: np.ndarray,
    normals: np.ndarray,
    top_z_percentile: float = 80.0,
    normal_z_min: float = 0.85,
) -> np.ndarray:
    """Return a bool mask over vertices that geometrically look like ceiling.

    Two conditions: vertex z is in the upper ``top_z_percentile`` of the cloud
    AND the vertex normal is close to vertical (|n_z| above ``normal_z_min``).
    Normal sign is ignored because Open3D's vertex-normal orientation isn't
    globally consistent on point clouds.
    """
    if len(verts) == 0:
        return np.zeros(0, dtype=bool)
    z_thresh = float(np.percentile(verts[:, 2], top_z_percentile))
    in_top = verts[:, 2] >= z_thresh
    if len(normals) == len(verts):
        horizontal = np.abs(normals[:, 2]) >= normal_z_min
    else:
        horizontal = np.ones(len(verts), dtype=bool)
    return in_top & horizontal


def _largest_xy_component(
    verts_xy: np.ndarray,
    voxel_size_m: float = 0.10,
) -> tuple[np.ndarray, dict]:
    """Return the indices of vertices that belong to the largest connected
    component when their XY positions are rasterised to a 2D grid.

    Drops thin stubs (aisle extending past a doorway) when the connecting
    cells are sparser than the room ceiling.
    """
    if len(verts_xy) == 0:
        return np.zeros(0, dtype=np.int64), {"n_components": 0}
    x = verts_xy[:, 0]
    y = verts_xy[:, 1]
    x0, x1 = float(x.min()), float(x.max())
    y0, y1 = float(y.min()), float(y.max())
    nx = max(1, int(np.ceil((x1 - x0) / voxel_size_m)) + 1)
    ny = max(1, int(np.ceil((y1 - y0) / voxel_size_m)) + 1)
    ix = np.clip(np.floor((x - x0) / voxel_size_m).astype(np.int64), 0, nx - 1)
    iy = np.clip(np.floor((y - y0) / voxel_size_m).astype(np.int64), 0, ny - 1)
    occupied = np.zeros((nx, ny), dtype=bool)
    occupied[ix, iy] = True
    structure = np.ones((3, 3), dtype=bool)
    labeled, n_components = ndi_label(occupied, structure=structure)
    if n_components == 0:
        return np.zeros(0, dtype=np.int64), {"n_components": 0}
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    largest = int(np.argmax(sizes))
    cell_label = labeled[ix, iy]
    keep_idx = np.where(cell_label == largest)[0]
    return keep_idx, {
        "n_components": int(n_components),
        "largest_component_cells": int(sizes[largest]),
        "voxel_size_m": float(voxel_size_m),
    }


def _estimate_bounds_from_ceiling(
    verts: np.ndarray,
    normals: np.ndarray,
    *,
    top_z_percentile: float = 80.0,
    normal_z_min: float = 0.85,
    rect_inset_percentile: float = 2.0,
    floor_band_m: float = 0.25,
    min_ceiling_vertices: int = 1000,
) -> tuple[np.ndarray, np.ndarray, dict] | None:
    """Derive cuboid bounds from a ceiling-projection rectangle.

    Returns ``(bounds_min, bounds_max, meta)`` or ``None`` if ceiling
    detection didn't find enough support to be trusted.
    """
    if len(verts) == 0:
        return None
    ceiling_mask = _detect_ceiling_vertices_geometric(
        verts, normals,
        top_z_percentile=top_z_percentile,
        normal_z_min=normal_z_min,
    )
    n_ceiling_raw = int(ceiling_mask.sum())
    if n_ceiling_raw < min_ceiling_vertices:
        return None

    ceiling_verts = verts[ceiling_mask]
    keep_local, comp_meta = _largest_xy_component(ceiling_verts[:, :2])
    if len(keep_local) < min_ceiling_vertices:
        return None
    room_ceiling_verts = ceiling_verts[keep_local]

    p = float(np.clip(rect_inset_percentile, 0.0, 10.0))
    x_min = float(np.percentile(room_ceiling_verts[:, 0], p))
    x_max = float(np.percentile(room_ceiling_verts[:, 0], 100.0 - p))
    y_min = float(np.percentile(room_ceiling_verts[:, 1], p))
    y_max = float(np.percentile(room_ceiling_verts[:, 1], 100.0 - p))
    ceiling_z = float(np.median(room_ceiling_verts[:, 2]))

    # Floor evidence: low-z vertices that fall inside the ceiling rectangle.
    inside_rect = (
        (verts[:, 0] >= x_min) & (verts[:, 0] <= x_max)
        & (verts[:, 1] >= y_min) & (verts[:, 1] <= y_max)
    )
    inside_verts = verts[inside_rect]
    floor_evidence_z: float | None = None
    if len(inside_verts):
        low_thresh = float(np.percentile(inside_verts[:, 2], 1.0)) + floor_band_m
        floor_band = inside_verts[inside_verts[:, 2] <= low_thresh]
        if len(floor_band) >= min_ceiling_vertices // 4:
            floor_evidence_z = float(np.median(floor_band[:, 2]))

    floor_fallback_z = ceiling_z - DEFAULT_ROOM_HEIGHT_M
    if floor_evidence_z is None:
        floor_z = floor_fallback_z
        floor_source = "domain_prior_fallback"
        height_disagreement = None
    else:
        floor_z = floor_evidence_z
        floor_source = "rectangle_low_z"
        height_disagreement = abs(floor_evidence_z - floor_fallback_z)

    bounds_min = np.array([x_min, y_min, floor_z], dtype=np.float64)
    bounds_max = np.array([x_max, y_max, ceiling_z], dtype=np.float64)
    meta = {
        "method": "ceiling_projection",
        "n_ceiling_candidates": n_ceiling_raw,
        "n_ceiling_in_largest_cc": int(len(keep_local)),
        "ceiling_z": round(ceiling_z, 4),
        "floor_z": round(floor_z, 4),
        "floor_source": floor_source,
        "floor_fallback_z": round(floor_fallback_z, 4),
        "height_disagreement_m": (
            None if height_disagreement is None else round(height_disagreement, 4)
        ),
        "height_disagreement_flagged": bool(
            height_disagreement is not None
            and height_disagreement > HEIGHT_DISAGREEMENT_FLAG_M
        ),
        "rect": {
            "x_min": round(x_min, 4), "x_max": round(x_max, 4),
            "y_min": round(y_min, 4), "y_max": round(y_max, 4),
        },
        "xy_components": comp_meta,
    }
    return bounds_min, bounds_max, meta


def estimate_room_shell_prior(
    mesh: o3d.geometry.TriangleMesh,
    config: StructuralProtectionConfig | None = None,
) -> RoomShellPrior:
    """Estimate a protected Manhattan cuboid shell from an aligned mesh.

    The mesh is assumed to be in the cleaner's room frame: floor near ``z=0``
    and walls roughly axis-aligned. Bounds are derived from the ceiling
    projection when enough ceiling vertices are detected geometrically;
    otherwise the percentile fallback (legacy behaviour) is used so the bounds
    don't get pulled by aisle bleed-through or window-side clutter.
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
    if len(normals) != n or not np.any(np.linalg.norm(normals, axis=1) > 1e-6):
        # Point-cloud input (no triangles → compute_vertex_normals is a no-op).
        # Estimate per-vertex normals via PCA on a kNN graph so the geometric
        # ceiling filter has signal.
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(verts)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.15, max_nn=24),
        )
        normals = np.asarray(pcd.normals, dtype=np.float64)
        if len(normals) != n:
            normals = np.zeros_like(verts)

    ceiling_result = _estimate_bounds_from_ceiling(verts, normals)
    if ceiling_result is not None:
        bounds_min, bounds_max, ceiling_meta = ceiling_result
        bounds_meta = ceiling_meta
    else:
        p = float(np.clip(cfg.outer_percentile, 0.0, 10.0))
        bounds_min = np.percentile(verts, p, axis=0)
        bounds_max = np.percentile(verts, 100.0 - p, axis=0)
        bounds_meta = {"method": "percentile_fallback", "outer_percentile": p}

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
        extra={"bounds": bounds_meta},
    )


def trim_outside_shell(
    mesh: o3d.geometry.TriangleMesh,
    prior: RoomShellPrior,
    margin_m: float = 0.05,
) -> tuple[o3d.geometry.TriangleMesh, dict]:
    """Drop vertices outside the prior's cuboid (plus a small margin).

    Triangles are kept only if all three vertices survive. Vertex colors and
    normals are carried through. Use this to strip aisle / window-side
    outliers once a robust ceiling-driven shell has been estimated.
    """
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    n_in = len(verts)
    if n_in == 0:
        return o3d.geometry.TriangleMesh(mesh), {"enabled": True, "n_in": 0, "n_out": 0, "n_dropped": 0}

    mn = prior.bounds_min.astype(np.float64) - margin_m
    mx = prior.bounds_max.astype(np.float64) + margin_m
    keep = (
        (verts[:, 0] >= mn[0]) & (verts[:, 0] <= mx[0])
        & (verts[:, 1] >= mn[1]) & (verts[:, 1] <= mx[1])
        & (verts[:, 2] >= mn[2]) & (verts[:, 2] <= mx[2])
    )
    n_out = int(keep.sum())
    if n_out == n_in:
        return o3d.geometry.TriangleMesh(mesh), {
            "enabled": True, "n_in": n_in, "n_out": n_in, "n_dropped": 0, "margin_m": margin_m,
        }

    old_to_new = np.full(n_in, -1, dtype=np.int64)
    old_to_new[keep] = np.arange(n_out)

    result = o3d.geometry.TriangleMesh()
    result.vertices = o3d.utility.Vector3dVector(verts[keep])

    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    if len(triangles):
        tri_keep = keep[triangles[:, 0]] & keep[triangles[:, 1]] & keep[triangles[:, 2]]
        new_tris = old_to_new[triangles[tri_keep]]
        valid = (new_tris >= 0).all(axis=1)
        new_tris = new_tris[valid]
        result.triangles = o3d.utility.Vector3iVector(new_tris)

    if mesh.has_vertex_colors():
        colors = np.asarray(mesh.vertex_colors)
        if len(colors) == n_in:
            result.vertex_colors = o3d.utility.Vector3dVector(colors[keep])
    if len(np.asarray(mesh.vertex_normals)) == n_in:
        normals = np.asarray(mesh.vertex_normals)
        result.vertex_normals = o3d.utility.Vector3dVector(normals[keep])
    elif len(np.asarray(result.triangles)) > 0:
        result.compute_vertex_normals()

    return result, {
        "enabled": True,
        "n_in": n_in,
        "n_out": n_out,
        "n_dropped": n_in - n_out,
        "margin_m": float(margin_m),
    }


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
