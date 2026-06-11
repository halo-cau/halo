"""Geometry-priors namer for class-agnostic 3D instances.

Takes a point cloud already partitioned into instances — Point-SAM masks,
Mask3D proposals, ground-truth instance ids, or DBSCAN clusters — and assigns
each *whole instance* a server-room semantic label (``floor`` / ``ceiling`` /
``wall`` / ``server rack`` / ``object``) using only geometry:

    orientation  — the instance's dominant surface normal vs. the up axis,
    height       — where the instance sits between floor and ceiling,
    position     — its footprint and whether it spans / touches the envelope.

No learned weights, so it runs anywhere (including the 8 GB GPU path) and is fully
deterministic. This is the *naming* stage that complements the instance
*proposal* stage: the proposer answers "which points form one thing", this
answers "what is that thing". It is backend-agnostic on purpose — the only
input contract is ``points (N, 3)`` plus an ``instance_ids (N,)`` partition.

Colors come from :data:`engine.vision.segmentor_base.LABEL_PALETTE` so a named
cloud renders identically to the rest of the CV pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from engine.vision.segmentor_base import label_to_color

# Canonical labels this namer can emit. ``object`` is the catch-all for clutter
# (boxes, equipment, AC units, noise) that doesn't match a structural prior.
FLOOR = "floor"
CEILING = "ceiling"
WALL = "wall"
RACK = "server rack"
OBJECT = "object"

NAMED_LABELS: tuple[str, ...] = (FLOOR, CEILING, WALL, RACK, OBJECT)


def _ramp(x: float, lo: float, hi: float) -> float:
    """Clamped linear ramp: 0 at/below ``lo``, 1 at/above ``hi``."""
    if hi <= lo:
        return 1.0 if x >= hi else 0.0
    return float(min(1.0, max(0.0, (x - lo) / (hi - lo))))


@dataclass(frozen=True)
class NamerConfig:
    """Thresholds for the geometry-prior decision list.

    Defaults are tuned for metre-scale server rooms with the up axis vertical
    (a ~2.5-3 m floor-to-ceiling slab). Every threshold is named so a reviewer
    can see exactly which orientation/height/position fact drives each label.
    """

    # --- room frame estimation ---
    floor_percentile: float = 1.0       # robust low/high percentiles of the
    ceiling_percentile: float = 99.0    # up coordinate define floor/ceiling z
    wall_percentile: float = 1.0        # lateral envelope percentiles

    # --- orientation gates (verticality = |surface_normal . up|) ---
    horiz_min: float = 0.70   # >= ⇒ a horizontal surface (floor / ceiling)
    vert_max: float = 0.50    # <= ⇒ a vertical surface (wall)

    # --- height membership (fraction of room height, 0=floor .. 1=ceiling) ---
    floor_h_max: float = 0.32     # centroid this low ⇒ floor-ish
    ceiling_h_min: float = 0.68   # centroid this high ⇒ ceiling-ish
    band_m: float = 0.30          # thickness of the floor/ceiling evidence band
    band_frac_hi: float = 0.55    # this share of points inside a band ⇒ commit
                                  # even when orientation/footprint are weak

    # --- footprint / span (position) ---
    struct_min_extent_m: float = 2.0   # a real floor/ceiling/big wall is wide
    wall_min_zspan: float = 0.45       # walls span a good chunk of room height
    wall_min_extent_m: float = 1.2     # ...and are laterally long

    # --- server rack: a free-standing tall box resting on the floor ---
    rack_h_min: float = 1.0     # shorter ⇒ a box, not a rack
    rack_h_max: float = 2.3     # taller-than-this + ceiling touch ⇒ wall
    rack_min_zspan: float = 0.35
    rack_foot_min: float = 0.12   # m^2 footprint window for a cabinet/row
    rack_foot_max: float = 3.5
    rack_min_lateral: float = 0.40
    rack_planar_max: float = 0.90  # a thin plane (planarity≈1) is a wall, not a rack

    # --- ambiguous room-spanning slab fallback (diagonal merged masks) ---
    bigwall_zspan: float = 0.80
    bigwall_extent_m: float = 2.5
    bigwall_planar: float = 0.60

    min_points: int = 40   # instances smaller than this are called ``object``


@dataclass(frozen=True)
class RoomFrame:
    """Estimated room envelope used as the reference for height/position."""

    up_axis: int
    floor: float
    ceiling: float
    lo: np.ndarray   # per-axis lower envelope (world coords)
    hi: np.ndarray   # per-axis upper envelope

    @property
    def height(self) -> float:
        return float(self.ceiling - self.floor)

    @property
    def lateral_axes(self) -> tuple[int, int]:
        return tuple(a for a in range(3) if a != self.up_axis)  # type: ignore[return-value]

    def height_fraction(self, z: float) -> float:
        h = self.height
        return 0.0 if h <= 1e-6 else float((z - self.floor) / h)


@dataclass
class InstanceFeatures:
    """Geometric description of one instance (all distances in metres)."""

    instance_id: int
    n_points: int
    centroid: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    normal: np.ndarray          # dominant surface normal (unit)
    verticality: float          # |normal . up|  (1=horizontal surface, 0=vertical)
    planarity: float            # 1 - lambda0/lambda1  (≈1 ⇒ flat sheet)
    height_center_frac: float   # centroid height as fraction of room height
    z_span_frac: float          # vertical extent / room height
    footprint_area: float       # lateral extent product (m^2)
    max_lateral_extent: float   # longer lateral side (m)
    floor_band_frac: float
    ceiling_band_frac: float
    rests_on_floor: bool
    reaches_ceiling: bool

    @property
    def extent(self) -> np.ndarray:
        return self.bbox_max - self.bbox_min


@dataclass
class NamedInstance:
    """An instance after naming: label, confidence, the deciding evidence."""

    features: InstanceFeatures
    label: str
    confidence: float
    reason: str
    color: tuple[float, float, float]

    @property
    def instance_id(self) -> int:
        return self.features.instance_id

    def to_json(self) -> dict:
        f = self.features
        return {
            "instance_id": f.instance_id,
            "label": self.label,
            "confidence": round(self.confidence, 3),
            "reason": self.reason,
            "n_points": f.n_points,
            "centroid": np.round(f.centroid, 3).tolist(),
            "extent": np.round(f.extent, 3).tolist(),
            "verticality": round(f.verticality, 3),
            "planarity": round(f.planarity, 3),
            "height_center_frac": round(f.height_center_frac, 3),
            "z_span_frac": round(f.z_span_frac, 3),
            "footprint_area": round(f.footprint_area, 3),
            "floor_band_frac": round(f.floor_band_frac, 3),
            "ceiling_band_frac": round(f.ceiling_band_frac, 3),
            "rests_on_floor": f.rests_on_floor,
            "reaches_ceiling": f.reaches_ceiling,
            "color_rgb": [round(c, 4) for c in self.color],
        }


@dataclass
class NamingResult:
    """Full output: per-instance names, the room frame, per-point arrays."""

    instances: list[NamedInstance]
    frame: RoomFrame
    point_labels: np.ndarray   # (N,) str — instance label broadcast to points
    point_colors: np.ndarray   # (N, 3) float in [0,1]
    config: NamerConfig

    def label_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for ni in self.instances:
            out[ni.label] = out.get(ni.label, 0) + 1
        return out

    def point_label_counts(self) -> dict[str, int]:
        u, c = np.unique(self.point_labels, return_counts=True)
        return {str(k): int(v) for k, v in zip(u, c)}

    def to_json(self) -> dict:
        return {
            "frame": {
                "up_axis": self.frame.up_axis,
                "floor": round(self.frame.floor, 4),
                "ceiling": round(self.frame.ceiling, 4),
                "height": round(self.frame.height, 4),
                "lo": np.round(self.frame.lo, 4).tolist(),
                "hi": np.round(self.frame.hi, 4).tolist(),
            },
            "n_instances": len(self.instances),
            "label_counts": self.label_counts(),
            "point_label_counts": self.point_label_counts(),
            "instances": [ni.to_json() for ni in self.instances],
        }


def infer_up_axis(points: np.ndarray) -> int:
    """Guess the up axis as the one with the smallest 1-99 percentile span.

    Rooms are wider than they are tall, so the vertical axis is the short one.
    Used only when the caller doesn't pass ``up_axis`` explicitly.
    """
    spans = [
        float(np.percentile(points[:, a], 99) - np.percentile(points[:, a], 1))
        for a in range(3)
    ]
    return int(np.argmin(spans))


def estimate_room_frame(
    points: np.ndarray,
    cfg: NamerConfig,
    up_axis: int | None = None,
) -> RoomFrame:
    """Robust floor/ceiling/envelope estimate from raw percentiles."""
    up = infer_up_axis(points) if up_axis is None else int(up_axis)
    floor = float(np.percentile(points[:, up], cfg.floor_percentile))
    ceiling = float(np.percentile(points[:, up], cfg.ceiling_percentile))
    lo = np.array(
        [float(np.percentile(points[:, a], cfg.wall_percentile)) for a in range(3)]
    )
    hi = np.array(
        [float(np.percentile(points[:, a], 100.0 - cfg.wall_percentile)) for a in range(3)]
    )
    lo[up], hi[up] = floor, ceiling
    return RoomFrame(up_axis=up, floor=floor, ceiling=ceiling, lo=lo, hi=hi)


def compute_features(
    instance_id: int,
    pts: np.ndarray,
    frame: RoomFrame,
    cfg: NamerConfig,
) -> InstanceFeatures:
    """Compute the orientation/height/position descriptor for one instance."""
    up = frame.up_axis
    lat = list(frame.lateral_axes)
    centroid = pts.mean(axis=0)
    bbox_min = pts.min(axis=0)
    bbox_max = pts.max(axis=0)
    extent = bbox_max - bbox_min

    # Dominant surface normal via PCA: the eigenvector of the smallest
    # eigenvalue of the covariance is the best-fit plane normal. Planarity is
    # how much flatter the smallest direction is than the middle one.
    centered = pts - centroid
    cov = centered.T @ centered / max(1, len(pts))
    evals, evecs = np.linalg.eigh(cov)  # ascending
    normal = evecs[:, 0]
    if normal[up] < 0:
        normal = -normal
    planarity = float(1.0 - evals[0] / max(evals[1], 1e-9))
    verticality = float(abs(normal[up]))

    height_center_frac = frame.height_fraction(float(centroid[up]))
    z_span_frac = float(extent[up] / max(frame.height, 1e-6))
    foot = (float(extent[lat[0]]), float(extent[lat[1]]))
    footprint_area = foot[0] * foot[1]
    max_lateral_extent = max(foot)

    zc = pts[:, up]
    floor_band_frac = float(np.mean(zc <= frame.floor + cfg.band_m))
    ceiling_band_frac = float(np.mean(zc >= frame.ceiling - cfg.band_m))
    rests_on_floor = bool(bbox_min[up] <= frame.floor + max(cfg.band_m, 0.35))
    reaches_ceiling = bool(bbox_max[up] >= frame.ceiling - max(cfg.band_m, 0.35))

    return InstanceFeatures(
        instance_id=instance_id,
        n_points=len(pts),
        centroid=centroid,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        normal=normal,
        verticality=verticality,
        planarity=planarity,
        height_center_frac=height_center_frac,
        z_span_frac=z_span_frac,
        footprint_area=footprint_area,
        max_lateral_extent=max_lateral_extent,
        floor_band_frac=floor_band_frac,
        ceiling_band_frac=ceiling_band_frac,
        rests_on_floor=rests_on_floor,
        reaches_ceiling=reaches_ceiling,
    )


def classify(f: InstanceFeatures, frame: RoomFrame, cfg: NamerConfig) -> tuple[str, float, str]:
    """Ordered geometry decision list → ``(label, confidence, reason)``.

    Rules are tried in priority order; the first match wins. Confidence is the
    product of the soft memberships the winning rule actually relied on, so it
    reflects *how cleanly* the geometry matched (not a learned probability).
    """
    if f.n_points < cfg.min_points:
        return OBJECT, 0.2, f"only {f.n_points} pts (< {cfg.min_points})"

    v = f.verticality
    h = f.height_center_frac
    horizontal = v >= cfg.horiz_min
    vertical = v <= cfg.vert_max
    big_foot = f.max_lateral_extent >= cfg.struct_min_extent_m
    height_m = float(f.extent[frame.up_axis])

    # 1 — CEILING: horizontal surface, high up, and either wide or strongly
    # banded against the ceiling.
    if horizontal and (f.ceiling_band_frac >= cfg.band_frac_hi or h >= cfg.ceiling_h_min) \
            and (big_foot or f.ceiling_band_frac >= cfg.band_frac_hi):
        conf = _ramp(v, cfg.vert_max, 0.95) * max(f.ceiling_band_frac, _ramp(h, 0.5, 0.85))
        return CEILING, round(min(1.0, conf), 3), (
            f"horizontal (|n·up|={v:.2f}) high in room (h={h:.2f}, ceil_band={f.ceiling_band_frac:.2f})"
        )

    # 2 — FLOOR: horizontal surface, low down, wide or strongly floor-banded.
    if horizontal and (f.floor_band_frac >= cfg.band_frac_hi or h <= cfg.floor_h_max) \
            and (big_foot or f.floor_band_frac >= cfg.band_frac_hi):
        conf = _ramp(v, cfg.vert_max, 0.95) * max(f.floor_band_frac, _ramp(0.45 - h, 0.0, 0.35))
        return FLOOR, round(min(1.0, conf), 3), (
            f"horizontal (|n·up|={v:.2f}) low in room (h={h:.2f}, floor_band={f.floor_band_frac:.2f})"
        )

    # 3 — WALL: a vertical sheet that is tall and laterally long.
    if vertical and f.z_span_frac >= cfg.wall_min_zspan and f.max_lateral_extent >= cfg.wall_min_extent_m:
        conf = (1.0 - _ramp(v, 0.0, cfg.horiz_min)) * _ramp(f.z_span_frac, cfg.wall_min_zspan, 0.9)
        return WALL, round(min(1.0, conf), 3), (
            f"vertical sheet (|n·up|={v:.2f}), z-span={f.z_span_frac:.2f}, extent={f.max_lateral_extent:.1f}m"
        )

    # 4 — SERVER RACK: free-standing tall box on the floor with a ceiling gap.
    if f.rests_on_floor and not f.reaches_ceiling \
            and cfg.rack_h_min <= height_m <= cfg.rack_h_max \
            and f.z_span_frac >= cfg.rack_min_zspan \
            and cfg.rack_foot_min <= f.footprint_area <= cfg.rack_foot_max \
            and f.max_lateral_extent >= cfg.rack_min_lateral \
            and f.planarity < cfg.rack_planar_max:
        conf = _ramp(height_m, cfg.rack_h_min, 1.9) * (1.0 - _ramp(f.planarity, cfg.rack_planar_max - 0.2, 1.0))
        return RACK, round(min(1.0, max(0.35, conf)), 3), (
            f"floor-standing box h={height_m:.2f}m, foot={f.footprint_area:.2f}m², gap to ceiling"
        )

    # 5 — WALL (fallback): a room-spanning slab whose merged mask gives a
    # diagonal normal (Point-SAM often fuses a wall with adjacent floor).
    if (f.z_span_frac >= cfg.bigwall_zspan and f.max_lateral_extent >= cfg.bigwall_extent_m) \
            or (f.planarity >= cfg.bigwall_planar and f.z_span_frac >= cfg.wall_min_zspan
                and f.max_lateral_extent >= cfg.struct_min_extent_m):
        conf = 0.5 * _ramp(f.z_span_frac, cfg.wall_min_zspan, 1.0) + 0.5 * _ramp(f.max_lateral_extent, 2.0, 5.0)
        return WALL, round(min(0.9, conf), 3), (
            f"room-spanning slab z-span={f.z_span_frac:.2f}, extent={f.max_lateral_extent:.1f}m (merged mask)"
        )

    # 6 — OBJECT: clutter / equipment / noise.
    return OBJECT, 0.3, (
        f"no structural match (|n·up|={v:.2f}, h={h:.2f}, z-span={f.z_span_frac:.2f})"
    )


def name_instances(
    points: np.ndarray,
    instance_ids: np.ndarray,
    cfg: NamerConfig | None = None,
    up_axis: int | None = None,
    frame: RoomFrame | None = None,
) -> NamingResult:
    """Name every instance in a partitioned point cloud by geometry priors.

    Parameters
    ----------
    points
        ``(N, 3)`` float array of point coordinates (metres).
    instance_ids
        ``(N,)`` integer array assigning each point to one instance. Negative
        ids are treated as unassigned and labelled ``object``.
    cfg
        Threshold bundle; defaults suit a metre-scale server room.
    up_axis
        Index (0/1/2) of the vertical axis. Inferred from the cloud when None.
    frame
        Pre-computed room frame. Estimated from ``points`` when None.
    """
    cfg = cfg or NamerConfig()
    points = np.asarray(points, dtype=np.float64)
    instance_ids = np.asarray(instance_ids).ravel()
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be (N, 3), got {points.shape}")
    if len(instance_ids) != len(points):
        raise ValueError("instance_ids and points length mismatch")

    frame = frame or estimate_room_frame(points, cfg, up_axis=up_axis)

    point_labels = np.full(len(points), OBJECT, dtype=object)
    point_colors = np.zeros((len(points), 3), dtype=np.float64)
    point_colors[:] = label_to_color(OBJECT)

    named: list[NamedInstance] = []
    for raw_id in sorted(int(i) for i in np.unique(instance_ids)):
        mask = instance_ids == raw_id
        pts = points[mask]
        if raw_id < 0:
            # Unassigned points stay ``object`` (default color already set).
            continue
        feats = compute_features(raw_id, pts, frame, cfg)
        label, conf, reason = classify(feats, frame, cfg)
        color = label_to_color(label)
        point_labels[mask] = label
        point_colors[mask] = color
        named.append(NamedInstance(feats, label, conf, reason, color))

    # Largest instances first — most useful ordering for reports/legends.
    named.sort(key=lambda ni: ni.features.n_points, reverse=True)
    return NamingResult(
        instances=named,
        frame=frame,
        point_labels=point_labels,
        point_colors=point_colors,
        config=cfg,
    )
