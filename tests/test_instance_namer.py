"""Unit tests for engine.vision.instance_namer — the geometry-priors namer.

Builds a synthetic server room with unambiguous instances (a floor slab, a
ceiling slab, a wall sheet, a free-standing rack box, a floating clutter box)
and asserts each instance gets the geometrically correct label. Deterministic
and dependency-light (pure numpy) — it pins the decision list independently of
any real, messy Point-SAM cloud.
"""

import numpy as np
import pytest

from engine.vision.instance_namer import (
    CEILING,
    FLOOR,
    OBJECT,
    RACK,
    WALL,
    NamerConfig,
    classify,
    compute_features,
    estimate_room_frame,
    name_instances,
)

RNG = np.random.default_rng(0)
FLOOR_Z, CEIL_Z = 0.0, 2.5
X_MAX, Y_MAX = 5.0, 8.0


def _plane(axis_fixed: int, fixed_val: float, a_rng, b_rng, n=1500, jitter=0.01):
    """A thin planar slab perpendicular to ``axis_fixed``."""
    pts = np.zeros((n, 3))
    free = [i for i in range(3) if i != axis_fixed]
    pts[:, axis_fixed] = fixed_val + RNG.normal(0, jitter, n)
    pts[:, free[0]] = RNG.uniform(*a_rng, n)
    pts[:, free[1]] = RNG.uniform(*b_rng, n)
    return pts


def _box(xr, yr, zr, n=1500):
    """A filled (volumetric) box."""
    return np.column_stack([
        RNG.uniform(*xr, n), RNG.uniform(*yr, n), RNG.uniform(*zr, n),
    ])


@pytest.fixture
def synthetic_room():
    floor = _plane(2, FLOOR_Z, (0, X_MAX), (0, Y_MAX))
    ceiling = _plane(2, CEIL_Z, (0, X_MAX), (0, Y_MAX))
    wall = _plane(0, 0.0, (0, Y_MAX), (0, CEIL_Z))          # x≈0 sheet
    rack = _box((2.0, 2.6), (3.0, 4.0), (0.0, 2.0))          # rests, ceiling gap
    clutter = _box((1.0, 1.3), (1.0, 1.3), (0.8, 1.1))       # small, floating

    chunks = [floor, ceiling, wall, rack, clutter]
    pts = np.vstack(chunks)
    ids = np.concatenate([np.full(len(c), i) for i, c in enumerate(chunks)])
    expected = {0: FLOOR, 1: CEILING, 2: WALL, 3: RACK, 4: OBJECT}
    return pts, ids, expected


def test_room_frame_recovers_floor_and_ceiling(synthetic_room):
    pts, _, _ = synthetic_room
    frame = estimate_room_frame(pts, NamerConfig(), up_axis=2)
    assert frame.up_axis == 2
    assert frame.floor == pytest.approx(FLOOR_Z, abs=0.05)
    assert frame.ceiling == pytest.approx(CEIL_Z, abs=0.05)
    assert frame.height == pytest.approx(CEIL_Z - FLOOR_Z, abs=0.05)


def test_each_instance_named_correctly(synthetic_room):
    pts, ids, expected = synthetic_room
    result = name_instances(pts, ids, up_axis=2)
    got = {ni.instance_id: ni.label for ni in result.instances}
    assert got == expected


def test_point_labels_broadcast_and_cover_all_points(synthetic_room):
    pts, ids, expected = synthetic_room
    result = name_instances(pts, ids, up_axis=2)
    assert len(result.point_labels) == len(pts)
    # Every floor-instance point should carry the floor label, etc.
    for iid, lbl in expected.items():
        assert set(result.point_labels[ids == iid]) == {lbl}
    assert result.point_colors.shape == (len(pts), 3)
    assert result.point_colors.min() >= 0.0 and result.point_colors.max() <= 1.0


def test_orientation_drives_floor_vs_wall(synthetic_room):
    """A horizontal slab is floor/ceiling; the same footprint stood vertical is a wall."""
    pts, ids, _ = synthetic_room
    frame = estimate_room_frame(pts, NamerConfig(), up_axis=2)
    cfg = NamerConfig()

    floor_feats = compute_features(0, pts[ids == 0], frame, cfg)
    wall_feats = compute_features(2, pts[ids == 2], frame, cfg)
    assert floor_feats.verticality > 0.9        # normal ≈ +up  → horizontal surface
    assert wall_feats.verticality < 0.2          # normal ⟂ up   → vertical surface
    assert classify(floor_feats, frame, cfg)[0] == FLOOR
    assert classify(wall_feats, frame, cfg)[0] == WALL


def test_small_instances_fall_through_to_object():
    pts = _box((0, 0.1), (0, 0.1), (0, 0.1), n=10)
    ids = np.zeros(len(pts), dtype=int)
    result = name_instances(pts, ids, up_axis=2)
    assert result.instances[0].label == OBJECT


def test_negative_ids_are_unassigned_objects():
    floor = _plane(2, 0.0, (0, 4), (0, 4))
    noise = _box((0, 4), (0, 4), (0, 2.5), n=200)
    pts = np.vstack([floor, noise])
    ids = np.concatenate([np.zeros(len(floor), int), np.full(len(noise), -1)])
    result = name_instances(pts, ids, up_axis=2)
    # The -1 points are not emitted as a named instance, but stay 'object'.
    assert all(ni.instance_id >= 0 for ni in result.instances)
    assert set(result.point_labels[ids == -1]) == {OBJECT}
