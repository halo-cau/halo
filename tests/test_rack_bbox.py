"""rack_bbox is the SINGLE rack->voxel geometry for the thermal engine.

The exhaust heat source (solver) and the intake/exhaust readout (metrics) must select exactly the
voxels the stamper (engine/vision/voxelizer._stamp_rack) laid on each rack, for every facing. These
tests pin that invariant so the solver/metrics copies that previously diverged can never silently
desync again, and so a rotated rack is read correctly.
"""
import numpy as np
import pytest

from engine.core.config import RACK_BODY, RACK_EXHAUST, RACK_INTAKE
from engine.core.data_types import Coordinate, RackFacing, RackPlacement
from engine.thermal._rack_geometry import rack_bbox, world_to_index

FACINGS = [RackFacing.PLUS_Y, RackFacing.MINUS_Y, RackFacing.PLUS_X, RackFacing.MINUS_X]
# pos is the intake-face center; each places the same body volume well inside a 40x40x30 grid.
POS = {
    RackFacing.PLUS_Y: (2.0, 2.45),
    RackFacing.MINUS_Y: (2.0, 1.55),
    RackFacing.PLUS_X: (2.45, 2.0),
    RackFacing.MINUS_X: (1.55, 2.0),
}


def _stamp(facing):
    """Stamp one 42U_real rack with the REAL stamper; return (grid, rack)."""
    voxelizer = pytest.importorskip("engine.vision.voxelizer")
    origin = np.zeros(3)
    shape = (40, 40, 30)
    px, py = POS[facing]
    rack = RackPlacement(position=Coordinate(px, py, 0.0), facing=facing, rack_type="42U_real")
    grid = np.zeros(shape, np.int8)
    voxelizer._stamp_rack(grid, rack, origin)
    return grid, rack, origin, shape


def test_single_definition_shared_by_solver_and_metrics():
    """Both readout sites resolve to the one rack_bbox — no duplicated copies left."""
    import engine.thermal.metrics as metrics
    import engine.thermal.solver as solver

    assert solver._rack_bbox is metrics._rack_bbox is rack_bbox


def test_world_to_index_matches_the_stamper():
    """The shared index helper must agree with the stamper's _world_to_index voxel-for-voxel (the
    consistency the refactor guarantees), including the floor quirk where 1.4/0.1 == 13.999... floors
    to 13 -- round would give 14 and drop the +facing exhaust voxel."""
    voxelizer = pytest.importorskip("engine.vision.voxelizer")
    o = np.zeros(3)
    for pos in [(1.4, 2.5, 0.0), (0.95, 2.95, 0.0), (6.7, 0.9, 0.0)]:
        assert world_to_index(*pos, o) == tuple(int(v) for v in voxelizer._world_to_index(*pos, o))
    assert world_to_index(1.4, 0.0, 0.0, o)[0] == 13   # floor, not round


@pytest.mark.parametrize("facing", FACINGS)
def test_bbox_captures_every_stamped_voxel(facing):
    """For each facing, the shared bbox contains every intake / exhaust / body voxel the stamper wrote,
    and is tight (its volume equals the rack's)."""
    grid, rack, origin, shape = _stamp(facing)
    x0, x1, y0, y1, z0, z1 = rack_bbox(rack, origin, shape)
    sub = grid[x0:x1, y0:y1, z0:z1]
    for vid in (RACK_INTAKE, RACK_EXHAUST, RACK_BODY):
        total = int((grid == vid).sum())
        assert total > 0, f"{vid} should have been stamped"
        assert int((sub == vid).sum()) == total, f"{vid} voxels fall outside the bbox for {facing}"
    # tight: no rack voxel outside, and no empty padding (every captured voxel belongs to the rack)
    assert int((sub != 0).sum()) == int((grid != 0).sum())
    assert sub.size == int((grid != 0).sum())


def test_180_rotation_swaps_intake_and_exhaust():
    """A 180° flip (PLUS_Y -> MINUS_Y, same body) trades intake and exhaust sides, and the bbox still
    captures both — the property the editor's rotate relies on."""
    g0, r0, o0, s0 = _stamp(RackFacing.PLUS_Y)
    g1, r1, o1, s1 = _stamp(RackFacing.MINUS_Y)

    def captured(g, r, o, s):
        x0, x1, y0, y1, z0, z1 = rack_bbox(r, o, s)
        sub = g[x0:x1, y0:y1, z0:z1]
        return (int((sub == RACK_INTAKE).sum()) == int((g == RACK_INTAKE).sum())
                and int((sub == RACK_EXHAUST).sum()) == int((g == RACK_EXHAUST).sum()))

    assert captured(g0, r0, o0, s0) and captured(g1, r1, o1, s1)
    iy0 = np.argwhere(g0 == RACK_INTAKE)[:, 1].mean()
    ey0 = np.argwhere(g0 == RACK_EXHAUST)[:, 1].mean()
    iy1 = np.argwhere(g1 == RACK_INTAKE)[:, 1].mean()
    ey1 = np.argwhere(g1 == RACK_EXHAUST)[:, 1].mean()
    assert iy0 > ey0 and iy1 < ey1     # intake/exhaust on opposite sides after the flip
