"""Unit tests for the CV-twin -> RL-env converter (engine/rl/twin_bridge)."""
import json

import numpy as np
import pytest

from engine.rl.twin_bridge import RL_GRID, _CELL_V, twin_dir_to_rl_input, twin_to_rl_input


def _toy_twin(vx=40, vy=24, vz=24):
    """A toy empty-room grid: floor + ceiling slabs, perimeter walls, one standing infra box."""
    g = np.zeros((vx, vy, vz), np.int8)
    g[:, :, 0] = 1            # floor slab (would falsely block everything if not excluded)
    g[:, :, -1] = 1           # ceiling slab
    g[0, :, :] = g[-1, :, :] = 1
    g[:, 0, :] = g[:, -1, :] = 1
    g[10:14, 10:14, :15] = 1  # a standing cabinet mid-room (voxels -> cells 5..6)
    manifest = {"instances": [
        {"name": "ac_unit 1", "kind": "box", "vox_id": 3, "center": [0.3, 1.2, 1.0], "dims": [0.5, 1.5, 2.0]},
        {"name": "server rack 1", "kind": "rack", "center": [1.0, 1.0, 1.0], "dims": [0.6, 0.9, 1.95]},
        {"name": "server rack 2", "kind": "rack", "center": [1.6, 1.0, 1.0], "dims": [0.6, 0.9, 1.95]},
    ]}
    return g, manifest


def test_output_shape_and_keys():
    g, m = _toy_twin()
    out = twin_to_rl_input(g, m, rack_num=5)
    assert set(out) == {"obstacle", "cooling_pos", "rack_num", "ceiling_m"}
    assert out["obstacle"].shape == (RL_GRID, RL_GRID)
    assert out["obstacle"].dtype == np.int8


def test_floor_ceiling_excluded_interior_is_free():
    # The floor & ceiling slabs must NOT make every cell an obstacle: a clear interior stays free.
    g, m = _toy_twin()
    obstacle = twin_to_rl_input(g, m, rack_num=5)["obstacle"]
    assert obstacle[8, 4] == 0          # interior away from walls/cabinet -> free
    assert int((obstacle == 0).sum()) > 50


def test_walls_and_infra_are_obstacles():
    g, m = _toy_twin()
    obstacle = twin_to_rl_input(g, m, rack_num=5)["obstacle"]
    assert obstacle[0, 3] == 1          # perimeter wall (voxel x=0 -> cell 0)
    assert obstacle[6, 6] == 1          # standing cabinet (voxels 10:14 -> cell 5..6)
    assert obstacle[RL_GRID - 1, RL_GRID - 1] == 1   # outside-the-room margin stays obstacle


def test_rack_num_defaults_to_manifest_count():
    g, m = _toy_twin()
    assert twin_to_rl_input(g, m)["rack_num"] == 2          # two racks in the manifest
    assert twin_to_rl_input(g, m, rack_num=12)["rack_num"] == 12   # explicit design target wins


def test_cooling_from_ac_is_free_and_bounded():
    g, m = _toy_twin()
    out = twin_to_rl_input(g, m, rack_num=5, max_coolers=6)
    cp = out["cooling_pos"]
    assert len(cp) >= 1 and len(cp) <= 6
    for gx, gy in cp:                                       # coolers must land on free cells
        assert out["obstacle"][gx, gy] == 0


def test_ceiling_m_is_room_height():
    g, m = _toy_twin(vz=24)
    assert twin_to_rl_input(g, m, rack_num=5)["ceiling_m"] == pytest.approx(2.4)


def test_room_larger_than_rl_grid_raises():
    g = np.zeros((RL_GRID * _CELL_V + 4, 24, 24), np.int8)  # > 10 m on one side
    g[:, :, 0] = 1
    with pytest.raises(ValueError):
        twin_to_rl_input(g, {"instances": []})


def test_short_object_threshold():
    """Objects >= 0.5 m are caught as obstacles; objects below the 0.5 m plane are not (point 1)."""
    vx, vy, vz = 40, 30, 24
    g = np.zeros((vx, vy, vz), np.int8)
    g[:, :, 0] = g[:, :, -1] = 1                # floor + ceiling slabs
    g[0, :, :] = g[-1, :, :] = g[:, 0, :] = g[:, -1, :] = 1
    g[20:24, 20:24, 0:7] = 1                    # 0.7 m box -> cell (10, 10), caught
    g[28:32, 10:14, 0:3] = 1                    # 0.3 m box (below 0.5 m) -> cells (14:16, 5:7), free
    ob = twin_to_rl_input(g, {"instances": []}, rack_num=5)["obstacle"]
    assert ob[10, 10] == 1
    assert ob[14, 5] == 0 and ob[15, 6] == 0


def test_cooling_override_is_verbatim():
    """An explicit cooling layout (a design target) overrides the scanned AC (point 2)."""
    g, m = _toy_twin()
    out = twin_to_rl_input(g, m, rack_num=5, cooling_pos=[[3, 3], [4, 4]])
    assert out["cooling_pos"].tolist() == [[3, 3], [4, 4]]


def test_ceiling_m_flows_through_reset():
    """The scanned ceiling reaches the env's 3-D thermal bridge via reset() (point 3)."""
    from engine.rl.datacenter import DataCenterEnv

    g, m = _toy_twin(vz=24)
    out = twin_to_rl_input(g, m, rack_num=4)
    env = DataCenterEnv(grid_size=50, rack_num=4, ceiling_m=3.0)
    env.reset(options=out)
    assert env.ceiling_m == pytest.approx(out["ceiling_m"])           # = 2.4 m
    assert env._bridge._nz == round(out["ceiling_m"] / 0.1)           # bridge rebuilt at the new height


def test_twin_dir_loader(tmp_path):
    g, m = _toy_twin()
    np.save(tmp_path / "voxel_empty_grid.npy", g)
    (tmp_path / "placements.json").write_text(json.dumps(m))
    out = twin_dir_to_rl_input(tmp_path, rack_num=8)
    assert out["rack_num"] == 8 and out["obstacle"].shape == (RL_GRID, RL_GRID)
