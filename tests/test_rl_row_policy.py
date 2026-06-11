"""Tests for the row-placement action space and the spatial Maskable policy (engine/rl)."""
import numpy as np
import pytest

# These need the RL stack (gymnasium always; torch + sb3 only for the policy test).
gym = pytest.importorskip("gymnasium")
from engine.rl.datacenter import _RACK_W_CELLS, DataCenterEnv  # noqa: E402


def _empty_room(env):
    """Reset to a fixed, fully open room with a known rack budget (bypass the random reset)."""
    env.reset(seed=0)
    env.obstacle[:] = 0
    env.rack_map[:] = 0
    env.rack_dir[:] = 0
    env.rack_occupied[:] = 0
    env.rack_power[:] = 0
    env.step_count = 0
    env.rack_num = 12
    env.cooling_pos = np.array([[2, 2], [2, 40]])
    env.num_cooler = 2


def test_row_action_places_an_aligned_row():
    env = DataCenterEnv(grid_size=50, rack_num=12, placement_mode="row")
    _empty_room(env)
    # Facing dir 2 (+Y exhaust) -> row spans X. Anchor near a corner so the row has room to extend.
    anchor = (5 * 50 + 5)
    env.step(anchor * 4 + 2)
    cells = np.argwhere(env.rack_map == 1)
    assert len(cells) >= 2, "a row must place multiple racks"
    # Every rack in the row shares the facing...
    assert {int(env.rack_dir[x, y]) for x, y in cells} == {2}
    # ...and lies on one line (same Y anchor coordinate), one rack-width apart along X.
    ys = {int(y) for _, y in cells}
    xs = sorted(int(x) for x, _ in cells)
    assert ys == {5}
    assert all(b - a == _RACK_W_CELLS for a, b in zip(xs, xs[1:]))


def test_row_respects_rack_budget():
    env = DataCenterEnv(grid_size=50, rack_num=4, placement_mode="row")
    _empty_room(env)
    env.rack_num = 4
    env.step((5 * 50 + 5) * 4 + 2)
    assert int(env.rack_map.sum()) <= 4


def test_reward_is_deterministic_for_a_fixed_layout():
    env = DataCenterEnv(grid_size=50, rack_num=12, placement_mode="row")
    _empty_room(env)
    env.step((5 * 50 + 5) * 4 + 2)
    r1, _ = env._finish_episode()
    r2, _ = env._finish_episode()
    assert r1 == r2  # bridge uses constant power; solver + metrics are deterministic


def test_obs_facing_channel_encodes_direction():
    env = DataCenterEnv(grid_size=50, rack_num=12, placement_mode="row")
    _empty_room(env)
    assert float(env._get_obs()[5].max()) == 0.0  # nothing placed -> empty facing field
    env.step((5 * 50 + 5) * 4 + 2)  # dir 2 -> (2+1)/4 = 0.75
    assert float(env._get_obs()[5].max()) == pytest.approx(0.75)


def test_mask_is_never_all_zero():
    env = DataCenterEnv(grid_size=50, rack_num=12, placement_mode="row")
    _empty_room(env)
    env.obstacle[:] = 1  # whole room blocked -> no valid placement
    mask = env.action_masks()
    assert mask.any(), "an all-zero mask would make the MaskableCategorical undefined"


def test_single_mode_still_works():
    env = DataCenterEnv(grid_size=50, rack_num=3, placement_mode="single")
    _empty_room(env)
    env.rack_num = 3
    obs, r, done, trunc, info = env.step((10 * 50 + 10) * 4 + 2)
    assert int(env.rack_map.sum()) == 1  # one rack per step in single mode
    assert not done


def test_spatial_policy_builds_and_predicts():
    pytest.importorskip("torch")
    pytest.importorskip("sb3_contrib")
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker

    from engine.rl.policy import SpatialMaskablePolicy

    base = DataCenterEnv(grid_size=50, rack_num=6, placement_mode="row")
    env = ActionMasker(base, lambda e: e.action_masks())
    model = MaskablePPO(SpatialMaskablePolicy, env, n_steps=16, batch_size=8, device="cpu")

    # Spatial action head: a 1x1 conv, NOT a dense 512->10000 layer.
    assert model.policy.action_net.__class__.__name__ == "_SpatialActionHead"
    n_action_params = sum(p.numel() for p in model.policy.action_net.parameters())
    assert n_action_params < 1000  # vs ~5.1M for NatureCNN's dense head

    obs, _ = base.reset(seed=0)
    mask = base.action_masks()
    action, _ = model.predict(obs, action_masks=mask, deterministic=True)
    assert bool(mask[int(action)])  # the prediction respects the mask
