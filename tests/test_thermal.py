"""Tests for engine.thermal.solver — ASHRAE-aware directional heat flow."""

import numpy as np
import pytest

from engine.core.config import (
    AIR_CP_J_KG_K,
    AIR_DENSITY_KG_M3,
    ASHRAE_RECOMMENDED_INLET,
    CFM_TO_M3_S,
    COOLING_AC_VENT,
    DEFAULT_RACK_POWER_KW,
    RACK_BODY,
    RACK_DIMENSIONS,
    RACK_EXHAUST,
    RACK_INTAKE,
    SPACE_EMPTY,
    VOXEL_SIZE,
)
from engine.core.data_types import (
    Coordinate,
    CoolingUnit,
    RackFacing,
    RackPlacement,
    ScanMetadata,
)
from engine.thermal.solver import (
    _exhaust_delta_t,
    _exhaust_plume_voxels,
    compute_thermal_field,
)
from engine.vision.voxelizer import _stamp_rack, _world_to_index


def _make_empty_room(
    w: float = 6.0, d: float = 4.0, h: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return an all-empty int8 grid and its origin at (0,0,0)."""
    gx = round(w / VOXEL_SIZE)
    gy = round(d / VOXEL_SIZE)
    gz = round(h / VOXEL_SIZE)
    grid = np.full((gx, gy, gz), SPACE_EMPTY, dtype=np.int8)
    origin = np.array([0.0, 0.0, 0.0])
    return grid, origin


class TestStampRack:
    """Verify that _stamp_rack places body / intake / exhaust correctly."""

    def test_rack_body_volume(self):
        grid, origin = _make_empty_room()
        rack = RackPlacement(
            position=Coordinate(3.0, 2.0, 0.0),
            facing=RackFacing.PLUS_Y,
        )
        _stamp_rack(grid, rack, origin)

        # Rack of type 42U = 0.6 w × 1.0 d × 2.0 h → 6×10×20 voxels
        body_count = (grid == RACK_BODY).sum()
        intake_count = (grid == RACK_INTAKE).sum()
        exhaust_count = (grid == RACK_EXHAUST).sum()
        total = body_count + intake_count + exhaust_count

        # Total voxels should be close to 6*10*20 = 1200
        assert 1000 <= total <= 1400

    def test_intake_faces_correct_direction(self):
        """Intake slab should be at the max-Y face for PLUS_Y facing."""
        grid, origin = _make_empty_room()
        rack = RackPlacement(
            position=Coordinate(3.0, 2.0, 0.0),
            facing=RackFacing.PLUS_Y,
        )
        _stamp_rack(grid, rack, origin)

        intake_ys = np.where(grid == RACK_INTAKE)[1]
        exhaust_ys = np.where(grid == RACK_EXHAUST)[1]
        assert intake_ys.max() > exhaust_ys.max(), "Intake should be at higher Y"

    def test_exhaust_opposite_to_intake(self):
        """Exhaust should be on the opposite face from intake."""
        grid, origin = _make_empty_room()
        rack = RackPlacement(
            position=Coordinate(3.0, 2.0, 0.0),
            facing=RackFacing.MINUS_X,
        )
        _stamp_rack(grid, rack, origin)

        intake_xs = np.where(grid == RACK_INTAKE)[0]
        exhaust_xs = np.where(grid == RACK_EXHAUST)[0]
        # MINUS_X facing: intake at min-X, exhaust at max-X
        assert intake_xs.min() < exhaust_xs.min()

    def test_four_facings_all_stamp(self):
        """All four facing directions should produce labelled voxels."""
        for facing in RackFacing:
            grid, origin = _make_empty_room()
            rack = RackPlacement(
                position=Coordinate(3.0, 2.0, 0.0),
                facing=facing,
            )
            _stamp_rack(grid, rack, origin)

            assert (grid == RACK_BODY).any(), f"No body for {facing}"
            assert (grid == RACK_INTAKE).any(), f"No intake for {facing}"
            assert (grid == RACK_EXHAUST).any(), f"No exhaust for {facing}"


class TestComputeThermalField:
    """Verify the thermal solver produces physically reasonable results."""

    def _place_rack_and_solve(
        self, facing: RackFacing = RackFacing.PLUS_Y,
        power_kw: float = 5.0,
        airflow_cfm: float = 800.0,
        add_ac: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, RackPlacement]:
        grid, origin = _make_empty_room()
        rack = RackPlacement(
            position=Coordinate(3.0, 2.0, 0.0),
            facing=facing,
            power_kw=power_kw,
            airflow_cfm=airflow_cfm,
        )
        _stamp_rack(grid, rack, origin)
        cooling_units: list[CoolingUnit] = []
        if add_ac:
            vx, vy, vz = _world_to_index(1.0, 1.0, 2.5, origin)
            if 0 <= vx < grid.shape[0] and 0 <= vy < grid.shape[1] and 0 <= vz < grid.shape[2]:
                grid[vx, vy, vz] = COOLING_AC_VENT
            cooling_units.append(
                CoolingUnit(
                    Coordinate(1.0, 1.0, 2.5),
                    capacity_kw=10.0,
                    supply_direction=(0, 0, -1),
                )
            )
        temp = compute_thermal_field(grid, [rack], origin, cooling_units=cooling_units or None)
        return temp, grid, rack

    def test_output_shape_and_dtype(self):
        temp, grid, _ = self._place_rack_and_solve()
        assert temp.shape == grid.shape
        assert temp.dtype == np.float32

    def test_exhaust_hotter_than_intake(self):
        """Exhaust side should be significantly hotter than intake side."""
        temp, grid, _ = self._place_rack_and_solve()

        exhaust_temp = temp[grid == RACK_EXHAUST].mean()
        intake_temp = temp[grid == RACK_INTAKE].mean()
        assert exhaust_temp > intake_temp + 2.0, (
            f"Exhaust ({exhaust_temp:.1f}) not hotter than intake ({intake_temp:.1f})"
        )

    def test_plume_extends_behind_exhaust(self):
        """Air directly behind the exhaust face should be warmer than ambient."""
        temp, grid, _ = self._place_rack_and_solve(RackFacing.PLUS_Y)

        ambient = ASHRAE_RECOMMENDED_INLET
        exhaust_ys = np.where(grid == RACK_EXHAUST)[1]
        if len(exhaust_ys) == 0:
            pytest.skip("No exhaust voxels found")
        min_exh_y = exhaust_ys.min()
        check_y = max(min_exh_y - 3, 0)
        plume_slice = temp[:, check_y:min_exh_y, :]
        air_behind = plume_slice[grid[:, check_y:min_exh_y, :] == SPACE_EMPTY]
        if len(air_behind) > 0:
            assert air_behind.mean() > ambient + 0.5

    def test_hot_air_rises(self):
        """Upper part of room should be warmer than lower part (buoyancy)."""
        temp, grid, _ = self._place_rack_and_solve()
        sz = grid.shape[2]
        mid_z = sz // 2
        air = grid == SPACE_EMPTY

        lower_air = temp[:, :, :mid_z][air[:, :, :mid_z]]
        upper_air = temp[:, :, mid_z:][air[:, :, mid_z:]]

        if len(lower_air) > 0 and len(upper_air) > 0:
            assert upper_air.mean() > lower_air.mean(), (
                f"Upper ({upper_air.mean():.2f}) should be warmer than lower ({lower_air.mean():.2f})"
            )

    def test_heat_spreads_across_room(self):
        """Temperature should vary noticeably across the room — not uniform."""
        temp, grid, _ = self._place_rack_and_solve()
        air_temps = temp[grid == SPACE_EMPTY]
        # Standard deviation should be significant (> 0.5°C across the room)
        assert air_temps.std() > 0.5, (
            f"Air temperature std is only {air_temps.std():.2f}°C — heat not spreading"
        )

    def test_ac_vent_cools(self):
        """Region near AC vent should be cooler than ambient."""
        temp, grid, _ = self._place_rack_and_solve(add_ac=True)
        ac_temp = temp[grid == COOLING_AC_VENT]
        assert ac_temp.mean() < ASHRAE_RECOMMENDED_INLET

    def test_ac_creates_cold_zone(self):
        """Air near the AC vent should also be cooled (influence radius)."""
        temp, grid, _ = self._place_rack_and_solve(add_ac=True)
        vx, vy, vz = _world_to_index(1.0, 1.0, 2.5, np.array([0., 0., 0.]))
        # Check a small cube around the AC vent
        r = 3
        x0, x1 = max(vx - r, 0), min(vx + r + 1, grid.shape[0])
        y0, y1 = max(vy - r, 0), min(vy + r + 1, grid.shape[1])
        z0, z1 = max(vz - r, 0), min(vz + r + 1, grid.shape[2])
        region = temp[x0:x1, y0:y1, z0:z1]
        air_region = region[grid[x0:x1, y0:y1, z0:z1] == SPACE_EMPTY]
        if len(air_region) > 0:
            assert air_region.mean() < ASHRAE_RECOMMENDED_INLET, (
                f"Air near AC vent should be cool, got {air_region.mean():.1f}"
            )

    def test_higher_power_means_hotter(self):
        """A 10 kW rack should produce hotter exhaust than a 5 kW rack."""
        temp_lo, grid_lo, _ = self._place_rack_and_solve(power_kw=5.0)
        temp_hi, grid_hi, _ = self._place_rack_and_solve(power_kw=10.0)
        assert temp_hi.max() > temp_lo.max()

    def test_solid_voxels_at_ambient(self):
        """Rack body and wall voxels should be at ambient temperature."""
        temp, grid, _ = self._place_rack_and_solve()
        body_temps = temp[grid == RACK_BODY]
        assert body_temps.mean() == pytest.approx(ASHRAE_RECOMMENDED_INLET, abs=0.1)

    def test_empty_room_stays_ambient(self):
        """A room with no racks should be at ambient everywhere."""
        grid, origin = _make_empty_room()
        temp = compute_thermal_field(grid, [], origin)
        assert temp.mean() == pytest.approx(ASHRAE_RECOMMENDED_INLET, abs=0.01)

    def test_diagonal_ac_cools_along_direction(self):
        """A wall-mounted AC blowing diagonally should cool voxels along
        that diagonal, not just straight out from the wall."""
        grid, origin = _make_empty_room()
        rack = RackPlacement(
            position=Coordinate(3.0, 2.0, 0.0),
            facing=RackFacing.PLUS_Y,
            power_kw=5.0,
        )
        _stamp_rack(grid, rack, origin)

        # Wall-mounted AC at Y=3.5 wall, blowing -Y and -Z (45° down into room)
        ac_pos = Coordinate(3.0, 3.5, 2.5)
        vx, vy, vz = _world_to_index(ac_pos.x, ac_pos.y, ac_pos.z, origin)
        if 0 <= vx < grid.shape[0] and 0 <= vy < grid.shape[1] and 0 <= vz < grid.shape[2]:
            grid[vx, vy, vz] = COOLING_AC_VENT
        cooling_units = [
            CoolingUnit(ac_pos, capacity_kw=10.0, supply_direction=(0, -1, -1)),
        ]
        temp = compute_thermal_field(grid, [rack], origin, cooling_units=cooling_units)

        # Air diagonally downstream (lower Y, lower Z) should be cooler than ambient
        target_y = max(vy - 10, 0)
        target_z = max(vz - 10, 0)
        downstream = temp[vx, target_y, target_z]
        assert downstream < ASHRAE_RECOMMENDED_INLET, (
            f"Downstream of diagonal AC should be cool, got {downstream:.1f}"
        )


class TestPhysicsBasedDeltaT:
    """Verify the first-principles ΔT = P / (V̇ · ρ · Cₚ) calculations."""

    def test_exhaust_delta_t_formula(self):
        """_exhaust_delta_t should match the analytical formula."""
        power_kw = 5.0
        cfm = 800.0
        expected = (power_kw * 1000.0) / (cfm * CFM_TO_M3_S * AIR_DENSITY_KG_M3 * AIR_CP_J_KG_K)
        assert _exhaust_delta_t(power_kw, cfm) == pytest.approx(expected, rel=1e-6)

    def test_higher_airflow_means_lower_delta_t(self):
        """Same power, twice the airflow → half the ΔT."""
        dt_lo = _exhaust_delta_t(5.0, 400.0)
        dt_hi = _exhaust_delta_t(5.0, 800.0)
        assert dt_lo == pytest.approx(2 * dt_hi, rel=1e-6)

    def test_zero_airflow_returns_zero(self):
        assert _exhaust_delta_t(10.0, 0.0) == 0.0

    def test_plume_depth_increases_with_airflow(self):
        """Higher CFM → higher exhaust velocity → longer plume."""
        short = _exhaust_plume_voxels(400.0, "42U")
        long = _exhaust_plume_voxels(1200.0, "42U")
        assert long > short

    def test_same_power_different_cfm_solver(self):
        """End-to-end: same power rack with less airflow should have a
        hotter exhaust region in the solved thermal field."""
        grid_lo, origin = _make_empty_room()
        grid_hi, _ = _make_empty_room()

        rack_lo_cfm = RackPlacement(
            Coordinate(3.0, 2.0, 0.0), RackFacing.PLUS_Y,
            power_kw=5.0, airflow_cfm=400.0,
        )
        rack_hi_cfm = RackPlacement(
            Coordinate(3.0, 2.0, 0.0), RackFacing.PLUS_Y,
            power_kw=5.0, airflow_cfm=1200.0,
        )

        _stamp_rack(grid_lo, rack_lo_cfm, origin)
        _stamp_rack(grid_hi, rack_hi_cfm, origin)

        temp_lo = compute_thermal_field(grid_lo, [rack_lo_cfm], origin)
        temp_hi = compute_thermal_field(grid_hi, [rack_hi_cfm], origin)

        # Lower airflow → hotter max temperature
        assert temp_lo.max() > temp_hi.max(), (
            f"Low-CFM max ({temp_lo.max():.1f}) should exceed "
            f"high-CFM max ({temp_hi.max():.1f})"
        )

    def test_per_unit_supply_temp(self):
        """AC with a warmer supply temp should cool less than a colder one."""
        grid_cold, origin = _make_empty_room()
        grid_warm, _ = _make_empty_room()

        rack = RackPlacement(Coordinate(3.0, 2.0, 0.0), RackFacing.PLUS_Y, power_kw=5.0)
        _stamp_rack(grid_cold, rack, origin)
        _stamp_rack(grid_warm, rack, origin)

        ac_pos = Coordinate(1.0, 1.0, 2.5)
        vx, vy, vz = _world_to_index(ac_pos.x, ac_pos.y, ac_pos.z, origin)
        for g in (grid_cold, grid_warm):
            if 0 <= vx < g.shape[0] and 0 <= vy < g.shape[1] and 0 <= vz < g.shape[2]:
                g[vx, vy, vz] = COOLING_AC_VENT

        cu_cold = CoolingUnit(ac_pos, capacity_kw=10.0, supply_direction=(0, 0, -1),
                              supply_temp_c=12.0)
        cu_warm = CoolingUnit(ac_pos, capacity_kw=10.0, supply_direction=(0, 0, -1),
                              supply_temp_c=18.0)

        temp_cold = compute_thermal_field(grid_cold, [rack], origin, cooling_units=[cu_cold])
        temp_warm = compute_thermal_field(grid_warm, [rack], origin, cooling_units=[cu_warm])

        # Colder supply → lower mean air temperature
        air = grid_cold == SPACE_EMPTY
        assert temp_cold[air].mean() < temp_warm[air].mean(), (
            f"Cold supply mean ({temp_cold[air].mean():.2f}) should be less than "
            f"warm supply mean ({temp_warm[air].mean():.2f})"
        )
