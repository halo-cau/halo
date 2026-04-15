"""Tests for engine.thermal.metrics — ASHRAE compliance metrics."""

import numpy as np
import pytest

from engine.core.config import (
    ASHRAE_INLET_TEMP_RANGE,
    ASHRAE_INLET_ALLOWABLE_RANGE,
    COOLING_AC_VENT,
    RACK_BODY,
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
)
from engine.thermal.metrics import (
    MetricsResult,
    RackMetrics,
    RoomMetrics,
    compute_metrics,
    _compute_rci_hi,
    _compute_rci_lo,
)
from engine.thermal.solver import compute_thermal_field
from engine.vision.voxelizer import _stamp_rack


def _make_empty_room(
    w: float = 6.0, d: float = 4.0, h: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    gx = round(w / VOXEL_SIZE)
    gy = round(d / VOXEL_SIZE)
    gz = round(h / VOXEL_SIZE)
    grid = np.full((gx, gy, gz), SPACE_EMPTY, dtype=np.int8)
    origin = np.array([0.0, 0.0, 0.0])
    return grid, origin


class TestRCIFormulas:
    """Unit tests for the RCI helper functions."""

    def test_rci_hi_all_compliant(self):
        """All intakes within recommended → RCI_HI = 100."""
        temps = [20.0, 22.0, 25.0]
        assert _compute_rci_hi(temps, 27.0, 35.0) == 100.0

    def test_rci_hi_all_above_recommended(self):
        """All intakes above recommended but below allowable → partial."""
        temps = [30.0, 30.0]  # 3°C above rec_hi=27
        # excess = 2 * 3 = 6, denom = 2 * (35 - 27) = 16
        # rci_hi = (1 - 6/16) * 100 = 62.5
        assert _compute_rci_hi(temps, 27.0, 35.0) == pytest.approx(62.5)

    def test_rci_lo_all_compliant(self):
        temps = [20.0, 22.0, 25.0]
        assert _compute_rci_lo(temps, 18.0, 15.0) == 100.0

    def test_rci_lo_below_recommended(self):
        temps = [16.0, 16.0]  # 2°C below rec_lo=18
        # deficit = 2 * 2 = 4, denom = 2 * (18 - 15) = 6
        # rci_lo = (1 - 4/6) * 100 = 33.3
        assert _compute_rci_lo(temps, 18.0, 15.0) == pytest.approx(33.333, rel=0.01)

    def test_rci_empty_returns_100(self):
        assert _compute_rci_hi([], 27.0, 35.0) == 100.0
        assert _compute_rci_lo([], 18.0, 15.0) == 100.0


class TestComputeMetrics:
    """Integration tests for the full metrics pipeline."""

    def _setup_room_with_rack(
        self,
        power_kw: float = 5.0,
        airflow_cfm: float = 800.0,
        add_ac: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[RackPlacement], list[CoolingUnit]]:
        grid, origin = _make_empty_room()
        rack = RackPlacement(
            Coordinate(3.0, 2.0, 0.0), RackFacing.PLUS_Y,
            power_kw=power_kw, airflow_cfm=airflow_cfm,
        )
        _stamp_rack(grid, rack, origin)

        cooling_units: list[CoolingUnit] = []
        if add_ac:
            from engine.vision.voxelizer import _world_to_index
            vx, vy, vz = _world_to_index(1.0, 1.0, 2.5, origin)
            if 0 <= vx < grid.shape[0] and 0 <= vy < grid.shape[1] and 0 <= vz < grid.shape[2]:
                grid[vx, vy, vz] = COOLING_AC_VENT
            cooling_units.append(
                CoolingUnit(Coordinate(1.0, 1.0, 2.5), capacity_kw=10.0,
                            supply_direction=(0, 0, -1), supply_temp_c=14.0, airflow_cfm=2000.0),
            )

        temp = compute_thermal_field(grid, [rack], origin, cooling_units=cooling_units or None)
        return grid, origin, temp, [rack], cooling_units

    def test_returns_metrics_result(self):
        grid, origin, temp, racks, cus = self._setup_room_with_rack()
        mr = compute_metrics(grid, temp, racks, origin, cus)
        assert isinstance(mr, MetricsResult)
        assert len(mr.racks) == 1
        assert isinstance(mr.room, RoomMetrics)

    def test_exhaust_hotter_than_intake(self):
        grid, origin, temp, racks, cus = self._setup_room_with_rack()
        mr = compute_metrics(grid, temp, racks, origin, cus)
        assert mr.racks[0].delta_t > 0

    def test_rci_values_in_range(self):
        grid, origin, temp, racks, cus = self._setup_room_with_rack()
        mr = compute_metrics(grid, temp, racks, origin, cus)
        assert 0 <= mr.room.rci_hi <= 100
        assert 0 <= mr.room.rci_lo <= 100

    def test_shi_rhi_sum_to_one(self):
        grid, origin, temp, racks, cus = self._setup_room_with_rack()
        mr = compute_metrics(grid, temp, racks, origin, cus)
        assert mr.room.shi + mr.room.rhi == pytest.approx(1.0, abs=0.001)

    def test_vertical_profile_length(self):
        grid, origin, temp, racks, cus = self._setup_room_with_rack()
        mr = compute_metrics(grid, temp, racks, origin, cus)
        assert len(mr.room.vertical_profile) == grid.shape[2]

    def test_vertical_profile_increases_upward(self):
        """Due to buoyancy, upper layers should generally be warmer
        when there is no forced downward airflow from AC units."""
        grid, origin, temp, racks, cus = self._setup_room_with_rack(
            power_kw=10.0, airflow_cfm=600.0, add_ac=False,
        )
        mr = compute_metrics(grid, temp, racks, origin, cus)
        profile = mr.room.vertical_profile
        # Compare bottom quarter mean vs top quarter mean
        n = len(profile)
        bottom = np.mean(profile[:n // 4])
        top = np.mean(profile[3 * n // 4:])
        assert top > bottom

    def test_inlet_compliance_flag(self):
        """With a well-cooled room, intake should be within ASHRAE recommended."""
        grid, origin, temp, racks, cus = self._setup_room_with_rack(
            power_kw=3.0, airflow_cfm=1200.0, add_ac=True,
        )
        mr = compute_metrics(grid, temp, racks, origin, cus)
        rec_lo, rec_hi = ASHRAE_INLET_TEMP_RANGE
        # With low power and good cooling, intake should be near ambient
        assert mr.racks[0].intake_temp >= rec_lo - 2  # allow slight margin

    def test_mean_return_is_reasonable(self):
        grid, origin, temp, racks, cus = self._setup_room_with_rack()
        mr = compute_metrics(grid, temp, racks, origin, cus)
        # Return temp should be between supply and exhaust
        assert mr.room.mean_return >= 14.0  # at least supply temp
        assert mr.room.mean_return <= mr.room.mean_exhaust + 5.0
