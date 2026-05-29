import gymnasium as gym
import numpy as np
from gymnasium import spaces
import scipy.ndimage

# ---------------------------------------------------------------------------
# 42U rack footprint constants (must be consistent with thermal_bridge.CELL_M)
#
# CELL_M = 0.2 m/cell  (thermal_bridge.py)
# 42U rack: 0.60 m wide × 1.00 m deep  (engine/core/config.py RACK_DIMENSIONS)
#
#   _RACK_W_CELLS = round(0.60 / 0.20) = 3  cells wide
#   _RACK_D_CELLS = round(1.00 / 0.20) = 5  cells deep
#   _RACK_HW      = 3 // 2             = 1  (half-width for centering)
#
# Convention: the agent selects the INTAKE FACE CENTER cell (gx, gy).
# The rack body extends D cells from (gx, gy) toward the exhaust direction,
# and spans W cells centered on the lateral coordinate.
# ---------------------------------------------------------------------------
_RACK_W_CELLS: int = 3  # 0.60 m / 0.20 m
_RACK_D_CELLS: int = 5  # 1.00 m / 0.20 m
_RACK_HW: int = _RACK_W_CELLS // 2  # = 1

# Exhaust face offset from the intake reference cell, by direction index.
# dir 0: +X exhaust → rack extends toward +X; exhaust at (gx + D-1, gy)
# dir 1: -X exhaust → rack extends toward -X; exhaust at (gx - D+1, gy)
# dir 2: +Y exhaust → exhaust at (gx, gy + D-1)
# dir 3: -Y exhaust → exhaust at (gx, gy - D+1)
_DIR_EXHAUST_OFFSET: list[tuple[int, int]] = [
    (_RACK_D_CELLS - 1, 0),
    (-(_RACK_D_CELLS - 1), 0),
    (0, _RACK_D_CELLS - 1),
    (0, -(_RACK_D_CELLS - 1)),
]


# ---------------------------------------------------------------------------
# Shared ASHRAE reward formula (used by both fast-2D and engine-3D paths)
# ---------------------------------------------------------------------------


def _ashrae_reward_from_metrics(metrics) -> float:
    """Compute the ASHRAE TC 9.9 reward from a MetricsResult object.

    Returns a scalar reward:
        - ~ +3.5 for an ideal layout (all racks compliant, good airflow)
        - ~  0   for a mediocre layout
        - ~ −5   for a layout with widespread allowable violations
    """
    intakes = np.array([r.intake_temp for r in metrics.racks])
    exhausts = np.array([r.exhaust_temp for r in metrics.racks])
    dts = exhausts - intakes

    s_rec = float(np.mean((intakes >= 18.0) & (intakes <= 27.0)))
    s_allow = float(np.mean((intakes >= 15.0) & (intakes <= 35.0)))

    p_allow = float(
        np.mean(np.maximum(intakes - 35.0, 0.0) + np.maximum(15.0 - intakes, 0.0))
        / 10.0
    )

    p_dt = float(
        np.mean(np.maximum(6.0 - dts, 0.0) / 6.0 + np.maximum(dts - 20.0, 0.0) / 10.0)
    )

    vp = np.array(metrics.room.vertical_profile)
    rci_hi_viol = float(np.mean(np.maximum(vp - 27.0, 0.0)) / 8.0)
    rci_lo_viol = float(np.mean(np.maximum(15.0 - vp, 0.0)) / 5.0)
    s_rci = 1.0 - 0.5 * (rci_hi_viol + rci_lo_viol)

    spread = float(np.percentile(vp, 95) - np.percentile(vp, 5))
    p_strat = max(0.0, (spread - 6.0) / 8.0)

    base = 2.0 * s_rec + 1.5 * s_rci - 3.0 * p_allow - 0.8 * p_dt - 0.5 * p_strat
    gate = -2.0 * (1.0 - s_allow)
    return float(base + gate)


class DataCenterEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, grid_size=50, rack_num=10, num_cooler=3, ceiling_m: float = 3.0):
        super().__init__()

        self.grid_size = grid_size
        self.rack_num = rack_num
        self.num_cooler = num_cooler
        self.ceiling_m = ceiling_m

        # rack_map[gx, gy] = 1 at the INTAKE FACE CENTER of each placed rack.
        # Used by ThermalBridge to locate racks and read their facing direction.
        self.rack_map = np.zeros((grid_size, grid_size))
        self.rack_dir = np.zeros((grid_size, grid_size))

        # rack_occupied marks ALL cells covered by any rack's physical footprint
        # (3 × 5 cells for a 42U). Used for collision detection and the observation.
        self.rack_occupied = np.zeros((grid_size, grid_size))

        self.temp = np.zeros((grid_size, grid_size))
        self.obstacle = np.zeros((grid_size, grid_size))

        self.cooling_pos = None

        self.cp = 1005.0
        self.rho = 1.2
        self.airflow = 1.0
        self.temp_decay = 0.95

        self.diff_sigma = 0.3
        self.cool_decay = 6.0
        self.cool_strength = 0.5
        self.cooling_power = 0.0

        # ── Thermal engine bridge ─────────────────────────────────────────
        # The authoritative 3-D thermal solver is invoked once per episode
        # (when done=True) to compute the final ASHRAE reward.
        # ceiling_m should match the scanned room's actual height; for training
        # with randomly generated layouts the default of 3.0 m is used.
        from engine.rl.thermal_bridge import ThermalBridge

        self._bridge = ThermalBridge(grid_size=grid_size, ceiling_m=ceiling_m)

        self.action_space = spaces.Discrete(grid_size * grid_size * 4)

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(8, grid_size, grid_size), dtype=np.float32
        )

        X, Y = np.meshgrid(np.arange(grid_size), np.arange(grid_size), indexing="ij")
        self.X = X.astype(np.float32)
        self.Y = Y.astype(np.float32)

        self.cached_flow = None
        self.flow_dirty = True

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.rack_map[:] = 0
        self.rack_dir[:] = 0
        self.rack_occupied[:] = 0
        self.temp[:] = 0
        self.rack_power = np.zeros((self.grid_size, self.grid_size))
        self.cached_flow = None
        self.flow_dirty = True

        if options is not None:
            self.obstacle = options.get("obstacle", self._generate_layout())
            self.cooling_pos = np.array(
                options.get("cooling_pos", self._generate_cooling(self.obstacle))
            )
            self.num_cooler = len(self.cooling_pos)
            self.rack_num = options.get("rack_num", self.rack_num)
        else:
            self.obstacle = self._generate_layout()
            free_mask = self.obstacle == 0
            free_cells = np.sum(free_mask)
            max_racks = max(
                2,
                free_cells
                // (
                    (_RACK_W_CELLS + 1) * (_RACK_D_CELLS + 1)
                ),  # for spacing between racks
            )  # rough upper bound
            self.rack_num = self.np_random.integers(1, max_racks)
            self.num_cooler = self.np_random.integers(2, 10)
            self.cooling_pos = np.array(self._generate_cooling(self.obstacle))

        self.step_count = 0

        return self._get_obs(), {}

    def step(self, action):
        pos = action // 4
        dir = action % 4

        x = pos // self.grid_size
        y = pos % self.grid_size

        # Validate the full 42U footprint before placing.
        footprint = self._rack_footprint(x, y, dir)
        for fx, fy in footprint:
            if not (0 <= fx < self.grid_size and 0 <= fy < self.grid_size):
                return self._get_obs(), -10.0, True, False, {}
            if self.rack_occupied[fx, fy] == 1 or self.obstacle[fx, fy] == 1:
                return self._get_obs(), -10.0, True, False, {}
            self.rack_occupied[fx, fy] = 1

        # Place the rack: record intake reference cell and mark full footprint.
        self.rack_map[x, y] = 1
        self.rack_dir[x, y] = dir
        self.rack_power[x, y] = np.random.uniform(0.8, 2.5)

        self.step_count += 1

        done = self.step_count >= self.rack_num

        self.flow_dirty = True

        if done:
            # Episode end: run the 3-D thermal engine once for the final reward.
            metrics = self._bridge.solve_metrics(
                self.rack_map,
                self.rack_dir,
                self.obstacle,
                self.cooling_pos,
            )
            reward = (
                _ashrae_reward_from_metrics(metrics) if metrics is not None else -10.0
            )
            info = {"metrics": metrics}
        else:
            reward = self._position_shaping_reward()
            info = {}

        return self._get_obs(), reward, done, False, info

    def _rack_footprint(self, gx: int, gy: int, dir_idx: int) -> list[tuple[int, int]]:
        """All (x, y) RL cells occupied by a 42U rack whose intake face is at (gx, gy).

        The rack extends _RACK_D_CELLS deep from the intake along the exhaust axis
        and spans _RACK_W_CELLS centered on the lateral coordinate.
        """
        cells: list[tuple[int, int]] = []
        if dir_idx in (0, 1):  # depth along X axis
            x_range = (
                range(gx, gx + _RACK_D_CELLS)
                if dir_idx == 0  # +X exhaust: body toward +X
                else range(gx - _RACK_D_CELLS + 1, gx + 1)  # -X exhaust: body toward -X
            )
            y_range = range(gy - _RACK_HW, gy - _RACK_HW + _RACK_W_CELLS)
        else:  # depth along Y axis
            x_range = range(gx - _RACK_HW, gx - _RACK_HW + _RACK_W_CELLS)
            y_range = (
                range(gy, gy + _RACK_D_CELLS)
                if dir_idx == 2
                else range(gy - _RACK_D_CELLS + 1, gy + 1)
            )
        for xi in x_range:
            for yi in y_range:
                cells.append((xi, yi))
        return cells

    def _get_obs(self):
        cooler_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        for cx, cy in self.cooling_pos:
            cooler_map[cx, cy] = 1.0

        dist_map = self._cooling_dist_map()

        density_map = scipy.ndimage.gaussian_filter(
            self.rack_occupied.astype(np.float32), sigma=1.5
        )

        remaining_ratio = (self.rack_num - self.step_count) / max(1, self.rack_num)
        rack_num_ch = np.full(
            (self.grid_size, self.grid_size), remaining_ratio, dtype=np.float32
        )
        cooling_num_ch = np.full(
            (self.grid_size, self.grid_size), (self.num_cooler / 10.0), dtype=np.float32
        )

        return np.stack(
            [
                self.rack_occupied.astype(np.float32),
                self.obstacle.astype(np.float32),
                density_map,
                cooler_map,
                dist_map,
                np.zeros_like(density_map),
                rack_num_ch,
                cooling_num_ch,
            ],
            axis=0,
        ).astype(np.float32)

    """
    def _update_temp(self):
        temp = self.temp.copy()

        self.rack_power = np.clip(self.rack_power, 0.5, 3.0)

        temp = temp * self.temp_decay + self._compute_exhaust()

        flow = self._compute_airflow()

        for _ in range(3):
            temp = self._advect(temp, flow)

        temp = scipy.ndimage.gaussian_filter(temp, sigma=self.diff_sigma)

        temp = self._apply_cooling(temp)

        temp[self.obstacle == 1] = 0

        self.temp = np.clip(temp, 0, 5)

    def _compute_exhaust(self):
        """ """Inject heat from each rack's exhaust face, not from the intake reference cell.""" """
        temp = np.zeros_like(self.temp)

        dirs = np.array([(1, 0), (-1, 0), (0, 1), (0, -1)])
        racks = np.argwhere(self.rack_map == 1)

        for x, y in racks:
            d_idx = int(self.rack_dir[x, y])
            d = dirs[d_idx]

            # Exhaust face center — D-1 cells from the intake reference along flow axis.
            ox, oy = _DIR_EXHAUST_OFFSET[d_idx]
            ex = int(np.clip(x + ox, 0, self.grid_size - 1))
            ey = int(np.clip(y + oy, 0, self.grid_size - 1))

            base_power = self.rack_power[x, y]

            intake_x = np.clip(x - d[0], 0, self.grid_size - 1)
            intake_y = np.clip(y - d[1], 0, self.grid_size - 1)
            intake_temp = self.temp[intake_x, intake_y]

            load = np.clip(base_power, 0.5, 3.0)
            power = base_power * (1 + 0.5 * intake_temp + 0.2 * load)

            delta_T = power / (self.airflow * self.cp + 1e-6)

            rx = self.X - ex
            ry = self.Y - ey

            proj = rx * d[0] + ry * d[1]
            perp = np.abs(rx * d[1] - ry * d[0])

            jet = np.zeros_like(self.temp)
            valid = proj > 0
            jet[valid] = (
                delta_T * np.exp(-(perp[valid] ** 2) / 4.0) / (proj[valid] + 1.0 + 1e-6)
            )

            temp += jet

        return temp

    def _compute_airflow(self):
        if not self.flow_dirty:
            return self.cached_flow

        flow = np.zeros((self.grid_size, self.grid_size, 2))

        for cx, cy in self.cooling_pos:
            dx = self.X - cx
            dy = self.Y - cy
            dist = np.sqrt(dx**2 + dy**2) + 1e-6
            flow[:, :, 0] -= dx / dist
            flow[:, :, 1] -= dy / dist

        dirs = np.array([(1, 0), (-1, 0), (0, 1), (0, -1)])
        rack_indices = np.argwhere(self.rack_map == 1)

        for x, y in rack_indices:
            d_idx = int(self.rack_dir[x, y])
            d = dirs[d_idx]
            # Airflow injection at the exhaust face, not the intake reference.
            ox, oy = _DIR_EXHAUST_OFFSET[d_idx]
            ex = int(np.clip(x + ox, 0, self.grid_size - 1))
            ey = int(np.clip(y + oy, 0, self.grid_size - 1))
            flow[ex, ey, 0] += d[0] * 2.0
            flow[ex, ey, 1] += d[1] * 2.0

        flow[self.obstacle == 1] = 0

        norm = np.linalg.norm(flow, axis=-1, keepdims=True)
        flow = flow / (norm + 1e-6)
        flow = flow * np.tanh(norm)
        flow = flow + 0.5

        self.cached_flow = flow
        self.flow_dirty = False

        return flow

    def _advect(self, temp, flow):
        x = self.X - flow[:, :, 0]
        y = self.Y - flow[:, :, 1]

        x0 = np.floor(x).astype(int)
        x1 = x0 + 1
        y0 = np.floor(y).astype(int)
        y1 = y0 + 1

        x0 = np.clip(x0, 0, self.grid_size - 1)
        x1 = np.clip(x1, 0, self.grid_size - 1)
        y0 = np.clip(y0, 0, self.grid_size - 1)
        y1 = np.clip(y1, 0, self.grid_size - 1)

        wx = x - x0
        wy = y - y0

        return (
            (1 - wx) * (1 - wy) * temp[x0, y0]
            + wx * (1 - wy) * temp[x1, y0]
            + (1 - wx) * wy * temp[x0, y1]
            + wx * wy * temp[x1, y1]
        )

    def _apply_cooling(self, temp):
        target_temp = 0.3
        error = np.mean(temp) - target_temp

        self.cooling_power = np.clip(self.cooling_power + 0.1 * error, 0.0, 2.0)

        for cx, cy in self.cooling_pos:
            d = np.sqrt((self.X - cx) ** 2 + (self.Y - cy) ** 2)
            effect = np.exp(-d / self.cool_decay)

            local_cooling = np.maximum(temp - target_temp, 0)

            flow = self._compute_airflow()
            flow_strength = np.linalg.norm(flow, axis=-1)

            temp -= (
                self.cooling_power * effect * local_cooling * (1 + 0.5 * flow_strength)
            )

        return temp
    """

    def _position_shaping_reward(self) -> float:
        """Lightweight per-step shaping reward (no thermal solve).

        Compares the newly placed rack's exhaust direction against all
        previously placed racks within aisle range (_RACK_D_CELLS + 2).
        Rewards hot-aisle alignment (exhausts same direction) and penalises
        intake-to-intake clashes, weighted by inverse reference-cell distance.

        The total shaping reward is intentionally small (|r| < 0.5) so it
        does not overwhelm the terminal ASHRAE reward.
        """
        racks = np.argwhere(self.rack_map == 1)
        if len(racks) == 0:
            return 0.0

        x, y = racks[-1]  # the rack just placed
        dirs = np.array([(1, 0), (-1, 0), (0, 1), (0, -1)])
        my_exhaust = dirs[int(self.rack_dir[x, y])]

        r_aisle = 0.0
        aisle_range = (
            _RACK_D_CELLS + 2
        )  # reference cells within which alignment matters
        for rx, ry in racks[:-1]:
            nb_exhaust = dirs[int(self.rack_dir[rx, ry])]
            dot = float(my_exhaust @ nb_exhaust)
            dist = max(1, abs(rx - x) + abs(ry - y))
            if dist <= aisle_range:
                r_aisle += 0.05 * dot / dist

        return float(r_aisle + 0.05)

    def _cooling_dist_map(self):
        dist = np.sqrt(
            (self.X[:, :, None] - self.cooling_pos[:, 0]) ** 2
            + (self.Y[:, :, None] - self.cooling_pos[:, 1]) ** 2
        )
        d = np.min(dist, axis=-1)
        return d / (np.max(d) + 1e-6)

    def _generate_layout(self):
        grid = np.zeros((self.grid_size, self.grid_size))

        margin_top = np.random.randint(0, 20)
        margin_bottom = np.random.randint(0, 20)
        margin_left = np.random.randint(0, 20)
        margin_right = np.random.randint(0, 20)

        if margin_top > 0:
            grid[:margin_top, :] = 1
        if margin_bottom > 0:
            grid[-margin_bottom:, :] = 1
        if margin_left > 0:
            grid[:, :margin_left] = 1
        if margin_right > 0:
            grid[:, -margin_right:] = 1

        return grid

    def _generate_cooling(self, obstacle):
        pos = []
        free = np.argwhere(obstacle == 0)
        idxs = np.random.choice(len(free), size=self.num_cooler, replace=False)
        for idx in idxs:
            pos.append(free[idx])
        return np.array(pos)

    def get_action_mask(self):
        """Mask any action whose 42U footprint overlaps an obstacle or placed rack."""
        mask = np.ones((self.grid_size, self.grid_size, 4), dtype=np.float32)

        blocked = (self.rack_occupied == 1) | (self.obstacle == 1)

        for d in range(4):
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    footprint = self._rack_footprint(i, j, d)
                    for fx, fy in footprint:
                        if (
                            not (0 <= fx < self.grid_size and 0 <= fy < self.grid_size)
                            or blocked[fx, fy]
                        ):
                            mask[i, j, d] = 0.0
                            break

        return mask.flatten()

    def action_masks(self):
        return self.get_action_mask()


def make_env_from_scan(
    scan_grid_shape: tuple[int, int, int],
    rack_num: int = 10,
    num_cooler: int = 3,
) -> DataCenterEnv:
    """Create a DataCenterEnv whose thermal bridge matches a scanned room.

    The scan's voxel-grid Z dimension encodes the actual room ceiling height.
    Passing it here ensures the 3-D thermal reconstruction at episode end uses
    the correct height rather than the hardcoded training default.

    Parameters
    ----------
    scan_grid_shape : (nx, ny, nz)
        ``PipelineResult.grid.shape`` or the ``shape`` field from
        ``ProcessScanResponse``.  All three values are in voxels at
        VOXEL_SIZE = 0.1 m.
    rack_num : int
        Number of racks the agent must place per episode.
    num_cooler : int
        Number of cooling units in the environment.
    """
    from engine.core.config import VOXEL_SIZE, MAX_ROOM_DIMENSIONS

    _nx, _ny, nz = scan_grid_shape
    ceiling_m = float(nz) * VOXEL_SIZE
    ceiling_m = min(ceiling_m, MAX_ROOM_DIMENSIONS[2])

    return DataCenterEnv(
        grid_size=50,
        rack_num=rack_num,
        num_cooler=num_cooler,
        ceiling_m=ceiling_m,
    )
