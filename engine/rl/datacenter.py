import gymnasium as gym
import numpy as np
from gymnasium import spaces
import scipy.ndimage


class DataCenterEnv(gym.Env):
    def __init__(self, grid_size=50, rack_num=10, num_cooler=3):
        super().__init__()

        self.grid_size = grid_size
        self.rack_num = rack_num
        self.num_cooler = num_cooler

        self.rack_map = np.zeros((grid_size, grid_size))
        self.rack_dir = np.zeros((grid_size, grid_size))
        self.temp = np.zeros((grid_size, grid_size))
        self.obstacle = np.zeros((grid_size, grid_size))

        self.cooling_pos = None

        self.cp = 1005.0
        self.rho = 1.2
        self.airflow = 1.0
        self.temp_decay = 0.95

        self.diff_sigma = 0.3
        self.cool_decay = 6.0

        # ── Thermal engine bridge ─────────────────────────────────────────
        # The authoritative 3-D thermal solver is invoked once per episode
        # (when done=True) to compute the final ASHRAE reward.
        # The 2-D field (self.temp) is still maintained for the observation.
        from engine.rl.thermal_bridge import ThermalBridge

        self._bridge = ThermalBridge(grid_size=grid_size)

        self.action_space = spaces.Discrete(grid_size * grid_size * 4)

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(8, grid_size, grid_size), dtype=np.float32
        )

        X, Y = np.meshgrid(np.arange(grid_size), np.arange(grid_size), indexing="ij")
        self.X = X.astype(np.float32)
        self.Y = Y.astype(np.float32)

        self.cached_flow = None
        self.flow_dirty = True

        self.solver_step = 5
        self.solver_reward_sacle = 0.2

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.rack_map[:] = 0
        self.rack_dir[:] = 0
        self.temp[:] = 0
        self.rack_power = np.zeros((self.grid_size, self.grid_size))
        self.cached_flow = None
        self.flow_dirty = True

        if options is not None:
            self.obstacle = options.get("obstacle", self._generate_layout())
            self.cooling_pos = options.get(
                "cooling_pos", self._generate_cooling(self.obstacle)
            )
            self.num_cooler = len(self.cooling_pos)
            self.rack_num = options.get("rack_num", self.rack_num)
        else:
            self.obstacle = self._generate_layout()
            free_mask = self.obstacle == 0
            free_cells = np.sum(free_mask)
            self.rack_num = self.np_random.integers(5, free_cells * 0.5)
            self.num_cooler = self.np_random.integers(2, 10)
            self.cooling_pos = self._generate_cooling(self.obstacle)

        self.step_count = 0

        return self._get_obs(), {}

    def step(self, action):
        pos = action // 4
        dir = action % 4

        x = pos // self.grid_size
        y = pos % self.grid_size

        if self.rack_map[x, y] == 1 or self.obstacle[x, y] == 1:
            return self._get_obs(), -100.0, False, False, {}
        self.rack_map[x, y] = 1
        self.rack_dir[x, y] = dir
        self.rack_power[x, y] = np.random.uniform(0.8, 2.5)

        self.step_count += 1

        self._update_temp()
        done = self.step_count >= self.rack_num

        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        flow = self._compute_airflow()

        flow_x = flow[:, :, 0]
        flow_y = flow[:, :, 1]

        dist = self._cooling_dist_map()

        remaining = (self.rack_num - self.step_count) / self.rack_num
        rack_num = np.ones((self.grid_size, self.grid_size)) * remaining

        cooling_num = np.ones((self.grid_size, self.grid_size)) * (
            self.num_cooler / 10.0
        )

        return np.stack(
            [
                self.rack_map,
                self.obstacle,
                self.temp,
                flow_x,
                flow_y,
                dist,
                rack_num,
                cooling_num,
            ],
            axis=0,
        ).astype(np.float32)

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
        temp = np.zeros_like(self.temp)

        dirs = np.array([(1, 0), (-1, 0), (0, 1), (0, -1)])
        racks = np.argwhere(self.rack_map == 1)

        for x, y in racks:
            d = dirs[int(self.rack_dir[x, y])]

            base_power = self.rack_power[x, y]

            intake_x = np.clip(x - d[0], 0, self.grid_size - 1)
            intake_y = np.clip(y - d[1], 0, self.grid_size - 1)
            intake_temp = self.temp[intake_x, intake_y]

            load = np.clip(base_power, 0.5, 3.0)
            power = base_power * (1 + 0.5 * intake_temp + 0.2 * load)

            delta_T = power / (self.airflow * self.cp + 1e-6)

            rx = self.X - x
            ry = self.Y - y

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
        racks = np.argwhere(self.rack_map == 1)

        for x, y in racks:
            d = dirs[int(self.rack_dir[x, y])]
            flow[x, y, 0] += d[0] * 2.0
            flow[x, y, 1] += d[1] * 2.0

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

    def _position_shaping_reward(self) -> float:
        """Lightweight per-step shaping reward (no thermal solve).

        Signals two layout quality proxies that are cheap to compute:

        1. Hot-/cold-aisle alignment — if any of the four orthogonal
           neighbours already has a rack, reward racks whose exhaust faces
           away from that neighbour's exhaust (back-to-back hot-aisle) and
           penalise intake-to-intake pairings (cold-aisle clash).

        2. Placement bonus — a fixed small positive for each valid placement
           to keep the reward from being dominated by violations alone.

        The total shaping reward is intentionally small (|r| < 0.5) so it
        does not overwhelm the terminal ASHRAE reward.
        """
        racks = np.argwhere(self.rack_map == 1)
        if len(racks) == 0:
            return 0.0

        x, y = racks[-1]  # the rack just placed
        dirs = np.array([(1, 0), (-1, 0), (0, 1), (0, -1)])
        my_dir = int(self.rack_dir[x, y])
        my_exhaust = dirs[my_dir]  # unit vector: direction exhaust blows

        # ── 1. Hot-/cold-aisle alignment ───────────────────────────────────────
        r_aisle = 0.0
        neighbour_offsets = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        for ox, oy in neighbour_offsets:
            nx, ny = x + ox, y + oy
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                if self.rack_map[nx, ny] == 1:
                    nb_exhaust = dirs[int(self.rack_dir[nx, ny])]
                    dot = float(my_exhaust @ nb_exhaust)
                    # exhausts pointing same direction (+1): hot-aisle alignment
                    # exhausts pointing toward each other (dot≈-1): intake clash
                    r_aisle += 0.05 * dot

        # ── 2. Placement bonus ────────────────────────────────────────────
        r_place = 0.05

        return float(r_aisle + r_place)

    def _cooling_dist_map(self):
        dist = np.sqrt(
            (self.X[:, :, None] - self.cooling_pos[:, 0]) ** 2
            + (self.Y[:, :, None] - self.cooling_pos[:, 1]) ** 2
        )

        d = np.min(dist, axis=-1)

        return d / (np.max(d) + 1e-6)

    def _generate_layout(self):
        grid = np.zeros((self.grid_size, self.grid_size))

        margin_top = np.random.randint(0, 15)
        margin_bottom = np.random.randint(0, 15)
        margin_left = np.random.randint(0, 15)
        margin_right = np.random.randint(0, 15)

        if margin_top > 0:
            grid[:margin_top, :] = 1

        if margin_bottom > 0:
            grid[-margin_bottom:, :] = 1

        if margin_left > 0:
            grid[:, :margin_left] = 1

        if margin_right > 0:
            grid[:, -margin_right:] = 1

        grid_x_size = self.grid_size - margin_top - margin_bottom
        grid_y_size = self.grid_size - margin_left - margin_right

        max_spacing = max(6, min(grid_x_size, grid_y_size) // 2)

        pillar = np.random.randint(5, max_spacing)

        for i in range(margin_top + pillar, self.grid_size - margin_bottom, pillar):
            for j in range(margin_left + pillar, self.grid_size - margin_right, pillar):
                if np.random.rand() < 0.8:
                    grid[i, j] = 1

        return grid

    def _generate_cooling(self, obstacle):
        pos = []
        free = np.argwhere(obstacle == 0)

        for _ in range(self.num_cooler):
            idx = np.random.choice(len(free))
            pos.append(free[idx])

        return np.array(pos)

    def get_action_mask(self):
        mask = np.ones(self.grid_size * self.grid_size * 4, dtype=np.float32)

        invalid = (self.rack_map == 1) | (self.obstacle == 1)
        invalid = invalid.flatten()

        for d in range(4):
            mask[d::4][invalid] = 0
        return mask

    def action_masks(self):
        return self.get_action_mask()

    def build_3D_scene(self):
        H = 5

        grid3D = np.zeros((self.grid_size, self.grid_size, H), dtype=np.int8)

        grid3D[:] = SPACE_EMPTY

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if self.obstacle[x, y] == 1:
                    grid3D[x, y, :] = OBSTACLE_WALL

        racks = []
        for x, y in np.argwhere(self.rack_map == 1):
            direction = int(self.rack_dir[x, y])

            facing = {
                0: RackFacing.PLUS_X,
                1: RackFacing.MINUS_X,
                2: RackFacing.PLUS_Y,
                3: RackFacing.MINUS_Y,
            }[direction]
        racks.append(
            RackPlacement(
                position=Coordinate(
                    x=float(x) * VOXEL_SIZE, y=float(y) * VOXEL_SIZE, z=0.0
                ),
                facing=facing,
                power_kw=5.0,
                airflow_cfm=800.0,
            )
        )

        cooling_units = []
        for cx, cy in self.cooling_pos:
            cooling_units.append(
                CoolingUnit(
                    position=Coordinate(
                        x=float(cx) * VOXEL_SIZE, y=float(cy) * VOXEL_SIZE, z=2.5
                    ),
                    supply_direction=(0, 0, -1),
                    supply_temp_c=14.0,
                    airflow_cfm=2000.0,
                )
            )

        origin = np.array([0.0, 0.0, 0.0])

        return grid3D, racks, cooling_units, origin

    def compute_solver_reward(self):
        grid3D, racks, cooling_units, origin = self.build_3D_scene()

        temp = compute_thermal_field(
            grid=grid3D, racks=racks, cooling_units=cooling_units, origin=origin
        )

        self.temp = np.mean(temp, axis=2)

        metrics = compute_metrics(
            grid=grid3D,
            temp=temp,
            racks=racks,
            origin=origin,
            cooling_units=cooling_units,
        )

        reward = (
            metrics.room.rci_hi * 0.1
            + metrics.room.rci_lo * 0.1
            - metrics.room.shi * 2.0
            - max(0, metrics.room.mean_intake - 27) * 2.0
        )
        return reward

    def _cooling_proxi_reward(self):
        rack_positions = np.argwhere(self.rack_map == 1)

        if len(rack_positions) == 0:
            return 0.0

        total = 0.0
        for x, y in rack_positions:
            d = np.min(
                np.sqrt(
                    (self.cooling_pos[:, 0] - x) ** 2
                    + (self.cooling_pos[:, 1] - y) ** 2
                )
            )

            total += np.exp(-d / 5.0)

        return total / len(rack_positions)

    def _airflow_alignment_reward(self):
        flow = self._compute_airflow()

        dirs = np.array([(1, 0), (-1, 0), (0, 1), (0, -1)])

        score = 0.0
        racks = np.argwhere(self.rack_map == 1)

        for x, y in racks:
            d = dirs[int(self.rack_dir[x, y])]
            f = flow[x, y]

            # cosine similarity
            dot = d[0] * f[0] + d[1] * f[1]
            score += dot

        return score / (len(racks) + 1e-6)

    def _density_penalty(self):
        rack_positions = np.argwhere(self.rack_map == 1)

        if len(rack_positions) < 2:
            return 0.0

        dist = np.linalg.norm(
            rack_positions[:, None] - rack_positions[None, :], axis=-1
        )

        dist += np.eye(len(rack_positions)) * 1e6

        mean_dist = np.mean(dist)

        return -np.exp(-mean_dist / 5.0)

    def _hotspot_penalty(self):
        heat = np.zeros((self.grid_size, self.grid_size))

        racks = np.argwhere(self.rack_map == 1)

        for x, y in racks:
            power = self.rack_power[x, y]

            dx = self.X - x
            dy = self.Y - y
            dist = np.sqrt(dx**2 + dy**2) + 1e-6

            heat += power * np.exp(-dist / 3.0)

        max_heat = np.max(heat)
        mean_heat = np.mean(heat)

        return -(0.5 * max_heat + 0.5 * mean_heat)

    def fast_reward(self):

        cooling = self._cooling_proxi_reward()

        airflow = self._airflow_alignment_reward()

        density = self._density_penalty()

        heat = np.mean(self.rack_power)

        if np.any(self.rack_map == 1):
            coords = np.argwhere(self.rack_map == 1)

            hotspot = 0.0
            for x, y in coords:
                d = np.min(
                    np.sqrt(
                        (self.cooling_pos[:, 0] - x) ** 2
                        + (self.cooling_pos[:, 1] - y) ** 2
                    )
                )
                hotspot += d

            hotspot /= len(coords)
        else:
            hotspot = 0.0

        return (
            2.0 * cooling + 1.5 * airflow - 1.0 * density - 1.2 * heat - 0.8 * hotspot
        )
