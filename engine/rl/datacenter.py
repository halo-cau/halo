import gymnasium as gym
import numpy as np
from gymnasium import spaces
import scipy.ndimage


# ---------------------------------------------------------------------------
# Shared ASHRAE reward formula (used by both fast-2D and engine-3D paths)
# ---------------------------------------------------------------------------

def _ashrae_reward_from_metrics(metrics) -> float:
    """Compute the ASHRAE TC 9.9 reward from a MetricsResult object.

    This is the canonical reward formula shared by both the fast 2-D path
    (which approximates intakes/exhausts from adjacent cells) and the
    authoritative 3-D engine path (which runs the full physics solver).
    Keeping it in one place ensures both paths produce comparable signals.

    Returns a scalar reward:
        - ~ +3.5 for an ideal layout (all racks compliant, good airflow)
        - ~  0   for a mediocre layout
        - ~ −5   for a layout with widespread allowable violations
    """
    intakes  = np.array([r.intake_temp  for r in metrics.racks])
    exhausts = np.array([r.exhaust_temp for r in metrics.racks])
    dts      = exhausts - intakes

    # ASHRAE compliance fractions
    s_rec   = float(np.mean((intakes >= 18.0) & (intakes <= 27.0)))
    s_allow = float(np.mean((intakes >= 15.0) & (intakes <= 35.0)))

    # Magnitude of allowable violations (per rack, normalised to 10 °C band)
    p_allow = float(np.mean(
        np.maximum(intakes - 35.0, 0.0) + np.maximum(15.0 - intakes, 0.0)
    ) / 10.0)

    # Delta-T health: ideal 6–20 °C; penalise outside that band
    p_dt = float(np.mean(
        np.maximum(6.0 - dts,  0.0) / 6.0
        + np.maximum(dts - 20.0, 0.0) / 10.0
    ))

    # RCI-style score from room vertical profile
    vp = np.array(metrics.room.vertical_profile)
    rci_hi_viol = float(np.mean(np.maximum(vp - 27.0, 0.0)) / 8.0)
    rci_lo_viol = float(np.mean(np.maximum(15.0 - vp, 0.0)) / 5.0)
    s_rci = 1.0 - 0.5 * (rci_hi_viol + rci_lo_viol)

    # Thermal stratification penalty (95th – 5th percentile spread > 6 °C)
    spread  = float(np.percentile(vp, 95) - np.percentile(vp, 5))
    p_strat = max(0.0, (spread - 6.0) / 8.0)

    base = (
        2.0 * s_rec
        + 1.5 * s_rci
        - 3.0 * p_allow
        - 0.8 * p_dt
        - 0.5 * p_strat
    )
    gate = -2.0 * (1.0 - s_allow)
    return float(base + gate)


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
            low=0, high=1, shape=(6, grid_size, grid_size), dtype=np.float32
        )

        X, Y = np.meshgrid(np.arange(grid_size), np.arange(grid_size), indexing="ij")
        self.X = X.astype(np.float32)
        self.Y = Y.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.rack_map[:] = 0
        self.rack_dir[:] = 0
        self.temp[:] = 0

        self.obstacle = self._generate_layout()
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

        self.step_count += 1

        self._update_temp()
        done = self.step_count >= self.rack_num

        if done:
            # Episode end: run the 3-D thermal engine once for the final reward
            metrics = self._bridge.solve_metrics(
                self.rack_map, self.rack_dir, self.obstacle, self.cooling_pos,
            )
            reward = _ashrae_reward_from_metrics(metrics) if metrics is not None else 0.0
        else:
            # Mid-episode: lightweight position shaping (no thermal solve)
            reward = self._position_shaping_reward()

        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        flow = self._compute_airflow()

        flow_x = flow[:, :, 0]
        flow_y = flow[:, :, 1]

        dist = self._cooling_dist_map()

        return np.stack(
            [self.rack_map, self.obstacle, self.temp, flow_x, flow_y, dist],
            axis=0,
        ).astype(np.float32)

    def _update_temp(self):
        temp = self.temp.copy()
        temp += self._compute_exhaust()

        flow = self._compute_airflow()

        for _ in range(5):
            temp = self._advect(temp, flow)

        temp = scipy.ndimage.gaussian_filter(temp, sigma=self.diff_sigma)

        temp = self._apply_cooling(temp)

        self.temp = np.clip(temp, 0, 5)

    def _compute_exhaust(self):
        temp = np.zeros_like(self.temp)

        dirs = np.array([(1, 0), (-1, 0), (0, 1), (0, -1)])

        racks = np.argwhere(self.rack_map == 1)

        for x, y in racks:
            d = dirs[int(self.rack_dir[x, y])]

            power = 1.0
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
        flow = np.zeros((self.grid_size, self.grid_size, 2))

        for cx, cy in self.cooling_pos:
            dx = self.X - cx
            dy = self.Y - cy

            dist = np.sqrt(dx**2 + dy**2) + 1e-6

            flow[:, :, 0] -= dx / dist
            flow[:, :, 1] -= dy / dist

        dirs = np.array([(1, 0), (-1, 0), (0, 1), (0, -1)])

        mask = self.rack_map == 1
        rack_indices = np.argwhere(mask)

        for x, y in rack_indices:
            d = dirs[int(self.rack_dir[x, y])]
            flow[x, y, 0] += d[0] * 2.0
            flow[x, y, 1] += d[1] * 2.0

        flow[self.obstacle == 1] = 0

        flow = flow / (np.linalg.norm(flow, axis=-1, keepdims=True) + 1e-6)

        return flow * 0.5

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
        for cx, cy in self.cooling_pos:
            d = np.sqrt((self.X - cx) ** 2 + (self.Y - cy) ** 2)
            effect = np.exp(-d / self.cool_decay)

            temp -= effect * (temp - 0.2)

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

        x, y = racks[-1]   # the rack just placed
        dirs = np.array([(1, 0), (-1, 0), (0, 1), (0, -1)])
        my_dir = int(self.rack_dir[x, y])
        my_exhaust = dirs[my_dir]   # unit vector: direction exhaust blows

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

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.rack_map[i, j] == 1 or self.obstacle[i, j] == 1:
                    for d in range(4):
                        idx = (i * self.grid_size + j) * 4 + d
                        mask[idx] = 0

        return mask

    def action_masks(self):
        return self.get_action_mask()
