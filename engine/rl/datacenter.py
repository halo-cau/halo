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
        if options is not None:
            self.obstacle = options.get("obstacle", self._generate_layout())
            self.cooling_pos = options.get(
                "cooling_pos", self._generate_cooling(self.obstacle)
            )
            self.num_cooler = len(self.cooling_pos)
            self.rack_num = options.get("rack_num", self.rack_num)
        else:
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
        reward = self._compute_reward()
        done = self.step_count >= self.rack_num

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
        temp = temp * self.temp_decay + self._compute_exhaust()

        flow = self._compute_airflow()

        for _ in range(5):
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

    def _compute_reward(self):
        T = self.temp

        real_T = 15 + T * 4

        rec = np.logical_and(real_T >= 18, real_T <= 27)
        C_rec = np.mean(rec)

        violation = np.maximum(0, real_T - 32)
        V = np.sum(violation)

        H = np.max(real_T)

        energy = np.mean(T)

        reward = 10 * C_rec - 0.2 * V - 0.3 * max(0, H - 27) - 0.05 * energy

        rack_positions = np.argwhere(self.rack_map == 1)

        if len(rack_positions) > 1:
            dists = np.linalg.norm(
                rack_positions[:, None] - rack_positions[None, :], axis=-1
            )
            dists = dists + np.eye(len(rack_positions)) * 1e6
            spread = np.mean(dists)
        else:
            spread = 0

        reward += -0.1 * spread

        return reward

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
