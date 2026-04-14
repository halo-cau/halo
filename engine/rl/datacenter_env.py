import gymnasium as gym
import numpy as np
from gymnasium import spaces
from collections import deque
import math
import scipy.ndimage


class DataCenterEnv(gym.Env):
    def __init__(
        self, grid_size=50, rack_num=10, num_coolers=3, T_max=0.7, obstacle_prob=0.1
    ):
        super().__init__()

        self.grid_size = grid_size
        self.rack_num = rack_num
        self.num_coolers = num_coolers
        self.obstacle_prob = obstacle_prob
        self.T_max = T_max

        self.rack_pos = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.obstacle = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.heat = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        self.sigma_heat = 2.0
        self.diffusion_sigma = 1.0
        self.cooling_decay = 5.0

        self.cooling_pos = np.array(
            [
                [self.grid_size // 4, self.grid_size // 4],
                [self.grid_size // 4, 3 * self.grid_size // 4],
                [3 * self.grid_size // 4, self.grid_size // 2],
            ]
        )

        self.rack_power = np.ones(self.rack_num)

        self.action_space = spaces.Discrete(self.grid_size * self.grid_size)

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(4, grid_size, grid_size), dtype=np.float32
        )

        X, Y = np.meshgrid(np.arange(grid_size), np.arange(grid_size), indexing="ij")
        self.X = X
        self.Y = Y
        self.coords = np.stack([X, Y], axis=-1)

    def step(self, action):
        # x = int(action[0] * self.grid_size)
        # y = int(action[1] * self.grid_size)

        # x = np.clip(x, 0, self.grid_size - 1)
        # y = np.clip(y, 0, self.grid_size - 1)

        x, y = self._decode(action)

        if self.rack_pos[x, y] == 1 or self.obstacle[x, y] == 1:
            reward = -10000.0
            # self.step_count += 1
        else:
            self.rack_pos[x, y] = 1
            self.step_count += 1

            self.compute_temp()

            reward = self.compute_reward()

        done = self.step_count >= self.rack_num

        return self._get_obs(), reward, done, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.rack_pos = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        # self.obstacle = self.generate_obstacle()
        self.heat = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        # self.cooling_pos = np.random.randint(
        #    0, self.grid_size, size=(self.num_coolers, 2)
        # )
        self.obstacle, self.cooling_pos = self.generate()
        self.dist_map = self.get_cooling_distance_map()

        self.step_count = 0

        return self._get_obs(), {}

    def _get_obs(self):
        # dist_map = self.get_cooling_distance_map()

        return np.stack(
            [self.rack_pos, self.obstacle, self.heat, self.dist_map], axis=0
        )

    def _decode(self, action):
        i = action // self.grid_size
        j = action % self.grid_size
        return i, j

    def get_action_mask(self):
        mask = np.ones(self.grid_size * self.grid_size, dtype=np.float32)

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.rack_pos[i, j] == 1 or self.obstacle[i, j] == 1:
                    idx = i * self.grid_size + j
                    mask[idx] = 0

        return mask

    def action_masks(self):
        return self.get_action_mask()

    """
    def upd_heat(self):
        heat = np.zeros_like(self.heat)

        rack_pos = np.argwhere(self.rack_pos == 1)

        for i, j in rack_pos:
            for k in range(self.grid_size):
                for o in range(self.grid_size):
                    dist = abs(i - k) + abs(j - o)
                    heat[k, o] += 1.0 / (1 + dist)

        heat = heat / (np.max(heat) + 1e-8)

        self.heat = heat
    """

    def compute_reward(self):
        # rack을 전부 멀리 떨어뜨려 배치하는 방향으로 학습될 수 있음
        # 냉각기의 위치를 고정하는 경우와 이것도 action으로 배치하는 경우로 나누어서 학습해기
        # 냉각기의 위치와 사용 에너지까지도 reward 계산에 넣어야함
        # 04/04 현재까지는 열의 분산과 hotspot으로만 reward 계산 -n> rack끼리 멀리 떨어뜨릴듯?
        # 실제로는 rack을 한곳에 모아서 냉각기로 한 부분만 집중적으로 냉각시키는 것이 더 효율적일수도?
        # 현재는 rack 개수 고정
        # 나중에는 rack 개수까지 에피소드마다 랜덤으로 설정해서 학습하도록 하기
        #
        # hot = np.var(self.heat)

        # hotspot = np.sum(self.heat > 0.7)

        # reward = -0.5 * (1 - math.exp(-hot)) - 0.5 * (1 - math.exp(-hotspot * 3))

        # temp = self.heat

        # violation = np.maximum(0, temp - self.T_max)
        # violation_penalty = np.sum(violation)

        energy = self.compute_energy(self.heat)

        # energy = np.log(1 + energy)

        # reward = -5.0 * (1 - math.exp(-energy))
        # reward = -10 * (1 - math.exp(-energy))
        reward = -energy / 1000.0

        return reward

    def generate_obstacle(self):
        while True:
            obstacle = (
                np.random.rand(self.grid_size, self.grid_size) < self.obstacle_prob
            ).astype(np.float32)

            if np.sum(obstacle == 0) < self.rack_num * 2:
                continue

            if self.is_connected(obstacle):
                return obstacle

    def is_connected(self, obstacle):
        visit = np.zeros_like(obstacle)
        empty = np.argwhere(obstacle == 0)

        if len(empty) == 0:
            return False

        queue = deque([tuple(empty[0])])
        visit[queue[0]] = 1

        direction = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        while queue:
            x, y = queue.popleft()

            for dx, dy in direction:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    if obstacle[nx, ny] == 0 and visit[nx, ny] == 0:
                        visit[nx, ny] = 1
                        queue.append((nx, ny))

        return np.sum(visit) == np.sum(obstacle == 0)

    def compute_heat_sources(self):
        heat_map = np.zeros((self.grid_size, self.grid_size))

        rack_coords = np.argwhere(self.rack_pos == 1)

        for idx, (x, y) in enumerate(rack_coords):
            power = self.rack_power[min(idx, len(self.rack_power) - 1)]
            d2 = (self.X - x) ** 2 + (self.Y - y) ** 2
            heat_map += power * np.exp(-d2 / (2 * self.sigma_heat**2))

        return heat_map

    def diffuse(self, heat):
        return scipy.ndimage.gaussian_filter(heat, sigma=self.diffusion_sigma)

    def compute_temp(self):
        heat = self.compute_heat_sources()
        heat = self.diffuse(heat)
        # heat = heat / (np.max(heat) + 1e-6)
        self.heat = np.clip(heat, 0, 5)

    def compute_energy(self, temp):

        dist = np.linalg.norm(
            self.coords[:, :, None, :] - self.cooling_pos[None, None, :, :], axis=-1
        )
        d = np.min(dist, axis=-1)

        efficiency = np.exp(-d / self.cooling_decay)

        base = temp
        excess = np.maximum(0, temp - self.T_max)

        cost = base + 3.0 * (excess**2)

        energy = cost / (efficiency + 1e-6)

        return np.sum(energy)

    def get_cooling_distance_map(self):
        dist = np.linalg.norm(
            self.coords[:, :, None, :] - self.cooling_pos[None, None, :, :],
            axis=-1,
        )
        d = np.min(dist, axis=-1)

        return d / (np.max(d) + 1e-6)

    def generate(self):
        wall = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        margin_1 = np.random.randint(1, 15)
        margin_2 = np.random.randint(1, 15)
        margin_3 = np.random.randint(1, 15)
        margin_4 = np.random.randint(1, 15)

        grid_x_size = self.grid_size - margin_1 - margin_2
        grid_y_size = self.grid_size - margin_3 - margin_4

        wall[:margin_1, :] = 1
        wall[-margin_2:, :] = 1
        wall[:, :margin_3] = 1
        wall[:, -margin_4:] = 1

        max_spacing = max(6, min(grid_x_size, grid_y_size) // 2)

        pillar = np.random.randint(5, max_spacing)

        for i in range(margin_1 + pillar, self.grid_size - margin_2, pillar):
            for j in range(margin_3 + pillar, self.grid_size - margin_4, pillar):
                if np.random.rand() < 0.8:
                    wall[i, j] = 1

        cooling_pos = np.zeros((self.num_coolers, 2), dtype=np.float32)
        used_positions = set()

        for i in range(self.num_coolers):
            for attempt in range(1000):
                x = np.random.randint(margin_1, self.grid_size - margin_2)
                y = np.random.randint(margin_3, self.grid_size - margin_4)

                if wall[x, y] == 0 and (x, y) not in used_positions:
                    cooling_pos[i] = [x, y]
                    used_positions.add((x, y))
                    break
            else:
                print("No space for more coolers!")

        return wall, cooling_pos
