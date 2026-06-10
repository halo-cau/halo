"""Macro-action environment for the layout policy.

Each step places a whole ENTITY rather than a single cell: a cooling unit, a server-rack row, or the
network rack. Composing a layout from a few structured macro actions makes row alignment a property of the
action space and keeps episodes short, which is the design used by structured-action agents generally.

Action space: ``Discrete(grid*grid * 6)``. Decoding ``ch = action % 6``, ``pos = action // 6``,
``gx = pos // grid``, ``gy = pos % grid``:
    ch 0      -> place a cooling unit (AC) at (gx, gy)
    ch 1..4   -> place a server-rack ROW anchored at (gx, gy), exhaust dir = ch-1
    ch 5      -> place the network rack at (gx, gy)

It reuses ``DataCenterEnv``'s observation, footprint, and stamping machinery; only the action semantics and
the per-entity episode are new. The 6-channel action space is consumed unchanged by
``SpatialMaskablePolicy`` (its head derives the channel count from ``action_space.n // grid**2``). The AC is
itself an action, so the room is reset with no cooler, which the base ``_cooling_dist_map`` now tolerates.
"""
from __future__ import annotations

import numpy as np
from gymnasium import spaces

from engine.rl.datacenter import _RACK_W_CELLS, DataCenterEnv

# Action channels.
CH_AC = 0
CH_ROW0 = 1            # ch 1..4 -> server row with exhaust dir 0..3
CH_NETRACK = 5
N_CHANNELS = 6


class MacroPlacementEnv(DataCenterEnv):
    """One macro action per step (cooling unit, rack row, or network rack) on a given room."""

    def __init__(self, grid_size=50, racks_per_row=6, n_actions=4, netrack_dir=1,
                 ceiling_m: float = 3.0):
        super().__init__(grid_size=grid_size, rack_num=racks_per_row * 2, num_cooler=1,
                         ceiling_m=ceiling_m, placement_mode="row")
        self.racks_per_row = racks_per_row
        self.n_actions = n_actions
        self._netrack_dir = int(netrack_dir)
        self._n_ch = N_CHANNELS
        self.action_space = spaces.Discrete(grid_size * grid_size * N_CHANNELS)
        self.netrack: list[tuple[int, int, int]] = []

    # ------------------------------------------------------------------ helpers
    def encode(self, gx: int, gy: int, channel: int) -> int:
        """(cell, channel) -> flat action index, matching the policy head's flatten order."""
        return int((gx * self.grid_size + gy) * self._n_ch + channel)

    def _place_row(self, gx: int, gy: int, d: int, count: int) -> int:
        """Place up to ``count`` racks from the anchor along the width axis (best effort)."""
        sx, sy = (0, _RACK_W_CELLS) if d in (0, 1) else (_RACK_W_CELLS, 0)
        placed = 0
        for k in range(count):
            if self._place_one(gx + sx * k, gy + sy * k, d, power=1.65):
                placed += 1
        return placed

    def _place_netrack(self, gx: int, gy: int, d: int) -> None:
        """Mark the network rack footprint occupied and record it (kept out of rack_map: non-thermal)."""
        for fx, fy in self._rack_footprint(gx, gy, d):
            if 0 <= fx < self.grid_size and 0 <= fy < self.grid_size:
                self.rack_occupied[fx, fy] = 1
        self.netrack.append((int(gx), int(gy), int(d)))

    # ------------------------------------------------------------------ gym API
    def reset(self, seed=None, options=None):
        # Skip DataCenterEnv.reset (it randomises the room); take the room from options, no cooler yet.
        super(DataCenterEnv, self).reset(seed=seed)
        g = self.grid_size
        self.rack_map[:] = 0
        self.rack_dir[:] = 0
        self.rack_occupied[:] = 0
        self.temp[:] = 0
        self.rack_power = np.zeros((g, g))
        self.cached_flow = None
        self.flow_dirty = True
        opts = options or {}
        self.obstacle = np.asarray(opts.get("obstacle", np.zeros((g, g))))
        self.rack_num = int(opts.get("rack_num", self.rack_num))
        cm = opts.get("ceiling_m")
        if cm is not None and abs(float(cm) - self.ceiling_m) > 1e-6:
            from engine.rl.thermal_bridge import ThermalBridge
            self.ceiling_m = float(cm)
            self._bridge = ThermalBridge(grid_size=g, ceiling_m=self.ceiling_m)
        self.cooling_pos = np.zeros((0, 2), dtype=int)   # the AC is an action, not a fixed input
        self.num_cooler = 0
        self.netrack = []
        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        action = int(action)
        pos, ch = action // self._n_ch, action % self._n_ch
        gx, gy = pos // self.grid_size, pos % self.grid_size

        if ch == CH_AC:
            cells = self.cooling_pos.tolist() if len(self.cooling_pos) else []
            cells.append([int(gx), int(gy)])
            self.cooling_pos = np.array(cells, dtype=int)
            self.num_cooler = len(self.cooling_pos)
        elif CH_ROW0 <= ch <= CH_ROW0 + 3:
            self._place_row(gx, gy, ch - CH_ROW0, self.racks_per_row)
        else:  # CH_NETRACK
            self._place_netrack(gx, gy, self._netrack_dir)

        self.step_count += 1
        self.flow_dirty = True
        done = self.step_count >= self.n_actions
        info = {"n_racks": int(self.rack_map.sum()), "n_cool": int(len(self.cooling_pos)),
                "n_netrack": len(self.netrack)} if done else {}
        return self._get_obs(), 0.0, done, False, info

    def action_masks(self):
        # No masking: the demonstrated actions are valid placements. Returning all ones keeps the
        # MaskableCategorical well defined for predict/rollout.
        return np.ones(self.grid_size * self.grid_size * self._n_ch, dtype=np.float32)
