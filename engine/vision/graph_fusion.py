"""Mesh-graph label fusion for safer structural cleanup.

The semantic segmentors operate per rendered view and can produce speckled
vertex labels.  This module adds a mesh-adjacency pass that treats the room
shell prior as seeds, restores near-shell vertices connected to those seeds,
and prunes structural islands that are not graph-connected to the shell.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import open3d as o3d

from engine.vision.segmentor_base import AMBIGUOUS_LABELS, STRUCTURAL_LABELS
from engine.vision.structural_priors import RoomShellPrior


@dataclass(frozen=True)
class GraphCleanupConfig:
    """Controls the mesh-graph fusion cleanup pass."""

    enabled: bool = False
    restore_band_m: float = 0.35
    prune_band_m: float = 0.65
    min_seed_vertices: int = 8
    min_seed_fraction: float = 0.002
    min_component_vertices: int = 48


@dataclass
class GraphCleanupResult:
    """Output labels and diagnostic counters from graph fusion."""

    vertex_labels: list[str]
    n_graph_restored: int
    n_graph_pruned: int
    n_components_total: int
    n_components_kept: int
    n_components_pruned: int
    config: GraphCleanupConfig

    def to_json(self) -> dict:
        return {
            "enabled": self.config.enabled,
            "restore_band_m": self.config.restore_band_m,
            "prune_band_m": self.config.prune_band_m,
            "min_seed_vertices": self.config.min_seed_vertices,
            "min_seed_fraction": self.config.min_seed_fraction,
            "min_component_vertices": self.config.min_component_vertices,
            "n_graph_restored": self.n_graph_restored,
            "n_graph_pruned": self.n_graph_pruned,
            "n_components_total": self.n_components_total,
            "n_components_kept": self.n_components_kept,
            "n_components_pruned": self.n_components_pruned,
        }


def _build_vertex_adjacency(n_vertices: int, triangles: np.ndarray) -> list[list[int]]:
    neighbours: list[set[int]] = [set() for _ in range(n_vertices)]
    for a, b, c in triangles.astype(np.int64, copy=False):
        neighbours[a].add(b)
        neighbours[a].add(c)
        neighbours[b].add(a)
        neighbours[b].add(c)
        neighbours[c].add(a)
        neighbours[c].add(b)
    return [list(ns) for ns in neighbours]


def _closest_shell_distance_and_label(
    verts: np.ndarray,
    prior: RoomShellPrior,
) -> tuple[np.ndarray, list[str]]:
    mn = prior.bounds_min.astype(float)
    mx = prior.bounds_max.astype(float)

    plane_distances = np.stack(
        [
            np.abs(verts[:, 2] - mn[2]),
            np.abs(verts[:, 2] - mx[2]),
            np.abs(verts[:, 0] - mn[0]),
            np.abs(verts[:, 0] - mx[0]),
            np.abs(verts[:, 1] - mn[1]),
            np.abs(verts[:, 1] - mx[1]),
        ],
        axis=1,
    )
    labels = np.array(["floor", "ceiling", "wall", "wall", "wall", "wall"], dtype=object)
    closest_idx = np.argmin(plane_distances, axis=1)
    return plane_distances[np.arange(len(verts)), closest_idx], labels[closest_idx].astype(str).tolist()


def apply_graph_cleanup(
    mesh: o3d.geometry.TriangleMesh,
    vertex_labels: list[str],
    prior: RoomShellPrior,
    config: GraphCleanupConfig | None = None,
) -> GraphCleanupResult:
    """Fuse AI labels with room-shell seeds over the mesh adjacency graph.

    The pass has two stages:
     1. Restore only ambiguous vertices that are near the estimated shell and
         graph-connected to protected shell seeds.  Explicit object/static labels
         are not converted into walls.
    2. Prune structural components that are not supported by shell seeds, plus
       long appendages that extend too far inward from the shell.
    """
    cfg = config or GraphCleanupConfig()
    labels = list(vertex_labels)
    n = len(labels)
    if not cfg.enabled or n == 0:
        return GraphCleanupResult(labels, 0, 0, 0, 0, 0, cfg)

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    adjacency = _build_vertex_adjacency(n, triangles)
    shell_distance, closest_shell_labels = _closest_shell_distance_and_label(verts, prior)

    protected = np.asarray(prior.protected_mask, dtype=bool)
    near_restore = shell_distance <= cfg.restore_band_m
    near_prune = shell_distance <= cfg.prune_band_m

    # Stage 1: shell-seeded restoration through near-shell graph regions.
    visited = np.zeros(n, dtype=bool)
    n_restored = 0
    seed_indices = np.where(protected & near_restore)[0]
    for seed in seed_indices:
        if visited[seed]:
            continue
        queue: deque[int] = deque([int(seed)])
        visited[seed] = True
        region: list[int] = []
        while queue:
            vi = queue.popleft()
            region.append(vi)
            for nb in adjacency[vi]:
                if visited[nb] or not near_restore[nb]:
                    continue
                visited[nb] = True
                queue.append(nb)

        for vi in region:
            if labels[vi] in AMBIGUOUS_LABELS:
                labels[vi] = closest_shell_labels[vi]
                n_restored += 1

    # Stage 2: connected-component pruning over currently structural labels.
    structural = np.array([lbl in STRUCTURAL_LABELS for lbl in labels], dtype=bool)
    component_seen = np.zeros(n, dtype=bool)
    n_components = 0
    n_kept_components = 0
    n_pruned_components = 0
    n_pruned_vertices = 0

    for start in np.where(structural)[0]:
        if component_seen[start]:
            continue
        n_components += 1
        queue = deque([int(start)])
        component_seen[start] = True
        component: list[int] = []
        while queue:
            vi = queue.popleft()
            component.append(vi)
            for nb in adjacency[vi]:
                if component_seen[nb] or not structural[nb]:
                    continue
                component_seen[nb] = True
                queue.append(nb)

        comp = np.asarray(component, dtype=np.int64)
        size = len(comp)
        seed_count = int(protected[comp].sum())
        seed_fraction = seed_count / max(1, size)
        close_count = int(near_prune[comp].sum())

        keep_component = (
            seed_count >= cfg.min_seed_vertices
            or seed_fraction >= cfg.min_seed_fraction
            or (size >= cfg.min_component_vertices and close_count / max(1, size) >= 0.80)
        )

        if not keep_component:
            for vi in comp:
                labels[int(vi)] = "object"
            n_pruned_vertices += size
            n_pruned_components += 1
            continue

        n_kept_components += 1
        far_appendage = comp[(~protected[comp]) & (~near_prune[comp])]
        for vi in far_appendage:
            labels[int(vi)] = "object"
        n_pruned_vertices += int(len(far_appendage))

    return GraphCleanupResult(
        vertex_labels=labels,
        n_graph_restored=n_restored,
        n_graph_pruned=n_pruned_vertices,
        n_components_total=n_components,
        n_components_kept=n_kept_components,
        n_components_pruned=n_pruned_components,
        config=cfg,
    )