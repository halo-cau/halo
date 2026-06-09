"""Shared multi-view scaffolding for the training-free grounding backends.

The DINOv3 / SAM3-concept backends all follow the same shape as
``DinoSam2Segmentor.run``:

    render N views  ->  per-view 2D ``label -> mask``  ->  backproject + vote
    ->  per-vertex argmax  ->  structural priors / cleanup  ->  SegmentorResult

Only the middle step (how each view's ``label -> mask`` dict is produced)
differs between backends. This module owns everything else so the backend
classes stay small, and it deliberately reuses the rendering / backprojection /
rack-prior helpers already living in ``segmentor_dino_sam`` rather than
re-implementing them.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import open3d as o3d

from engine.vision.graph_fusion import GraphCleanupConfig, apply_graph_cleanup
from engine.vision.segmentor_base import (
    PRESERVED_LABELS,
    STRUCTURAL_LABELS,
    SegmentorResult,
    label_to_color,
    strip_non_structural,
)
from engine.vision.segmentor_dino_sam import (
    _RenderView,
    _apply_server_rack_geometry_prior,
    _build_camera_poses,
    _render_views,
    _visible_vertices_in_view,
)
from engine.vision.structural_priors import (
    StructuralProtectionConfig,
    append_virtual_room_shell,
    apply_structural_protection,
    estimate_room_shell_prior,
    flatten_structural_labels_to_room_shell,
)

log = logging.getLogger(__name__)

#: Object classes we try to *ground* via the 2D foundation models. Structural
#: wall/floor/ceiling are left to the geometric shell prior, matching the
#: existing DINO+SAM default (which disables structural prompts). Each entry is
#: ``(prompt_text, canonical_label)``; the canonical label must exist in the
#: viewer palette (see ``segmentor_base.LABEL_PALETTE``).
GROUNDING_TARGETS: tuple[tuple[str, str], ...] = (
    ("server rack", "server rack"),
    ("air conditioning unit", "ac_unit"),
    ("air conditioner", "ac_unit"),
    ("HVAC unit", "ac_unit"),
    ("cable tray", "cable tray"),
    ("cardboard box", "cardboard box"),
    ("chair", "chair"),
    ("office chair", "chair"),
    ("trash can", "trash can"),
    ("garbage bin", "trash can"),
    ("waste basket", "trash can"),
)


def render_mesh_views(
    mesh: o3d.geometry.TriangleMesh,
    n_horizontal: int = 8,
    n_oblique: int = 4,
    render_width: int = 800,
    render_height: int = 600,
) -> tuple[list[_RenderView], np.ndarray]:
    """Render a ring of RGB+depth views around the mesh.

    Returns ``(views, verts)`` where ``verts`` is the ``(V, 3)`` float32 vertex
    array used for backprojection.
    """
    mesh.compute_vertex_normals()
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    bbox = mesh.get_axis_aligned_bounding_box()
    center = np.asarray(bbox.get_center(), dtype=np.float64)
    extent = np.asarray(bbox.get_extent(), dtype=np.float64)
    radius = float(np.linalg.norm(extent)) * 0.9
    log.info(
        "Rendering %d views (radius=%.2f m)…",
        1 + n_horizontal + n_oblique, radius,
    )
    poses = _build_camera_poses(center, radius, n_horizontal, n_oblique)
    views = _render_views(mesh, poses, render_width, render_height)
    return views, verts


class VoteAccumulator:
    """Accumulate per-vertex label votes across rendered views."""

    def __init__(self, n_vertices: int) -> None:
        self.n = n_vertices
        self.votes: dict[str, np.ndarray] = {}
        self.seen = np.zeros(n_vertices, dtype=np.int32)

    def add_view(
        self,
        verts: np.ndarray,
        view: _RenderView,
        label_masks: dict[str, np.ndarray],
    ) -> None:
        """Backproject one view's ``label -> mask2d`` dict into vote counts."""
        px, py, depth_valid = _visible_vertices_in_view(verts, view)
        self.seen[depth_valid] += 1
        idx = np.where(depth_valid)[0]
        H, W = view.depth.shape
        if len(idx) == 0:
            for label in label_masks:
                self.votes.setdefault(label, np.zeros(self.n, dtype=np.int32))
            return
        vy = np.clip(py[idx], 0, H - 1)
        vx = np.clip(px[idx], 0, W - 1)
        for label, mask2d in label_masks.items():
            arr = self.votes.setdefault(label, np.zeros(self.n, dtype=np.int32))
            in_mask = mask2d[vy, vx]
            arr[idx[in_mask]] += 1


def _vote_labels(
    accumulator: VoteAccumulator,
    pre_prior,
    min_vote_fraction: float,
    use_structural_fallback: bool,
) -> list[str]:
    """Per-vertex argmax over votes, with optional structural-prior fallback."""
    n = accumulator.n
    labels = ["unknown"] * n
    if accumulator.votes:
        names = list(accumulator.votes.keys())
        stacked = np.stack([accumulator.votes[name] for name in names], axis=1)  # (N, K)
        best = stacked.argmax(axis=1)
        best_count = stacked.max(axis=1)
        frac = best_count / np.maximum(1, accumulator.seen)
        accept = (best_count > 0) & (frac >= min_vote_fraction)
    else:
        accept = np.zeros(n, dtype=bool)
        best = np.zeros(n, dtype=np.int64)
        names = []

    for vi in range(n):
        if accept[vi]:
            labels[vi] = names[best[vi]]
        elif (
            use_structural_fallback
            and pre_prior.protected_mask[vi]
            and pre_prior.protected_labels[vi] in STRUCTURAL_LABELS
        ):
            labels[vi] = pre_prior.protected_labels[vi]
    return labels


def finalize_grounding_result(
    mesh: o3d.geometry.TriangleMesh,
    accumulator: VoteAccumulator,
    *,
    backend: str,
    min_vote_fraction: float = 0.30,
    protection_config: StructuralProtectionConfig | None = None,
    graph_config: GraphCleanupConfig | None = None,
    remove_movable: bool = False,
    apply_semantic_priors: bool = True,
    apply_rack_prior: bool = True,
    color_fn: Callable[[str], tuple[float, float, float]] = label_to_color,
    extra: dict | None = None,
) -> SegmentorResult:
    """Turn accumulated votes into a SegmentorResult via the standard prior chain.

    When ``apply_semantic_priors`` is False (unsupervised k-means clusters) the
    structural / rack / flatten priors are skipped — the cluster labels are kept
    verbatim and colored by ``color_fn`` so the viewer shows raw feature groups.

    ``apply_rack_prior`` (only meaningful when ``apply_semantic_priors`` is True)
    gates the 3D server-rack geometry prior. Leave it on for rack-focused runs;
    turn it off to inspect the model's own labels — including AC / box / cable
    tray, which the rack prior would otherwise overwrite on interior verticals.
    """
    protection_config = protection_config or StructuralProtectionConfig()
    graph_config = graph_config or GraphCleanupConfig()

    # The shell prior used for vote-fallback and the rack prior must not be the
    # "off" no-op (which protects nothing); force a usable level like DinoSam.
    prior_cfg = protection_config
    if prior_cfg.level == "off":
        prior_cfg = StructuralProtectionConfig(
            level="balanced",
            outer_percentile=prior_cfg.outer_percentile,
            plane_tolerance_m=prior_cfg.plane_tolerance_m,
            normal_cos_min=prior_cfg.normal_cos_min,
        )
    pre_prior = estimate_room_shell_prior(mesh, prior_cfg)

    labels = _vote_labels(
        accumulator, pre_prior, min_vote_fraction,
        use_structural_fallback=apply_semantic_priors,
    )

    extra_stats: dict = {}
    flattened_mesh = mesh
    if apply_semantic_priors:
        if apply_rack_prior:
            labels, rack_stats = _apply_server_rack_geometry_prior(mesh, labels, pre_prior)
        else:
            rack_stats = {"enabled": False}
        protection = apply_structural_protection(mesh, labels, protection_config)
        labels = protection.vertex_labels
        graph = apply_graph_cleanup(mesh, labels, protection.prior, graph_config)
        labels = graph.vertex_labels
        flattened_mesh, flatten_stats = flatten_structural_labels_to_room_shell(
            mesh, labels, protection.prior,
        )
        review_mesh, vshell_stats = append_virtual_room_shell(flattened_mesh, protection.prior)
        strip_prior = protection.prior
        extra_stats = {
            "server_rack_geometry_prior": rack_stats,
            "structural_protection": protection.to_json(),
            "graph_cleanup": graph.to_json(),
            "structural_flattening": flatten_stats,
            "virtual_room_shell": vshell_stats,
        }
    else:
        review_mesh = mesh
        strip_prior = pre_prior
        extra_stats = {"semantic_priors": "skipped (unsupervised clusters)"}

    label_colors = np.array([color_fn(lbl) for lbl in labels], dtype=np.float32)
    label_map: dict[str, list[int]] = {}
    for vi, lbl in enumerate(labels):
        label_map.setdefault(lbl, []).append(vi)

    n_removable = sum(1 for lbl in labels if lbl not in PRESERVED_LABELS)
    n_removed = n_removable if remove_movable else 0
    if remove_movable and apply_semantic_priors:
        structural_mesh = append_virtual_room_shell(
            strip_non_structural(flattened_mesh, labels), strip_prior,
        )[0]
    else:
        structural_mesh = review_mesh

    merged_extra = {
        "n_views": int(accumulator.seen.max()) if accumulator.n else 0,
        "min_vote_fraction": min_vote_fraction,
        "remove_movable": remove_movable,
        "n_removable_candidates": n_removable,
        **extra_stats,
    }
    if extra:
        merged_extra.update(extra)

    return SegmentorResult(
        structural_mesh=structural_mesh,
        vertex_labels=labels,
        label_colors=label_colors,
        label_map=label_map,
        n_removed=n_removed,
        backend=backend,
        extra=merged_extra,
    )
