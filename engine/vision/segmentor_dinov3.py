"""DINOv3 grounding backends — standalone and DINOv3+SAM3.

Two backends that test Meta's DINOv3 self-supervised feature backbone for
grounding server racks / AC / objects, lifted from multi-view renders to 3D
vertex labels:

``Dinov3Segmentor`` (``dinov3``)
    Pure DINOv3, no masks and no labels. Patch features from every view are
    clustered with a single shared k-means; each cluster gets a distinct color.
    Output is *unnamed* groups — it tests whether DINOv3 features alone separate
    racks / AC / shell (the user names clusters by eye in the viewer).

``Dinov3Sam3Segmentor`` (``dinov3_sam3``)
    SAM3 supplies a few exemplar regions per class (via concept prompts);
    DINOv3 features pooled inside those regions become per-class **prototypes**.
    Every patch in every view is then labeled by nearest-prototype cosine
    similarity, so DINOv3 — not SAM3 — decides each vertex's class. SAM3 is freed
    before DINOv3 loads to stay within 8 GB VRAM.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import open3d as o3d

from engine.vision.dinov3_features import (
    DEFAULT_DINOV3_MODEL,
    Dinov3FeatureExtractor,
    build_proto_matrix,
    cluster_color,
)
from engine.vision.graph_fusion import GraphCleanupConfig
from engine.vision.multiview_common import (
    GROUNDING_TARGETS,
    VoteAccumulator,
    finalize_grounding_result,
    render_mesh_views,
)
from engine.vision.segmentor_base import BaseSegmentor, SegmentorResult
from engine.vision.segmentor_dino_sam import _load_sam3
from engine.vision.segmentor_sam3_concept import segment_view_sam3_concept
from engine.vision.structural_priors import StructuralProtectionConfig

log = logging.getLogger(__name__)


def _cluster_color_fn(label: str) -> tuple[float, float, float]:
    """Color ``cluster_<k>`` labels with the distinct cluster palette."""
    if label.startswith("cluster_"):
        try:
            return cluster_color(int(label.split("_", 1)[1]))
        except (ValueError, IndexError):
            pass
    return (0.1, 0.1, 0.1)


def _mean_unit(vectors: np.ndarray) -> np.ndarray:
    """Mean of stacked vectors, L2-normalized."""
    vec = vectors.mean(axis=0)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-8 else vec


class Dinov3Segmentor(BaseSegmentor):
    """DINOv3-only unsupervised feature clustering (no SAM, no labels)."""

    def __init__(
        self,
        model_name: str = DEFAULT_DINOV3_MODEL,
        device: str | None = None,
        half: bool = True,
        hf_token: str | None = None,
        render_width: int = 800,
        render_height: int = 600,
        n_horizontal: int = 8,
        n_oblique: int = 4,
        n_clusters: int = 8,
        min_vote_fraction: float = 0.30,
        protection_config: StructuralProtectionConfig | None = None,
        graph_cleanup_config: GraphCleanupConfig | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.half = half
        self.hf_token = hf_token
        self.render_width = render_width
        self.render_height = render_height
        self.n_horizontal = n_horizontal
        self.n_oblique = n_oblique
        self.n_clusters = n_clusters
        self.min_vote_fraction = min_vote_fraction
        self.protection_config = protection_config or StructuralProtectionConfig()
        self.graph_cleanup_config = graph_cleanup_config or GraphCleanupConfig()
        self._extractor: Dinov3FeatureExtractor | None = None

    @property
    def name(self) -> str:
        return "dinov3"

    def _ensure_model(self) -> None:
        if self._extractor is None:
            self._extractor = Dinov3FeatureExtractor(
                self.model_name, self.device, self.half, self.hf_token,
            )

    def run(
        self,
        mesh: o3d.geometry.TriangleMesh,
        source_path: Path,
    ) -> SegmentorResult:
        self._ensure_model()
        assert self._extractor is not None
        views, verts = render_mesh_views(
            mesh, self.n_horizontal, self.n_oblique,
            self.render_width, self.render_height,
        )
        grids = [self._extractor.patch_features(v.rgb) for v in views]
        feat_grids = [g for g, _ in grids]
        centroids = self._extractor.fit_kmeans(feat_grids, k=self.n_clusters)

        accumulator = VoteAccumulator(len(verts))
        for view, (grid, _) in zip(views, grids):
            id_grid = self._extractor.assign_kmeans(grid, centroids)
            H, W = view.rgb.shape[:2]
            full = self._extractor.upsample_ids(id_grid, (H, W))
            label_masks = {
                f"cluster_{int(k)}": (full == k) for k in np.unique(full)
            }
            accumulator.add_view(verts, view, label_masks)

        self._extractor.free()
        return finalize_grounding_result(
            mesh,
            accumulator,
            backend=self.name,
            min_vote_fraction=self.min_vote_fraction,
            protection_config=self.protection_config,
            graph_config=self.graph_cleanup_config,
            remove_movable=False,
            apply_semantic_priors=False,   # clusters are unnamed — keep verbatim
            color_fn=_cluster_color_fn,
            extra={
                "mode": "kmeans",
                "n_clusters": self.n_clusters,
                "model": self.model_name,
                "n_views_rendered": len(views),
            },
        )


class Dinov3Sam3Segmentor(BaseSegmentor):
    """SAM3-seeded prototypes + dense DINOv3 nearest-prototype labeling."""

    def __init__(
        self,
        sam3_checkpoint: str | Path | None = None,
        model_name: str = DEFAULT_DINOV3_MODEL,
        device: str | None = None,
        half: bool = True,
        hf_token: str | None = None,
        render_width: int = 800,
        render_height: int = 600,
        n_horizontal: int = 8,
        n_oblique: int = 4,
        sam3_confidence_threshold: float = 0.5,
        min_similarity: float = 0.45,
        min_vote_fraction: float = 0.30,
        targets: tuple[tuple[str, str], ...] = GROUNDING_TARGETS,
        protection_config: StructuralProtectionConfig | None = None,
        graph_cleanup_config: GraphCleanupConfig | None = None,
        remove_movable: bool = False,
        apply_rack_prior: bool = True,
    ) -> None:
        self.sam3_checkpoint = str(sam3_checkpoint) if sam3_checkpoint else None
        self.model_name = model_name
        self.device = device
        self.half = half
        self.hf_token = hf_token
        self.render_width = render_width
        self.render_height = render_height
        self.n_horizontal = n_horizontal
        self.n_oblique = n_oblique
        self.sam3_confidence_threshold = sam3_confidence_threshold
        self.min_similarity = min_similarity
        self.min_vote_fraction = min_vote_fraction
        self.targets = targets
        self.protection_config = protection_config or StructuralProtectionConfig()
        self.graph_cleanup_config = graph_cleanup_config or GraphCleanupConfig()
        self.remove_movable = remove_movable
        self.apply_rack_prior = apply_rack_prior

    @property
    def name(self) -> str:
        return "dinov3_sam3"

    def run(
        self,
        mesh: o3d.geometry.TriangleMesh,
        source_path: Path,
    ) -> SegmentorResult:
        import torch

        views, verts = render_mesh_views(
            mesh, self.n_horizontal, self.n_oblique,
            self.render_width, self.render_height,
        )

        # 1) SAM3 concept pass → per-view exemplar regions, then free SAM3.
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        processor = _load_sam3(self.sam3_checkpoint, device, self.sam3_confidence_threshold)
        per_view_regions = [
            segment_view_sam3_concept(
                processor, v.rgb, self.targets, self.sam3_confidence_threshold,
            )
            for v in views
        ]
        del processor
        if device == "cuda":
            torch.cuda.empty_cache()

        # 2) DINOv3 features; pool inside SAM3 regions to seed class prototypes.
        extractor = Dinov3FeatureExtractor(self.model_name, device, self.half, self.hf_token)
        grids = [extractor.patch_features(v.rgb) for v in views]

        proto_accum: dict[str, list[np.ndarray]] = {}
        for (grid, _), regions in zip(grids, per_view_regions):
            for label, mask in regions.items():
                vec = extractor.pool_mask_feature(grid, mask)
                if vec is not None:
                    proto_accum.setdefault(label, []).append(vec)
        prototypes = {l: _mean_unit(np.stack(v)) for l, v in proto_accum.items()}

        # 3) Dense DINOv3 nearest-prototype labeling of every view.
        accumulator = VoteAccumulator(len(verts))
        if prototypes:
            proto_matrix, proto_labels = build_proto_matrix(prototypes)
            for view, (grid, _) in zip(views, grids):
                id_grid = extractor.assign_by_prototypes(
                    grid, proto_matrix, min_similarity=self.min_similarity,
                )
                H, W = view.rgb.shape[:2]
                full = extractor.upsample_ids(id_grid, (H, W))
                label_masks = {
                    proto_labels[li]: mask
                    for li in range(len(proto_labels))
                    if (mask := (full == li)).any()
                }
                accumulator.add_view(verts, view, label_masks)
        else:
            log.warning(
                "dinov3_sam3: SAM3 found no regions to seed DINOv3 prototypes; "
                "result falls back to the structural shell prior only.",
            )
            for view in views:
                accumulator.add_view(verts, view, {})

        extractor.free()
        return finalize_grounding_result(
            mesh,
            accumulator,
            backend=self.name,
            min_vote_fraction=self.min_vote_fraction,
            protection_config=self.protection_config,
            graph_config=self.graph_cleanup_config,
            remove_movable=self.remove_movable,
            apply_semantic_priors=True,
            apply_rack_prior=self.apply_rack_prior,
            extra={
                "prototype_labels": list(prototypes.keys()),
                "min_similarity": self.min_similarity,
                "model": self.model_name,
                "sam3_confidence_threshold": self.sam3_confidence_threshold,
                "n_views_rendered": len(views),
            },
        )
