"""SAM3-only grounding backend (no GroundingDINO).

SAM3 (Meta, 2025) supports *promptable concept segmentation*: given a text
concept (e.g. "server rack") it returns masks for every matching instance in
the image. That removes the need for a separate open-vocabulary detector —
where the existing ``dino_sam3`` backend used GroundingDINO to make boxes and
then prompted SAM3 with each box, this backend prompts SAM3 directly with the
target class names.

Per rendered view, for each target concept we take the union of the
above-threshold instance masks as that class's region, backproject to vertices,
vote across views, then hand off to the shared structural-prior chain. Wall /
floor / ceiling are recovered by the geometric shell prior, not prompted.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image

from engine.vision.graph_fusion import GraphCleanupConfig
from engine.vision.multiview_common import (
    GROUNDING_TARGETS,
    VoteAccumulator,
    finalize_grounding_result,
    render_mesh_views,
)
from engine.vision.segmentor_base import BaseSegmentor, SegmentorResult
from engine.vision.segmentor_dino_sam import (
    _clone_sam3_image_state,
    _load_sam3,
    _sam3_masks_to_numpy,
    _sam3_scores_to_numpy,
)
from engine.vision.structural_priors import StructuralProtectionConfig

log = logging.getLogger(__name__)


def segment_view_sam3_concept(
    processor,
    rgb: np.ndarray,
    targets: tuple[tuple[str, str], ...],
    confidence_threshold: float,
    thr_override: dict[str, float] | None = None,
) -> dict[str, np.ndarray]:
    """Run SAM3 concept segmentation on one RGB view.

    For each ``(prompt, canonical)`` target, prompt SAM3 with the text concept
    and take the union of above-threshold instance masks. Returns
    ``canonical_label -> mask (H, W) bool``. Shared by the SAM3-only backend and
    the DINOv3+SAM3 backend (which uses these regions to seed prototypes).

    ``thr_override`` maps a canonical label to a per-class score threshold,
    overriding ``confidence_threshold`` for just that concept. Use it to demand a
    higher score from semantically broad prompts (e.g. "server rack" over-fires
    on AC units / fire-hose boxes) without lowering recall on the others.
    """
    H, W = rgb.shape[:2]
    base_state = processor.set_image(Image.fromarray(rgb))
    out: dict[str, np.ndarray] = {}
    for prompt, canonical in targets:
        state = _clone_sam3_image_state(base_state)
        try:
            state = processor.set_text_prompt(str(prompt), state)
        except Exception as exc:  # noqa: BLE001 — skip this concept, keep going
            log.warning("SAM3 concept prompt %r failed: %s", prompt, exc)
            continue
        masks = _sam3_masks_to_numpy(state, (H, W))
        if len(masks) == 0:
            continue
        scores = _sam3_scores_to_numpy(state, len(masks))
        thr = (thr_override or {}).get(canonical, confidence_threshold)
        if len(scores) == len(masks):
            keep = scores >= thr
            masks = masks[keep] if keep.any() else masks[:0]
        if len(masks) == 0:
            continue
        union = np.any(masks, axis=0)
        out[canonical] = union | out[canonical] if canonical in out else union
    return out


class Sam3ConceptSegmentor(BaseSegmentor):
    """Multi-view SAM3 promptable-concept segmentation backend."""

    def __init__(
        self,
        sam3_checkpoint: str | Path | None = None,
        device: str | None = None,
        render_width: int = 800,
        render_height: int = 600,
        n_horizontal: int = 8,
        n_oblique: int = 4,
        sam3_confidence_threshold: float = 0.5,
        min_vote_fraction: float = 0.30,
        targets: tuple[tuple[str, str], ...] = GROUNDING_TARGETS,
        protection_config: StructuralProtectionConfig | None = None,
        graph_cleanup_config: GraphCleanupConfig | None = None,
        remove_movable: bool = False,
        apply_rack_prior: bool = True,
    ) -> None:
        import torch

        self.sam3_checkpoint = str(sam3_checkpoint) if sam3_checkpoint else None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.render_width = render_width
        self.render_height = render_height
        self.n_horizontal = n_horizontal
        self.n_oblique = n_oblique
        self.sam3_confidence_threshold = sam3_confidence_threshold
        self.min_vote_fraction = min_vote_fraction
        self.targets = targets
        self.protection_config = protection_config or StructuralProtectionConfig()
        self.graph_cleanup_config = graph_cleanup_config or GraphCleanupConfig()
        self.remove_movable = remove_movable
        self.apply_rack_prior = apply_rack_prior
        self._processor = None

    @property
    def name(self) -> str:
        return "sam3_concept"

    def _ensure_model(self) -> None:
        if self._processor is None:
            self._processor = _load_sam3(
                self.sam3_checkpoint, self.device, self.sam3_confidence_threshold,
            )

    def _segment_view(self, rgb: np.ndarray) -> dict[str, np.ndarray]:
        """Return ``canonical_label -> union mask (H, W) bool`` for one view."""
        return segment_view_sam3_concept(
            self._processor, rgb, self.targets, self.sam3_confidence_threshold,
        )

    def run(
        self,
        mesh: o3d.geometry.TriangleMesh,
        source_path: Path,
    ) -> SegmentorResult:
        self._ensure_model()
        views, verts = render_mesh_views(
            mesh, self.n_horizontal, self.n_oblique,
            self.render_width, self.render_height,
        )
        accumulator = VoteAccumulator(len(verts))
        for view_idx, view in enumerate(views):
            log.debug("  SAM3 concept view %d / %d", view_idx + 1, len(views))
            accumulator.add_view(verts, view, self._segment_view(view.rgb))

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
                "targets": [c for _, c in self.targets],
                "sam3_confidence_threshold": self.sam3_confidence_threshold,
                "n_views_rendered": len(views),
            },
        )
