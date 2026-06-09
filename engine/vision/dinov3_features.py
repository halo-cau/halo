"""DINOv3 dense-feature extraction and training-free labeling helpers.

DINOv3 (Meta, 2025) is a **self-supervised ViT backbone**: it emits dense
per-patch feature vectors but has *no* language head and *no* detection head.
It is therefore not a drop-in replacement for GroundingDINO (a text->box
detector). To turn DINOv3 features into named labels we need either

  * **prototypes** — a mean feature vector per class, computed from a handful
    of exemplar regions (here: seeded from SAM3 concept masks), then every
    patch is labeled by nearest-prototype cosine similarity, or
  * **unsupervised clusters** — k-means over patch features, producing
    *unnamed* groups the user labels by eye (tests feature separability of
    racks / AC / shell without any supervision).

This module is intentionally numpy-facing: ``torch``/``transformers`` are
imported lazily inside the loader so the module can be imported in a base
environment without the heavy ML stack.

Loading
-------
DINOv3 ships in ``transformers`` (>= 4.53) as ``DINOv3ViTModel`` /
``DINOv3ViTImageProcessorFast`` (model_type ``dinov3_vit``), so it loads via
the Auto* classes::

    AutoModel.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m")
    AutoImageProcessor.from_pretrained(...)

The checkpoints are gated on Hugging Face — accept the license and provide a
token (``HF_TOKEN`` env or ``huggingface-cli login``) before first use.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

#: Default DINOv3 ViT checkpoint. Base/16 fits comfortably in 8 GB VRAM for
#: inference and is fast enough for a multi-view test; bump to ``vitl16`` for
#: stronger features when memory allows.
DEFAULT_DINOV3_MODEL = "facebook/dinov3-vitb16-pretrain-lvd1689m"

#: Distinct RGB palette (in [0, 1]) for coloring unsupervised k-means clusters
#: in the viewer. 20 visually separable hues (tab20-style).
CLUSTER_PALETTE: tuple[tuple[float, float, float], ...] = (
    (0.122, 0.467, 0.706), (1.000, 0.498, 0.055), (0.173, 0.627, 0.173),
    (0.839, 0.153, 0.157), (0.580, 0.404, 0.741), (0.549, 0.337, 0.294),
    (0.890, 0.467, 0.761), (0.498, 0.498, 0.498), (0.737, 0.741, 0.133),
    (0.090, 0.745, 0.812), (0.682, 0.780, 0.910), (1.000, 0.733, 0.471),
    (0.596, 0.875, 0.541), (1.000, 0.596, 0.588), (0.773, 0.690, 0.835),
    (0.769, 0.612, 0.580), (0.969, 0.714, 0.824), (0.780, 0.780, 0.780),
    (0.859, 0.859, 0.553), (0.620, 0.855, 0.898),
)


def cluster_color(cluster_id: int) -> tuple[float, float, float]:
    """Deterministic RGB for an unsupervised cluster id."""
    if cluster_id < 0:
        return (0.1, 0.1, 0.1)
    return CLUSTER_PALETTE[cluster_id % len(CLUSTER_PALETTE)]


class Dinov3FeatureExtractor:
    """Dense DINOv3 patch features + prototype / cluster labeling.

    All public methods take/return numpy arrays. One forward pass per image;
    load a single model instance and reuse it across rendered views.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_DINOV3_MODEL,
        device: str | None = None,
        half: bool = True,
        hf_token: str | None = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise ImportError(
                "DINOv3 needs torch + transformers (>=4.53). Install the "
                "vision-ai stack: pip install -r requirements-vision-ai.txt. "
                f"Error: {exc}"
            ) from exc

        self._torch = torch
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # fp16 only helps on CUDA; keep fp32 on CPU for correctness.
        self.dtype = torch.float16 if (half and self.device == "cuda") else torch.float32
        token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

        logger.info("Loading DINOv3 %s on %s (%s)…", model_name, self.device, self.dtype)
        self.processor = AutoImageProcessor.from_pretrained(model_name, token=token)
        self.model = (
            AutoModel.from_pretrained(model_name, token=token, torch_dtype=self.dtype)
            .eval()
            .to(self.device)
        )
        cfg = self.model.config
        self.patch_size = int(getattr(cfg, "patch_size", 16))
        self.n_register = int(getattr(cfg, "num_register_tokens", 0))
        self.hidden_size = int(getattr(cfg, "hidden_size", 768))

    # -- core feature extraction --------------------------------------------

    def patch_features(self, image_rgb: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        """Return L2-normalized patch features for one RGB image.

        Parameters
        ----------
        image_rgb:
            ``(H, W, 3)`` uint8 array.

        Returns
        -------
        ``(feat_grid, (Hp, Wp))`` where ``feat_grid`` is ``(Hp, Wp, D)``
        float32, unit-normalized along D. ``Hp/Wp`` are the patch-grid dims at
        the processor's internal resolution (not the input H/W).
        """
        torch = self._torch
        from PIL import Image

        pil = Image.fromarray(np.ascontiguousarray(image_rgb[..., :3]))
        inputs = self.processor(images=pil, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device, self.dtype)
        ph, pw = int(pixel_values.shape[-2]), int(pixel_values.shape[-1])
        hp, wp = ph // self.patch_size, pw // self.patch_size

        with torch.no_grad():
            out = self.model(pixel_values=pixel_values)
        tokens = out.last_hidden_state[0]  # (seq, D)
        n_patches = hp * wp
        seq = tokens.shape[0]
        # Layout is [CLS] + [registers] + [patches]; tolerate odd configs by
        # falling back to the trailing n_patches tokens.
        start = seq - n_patches if seq >= n_patches else 0
        if start != 1 + self.n_register:
            logger.debug(
                "DINOv3 token layout: seq=%d expected_start=%d using_start=%d",
                seq, 1 + self.n_register, start,
            )
        patches = tokens[start:start + n_patches].float()  # (n_patches, D)
        patches = torch.nn.functional.normalize(patches, dim=-1)
        grid = patches.reshape(hp, wp, -1).cpu().numpy().astype(np.float32)
        return grid, (hp, wp)

    # -- prototype labeling -------------------------------------------------

    @staticmethod
    def assign_by_prototypes(
        feat_grid: np.ndarray,           # (Hp, Wp, D), unit-normalized
        proto_matrix: np.ndarray,        # (K, D), unit-normalized
        min_similarity: float = 0.0,
    ) -> np.ndarray:
        """Per-patch nearest-prototype ids via cosine similarity.

        Returns ``(Hp, Wp)`` int array; patches whose best similarity is below
        ``min_similarity`` are set to ``-1`` (unknown).
        """
        hp, wp, d = feat_grid.shape
        flat = feat_grid.reshape(-1, d)
        sims = flat @ proto_matrix.T          # (HpWp, K), cosine (both unit norm)
        ids = sims.argmax(axis=1)
        if min_similarity > 0.0:
            best = sims[np.arange(len(ids)), ids]
            ids = np.where(best >= min_similarity, ids, -1)
        return ids.reshape(hp, wp)

    @staticmethod
    def pool_mask_feature(feat_grid: np.ndarray, mask_hw: np.ndarray) -> np.ndarray | None:
        """Mean (unit-normalized) feature of patches covered by a 2D mask.

        ``mask_hw`` is a full-resolution boolean mask; it is nearest-resized to
        the patch grid. Returns ``None`` if no patch is covered.
        """
        import cv2

        hp, wp, d = feat_grid.shape
        small = cv2.resize(
            mask_hw.astype(np.uint8), (wp, hp), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        if not small.any():
            return None
        vec = feat_grid[small].mean(axis=0)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 1e-8 else None

    # -- unsupervised clustering --------------------------------------------

    @staticmethod
    def fit_kmeans(
        feat_grids: Sequence[np.ndarray],
        k: int = 8,
        sample_per_image: int = 4000,
        seed: int = 0,
    ) -> np.ndarray:
        """Fit one k-means over patch features pooled across views.

        Returns ``(k, D)`` unit-normalized centroids so cluster ids are shared
        across all views.
        """
        from sklearn.cluster import KMeans

        rng = np.random.default_rng(seed)
        chunks: list[np.ndarray] = []
        for g in feat_grids:
            flat = g.reshape(-1, g.shape[-1])
            if len(flat) > sample_per_image:
                flat = flat[rng.choice(len(flat), sample_per_image, replace=False)]
            chunks.append(flat)
        data = np.concatenate(chunks, axis=0)
        km = KMeans(n_clusters=k, random_state=seed, n_init=4)
        km.fit(data)
        centroids = km.cluster_centers_.astype(np.float32)
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        return centroids / np.clip(norms, 1e-8, None)

    @staticmethod
    def assign_kmeans(feat_grid: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Per-patch cluster ids (nearest centroid, cosine)."""
        hp, wp, d = feat_grid.shape
        sims = feat_grid.reshape(-1, d) @ centroids.T
        return sims.argmax(axis=1).reshape(hp, wp)

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def upsample_ids(id_grid: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
        """Nearest-upsample a patch-grid int id map to full image resolution."""
        import cv2

        h, w = out_hw
        return cv2.resize(
            id_grid.astype(np.int32), (w, h), interpolation=cv2.INTER_NEAREST
        ).astype(np.int32)

    def free(self) -> None:
        """Release the model and CUDA cache (useful before loading SAM3)."""
        try:
            self.model = None
            if self.device == "cuda":
                self._torch.cuda.empty_cache()
        except Exception:  # pragma: no cover
            pass


def build_proto_matrix(prototypes: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    """Stack a ``label -> vector`` dict into ``(matrix (K, D), labels)``."""
    labels = list(prototypes.keys())
    matrix = np.stack([prototypes[l] for l in labels], axis=0).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.clip(norms, 1e-8, None)
    return matrix, labels
