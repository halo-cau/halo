"""Grounded-DINO + SAM2/SAM3 segmentation backends.

Renders the aligned 3D mesh from multiple viewpoints, runs open-vocabulary
2D detection (Grounded-DINO) and pixel-precise segmentation (SAM2 or SAM3)
per view,
then backprojects the 2D masks to 3D vertex labels via the depth buffer.
Multi-view votes are aggregated per-vertex; the majority structural / non-
structural label wins.

Requirements
------------
    pip install groundingdino-py
    pip install sam2            # optional: Meta's SAM2 package
    pip install sam3            # optional: Meta's SAM3 package
    pip install pyrender        # headless mesh rasteriser (EGL / OSMesa)

For headless rendering set:
    export PYOPENGL_PLATFORM=egl    # on GPU servers
    # OR
    export PYOPENGL_PLATFORM=osmesa # on CPU-only / remote

Usage
-----
    from engine.vision.segmentor_dino_sam import DinoSam3Segmentor
    seg = DinoSam3Segmentor(
        grounding_dino_config="path/to/GroundingDINO_SwinT_OGC.py",
        grounding_dino_checkpoint="path/to/groundingdino_swint_ogc.pth",
        sam3_checkpoint="path/to/sam3.pt",
    )
    result = seg.run(mesh_aligned, source_path)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import open3d as o3d
import torch
from PIL import Image

from engine.vision.segmentor_base import (
    PRESERVED_LABELS,
    STRUCTURAL_LABELS,
    BaseSegmentor,
    SegmentorResult,
    label_to_color,
    strip_non_structural,
)
from engine.vision.graph_fusion import GraphCleanupConfig, apply_graph_cleanup
from engine.vision.structural_priors import (
    StructuralProtectionConfig,
    apply_structural_protection,
    estimate_room_shell_prior,
    append_virtual_room_shell,
    flatten_structural_labels_to_room_shell,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default text prompts for open-vocab detection
# ---------------------------------------------------------------------------

DEFAULT_STRUCTURAL_PROMPTS: list[str] = [
    "wall", "floor", "ceiling",
]

DEFAULT_STATIC_INFRASTRUCTURE_PROMPTS: list[str] = [
    # Disabled by default because GroundingDINO tends to return one very large
    # combined "network rack / equipment rack / cable tray" box for this scan.
    # Rack preservation is handled by the 3D geometry prior below instead.
]

DEFAULT_MOVABLE_PROMPTS: list[str] = [
    "chair",
    "desk",
    "table",
    "door",
    "window",
    "curtain",
    "cardboard box",
    "fence",
    "monitor",
    "keyboard",
    "trash can",
    "fire extinguisher",
    "box",
]

# Single period-separated prompt string that Grounded-DINO expects
def _build_prompt(labels: list[str]) -> str:
    return ". ".join(labels) + "."


def _canonicalize_dino_label(label: str) -> str:
    """Map noisy GroundingDINO phrases to stable project labels."""
    text = label.lower().replace(".", " ").replace(",", " ").strip()
    text = " ".join(text.split())
    if "rack" in text:
        return "server rack"
    if "cable tray" in text:
        return "cable tray"
    if "cabinet" in text:
        return "cabinet"
    if "air conditioning" in text or "ac unit" in text or text == "unit":
        return "air conditioning unit"
    if "cardboard" in text or text == "box":
        return "cardboard box"
    if "trash" in text:
        return "trash can"
    if "fire extinguisher" in text:
        return "fire extinguisher"
    if "monitor" in text:
        return "monitor"
    if "keyboard" in text:
        return "keyboard"
    if "chair" in text:
        return "chair"
    if "desk" in text:
        return "desk"
    if "table" in text:
        return "table"
    if "door" in text:
        return "door"
    if "window" in text:
        return "window"
    if "curtain" in text:
        return "curtain"
    if "fence" in text:
        return "fence"
    if "wall" in text:
        return "wall"
    if "floor" in text:
        return "floor"
    if "ceiling" in text:
        return "ceiling"
    return text or "unknown"


def _apply_server_rack_geometry_prior(
    mesh: o3d.geometry.TriangleMesh,
    vertex_labels: list[str],
    prior,
    min_shell_distance_m: float = 0.35,
    max_normal_z: float = 0.45,
    max_brightness: float = 0.52,
) -> tuple[list[str], dict]:
    """Protect likely server-rack front faces using 3D evidence.

    Racks are interior, mostly vertical, dark surfaces.  This avoids relying on
    ScanNet classes or overbroad open-vocabulary DINO boxes.
    """
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    if len(verts) == 0:
        return vertex_labels, {"enabled": True, "n_labeled": 0}
    mesh.compute_vertex_normals()
    normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
    if len(normals) != len(verts):
        normals = np.zeros_like(verts)

    z = verts[:, 2]
    ceiling = float(prior.bounds_max[2])
    distance = np.asarray(prior.distance_to_shell, dtype=np.float64)
    interior_vertical = (
        (distance >= min_shell_distance_m)
        & (z >= 0.20)
        & (z <= ceiling - 0.15)
        & (np.abs(normals[:, 2]) <= max_normal_z)
    )

    if mesh.has_vertex_colors():
        colors = np.asarray(mesh.vertex_colors, dtype=np.float64)
        brightness = colors.mean(axis=1) if len(colors) == len(verts) else np.ones(len(verts))
        interior_vertical &= brightness <= max_brightness

    labels = list(vertex_labels)
    n_labeled = 0
    for vi in np.where(interior_vertical)[0]:
        if labels[int(vi)] != "server rack":
            labels[int(vi)] = "server rack"
            n_labeled += 1

    return labels, {
        "enabled": True,
        "n_labeled": n_labeled,
        "min_shell_distance_m": min_shell_distance_m,
        "max_normal_z": max_normal_z,
        "max_brightness": max_brightness,
    }


# ---------------------------------------------------------------------------
# Camera rig: positions around and above the scene
# ---------------------------------------------------------------------------

def _build_camera_poses(
    center: np.ndarray,
    radius: float,
    n_horizontal: int = 8,
    n_oblique: int = 4,
) -> list[np.ndarray]:
    """Return a list of 4×4 camera-to-world matrices surrounding `center`.

    Cameras always look toward `center`.  Includes:
    - 1 top-down view
    - `n_horizontal` views at eye-level (30° elevation)
    - `n_oblique` views at 60° elevation
    """
    poses: list[np.ndarray] = []

    def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray = np.array([0, 0, 1])) -> np.ndarray:
        z = eye - target; z /= np.linalg.norm(z)
        x = np.cross(up, z);
        if np.linalg.norm(x) < 1e-6:
            up = np.array([0, 1, 0])
            x = np.cross(up, z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        mat = np.eye(4)
        mat[:3, :3] = np.stack([x, y, z], axis=1)
        mat[:3, 3]  = eye
        return mat

    # Top-down
    poses.append(_look_at(center + np.array([0, 0, radius]), center))

    for ring_elev, n_ring in [(30, n_horizontal), (60, n_oblique)]:
        elev_rad = np.radians(ring_elev)
        for i in range(n_ring):
            az = 2 * np.pi * i / n_ring
            eye = center + radius * np.array([
                np.cos(az) * np.cos(elev_rad),
                np.sin(az) * np.cos(elev_rad),
                np.sin(elev_rad),
            ])
            poses.append(_look_at(eye, center))

    return poses


# ---------------------------------------------------------------------------
# Renderer wrapper (pyrender)
# ---------------------------------------------------------------------------

@dataclass
class _RenderView:
    rgb:   np.ndarray   # (H, W, 3) uint8
    depth: np.ndarray   # (H, W) float32, metres; 0 = no geometry
    proj:  np.ndarray   # (4, 4) projection matrix
    view:  np.ndarray   # (4, 4) view (world→camera) matrix


def _render_views(
    mesh: o3d.geometry.TriangleMesh,
    poses: list[np.ndarray],
    width: int = 800,
    height: int = 600,
    fov_deg: float = 60.0,
) -> list[_RenderView]:
    """Render RGB + depth from each camera pose using pyrender."""
    try:
        import pyrender  # type: ignore[import]
        import trimesh as tm  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "pyrender and trimesh are required for the DINO+SAM2 backend. "
            "Install with: pip install pyrender trimesh. Error: " + str(exc)
        ) from exc

    # Convert Open3D mesh → trimesh → pyrender
    verts  = np.asarray(mesh.vertices)
    tris   = np.asarray(mesh.triangles)
    colors = (np.asarray(mesh.vertex_colors) * 255).astype(np.uint8) \
             if mesh.has_vertex_colors() else None

    tm_mesh = tm.Trimesh(vertices=verts, faces=tris, vertex_colors=colors)
    pr_mesh = pyrender.Mesh.from_trimesh(tm_mesh)

    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.6, 0.6, 0.6])
    mesh_node = scene.add(pr_mesh)

    camera = pyrender.PerspectiveCamera(yfov=np.radians(fov_deg), aspectRatio=width / height)
    cam_node = scene.add(camera)

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    light_node = scene.add(light)

    renderer = pyrender.OffscreenRenderer(width, height)
    fx = fy = height / (2 * np.tan(np.radians(fov_deg) / 2))
    proj = np.array([
        [fx, 0, width  / 2, 0],
        [0, fy, height / 2, 0],
        [0,  0,           1, 0],
        [0,  0,           0, 1],
    ], dtype=np.float32)

    views: list[_RenderView] = []
    for c2w in poses:
        # `_look_at()` already returns a pyrender/OpenGL-compatible camera pose:
        # camera +Y is up and camera looks along local -Z.
        pose_gl = c2w

        scene.set_pose(cam_node, pose_gl)
        scene.set_pose(light_node, pose_gl)   # keep light with camera

        rgb, depth = renderer.render(scene)
        view_mat = np.linalg.inv(pose_gl)

        views.append(_RenderView(
            rgb=rgb.astype(np.uint8),
            depth=depth.astype(np.float32),
            proj=proj,
            view=view_mat,
        ))

    renderer.delete()
    return views


# ---------------------------------------------------------------------------
# Detection helpers (Grounded-DINO)
# ---------------------------------------------------------------------------

def _load_grounding_dino(config_path: str, checkpoint_path: str, device: str):
    """Load GroundingDINO model."""
    try:
        from groundingdino.util.inference import load_model  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "groundingdino-py is not installed. Run: pip install groundingdino-py. "
            "Error: " + str(exc)
        ) from exc
    return load_model(config_path, checkpoint_path, device=device)


def _detect_boxes(
    model,
    image_rgb: np.ndarray,  # (H, W, 3) uint8
    prompt: str,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Run Grounded-DINO and return (boxes_xyxy, phrases, logits)."""
    from groundingdino.util.inference import predict  # type: ignore[import]
    try:
        from groundingdino.util import transforms as T  # type: ignore[import]
    except ImportError:
        from groundingdino.datasets import transforms as T  # type: ignore[import]

    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    pil_img = Image.fromarray(image_rgb)
    img_tensor, _ = transform(pil_img, None)

    boxes, logits, phrases = predict(
        model=model,
        image=img_tensor,
        caption=prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )

    # Convert from CX/CY/W/H (normalised) → XYXY (pixel)
    H, W = image_rgb.shape[:2]
    if len(boxes) == 0:
        return np.empty((0, 4)), [], np.empty(0)

    cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = ((cx - bw / 2) * W).numpy().astype(np.int32)
    y1 = ((cy - bh / 2) * H).numpy().astype(np.int32)
    x2 = ((cx + bw / 2) * W).numpy().astype(np.int32)
    y2 = ((cy + bh / 2) * H).numpy().astype(np.int32)
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    return boxes_xyxy, phrases, logits.numpy()


# ---------------------------------------------------------------------------
# SAM2 mask prediction
# ---------------------------------------------------------------------------

def _load_sam2(checkpoint: str, config: str, device: str):
    try:
        from sam2.build_sam import build_sam2           # type: ignore[import]
        from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "sam2 is not installed. Run: pip install sam2. Error: " + str(exc)
        ) from exc

    sam2_model = build_sam2(config, checkpoint, device=device)
    return SAM2ImagePredictor(sam2_model)


def _load_sam3(checkpoint: str | None, device: str, confidence_threshold: float):
    """Load SAM3 image model + processor.

    SAM3 checkpoints are gated on Hugging Face.  If *checkpoint* is None, the
    upstream builder attempts to download ``facebook/sam3`` and will raise a
    clear authentication error if the user has not accepted/accessed the model.
    """
    try:
        from sam3 import build_sam3_image_model  # type: ignore[import]
        from sam3.model.sam3_image_processor import Sam3Processor  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "sam3 is not installed. Run: pip install sam3. Error: " + str(exc)
        ) from exc

    model = build_sam3_image_model(
        device=device,
        eval_mode=True,
        checkpoint_path=checkpoint,
        load_from_HF=checkpoint is None,
        enable_segmentation=True,
        enable_inst_interactivity=False,
    )
    return Sam3Processor(
        model,
        device=device,
        confidence_threshold=confidence_threshold,
    )


def _predict_masks_sam2(
    predictor,
    image_rgb: np.ndarray,
    boxes_xyxy: np.ndarray,
) -> np.ndarray:
    """Return binary masks (N, H, W) bool for each box."""
    if len(boxes_xyxy) == 0:
        return np.empty((0, *image_rgb.shape[:2]), dtype=bool)

    predictor.set_image(image_rgb)
    boxes_tensor = torch.tensor(boxes_xyxy, dtype=torch.float32)

    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=boxes_tensor,
        multimask_output=False,
    )
    masks_np = np.asarray(masks)
    if masks_np.ndim == 4:
        masks_np = masks_np[:, 0, :, :]
    elif masks_np.ndim == 2:
        masks_np = masks_np[None, :, :]
    if masks_np.ndim != 3:
        raise ValueError(f"Unexpected SAM2 mask shape: {masks_np.shape}")
    return masks_np.astype(bool)   # (N, H, W)


def _clone_sam3_image_state(state: dict) -> dict:
    """Cheaply clone SAM3's per-image state for per-box prompting."""
    cloned = dict(state)
    if "backbone_out" in cloned:
        cloned["backbone_out"] = dict(cloned["backbone_out"])
    return cloned


def _sam3_masks_to_numpy(state: dict, image_shape: tuple[int, int]) -> np.ndarray:
    """Extract SAM3 boolean masks from a processor state as (N, H, W)."""
    masks = state.get("masks")
    if masks is None:
        return np.empty((0, *image_shape), dtype=bool)
    if isinstance(masks, torch.Tensor):
        masks_np = masks.detach().cpu().numpy()
    else:
        masks_np = np.asarray(masks)

    # SAM3 commonly returns (N, 1, H, W); tolerate (N, H, W) and (H, W).
    if masks_np.ndim == 4:
        masks_np = masks_np[:, 0, :, :]
    elif masks_np.ndim == 2:
        masks_np = masks_np[None, :, :]
    return masks_np.astype(bool)


def _sam3_scores_to_numpy(state: dict, n_masks: int) -> np.ndarray:
    scores = state.get("scores")
    if scores is None:
        return np.ones(n_masks, dtype=np.float32)
    if isinstance(scores, torch.Tensor):
        scores_np = scores.detach().cpu().numpy()
    else:
        scores_np = np.asarray(scores)
    return scores_np.reshape(-1).astype(np.float32)


def _xyxy_to_normalized_cxcywh(box_xyxy: np.ndarray, width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = box_xyxy.astype(float).tolist()
    x1 = float(np.clip(x1, 0, width - 1))
    x2 = float(np.clip(x2, 0, width - 1))
    y1 = float(np.clip(y1, 0, height - 1))
    y2 = float(np.clip(y2, 0, height - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    bw = max(1.0, x2 - x1) / width
    bh = max(1.0, y2 - y1) / height
    cx = ((x1 + x2) * 0.5) / width
    cy = ((y1 + y2) * 0.5) / height
    return [cx, cy, bw, bh]


def _predict_masks_sam3(
    processor,
    image_rgb: np.ndarray,
    boxes_xyxy: np.ndarray,
    phrases: list[str],
) -> np.ndarray:
    """Return one SAM3 mask per GroundingDINO box.

    GroundingDINO supplies the class phrase and coarse box.  SAM3 receives both
    the phrase and normalized box as prompts, then the highest-scoring mask is
    kept for that DINO detection.  This keeps labels deterministic while using
    SAM3 for pixel-accurate masks.
    """
    H, W = image_rgb.shape[:2]
    if len(boxes_xyxy) == 0:
        return np.empty((0, H, W), dtype=bool)

    # SAM3's ndarray path currently reads H/W from image.shape[-2:], so use PIL.
    pil_image = Image.fromarray(image_rgb)
    base_state = processor.set_image(pil_image)

    out_masks: list[np.ndarray] = []
    for box, phrase in zip(boxes_xyxy, phrases):
        state = _clone_sam3_image_state(base_state)
        # Text prompt is useful when SAM3 has to disambiguate multiple objects
        # inside one GroundingDINO box.  If text grounding fails, the geometric
        # box prompt still provides a precise fallback.
        try:
            state = processor.set_text_prompt(str(phrase), state)
        except Exception:
            state = _clone_sam3_image_state(base_state)

        norm_box = _xyxy_to_normalized_cxcywh(box, W, H)
        state = processor.add_geometric_prompt(norm_box, True, state)
        masks = _sam3_masks_to_numpy(state, (H, W))
        if len(masks) == 0:
            out_masks.append(np.zeros((H, W), dtype=bool))
            continue
        scores = _sam3_scores_to_numpy(state, len(masks))
        best_idx = int(np.argmax(scores[: len(masks)])) if len(scores) else 0
        out_masks.append(masks[best_idx])

    return np.stack(out_masks, axis=0).astype(bool)


# ---------------------------------------------------------------------------
# Backprojection: 2D mask pixel → 3D vertex
# ---------------------------------------------------------------------------

def _backproject_masks_to_vertices(
    mesh_verts: np.ndarray,        # (V, 3) float32
    view: _RenderView,
    box_phrases: list[tuple[str, np.ndarray]],   # [(label, mask HxW bool), ...]
    depth_tolerance: float = 0.05,               # metres
) -> dict[str, np.ndarray]:
    """Map 2D segmentation masks back to mesh vertex indices.

    Returns dict: label → bool mask over vertices (V,).
    """
    H, W = view.depth.shape
    px, py, depth_valid = _visible_vertices_in_view(mesh_verts, view, depth_tolerance)

    label_vertex_masks: dict[str, np.ndarray] = {}
    for label, mask2d in box_phrases:
        label = _canonicalize_dino_label(label)
        vert_mask = np.zeros(len(mesh_verts), dtype=bool)
        # Vectorised: check which depth-valid vertices fall inside this 2D mask
        valid_idx = np.where(depth_valid)[0]
        if len(valid_idx) == 0:
            if label in label_vertex_masks:
                label_vertex_masks[label] |= vert_mask
            else:
                label_vertex_masks[label] = vert_mask
            continue
        vy = py[valid_idx]
        vx = px[valid_idx]
        vy_c = np.clip(vy, 0, H - 1)
        vx_c = np.clip(vx, 0, W - 1)
        in_mask = mask2d[vy_c, vx_c]
        vert_mask[valid_idx[in_mask]] = True
        if label in label_vertex_masks:
            label_vertex_masks[label] |= vert_mask
        else:
            label_vertex_masks[label] = vert_mask

    return label_vertex_masks


def _visible_vertices_in_view(
    mesh_verts: np.ndarray,
    view: _RenderView,
    depth_tolerance: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project vertices and return pixel coordinates plus visibility mask."""
    H, W = view.depth.shape
    vm = view.view
    proj = view.proj

    verts_h = np.hstack([mesh_verts, np.ones((len(mesh_verts), 1))])
    cam_coords = (vm @ verts_h.T).T
    z_depth = -cam_coords[:, 2]
    fx = float(proj[0, 0])
    fy = float(proj[1, 1])
    cx = float(proj[0, 2])
    cy = float(proj[1, 2])
    z_safe = np.where(np.abs(z_depth) < 1e-6, 1e-6, z_depth)
    px = (fx * (cam_coords[:, 0] / z_safe) + cx).astype(np.int32)
    py = (fy * (-cam_coords[:, 1] / z_safe) + cy).astype(np.int32)

    visible = (
        (z_depth > 0)
        & (px >= 0) & (px < W)
        & (py >= 0) & (py < H)
    )
    depth_at_px = np.where(
        visible,
        view.depth[np.clip(py, 0, H - 1), np.clip(px, 0, W - 1)],
        0.0,
    )
    depth_valid = visible & (np.abs(depth_at_px - z_depth) < depth_tolerance) & (depth_at_px > 0)
    return px, py, depth_valid


# ---------------------------------------------------------------------------
# Main segmentor class
# ---------------------------------------------------------------------------

class DinoSam2Segmentor(BaseSegmentor):
    """Multi-view Grounded-DINO + SAM2 segmentation backend."""

    def __init__(
        self,
        grounding_dino_config: str | Path,
        grounding_dino_checkpoint: str | Path,
        sam2_checkpoint: str | Path,
        sam2_config: str = "sam2_hiera_l.yaml",
        device: str | None = None,
        render_width: int = 800,
        render_height: int = 600,
        n_horizontal: int = 8,
        n_oblique: int = 4,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        min_vote_fraction: float = 0.3,
        min_movable_votes: int = 2,
        structural_prompts: list[str] | None = None,
        static_prompts: list[str] | None = None,
        movable_prompts: list[str] | None = None,
        protection_config: StructuralProtectionConfig | None = None,
        graph_cleanup_config: GraphCleanupConfig | None = None,
        static_min_shell_distance_m: float = 0.18,
        remove_movable: bool = False,
    ) -> None:
        self.dino_config = str(grounding_dino_config)
        self.dino_checkpoint = str(grounding_dino_checkpoint)
        self.sam2_checkpoint = str(sam2_checkpoint)
        self.sam2_config = sam2_config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.render_width = render_width
        self.render_height = render_height
        self.n_horizontal = n_horizontal
        self.n_oblique = n_oblique
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.min_vote_fraction = min_vote_fraction
        self.min_movable_votes = min_movable_votes
        self.protection_config = protection_config or StructuralProtectionConfig()
        self.graph_cleanup_config = graph_cleanup_config or GraphCleanupConfig()
        self.static_min_shell_distance_m = static_min_shell_distance_m
        self.remove_movable = remove_movable

        self._structural_prompts = structural_prompts or DEFAULT_STRUCTURAL_PROMPTS
        self._static_prompts = static_prompts if static_prompts is not None else DEFAULT_STATIC_INFRASTRUCTURE_PROMPTS
        self._movable_prompts = movable_prompts or DEFAULT_MOVABLE_PROMPTS

        self._dino_model = None
        self._sam_model = None

    @property
    def name(self) -> str:
        return "dino_sam2"

    def _ensure_models(self) -> None:
        if self._dino_model is None:
            self._dino_model = _load_grounding_dino(
                self.dino_config, self.dino_checkpoint, self.device
            )
        if self._sam_model is None:
            self._sam_model = _load_sam2(
                self.sam2_checkpoint, self.sam2_config, self.device
            )

    def _predict_masks(
        self,
        image_rgb: np.ndarray,
        boxes: np.ndarray,
        phrases: list[str],
    ) -> np.ndarray:
        return _predict_masks_sam2(self._sam_model, image_rgb, boxes)

    def run(
        self,
        mesh: o3d.geometry.TriangleMesh,
        source_path: Path,
    ) -> SegmentorResult:
        self._ensure_models()
        mesh.compute_vertex_normals()

        verts = np.asarray(mesh.vertices, dtype=np.float32)
        n_total = len(verts)
        bbox = mesh.get_axis_aligned_bounding_box()
        center = np.asarray(bbox.get_center(), dtype=np.float64)
        extent = np.asarray(bbox.get_extent(), dtype=np.float64)
        radius = float(np.linalg.norm(extent)) * 0.9

        log.info(
            "Rendering %d views (radius=%.2f m) …",
            1 + self.n_horizontal + self.n_oblique,
            radius,
        )
        poses = _build_camera_poses(center, radius, self.n_horizontal, self.n_oblique)
        views = _render_views(mesh, poses, self.render_width, self.render_height)

        votes_static = np.zeros(n_total, dtype=np.int32)
        votes_movable = np.zeros(n_total, dtype=np.int32)
        votes_seen = np.zeros(n_total, dtype=np.int32)
        best_static_label = ["unknown"] * n_total
        best_movable_label = ["unknown"] * n_total

        static_prompt = _build_prompt(self._static_prompts) if self._static_prompts else ""
        movable_prompt = _build_prompt(self._movable_prompts)

        for view_idx, view in enumerate(views):
            log.debug("  View %d / %d", view_idx + 1, len(views))
            _, _, visible_vertices = _visible_vertices_in_view(verts, view)
            votes_seen[visible_vertices] += 1
            prompt_jobs = [(movable_prompt, votes_movable, best_movable_label)]
            if static_prompt:
                prompt_jobs.insert(0, (static_prompt, votes_static, best_static_label))
            for prompt, vote_arr, best_arr in prompt_jobs:
                boxes, phrases, _ = _detect_boxes(
                    self._dino_model,
                    view.rgb,
                    prompt,
                    self.box_threshold,
                    self.text_threshold,
                )
                if len(boxes) == 0:
                    continue

                masks = self._predict_masks(view.rgb, boxes, phrases)
                if len(masks) == 0:
                    continue

                phrase_mask_pairs = list(zip(phrases, masks))
                label_verts = _backproject_masks_to_vertices(verts, view, phrase_mask_pairs)

                for label, vert_mask in label_verts.items():
                    vote_arr[vert_mask] += 1
                    for vi in np.where(vert_mask)[0]:
                        if best_arr[vi] == "unknown":
                            best_arr[vi] = label

        prior_config = self.protection_config
        if prior_config.level == "off":
            prior_config = StructuralProtectionConfig(
                level="balanced",
                outer_percentile=prior_config.outer_percentile,
                plane_tolerance_m=prior_config.plane_tolerance_m,
                normal_cos_min=prior_config.normal_cos_min,
            )
        pre_prior = estimate_room_shell_prior(mesh, prior_config)
        vertex_labels: list[str] = []
        z_max = float(verts[:, 2].max()) if n_total else 0.0
        for vi in range(n_total):
            if votes_static[vi] > 0 and (
                votes_static[vi] / max(1, votes_seen[vi]) >= self.min_vote_fraction
                and float(pre_prior.distance_to_shell[vi]) >= self.static_min_shell_distance_m
            ):
                vertex_labels.append(best_static_label[vi])
            elif votes_movable[vi] > 0 and (
                votes_movable[vi] >= self.min_movable_votes
                and votes_movable[vi] / max(1, votes_seen[vi]) >= self.min_vote_fraction
            ):
                vertex_labels.append(best_movable_label[vi])
            elif pre_prior.protected_mask[vi] and pre_prior.protected_labels[vi] in STRUCTURAL_LABELS:
                vertex_labels.append(pre_prior.protected_labels[vi])
            else:
                vertex_labels.append("unknown")

        vertex_labels, rack_prior_stats = _apply_server_rack_geometry_prior(
            mesh,
            vertex_labels,
            pre_prior,
        )

        protection = apply_structural_protection(
            mesh,
            vertex_labels,
            self.protection_config,
            movable_votes=votes_movable,
        )
        vertex_labels = protection.vertex_labels
        if protection.n_restored:
            log.info(
                "%s: structural prior restored %d protected shell vertices",
                self.name,
                protection.n_restored,
            )

        graph_cleanup = apply_graph_cleanup(
            mesh,
            vertex_labels,
            protection.prior,
            self.graph_cleanup_config,
        )
        vertex_labels = graph_cleanup.vertex_labels
        if graph_cleanup.n_graph_restored or graph_cleanup.n_graph_pruned:
            log.info(
                "%s: graph cleanup restored %d and pruned %d vertices",
                self.name,
                graph_cleanup.n_graph_restored,
                graph_cleanup.n_graph_pruned,
            )

        flattened_mesh, flatten_stats = flatten_structural_labels_to_room_shell(
            mesh,
            vertex_labels,
            protection.prior,
        )
        review_mesh, virtual_shell_stats = append_virtual_room_shell(flattened_mesh, protection.prior)

        label_colors = np.array(
            [label_to_color(lbl) for lbl in vertex_labels], dtype=np.float32
        )
        label_map: dict[str, list[int]] = {}
        for vi, lbl in enumerate(vertex_labels):
            label_map.setdefault(lbl, []).append(vi)

        n_removable_candidates = sum(1 for lbl in vertex_labels if lbl not in PRESERVED_LABELS)
        n_removed = n_removable_candidates if self.remove_movable else 0
        log.info(
            "%s: %d / %d vertices explicit movable candidates (%.1f%%) — %s",
            self.name,
            n_removable_candidates,
            n_total,
            100.0 * n_removable_candidates / max(1, n_total),
            "removing" if self.remove_movable else "preserving for review",
        )

        structural_mesh = (
            append_virtual_room_shell(strip_non_structural(flattened_mesh, vertex_labels), protection.prior)[0]
            if self.remove_movable
            else review_mesh
        )

        return SegmentorResult(
            structural_mesh=structural_mesh,
            vertex_labels=vertex_labels,
            label_colors=label_colors,
            label_map=label_map,
            n_removed=n_removed,
            backend=self.name,
            extra={
                "votes_movable": votes_movable.tolist(),
                "votes_static": votes_static.tolist(),
                "votes_seen": votes_seen.tolist(),
                "n_views": len(views),
                "structural_protection": protection.to_json(),
                "graph_cleanup": graph_cleanup.to_json(),
                "structural_flattening": flatten_stats,
                "virtual_room_shell": virtual_shell_stats,
                "server_rack_geometry_prior": rack_prior_stats,
                "min_movable_votes": self.min_movable_votes,
                "remove_movable": self.remove_movable,
                "n_removable_candidates": n_removable_candidates,
            },
        )


class DinoSam3Segmentor(DinoSam2Segmentor):
    """Multi-view Grounded-DINO + SAM3 segmentation backend."""

    def __init__(
        self,
        grounding_dino_config: str | Path,
        grounding_dino_checkpoint: str | Path,
        sam3_checkpoint: str | Path | None = None,
        device: str | None = None,
        render_width: int = 800,
        render_height: int = 600,
        n_horizontal: int = 8,
        n_oblique: int = 4,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        min_vote_fraction: float = 0.3,
        min_movable_votes: int = 2,
        sam3_confidence_threshold: float = 0.5,
        structural_prompts: list[str] | None = None,
        static_prompts: list[str] | None = None,
        movable_prompts: list[str] | None = None,
        protection_config: StructuralProtectionConfig | None = None,
        graph_cleanup_config: GraphCleanupConfig | None = None,
        remove_movable: bool = False,
    ) -> None:
        super().__init__(
            grounding_dino_config=grounding_dino_config,
            grounding_dino_checkpoint=grounding_dino_checkpoint,
            sam2_checkpoint="",
            sam2_config="",
            device=device,
            render_width=render_width,
            render_height=render_height,
            n_horizontal=n_horizontal,
            n_oblique=n_oblique,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            min_vote_fraction=min_vote_fraction,
            min_movable_votes=min_movable_votes,
            structural_prompts=structural_prompts,
            static_prompts=static_prompts,
            movable_prompts=movable_prompts,
            protection_config=protection_config,
            graph_cleanup_config=graph_cleanup_config,
            remove_movable=remove_movable,
        )
        self.sam3_checkpoint = str(sam3_checkpoint) if sam3_checkpoint else None
        self.sam3_confidence_threshold = sam3_confidence_threshold

    @property
    def name(self) -> str:
        return "dino_sam3"

    def _ensure_models(self) -> None:
        if self._dino_model is None:
            self._dino_model = _load_grounding_dino(
                self.dino_config, self.dino_checkpoint, self.device
            )
        if self._sam_model is None:
            self._sam_model = _load_sam3(
                self.sam3_checkpoint,
                self.device,
                self.sam3_confidence_threshold,
            )

    def _predict_masks(
        self,
        image_rgb: np.ndarray,
        boxes: np.ndarray,
        phrases: list[str],
    ) -> np.ndarray:
        return _predict_masks_sam3(self._sam_model, image_rgb, boxes, phrases)
