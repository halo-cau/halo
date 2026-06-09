"""Shared helpers for Sonata (PTv3) finetune + predict on the server-room data.

Runs in the isolated `sonata` conda env (see setup_sonata_env.sh), NOT `halo`.
Keeps the encoder-load / feature-unpool / inference / PLY-export logic in one
place so the finetune and predict scripts stay small and consistent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
# Fallback so `import sonata` works even if `pip install -e opt/sonata` wasn't run.
sys.path.insert(0, str(_ROOT / "opt" / "sonata"))
import sonata  # noqa: E402

# Head label space — MUST match prepare_las6_for_sonata.py.
CLASS_NAMES = ["wall", "floor", "ceiling", "server_rack", "box_clutter", "ac_unit"]
# Colors mirror engine.vision.segmentor_base.LABEL_PALETTE so views match the pipeline.
CLASS_COLORS = {
    "wall": (0.216, 0.541, 0.867),
    "floor": (0.706, 0.698, 0.663),
    "ceiling": (0.910, 0.898, 0.867),
    "server_rack": (0.114, 0.620, 0.459),
    "box_clutter": (0.910, 0.620, 0.310),
    "ac_unit": (0.141, 0.663, 0.882),
    "ignore": (0.40, 0.39, 0.37),
}


def load_encoder(enable_flash: bool = False, patch: int = 1024, device: str = "cuda"):
    """Load the pretrained Sonata PTv3 encoder (weights auto-download from HF)."""
    custom_config = dict(enable_flash=bool(enable_flash), enc_patch_size=[patch] * 5)
    model = sonata.load("sonata", repo_id="facebook/sonata", custom_config=custom_config)
    return model.to(device)


def unpool_to_full(point):
    """Concat hierarchical encoder features back to input resolution (Sonata README)."""
    while "pooling_parent" in point.keys():
        assert "pooling_inverse" in point.keys()
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
        point = parent
    return point


def to_device(point_dict: dict, device: str) -> dict:
    for k in point_dict:
        if torch.is_tensor(point_dict[k]):
            point_dict[k] = point_dict[k].to(device, non_blocking=True)
    return point_dict


def feature_dim(model, raw_point: dict, device: str = "cuda") -> int:
    """One forward pass on a small slice to read the unpooled feature width."""
    sub = {k: v[:20000].copy() for k, v in raw_point.items() if isinstance(v, np.ndarray)}
    transform = sonata.transform.default()
    pt = to_device(transform(sub), device)
    model.eval()
    with torch.inference_mode():
        out = unpool_to_full(model(pt))
        return int(out.feat.shape[1])


@torch.inference_mode()
def predict_labels(model, head, raw_point: dict, device: str = "cuda") -> np.ndarray:
    """Per-ORIGINAL-point class ids for a {coord,color,normal} cloud.

    Uses Sonata's default (single grid-sample) transform, unpools to the
    grid-sampled resolution, applies the linear head, then scatters the argmax
    back to the original points via the GridSample ``inverse`` map.
    """
    transform = sonata.transform.default()
    pt = transform({k: v.copy() for k, v in raw_point.items() if isinstance(v, np.ndarray)})
    inverse = pt["inverse"].cpu().numpy()
    pt = to_device(pt, device)
    model.eval(); head.eval()
    out = unpool_to_full(model(pt))
    pred_grid = head(out.feat).argmax(dim=-1).cpu().numpy()
    return pred_grid[inverse]


def colorize(pred: np.ndarray) -> np.ndarray:
    colors = np.zeros((len(pred), 3), dtype=np.float64)
    for c, name in enumerate(CLASS_NAMES):
        colors[pred == c] = CLASS_COLORS[name]
    return colors


def save_labeled_ply(coord: np.ndarray, pred: np.ndarray, out_path: Path) -> None:
    """Write a class-colored PLY + sibling .legend.json (for tools/ply_viewer.html)."""
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colorize(pred))
    out_path = Path(out_path)
    o3d.io.write_point_cloud(str(out_path), pcd)
    counts = {CLASS_NAMES[c]: int((pred == c).sum()) for c in range(len(CLASS_NAMES))}
    legend = {
        "label_counts": {k: v for k, v in counts.items() if v > 0},
        "palette": {k: list(CLASS_COLORS[k]) for k in counts},
        "n_points": int(len(pred)),
        "source": "sonata_ptv3_las6",
    }
    out_path.with_suffix(".legend.json").write_text(json.dumps(legend, indent=2))


def per_class_iou(pred: np.ndarray, gt: np.ndarray, ignore: int = -1) -> dict:
    """mIoU + per-class IoU over points where gt != ignore."""
    out = {}
    ious = []
    for c in range(len(CLASS_NAMES)):
        gt_c = (gt == c)
        if gt_c.sum() == 0:
            out[CLASS_NAMES[c]] = None
            continue
        pred_c = (pred == c)
        inter = np.logical_and(pred_c, gt_c).sum()
        union = np.logical_or(pred_c, gt_c & (gt != ignore)).sum()
        iou = float(inter / max(1, union))
        out[CLASS_NAMES[c]] = round(iou, 4)
        ious.append(iou)
    out["mIoU"] = round(float(np.mean(ious)), 4) if ious else 0.0
    return out
