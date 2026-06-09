"""Run the HALO-finetuned Mask3D on a target point cloud and write a colored PLY.

Uses ``mask3d.get_model`` with our new ``halo`` dataset branch (added to
``opt/Mask3D/mask3d/__init__.py``) so the model architecture matches the
training-time config (num_targets=7, num_queries=100, num_labels=6).

Maps the model's per-instance predictions back onto every input vertex and
colors them by the HALO 6-class palette.

Usage::

    python scripts/predict_halo.py \\
        --checkpoint opt/Mask3D/checkpoints/halo_overfit_v1/last-epoch.ckpt \\
        --input server_room_phone/pipeline_vis_lidar_laz/s3_manhattan.ply \\
        --output server_room_phone/pipeline_vis_lidar_laz/s3_halo_pred.ply \\
        --confidence-threshold 0.5

The checkpoint filename must start with ``halo_`` so ``get_model`` picks the
halo branch. The script will rename a passed checkpoint if needed.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from shutil import copyfile

import numpy as np
import open3d as o3d
import torch

# ``mask3d/`` is importable as a package after ``-e ./opt/Mask3D`` install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "opt" / "Mask3D"))

# Standard HALO 6-class palette + ignore.
CLASS_NAMES = ["wall", "floor", "ceiling", "server_rack", "box_clutter", "ac_unit"]
CLASS_COLORS_BGR_255 = [
    (55, 138, 221),    # wall
    (180, 178, 169),   # floor
    (232, 229, 221),   # ceiling
    (29, 158, 117),    # server_rack
    (232, 158, 79),    # box_clutter
    (36, 169, 225),    # ac_unit
]
IGNORE_COLOR = (110, 110, 110)


def _ensure_halo_named_checkpoint(ckpt_path: Path) -> Path:
    """Ensure the file name starts with ``halo_`` so get_model picks the halo branch."""
    if ckpt_path.name.startswith("halo_"):
        return ckpt_path
    # Copy to a temp file with the right prefix; symlink would be cleaner but
    # not all filesystems support it.
    tmp = Path(tempfile.gettempdir()) / f"halo_{ckpt_path.name}"
    if not tmp.exists() or tmp.stat().st_size != ckpt_path.stat().st_size:
        copyfile(str(ckpt_path), str(tmp))
    return tmp


def predict(
    checkpoint: Path,
    input_ply: Path,
    output_ply: Path,
    confidence_threshold: float = 0.5,
    mask_threshold: float = 0.5,
) -> dict:
    from mask3d import get_model, prepare_data  # uses cvg fork

    # The cvg fork's get_model dispatches on the FIRST underscore-separated
    # token of the checkpoint filename, so a name like
    # ``halo_overfit_v1.ckpt`` selects our halo branch (num_targets=7 etc.).
    halo_ckpt = _ensure_halo_named_checkpoint(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model with halo branch from {halo_ckpt}")
    model = get_model(checkpoint_path=str(halo_ckpt))
    model.eval()
    model.to(device)

    print(f"Reading input PLY: {input_ply}")
    mesh = o3d.io.read_triangle_mesh(str(input_ply))
    n_verts = len(np.asarray(mesh.vertices))
    print(f"  vertices: {n_verts:,}  triangles: {len(np.asarray(mesh.triangles)):,}")
    if not mesh.has_vertex_colors():
        # prepare_data expects per-vertex colors; default to a neutral grey.
        mesh.vertex_colors = o3d.utility.Vector3dVector(
            np.full((n_verts, 3), 0.6, dtype=np.float64)
        )

    # Build sparse tensor input via the cvg fork's helper.
    data, points, colors, features, unique_map, inverse_map = prepare_data(mesh, device)
    print(f"  sparse-quantized to {len(unique_map):,} unique voxels")

    with torch.no_grad():
        outputs = model(data, raw_coordinates=features)

    pred_logits = outputs["pred_logits"][0].detach().cpu()  # (Q, num_targets)
    pred_masks  = outputs["pred_masks"][0].detach().cpu()    # (V_quantized, Q)

    n_targets = pred_logits.shape[-1]
    n_classes = len(CLASS_NAMES)
    assert n_targets >= n_classes + 1, f"Expected num_targets>={n_classes+1}, got {n_targets}"

    # Per-vertex label assignment, preferring higher-confidence instances.
    point_labels = -1 * np.ones(n_verts, dtype=np.int32)
    point_confidences = np.zeros(n_verts, dtype=np.float32)
    instance_count = 0

    softmax = torch.softmax(pred_logits, dim=-1)
    sigmoid_masks = torch.sigmoid(pred_masks)

    for q in range(pred_logits.shape[0]):
        # Argmax over FOREGROUND classes only — under-trained models put too
        # much mass on the "no-object" class and the standard argmax would
        # reject every query. We still gate on a confidence_threshold that
        # applies to the foreground probability (i.e. how confident the model
        # is in the best object class assuming SOMETHING is here).
        fg_probs = softmax[q, :n_classes]
        cls = int(fg_probs.argmax())
        cls_conf = float(fg_probs.max())
        if cls_conf < confidence_threshold:
            continue
        m_quantized = sigmoid_masks[:, q] > mask_threshold
        if not m_quantized.any():
            continue
        # Expand from quantized voxels back to full point cloud via inverse_map.
        m_full = m_quantized[inverse_map].numpy()
        # Per-point mask confidence
        mask_conf = float(sigmoid_masks[m_quantized, q].mean())
        score = cls_conf * mask_conf
        # Higher score overwrites prior assignments
        overwrite = m_full & (score > point_confidences)
        point_labels[overwrite] = cls
        point_confidences[overwrite] = score
        instance_count += 1

    print(f"  accepted {instance_count} instance proposals "
          f"(threshold class={confidence_threshold}, mask={mask_threshold})")

    # Assignment distribution
    coverage = (point_labels >= 0).sum()
    print(f"  point coverage: {coverage:,}/{n_verts:,} ({100*coverage/n_verts:.1f}%)")
    for cls in range(n_classes):
        n = int((point_labels == cls).sum())
        if n:
            print(f"    {CLASS_NAMES[cls]:<12} cls={cls}  n={n:,}  ({100*n/n_verts:.1f}%)")

    # Build colored PLY
    colors_out = np.tile(np.asarray(IGNORE_COLOR, dtype=np.uint8), (n_verts, 1))
    for cls in range(n_classes):
        m = point_labels == cls
        if m.any():
            colors_out[m] = CLASS_COLORS_BGR_255[cls]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors_out.astype(np.float64) / 255.0)
    output_ply.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(output_ply), pcd, write_ascii=False)
    print(f"  wrote {output_ply}")

    return {
        "n_vertices": int(n_verts),
        "coverage": int(coverage),
        "instances_accepted": int(instance_count),
        "labels": point_labels,
        "confidences": point_confidences,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--input", required=True, type=Path, help="Input PLY (e.g. s3_manhattan.ply)")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--confidence-threshold", type=float, default=0.5)
    p.add_argument("--mask-threshold", type=float, default=0.5)
    args = p.parse_args()
    predict(
        checkpoint=args.checkpoint,
        input_ply=args.input,
        output_ply=args.output,
        confidence_threshold=args.confidence_threshold,
        mask_threshold=args.mask_threshold,
    )


if __name__ == "__main__":
    main()
