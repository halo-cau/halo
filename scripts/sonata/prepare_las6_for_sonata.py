"""Convert las6_corrected.npy → Sonata/Pointcept point-dict format.

`data/mask3d_server_room/train/las6_corrected.npy` is the Mask3D workflow-spec
array (N, 12): [xyz, rgb(0-255), normal(zeros), segment_id, semantic(1-6/255),
instance]. Sonata's PTv3 expects a dict with **real normals** (its input feat is
coord+color+normal, 9-ch) — the npy's normal columns are all zero, so we compute
them here with Open3D. We remap the 1-based semantic ids to a contiguous
0..K-1 head label space (ignore=255 → -1) and add a class-stratified train/val
split (single scene → the split measures feature separability, not spatial
generalization; the real test is a held-out scan / the MVS cloud).

Run in the `halo` env (numpy + open3d). Output: data/sonata_las6/las6.npz + meta.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import open3d as o3d

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "data" / "mask3d_server_room" / "train" / "las6_corrected.npy"
_OUT_DIR = _ROOT / "data" / "sonata_las6"

# las6 semantic id (1-based, 255=ignore) → contiguous head label (0-based).
# Order fixes the head's output channel meaning; keep it stable across train/predict.
SRC_ID_TO_NAME = {1: "wall", 2: "floor", 3: "ceiling", 4: "server_rack", 5: "box_clutter", 6: "ac_unit"}
CLASS_NAMES = ["wall", "floor", "ceiling", "server_rack", "box_clutter", "ac_unit"]
NAME_TO_HEAD = {n: i for i, n in enumerate(CLASS_NAMES)}
IGNORE_LABEL = -1
IGNORE_SRC = 255
VAL_FRACTION = 0.2
NORMAL_KNN = 32
SEED = 0


def main() -> None:
    if not _SRC.exists():
        raise SystemExit(f"Not found: {_SRC}")
    arr = np.load(_SRC).astype(np.float64)
    assert arr.shape[1] == 12, f"expected (N,12), got {arr.shape}"
    coord = arr[:, 0:3].astype(np.float32)
    color = arr[:, 3:6].astype(np.float32)  # 0-255, NormalizeColor divides later
    semantic_src = arr[:, 10].astype(np.int64)

    # --- map semantic ids → head labels (ignore → -1) ---
    head = np.full(len(arr), IGNORE_LABEL, dtype=np.int64)
    for src_id, name in SRC_ID_TO_NAME.items():
        head[semantic_src == src_id] = NAME_TO_HEAD[name]
    n_ignore = int((semantic_src == IGNORE_SRC).sum())
    # sanity vs the label summary (ac_unit == 28545)
    assert int((semantic_src == 6).sum()) == int((head == NAME_TO_HEAD["ac_unit"]).sum())

    # --- real normals via Open3D PCA (the npy normals are all zero) ---
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord.astype(np.float64))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=NORMAL_KNN))
    pcd.orient_normals_towards_camera_location(camera_location=coord.mean(0).astype(np.float64))
    normal = np.asarray(pcd.normals, dtype=np.float32)

    # --- class-stratified random train/val split (keeps the lone AC in both) ---
    rng = np.random.default_rng(SEED)
    split = np.zeros(len(arr), dtype=np.int8)  # 0=train, 1=val
    for c in range(len(CLASS_NAMES)):
        idx = np.where(head == c)[0]
        if len(idx) == 0:
            continue
        n_val = max(1, int(round(len(idx) * VAL_FRACTION)))
        split[rng.choice(idx, size=n_val, replace=False)] = 1

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_npz = _OUT_DIR / "las6.npz"
    np.savez(
        out_npz,
        coord=coord, color=color, normal=normal,
        segment=head.astype(np.int64), split=split,
    )

    counts = {CLASS_NAMES[c]: int((head == c).sum()) for c in range(len(CLASS_NAMES))}
    # inverse-frequency class weights (upweights the scarce AC class) for the loss
    freqs = np.array([max(1, counts[n]) for n in CLASS_NAMES], dtype=np.float64)
    weights = (freqs.sum() / (len(freqs) * freqs))
    weights = (weights / weights.mean()).round(3)
    meta = {
        "source": str(_SRC),
        "n_points": int(len(arr)),
        "class_names": CLASS_NAMES,
        "ignore_label": IGNORE_LABEL,
        "n_ignore": n_ignore,
        "class_counts": counts,
        "class_weights_inv_freq": {CLASS_NAMES[i]: float(weights[i]) for i in range(len(CLASS_NAMES))},
        "val_fraction": VAL_FRACTION,
        "n_train": int((split == 0).sum()),
        "n_val": int((split == 1).sum()),
        "normal_knn": NORMAL_KNN,
        "coord_bbox_min": coord.min(0).round(3).tolist(),
        "coord_bbox_max": coord.max(0).round(3).tolist(),
    }
    (_OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"Wrote {out_npz} ({out_npz.stat().st_size/1e6:.1f} MB)")
    print(f"  points={meta['n_points']:,}  ignore={n_ignore:,}  train={meta['n_train']:,} val={meta['n_val']:,}")
    print(f"  normals: mean|n|={np.linalg.norm(normal,axis=1).mean():.3f} (should be ≈1)")
    print("  class counts:", counts)
    print("  inv-freq weights:", meta["class_weights_inv_freq"])
    print("  → AC upweighted ×%.1f vs wall" % (weights[NAME_TO_HEAD['ac_unit']]/weights[NAME_TO_HEAD['wall']]))


if __name__ == "__main__":
    main()
