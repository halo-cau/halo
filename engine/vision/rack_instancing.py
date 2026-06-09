"""Per-instance separation for server racks, with a scale-free 42U size prior.

Pi3 (and most feed-forward MVS) clouds are *up-to-scale*, not metric, so a rack
prior can't be expressed in millimetres. But the 19"/42U standard is a fixed
*aspect*: a populated enclosure is ~600 mm wide x ~2000 mm tall x ~1000 mm deep,
so width:height ~= 0.30 and a real rack is a deep box, not a thin slab. Those
ratios are scale-invariant, so they apply directly to the Pi3 cloud.

Two consumers (see ``scripts/recon/segment_photos_sam3.py --rack-instancing``):

* ``geometric`` — door-reject by the slab test, then split each merged rack
  component along its row axis at the standard 42U pitch.
* ``sam3`` — keep SAM3's per-image rack instance masks separate, associate the
  lifted blobs across views (same 3D world frame), then door-reject; the 42U
  pitch is only a backstop to re-split any pair SAM3 saw as one.

All inputs are gravity-aligned (``+Z`` up), matching the labeler's cloud.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

#: enclosure width / height for a standard 19"/42U rack (600 mm / ~2000 mm).
RACK_WIDTH_RATIO = 0.30
#: a real rack is a deep box; a leaning door/panel is a slab. Reject a rack
#: component when its thinnest extent / height falls below this (door ~0.075,
#: real racks ~0.19-0.23 in measured Pi3 units -> 0.12 cleanly separates them).
DOOR_THICK_RATIO = 0.12


def obb_metrics(pts: np.ndarray) -> dict:
    """Gravity-aligned (Z up) oriented-extent metrics for one point blob.

    Returns height (Z extent), rowlen/depth (the long/short horizontal extents),
    thick (smallest principal extent ~ slab thickness), tilt (|Z| of the thinnest
    direction's normal: 0 = vertical panel, 1 = horizontal slab), and the
    horizontal row axis + centroid used for splitting.
    """
    Q = np.asarray(pts, float)
    c = Q.mean(0)
    H = float(Q[:, 2].max() - Q[:, 2].min())
    XY = Q[:, :2] - c[:2]
    _, _, vt = np.linalg.svd(XY, full_matrices=False)
    proj = XY @ vt.T
    rowlen = float(proj[:, 0].ptp())
    depth = float(proj[:, 1].ptp())
    Qc = Q - c
    _, s3, vt3 = np.linalg.svd(Qc, full_matrices=False)
    ext = s3 / np.sqrt(max(len(Qc), 1))
    thick = float(2 * ext.min())
    normal = vt3[int(np.argmin(ext))]
    tilt = float(abs(normal[2]))
    row_axis = np.array([vt[0, 0], vt[0, 1], 0.0])
    n = np.linalg.norm(row_axis)
    row_axis = row_axis / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
    return {"height": H, "rowlen": rowlen, "depth": depth, "thick": thick,
            "tilt": tilt, "row_axis": row_axis, "center": c}


def is_door(pts: np.ndarray, thick_ratio_max: float = DOOR_THICK_RATIO) -> bool:
    """True if a rack-class blob is a thin/leaning panel (door), not a real rack."""
    m = obb_metrics(pts)
    if m["height"] < 1e-6:
        return True
    return (m["thick"] / m["height"]) < thick_ratio_max


def standard_width(heights, width_ratio: float = RACK_WIDTH_RATIO) -> float:
    """Scale-free standard rack width = ratio * (median real-rack height).

    Using a shared median height (racks are uniform 42U) gives one stable pitch
    for the whole room instead of a per-component height that noise can inflate.
    """
    h = np.asarray([x for x in heights if x > 1e-6], float)
    if len(h) == 0:
        return 0.0
    return width_ratio * float(np.median(h))


def split_geometric(pts: np.ndarray, std_width: float,
                    metrics: dict | None = None) -> np.ndarray:
    """Split one rack component into N instances along its row axis at std_width.

    N = round(rowlen / std_width), clamped to >= 1. Returns a 0..N-1 label per
    point (equal-extent bins along the principal horizontal axis).
    """
    m = metrics or obb_metrics(pts)
    if std_width <= 1e-6:
        return np.zeros(len(pts), np.int32)
    n = max(1, int(round(m["rowlen"] / std_width)))
    if n == 1:
        return np.zeros(len(pts), np.int32)
    t = (np.asarray(pts, float) - m["center"]) @ m["row_axis"]
    edges = np.linspace(t.min(), t.max(), n + 1)
    return np.clip(np.digitize(t, edges[1:-1]), 0, n - 1).astype(np.int32)


def associate_blobs(blobs: list[np.ndarray], view_ids: list[int], voxel: float,
                    overlap_thr: float = 0.20) -> np.ndarray:
    """Union-find merge of per-view SAM3 instance blobs into global instances.

    Two blobs merge when their voxel sets overlap (overlap coefficient =
    |A intersect B| / min(|A|,|B|) >= overlap_thr) AND they come from *different*
    views — SAM3 already declared same-view masks distinct, so we never merge
    those. Returns a global instance id per input blob.
    """
    n = len(blobs)
    vsets = [set(map(tuple, np.floor(b / voxel).astype(np.int64).tolist())) for b in blobs]
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if view_ids[i] == view_ids[j]:
                continue
            if not vsets[i] or not vsets[j]:
                continue
            inter = len(vsets[i] & vsets[j])
            if inter == 0:
                continue
            if inter / min(len(vsets[i]), len(vsets[j])) >= overlap_thr:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    roots = [find(i) for i in range(n)]
    remap = {r: k for k, r in enumerate(sorted(set(roots)))}
    return np.array([remap[r] for r in roots], np.int32)


def assign_points_to_instances(rack_pts: np.ndarray, inst_pts: np.ndarray,
                               inst_lab: np.ndarray, voxel: float,
                               max_dist: float | None = None) -> np.ndarray:
    """Label each recon rack point by its nearest SAM3-instance representative.

    ``inst_pts``/``inst_lab`` are the lifted instance points and their global
    ids; we voxel-downsample them (bounded size) and propagate via a KD-tree —
    same map-back trick the dense-cloud clustering uses. Points farther than
    ``max_dist`` from any kept instance get -1 (e.g. door points whose own
    instance was rejected, so they aren't absorbed into a neighbouring rack).
    """
    if len(inst_pts) == 0:
        return np.full(len(rack_pts), -1, np.int32)
    key = np.floor(inst_pts / voxel).astype(np.int64)
    _, keep = np.unique(key, axis=0, return_index=True)
    tree = cKDTree(inst_pts[keep])
    dist, nn = tree.query(rack_pts, k=1, workers=-1)
    out = inst_lab[keep][nn].astype(np.int32)
    if max_dist is not None:
        out[dist > max_dist] = -1
    return out
