"""Unified digital-twin builder: ANY input converges on ONE labeled-cloud -> voxelize tail.

Two fronts produce the same intermediate -- a ``labeled.ply`` (per-point colours) plus a
``point_labels.npz`` sidecar (per-point instance names) in a run directory:

* PRIMARY (multi-view, used for the demo): N images -> pi3 reconstruction + SAM3 segmentation, run by
  ``scripts/recon/pipeline_web.py`` in the ``halo`` env, which writes ``labeled.ply`` + ``point_labels.npz``.
* SECONDARY (geometry): one ``.obj/.ply/.las/.laz`` scan -> clean & align -> cluster -> the geometry-
  priors namer (:func:`engine.vision.instance_namer.name_instances`) -> the same two files.

Both then call the SHARED tail :func:`voxelize_labeled_cloud.voxelize_labeled`. Keeping the seam at the
labeled cloud is what lets one voxelizer serve every input type.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".bmp", ".tif", ".tiff", ".webp"}
GEOMETRY_SUFFIXES = {".obj", ".ply", ".las", ".laz"}
_STRUCTURAL = {"wall", "floor", "ceiling"}


def detect_input_kind(paths) -> str:
    """Dispatch by input: ``"images"`` (>=2 image files) or ``"geometry"`` (exactly one mesh/cloud)."""
    paths = [Path(p) for p in (paths if isinstance(paths, (list, tuple)) else [paths])]
    imgs = [p for p in paths if p.suffix.lower() in IMAGE_SUFFIXES]
    geom = [p for p in paths if p.suffix.lower() in GEOMETRY_SUFFIXES]
    if len(imgs) >= 2 and not geom:
        return "images"
    if len(geom) == 1 and not imgs:
        return "geometry"
    raise ValueError(
        f"cannot dispatch input: {len(imgs)} image(s), {len(geom)} geometry file(s) -- need >= 2 "
        f"images for the multi-view front OR exactly one .obj/.ply/.las/.laz for the geometry front")


def geometry_to_labeled_cloud(scan_path, run_dir) -> Path:
    """SECONDARY front: a geometry scan -> ``labeled.ply`` + ``point_labels.npz`` + ``labeled.legend.json``.

    Cleans/aligns the scan (OBJ/PLY mesh or LAS/LAZ cloud), clusters it (scale-robust DBSCAN), names each
    cluster with the geometry-priors namer (floor/ceiling/wall/server rack/object), and broadcasts a per-
    point instance name (objects get a per-cluster index so the voxelizer can split rack rows; structural
    surfaces keep their bare class). Writes the same contract the multi-view front emits.
    """
    from engine.vision.cleaner import clean_and_align_meshes
    from engine.vision.instance_namer import name_instances
    import open3d as o3d
    import trimesh

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    _, cleaned = clean_and_align_meshes(Path(scan_path))
    pts = np.asarray(cleaned.vertices, np.float64)
    if len(pts) < 100:
        raise ValueError(f"cleaned scan has too few vertices ({len(pts)}) to label")

    # scale-robust clustering: eps as a fraction of the cloud diagonal (the scan may be up-to-scale).
    diag = float(np.linalg.norm(pts.max(0) - pts.min(0))) or 1.0
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    labels = np.asarray(pcd.cluster_dbscan(eps=max(diag * 0.015, 1e-6), min_points=30))
    if (labels >= 0).any():
        labels[labels < 0] = int(labels.max()) + 1          # noise -> its own bucket, never dropped
    else:
        labels[:] = 0

    res = name_instances(pts, labels)                        # per-point class label + colour
    names = res.point_labels.astype("<U24").copy()
    per_class_idx: dict[str, int] = {}
    for cl in np.unique(labels):
        m = labels == cl
        cls = str(res.point_labels[m][0])
        if cls in _STRUCTURAL:
            continue                                         # walls/floor/ceiling stay bare (the shell)
        per_class_idx[cls] = per_class_idx.get(cls, 0) + 1   # objects -> "<class> <n>" per cluster
        names[m] = f"{cls} {per_class_idx[cls]}"[:24]

    np.savez(run_dir / "point_labels.npz", names=names.astype("<U24"))
    trimesh.PointCloud(vertices=pts.astype(np.float32),
                       colors=(res.point_colors * 255).astype(np.uint8)).export(str(run_dir / "labeled.ply"))
    palette = {}
    for cl in np.unique(labels):
        m = labels == cl
        palette[str(res.point_labels[m][0])] = [round(float(x), 3) for x in res.point_colors[m][0]]
    u, c = np.unique(names, return_counts=True)
    (run_dir / "labeled.legend.json").write_text(json.dumps(
        {"backend": "geometry_namer", "label_counts": {str(k): int(v) for k, v in zip(u, c)},
         "palette": palette}, indent=2))
    return run_dir / "labeled.ply"


def build_twin(run_dir, *, scan=None, **voxel_kw):
    """Run the SHARED voxelize tail for either front, returning ``(grid, placements, origin)``.

    Pass ``scan=<geometry file>`` to build the labeled cloud from a scan first (secondary front);
    otherwise ``<run_dir>/labeled.ply`` must already exist (produced by the multi-view front).
    Extra keyword arguments (``rack_type``, ``aisle``, ``room_depth`` ...) pass through to the tail.
    """
    run_dir = Path(run_dir)
    if scan is not None:
        geometry_to_labeled_cloud(scan, run_dir)
        # LAS/LAZ scans are already in metres -> skip the up-to-scale rack-width anchor.
        voxel_kw.setdefault("metric", Path(scan).suffix.lower() in {".las", ".laz"})
    if not (run_dir / "labeled.ply").exists():
        raise FileNotFoundError(
            f"no labeled.ply in {run_dir}: run the multi-view front first, or pass scan=<geometry file>")
    sys.path.insert(0, str(_REPO / "scripts" / "recon"))
    from voxelize_labeled_cloud import voxelize_labeled
    return voxelize_labeled(run_dir, **voxel_kw)
