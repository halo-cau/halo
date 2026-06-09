"""End-to-end CV demo: raw scan → segmentation → voxelized RL observation.

Runs ``engine.vision.pipeline.run_pipeline`` on a real scan with minimal
metadata, then writes the demo deliverables:

  * ``padded_grid.npy``           — int8 array, shape (200, 200, 60).
  * ``segmentation_labels.ply``   — per-vertex labels from the chosen backend.
  * ``segmentation_summary.json`` — label counts + shell/component stats.

What this exercises:
  1. cleaner.clean_and_align_meshes_staged — SOR + RANSAC floor + Manhattan.
  2. segmentor — geometric (default) or mask3d, picked by ``--backend``.
  3. pipeline — appends the room-shell cuboid centrally before voxelization.
  4. voxelizer.voxelize_and_stamp_metadata — surface voxelize + closing +
     metadata-driven semantic labels (racks, AC, legacy heat, workspaces).
  5. voxelizer.pad_to_fixed_shape — zero-pad to RL observation shape.

Usage::

    python scripts/smoke_test_pipeline.py
    python scripts/smoke_test_pipeline.py --backend mask3d
    python scripts/smoke_test_pipeline.py --scan path/to/other.las --out-dir out/

Optional companion (mesh inputs only, e.g. .obj):

    python scripts/visualize_pipeline.py --scan server_room_phone/textured_output.obj
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.core.config import (
    COOLING_AC_VENT,
    GRID_SHAPE,
    HEAT_LEGACY_SERVER,
    HUMAN_WORKSPACE,
    MAX_ROOM_DIMENSIONS,
    OBSTACLE_WALL,
    RACK_BODY,
    RACK_EXHAUST,
    RACK_INTAKE,
    SEMANTIC_LABELS,
    SPACE_EMPTY,
    VOXEL_SIZE,
)
from engine.core.data_types import (
    Coordinate,
    CoolingUnit,
    RackFacing,
    RackPlacement,
    ScanMetadata,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCAN = PROJECT_ROOT / "server_room_phone" / "server_room_6.las"


def _build_demo_metadata() -> ScanMetadata:
    """Minimal metadata exercising every stamp path.

    Coordinates roughly match the demo server room as labelled in the
    Mask3D dataset. In production the operator-tagged equipment list
    arrives from the frontend; the values below are placeholders so the
    voxel-grid sanity asserts have something to land on.
    """
    return ScanMetadata(
        cooling_units=[
            CoolingUnit(position=Coordinate(x=0.0, y=3.0, z=1.5)),
        ],
        racks=[
            RackPlacement(
                position=Coordinate(x=-0.5, y=-3.5, z=0.0),
                facing=RackFacing.PLUS_X,
                rack_type="42U",
            ),
            RackPlacement(
                position=Coordinate(x=1.2, y=-3.5, z=0.0),
                facing=RackFacing.MINUS_X,
                rack_type="42U",
            ),
        ],
        legacy_servers=[Coordinate(x=-0.5, y=-2.0, z=1.0)],
        human_workspaces=[Coordinate(x=1.5, y=2.5, z=0.0)],
    )


def _write_labels_ply(
    mesh: o3d.geometry.TriangleMesh,
    label_colors: np.ndarray,
    path: Path,
) -> None:
    points = np.asarray(mesh.vertices, dtype=np.float64)
    if len(points) == 0 or len(label_colors) == 0:
        return
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    n = min(len(points), len(label_colors))
    pcd.colors = o3d.utility.Vector3dVector(np.clip(label_colors[:n].astype(np.float64), 0.0, 1.0))
    o3d.io.write_point_cloud(str(path), pcd, write_ascii=False)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--scan",
        type=Path,
        default=DEFAULT_SCAN,
        help="Path to a scan file (.obj/.ply/.las/.laz). Default: server_room_6.las.",
    )
    p.add_argument(
        "--backend",
        choices=(
            "geometric", "mask3d", "dino_sam3",
            "sam3_concept", "dinov3", "dinov3_sam3", "none",
        ),
        default="geometric",
        help="Segmentor backend (sets HALO_SEGMENTOR_BACKEND). The SAM3/DINOv3 "
             "backends need the vision-ai stack + HF_TOKEN.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for padded_grid.npy + viewer artifacts. "
             "Default: <scan parent>/pipeline_vis_demo.",
    )
    args = p.parse_args()

    # Set the backend env var BEFORE importing the pipeline so the factory
    # cache resolves with the user's choice.
    os.environ["HALO_SEGMENTOR_BACKEND"] = args.backend

    scan_path = args.scan.expanduser().resolve()
    if not scan_path.exists():
        raise SystemExit(f"Scan not found: {scan_path}")

    out_dir = (args.out_dir if args.out_dir is not None else scan_path.parent / "pipeline_vis_demo").expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Import after env var is set.
    from engine.vision.cleaner import clean_and_align_meshes
    from engine.vision.pipeline import run_pipeline
    from engine.vision.segmentor_factory import get_default_segmentor
    from engine.vision.structural_priors import estimate_room_shell_prior

    metadata = _build_demo_metadata()

    print(f"=== HALO CV demo pipeline ===")
    print(f"  Scan       : {scan_path.name}")
    print(f"  Backend    : {args.backend}")
    print(f"  Out dir    : {out_dir}")
    print(f"  Voxel size : {VOXEL_SIZE} m  →  fixed grid {GRID_SHAPE}")
    print(f"  Metadata   : {len(metadata.cooling_units)} cooling, "
          f"{len(metadata.racks)} racks, "
          f"{len(metadata.legacy_servers)} legacy, "
          f"{len(metadata.human_workspaces)} workspaces")
    print()

    segmentor = get_default_segmentor()

    t0 = time.time()
    result = run_pipeline(scan_path, metadata, segmentor=segmentor)
    elapsed = time.time() - t0

    grid = result.grid
    padded_grid = result.padded_grid
    origin = result.origin
    grid_offset = result.grid_offset

    print(f"Pipeline completed in {elapsed:.2f}s")
    print()
    print(f"=== Output checks ===")
    print(f"  Room grid shape    : {grid.shape}   dtype={grid.dtype}")
    print(f"  Padded grid shape  : {padded_grid.shape}   dtype={padded_grid.dtype}")
    print(f"  World origin (m)   : {origin.round(3).tolist()}")
    print(f"  Grid offset (vox)  : {grid_offset}")
    print()

    assert padded_grid.shape == GRID_SHAPE, (
        f"Padded shape {padded_grid.shape} does not match GRID_SHAPE {GRID_SHAPE}"
    )
    assert padded_grid.dtype == np.int8, f"Expected int8, got {padded_grid.dtype}"

    # Cuboid extent sanity: the room shell must fit within the configured
    # max room dimensions. A blown extent indicates a bad shell estimate
    # (e.g. an outlier was not trimmed) and will propagate noise to RL.
    _, cleaned_mesh = clean_and_align_meshes(scan_path)
    shell_prior = estimate_room_shell_prior(cleaned_mesh)
    extents = (shell_prior.bounds_max - shell_prior.bounds_min).tolist()
    print(f"  Shell extents (m)  : {[round(v, 2) for v in extents]}  "
          f"(MAX={MAX_ROOM_DIMENSIONS})")
    for axis, (extent, limit) in enumerate(zip(extents, MAX_ROOM_DIMENSIONS)):
        assert extent <= limit, (
            f"Shell extent on axis {axis} = {extent:.2f}m exceeds "
            f"MAX_ROOM_DIMENSIONS[{axis}] = {limit:.2f}m"
        )

    # Save padded grid for downstream RL.
    grid_path = out_dir / "padded_grid.npy"
    np.save(grid_path, padded_grid)
    print(f"  Saved             : {grid_path.relative_to(PROJECT_ROOT)}")

    # If the backend produced per-vertex labels, save them as a colored PLY
    # alongside a JSON summary. This is the deliverable that visually shows
    # "the segmentor worked" on the demo scan.
    seg_summary: dict = {
        "backend": args.backend,
        "scan": scan_path.name,
    }
    if segmentor is not None:
        try:
            seg_result = segmentor.run(cleaned_mesh, scan_path)
            labels_ply = out_dir / "segmentation_labels.ply"
            _write_labels_ply(cleaned_mesh, seg_result.label_colors, labels_ply)
            from collections import Counter
            seg_summary["n_total"] = seg_result.n_total
            seg_summary["n_removed"] = seg_result.n_removed
            seg_summary["removal_fraction"] = round(seg_result.removal_fraction, 4)
            seg_summary["label_counts"] = dict(Counter(seg_result.vertex_labels))
            seg_summary["extra"] = seg_result.extra
            print(f"  Saved             : {labels_ply.relative_to(PROJECT_ROOT)}")
        except Exception as exc:  # noqa: BLE001
            print(f"  ⚠ Could not write segmentation PLY: {exc}")
            seg_summary["error"] = str(exc)

    summary_path = out_dir / "segmentation_summary.json"
    summary_path.write_text(json.dumps(seg_summary, indent=2, default=str))
    print(f"  Saved             : {summary_path.relative_to(PROJECT_ROOT)}")
    print()

    # Semantic distribution
    print(f"=== Label distribution (padded grid) ===")
    total = padded_grid.size
    for sid, name in SEMANTIC_LABELS.items():
        n = int((padded_grid == sid).sum())
        if n == 0:
            continue
        pct = 100.0 * n / total
        print(f"  {sid:>2} {name:<18} {n:>10,}  ({pct:>5.2f}%)")

    # Required label coverage: walls/floor/ceiling come from the cuboid +
    # surface voxelization; racks come from metadata stamping. If any of
    # these is missing the demo is broken.
    seen = set(np.unique(padded_grid).tolist())
    must_be_present: list[tuple[int, str]] = [
        (OBSTACLE_WALL, "wall"),
    ]
    if metadata.cooling_units:
        must_be_present.append((COOLING_AC_VENT, "cooling_units"))
    if metadata.legacy_servers:
        must_be_present.append((HEAT_LEGACY_SERVER, "legacy_servers"))
    if metadata.human_workspaces:
        must_be_present.append((HUMAN_WORKSPACE, "human_workspaces"))
    if metadata.racks:
        must_be_present.extend(
            [(RACK_BODY, "rack_body"), (RACK_INTAKE, "rack_intake"), (RACK_EXHAUST, "rack_exhaust")]
        )

    print()
    print(f"=== Stamping sanity ===")
    failed: list[str] = []
    for sid, name in must_be_present:
        if sid in seen:
            print(f"  ✓ {name:<18} (id={sid}) present")
        else:
            print(f"  ✗ {name:<18} (id={sid}) MISSING — metadata coord may be outside grid bounds")
            failed.append(name)

    print()
    if failed:
        print(f"⚠ Pipeline ran but {len(failed)} required classes did not land in the grid.")
        print(f"  This usually means metadata coords sit outside the room's voxel bounds.")
        print(f"  Adjust _build_demo_metadata() to match the scan's actual room dimensions.")
        raise SystemExit(1)
    print(f"✓ End-to-end demo pipeline passed.")
    print(f"  Wall voxels (OBSTACLE_WALL=1): {int((padded_grid == OBSTACLE_WALL).sum()):,}")
    print(f"  Empty voxels (SPACE_EMPTY=0): {int((padded_grid == SPACE_EMPTY).sum()):,}")


if __name__ == "__main__":
    main()
