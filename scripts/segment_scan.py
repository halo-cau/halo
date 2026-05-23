"""Run one or both segmentation backends on the server room scan and export
colored PLY files + label JSON for the Three.js pipeline viewer.

Outputs (to server_room_phone/pipeline_vis/):
  s3_seg_mask3d.ply      – mesh colored by Mask3D semantic label
  s3_seg_mask3d.json     – label counts + removal stats
    s3_seg_dino_sam3.ply   – mesh colored by Grounded-DINO+SAM3 label
    s3_seg_dino_sam3.json  – label counts + removal stats

After running, reload pipeline_viewer.html — both segmentation stages appear
between Stage 2 (aligned) and Stage 4 (voxelized).

Usage
-----
  # Run Mask3D only (checkpoint already on research server):
  conda run -n halo python scripts/segment_scan.py \\
      --backend mask3d \\
      --mask3d-checkpoint /path/to/mask3d.ckpt \\
      --mask3d-repo /path/to/mask3d_repo   # if not pip-installed

  # Run DINO+SAM3 only:
  conda run -n halo python scripts/segment_scan.py \\
      --backend dino_sam3 \
      --dino-config /path/to/GroundingDINO_SwinT_OGC.py \\
      --dino-checkpoint /path/to/groundingdino_swint_ogc.pth \\
      --sam3-checkpoint /path/to/sam3.pt

  # Run both back-to-back:
  conda run -n halo python scripts/segment_scan.py --backend both ...all flags...
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.vision.cleaner import clean_and_align_meshes
from engine.vision.graph_fusion import GraphCleanupConfig, apply_graph_cleanup
from engine.vision.segmentor_base import (
    PRESERVED_LABELS,
    SegmentorResult,
    label_to_color,
    strip_non_structural,
)
from engine.vision.structural_priors import (
    StructuralProtectionConfig,
    apply_structural_protection,
    build_cuboid_shell_mesh,
    append_virtual_room_shell,
    flatten_structural_labels_to_room_shell,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(name)s — %(message)s",
)
log = logging.getLogger("segment_scan")

OBJ_PATH = Path(__file__).resolve().parents[1] / "server_room_phone" / "textured_output.obj"
OUT_DIR  = OBJ_PATH.parent / "pipeline_vis"
OUT_DIR.mkdir(exist_ok=True)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MASK3D_CHECKPOINT = PROJECT_ROOT / "opt" / "Mask3D" / "checkpoints" / "scannet200" / "scannet200_benchmark.ckpt"
DEFAULT_DINO_CONFIG = PROJECT_ROOT / "opt" / "checkpoints" / "groundingdino" / "GroundingDINO_SwinT_OGC.py"
DEFAULT_DINO_CHECKPOINT = PROJECT_ROOT / "opt" / "checkpoints" / "groundingdino" / "groundingdino_swint_ogc.pth"
DEFAULT_SAM2_CHECKPOINT = PROJECT_ROOT / "opt" / "checkpoints" / "sam2" / "sam2.1_hiera_large.pt"
DEFAULT_SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEFAULT_SAM3_CHECKPOINT = PROJECT_ROOT / "opt" / "checkpoints" / "sam3" / "sam3.pt"

SEP = "─" * 58


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def _export_colored_ply(
    mesh: o3d.geometry.TriangleMesh,
    label_colors: np.ndarray,   # (N, 3) float32 in [0, 1]
    path: Path,
) -> None:
    """Write a PLY with per-vertex semantic colors."""
    exp = o3d.geometry.TriangleMesh()
    exp.vertices       = mesh.vertices
    exp.triangles      = mesh.triangles
    exp.vertex_normals = mesh.vertex_normals
    exp.vertex_colors  = o3d.utility.Vector3dVector(
        np.clip(label_colors, 0.0, 1.0).astype(np.float64)
    )
    o3d.io.write_triangle_mesh(str(path), exp, write_ascii=False)
    n_v = len(np.asarray(exp.vertices))
    n_t = len(np.asarray(exp.triangles))
    sz  = path.stat().st_size / 1024
    log.info("  → %s  (%d verts, %d tris, %.0f KB)", path.name, n_v, n_t, sz)


def _export_structural_ply(result: SegmentorResult, path: Path) -> None:
    """Write the stripped structural-only mesh with normal-derived colors."""
    m = result.structural_mesh
    n_v = len(np.asarray(m.vertices))
    n_t = len(np.asarray(m.triangles))
    if n_v == 0:
        log.warning("  → %s skipped (structural mesh is empty)", path.name)
        return
    m.compute_vertex_normals()
    normals = np.asarray(m.vertex_normals)
    colors  = np.clip(normals * 0.5 + 0.5, 0.0, 1.0)
    m.vertex_colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_triangle_mesh(str(path), m, write_ascii=False)
    sz  = path.stat().st_size / 1024
    log.info("  → %s  (%d verts, %d tris, %.0f KB)", path.name, n_v, n_t, sz)


def _export_label_json(result: SegmentorResult, path: Path) -> None:
    """Write a JSON summary of label distribution for the viewer info panel."""
    label_counts = {lbl: len(idxs) for lbl, idxs in result.label_map.items()}
    data = {
        "backend":          result.backend,
        "n_total":          result.n_total,
        "n_removed":        result.n_removed,
        "removal_fraction": round(result.removal_fraction, 4),
        "label_counts":     label_counts,
        "extra":            result.extra,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    log.info("  → %s", path.name)


def _write_result_outputs(
    mesh: o3d.geometry.TriangleMesh,
    result: SegmentorResult,
    tag: str,
) -> None:
    """Write colored labels, stripped structural mesh, and JSON stats."""
    _export_colored_ply(
        mesh,
        result.label_colors,
        OUT_DIR / f"s3_seg_{tag}_labels.ply",
    )
    _export_structural_ply(result, OUT_DIR / f"s3_seg_{tag}_structural.ply")
    _export_label_json(result, OUT_DIR / f"s3_seg_{tag}.json")


# ---------------------------------------------------------------------------
# Run a single backend
# ---------------------------------------------------------------------------

def run_mask3d(mesh: o3d.geometry.TriangleMesh, args: argparse.Namespace) -> SegmentorResult:
    from engine.vision.segmentor_mask3d import Mask3DSegmentor
    seg = Mask3DSegmentor(
        checkpoint_path=args.mask3d_checkpoint,
        protection_config=_build_protection_config(args),
        graph_cleanup_config=_build_graph_config(args),
    )
    return seg.run(mesh, args.scan_path)


def run_dino_sam2(mesh: o3d.geometry.TriangleMesh, args: argparse.Namespace) -> SegmentorResult:
    from engine.vision.segmentor_dino_sam import DinoSam2Segmentor
    seg = DinoSam2Segmentor(
        grounding_dino_config=args.dino_config,
        grounding_dino_checkpoint=args.dino_checkpoint,
        sam2_checkpoint=args.sam2_checkpoint,
        sam2_config=args.sam2_config,
        n_horizontal=args.n_views_horizontal,
        n_oblique=args.n_views_oblique,
        render_width=args.render_width,
        render_height=args.render_height,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        min_vote_fraction=args.min_vote_fraction,
        min_movable_votes=args.min_movable_votes,
        protection_config=_build_protection_config(args),
        graph_cleanup_config=_build_graph_config(args),
        remove_movable=args.remove_movable,
    )
    return seg.run(mesh, args.scan_path)


def run_dino_sam3(mesh: o3d.geometry.TriangleMesh, args: argparse.Namespace) -> SegmentorResult:
    from engine.vision.segmentor_dino_sam import DinoSam3Segmentor
    sam3_checkpoint = args.sam3_checkpoint or ""
    seg = DinoSam3Segmentor(
        grounding_dino_config=args.dino_config,
        grounding_dino_checkpoint=args.dino_checkpoint,
        sam3_checkpoint=sam3_checkpoint or None,
        n_horizontal=args.n_views_horizontal,
        n_oblique=args.n_views_oblique,
        render_width=args.render_width,
        render_height=args.render_height,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        min_vote_fraction=args.min_vote_fraction,
        min_movable_votes=args.min_movable_votes,
        sam3_confidence_threshold=args.sam3_confidence_threshold,
        protection_config=_build_protection_config(args),
        graph_cleanup_config=_build_graph_config(args),
        remove_movable=args.remove_movable,
    )
    return seg.run(mesh, args.scan_path)


def _build_protection_config(args: argparse.Namespace) -> StructuralProtectionConfig:
    min_votes = args.protection_min_votes
    return StructuralProtectionConfig(
        level=args.protection_level,
        outer_percentile=args.protection_outer_percentile,
        plane_tolerance_m=args.protection_tolerance,
        normal_cos_min=args.protection_normal_cos,
        protected_min_movable_votes=min_votes,
    )


def _build_graph_config(args: argparse.Namespace) -> GraphCleanupConfig:
    return GraphCleanupConfig(
        enabled=args.graph_cleanup,
        restore_band_m=args.graph_restore_band,
        prune_band_m=args.graph_prune_band,
        min_seed_vertices=args.graph_min_seed_vertices,
        min_seed_fraction=args.graph_min_seed_fraction,
        min_component_vertices=args.graph_min_component_vertices,
    )


def _make_output_tag(backend: str, args: argparse.Namespace) -> str:
    suffix = (args.output_suffix or "").strip().strip("_")
    if not suffix and args.graph_cleanup:
        suffix = f"{args.protection_level}_graph"
    return f"{backend}_{suffix}" if suffix else backend


def _reapply_protection_variant(
    mesh: o3d.geometry.TriangleMesh,
    raw_result: SegmentorResult,
    config: StructuralProtectionConfig,
    graph_config: GraphCleanupConfig | None = None,
) -> SegmentorResult:
    """Re-run only the cheap protection/stripping stage on raw AI labels."""
    movable_votes = np.asarray(raw_result.extra.get("votes_movable", []), dtype=np.int32)
    if len(movable_votes) != raw_result.n_total:
        movable_votes = np.zeros(raw_result.n_total, dtype=np.int32)

    protection = apply_structural_protection(
        mesh,
        raw_result.vertex_labels,
        config,
        movable_votes=movable_votes,
    )
    vertex_labels = protection.vertex_labels

    graph_cleanup = apply_graph_cleanup(
        mesh,
        vertex_labels,
        protection.prior,
        graph_config or GraphCleanupConfig(),
    )
    vertex_labels = graph_cleanup.vertex_labels
    label_colors = np.array([label_to_color(lbl) for lbl in vertex_labels], dtype=np.float32)

    label_map: dict[str, list[int]] = {}
    for vi, lbl in enumerate(vertex_labels):
        label_map.setdefault(lbl, []).append(vi)

    n_removable_candidates = sum(1 for lbl in vertex_labels if lbl not in PRESERVED_LABELS)
    remove_movable = bool(getattr(raw_result, "extra", {}).get("remove_movable", False))
    n_removed = n_removable_candidates if remove_movable else 0
    if config.level == "shell":
        structural_mesh = build_cuboid_shell_mesh(protection.prior)
        flatten_stats = {"enabled": False, "reason": "shell mode uses cuboid prior mesh"}
    else:
        flattened_mesh, flatten_stats = flatten_structural_labels_to_room_shell(
            mesh,
            vertex_labels,
            protection.prior,
        )
        review_mesh, virtual_shell_stats = append_virtual_room_shell(flattened_mesh, protection.prior)
        structural_mesh = (
            append_virtual_room_shell(strip_non_structural(flattened_mesh, vertex_labels), protection.prior)[0]
            if remove_movable
            else review_mesh
        )

    extra = dict(raw_result.extra)
    extra["structural_protection"] = protection.to_json()
    extra["graph_cleanup"] = graph_cleanup.to_json()
    extra["structural_flattening"] = flatten_stats
    if config.level != "shell":
        extra["virtual_room_shell"] = virtual_shell_stats
    extra["protection_sweep_source"] = raw_result.backend
    extra["remove_movable"] = remove_movable
    extra["n_removable_candidates"] = n_removable_candidates
    if config.level == "shell":
        extra["shell_output"] = "cuboid room-shell mesh from estimated protected bounds"

    return SegmentorResult(
        structural_mesh=structural_mesh,
        vertex_labels=vertex_labels,
        label_colors=label_colors,
        label_map=label_map,
        n_removed=n_removed,
        backend=raw_result.backend,
        extra=extra,
    )


def _run_dino_sam2_protection_sweep(
    mesh: o3d.geometry.TriangleMesh,
    args: argparse.Namespace,
) -> None:
    """Run DINO+SAM2 once, then export every protection level variant."""
    raw_args = copy.copy(args)
    raw_args.protection_level = "off"
    raw_args.protection_min_votes = None
    raw_args.graph_cleanup = False

    print(f"\n{SEP}")
    print("  Protection sweep: DINO+SAM2 raw semantic pass")
    print(SEP)
    raw_result = run_dino_sam2(mesh, raw_args)

    for level in args.protection_levels:
        variant_args = copy.copy(args)
        variant_args.protection_level = level
        config = _build_protection_config(variant_args)
        graph_config = _build_graph_config(variant_args)
        result = _reapply_protection_variant(mesh, raw_result, config, graph_config)
        tag = f"dino_sam2_{level}{'_graph' if graph_config.enabled else ''}"

        protection = result.extra.get("structural_protection", {})
        print(f"\n  Results — dino_sam2 / protection={level}:")
        print(f"    Total vertices : {result.n_total:,}")
        print(f"    Protected      : {protection.get('n_protected', 0):,}")
        print(f"    Restored       : {protection.get('n_restored', 0):,}")
        if graph_config.enabled:
            graph = result.extra.get("graph_cleanup", {})
            print(f"    Graph restored : {graph.get('n_graph_restored', 0):,}")
            print(f"    Graph pruned   : {graph.get('n_graph_pruned', 0):,}")
        print(f"    Removed        : {result.n_removed:,}  ({result.removal_fraction*100:.1f}%)")
        print("    Label breakdown:")
        for lbl, idxs in sorted(result.label_map.items(), key=lambda x: -len(x[1])):
            pct = 100.0 * len(idxs) / max(1, result.n_total)
            print(f"      {lbl:<28} {len(idxs):>7,}  ({pct:.1f}%)")

        _write_result_outputs(mesh, result, tag)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run 3D segmentation on the server room scan.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--scan", default=str(OBJ_PATH), help="Path to OBJ/PLY scan file.")
    p.add_argument(
        "--backend",
        choices=["mask3d", "dino_sam2", "dino_sam3", "both"],
        default="both",
        help="Which segmentation backend(s) to run.",
    )
    p.add_argument(
        "--output-suffix",
        default="",
        help="Optional suffix appended to output filenames, e.g. 'balanced' writes s3_seg_dino_sam2_balanced_*.ply.",
    )
    p.add_argument(
        "--output-dir",
        default=str(OUT_DIR),
        help="Directory for viewer segmentation artifacts.",
    )

    # ── Mask3D ────────────────────────────────────────────────────────────
    m = p.add_argument_group("Mask3D options")
    m.add_argument("--mask3d-checkpoint", default=str(DEFAULT_MASK3D_CHECKPOINT), help="Path to Mask3D .ckpt/.pth file.")

    # ── Grounded-DINO + SAM2 ─────────────────────────────────────────────
    d = p.add_argument_group("Grounded-DINO + SAM options")
    d.add_argument("--dino-config",       default=str(DEFAULT_DINO_CONFIG), help="GroundingDINO config .py path.")
    d.add_argument("--dino-checkpoint",   default=str(DEFAULT_DINO_CHECKPOINT), help="GroundingDINO .pth checkpoint.")
    d.add_argument("--sam2-checkpoint",   default=str(DEFAULT_SAM2_CHECKPOINT), help="SAM2 .pt checkpoint.")
    d.add_argument("--sam2-config",       default=DEFAULT_SAM2_CONFIG)
    d.add_argument("--sam3-checkpoint",   default=str(DEFAULT_SAM3_CHECKPOINT), help="SAM3 .pt checkpoint. If missing/empty, SAM3 tries Hugging Face download.")
    d.add_argument("--sam3-confidence-threshold", type=float, default=0.50)
    d.add_argument("--n-views-horizontal", type=int, default=8)
    d.add_argument("--n-views-oblique",    type=int, default=4)
    d.add_argument("--render-width",       type=int, default=800)
    d.add_argument("--render-height",      type=int, default=600)
    d.add_argument("--box-threshold",      type=float, default=0.35)
    d.add_argument("--text-threshold",     type=float, default=0.25)
    d.add_argument("--min-vote-fraction",  type=float, default=0.30,
                   help="Fraction of views that must agree to label a vertex non-structural.")
    d.add_argument("--min-movable-votes", type=int, default=2,
                   help="Minimum distinct rendered-view votes required before a vertex can be removed as movable clutter.")
    d.add_argument("--remove-movable", action="store_true",
                   help="Actually delete explicit movable labels. Disabled by default so the viewer/voxelizer keep the full picture before destructive cleanup.")

    # ── Structural shell protection ──────────────────────────────────────
    s = p.add_argument_group("Structural room-shell protection")
    s.add_argument(
        "--protection-level",
        choices=["off", "light", "balanced", "strong", "shell"],
        default="balanced",
        help="How strongly the outer wall/floor/ceiling cuboid prior protects geometry.",
    )
    s.add_argument(
        "--protection-tolerance",
        type=float,
        default=0.12,
        help="Distance in metres from an estimated shell plane for protected vertices.",
    )
    s.add_argument(
        "--protection-outer-percentile",
        type=float,
        default=0.5,
        help="Robust percentile used for min/max room shell bounds.",
    )
    s.add_argument(
        "--protection-normal-cos",
        type=float,
        default=0.35,
        help="Minimum absolute normal component along a shell axis to protect a vertex.",
    )
    s.add_argument(
        "--protection-min-votes",
        type=int,
        default=None,
        help="Override movable votes required to delete a protected vertex.",
    )
    s.add_argument(
        "--protection-sweep",
        action="store_true",
        help="For DINO+SAM2, run semantic inference once and export one viewer asset set per protection level.",
    )
    s.add_argument(
        "--protection-levels",
        nargs="+",
        choices=["off", "light", "balanced", "strong", "shell"],
        default=["off", "light", "balanced", "strong", "shell"],
        help="Levels exported when --protection-sweep is enabled.",
    )

    # ── Graph-based label fusion ────────────────────────────────────────
    g = p.add_argument_group("Graph-based structural fusion")
    g.add_argument(
        "--graph-cleanup",
        action="store_true",
        help="Restore shell-connected near-wall/floor/ceiling vertices and prune disconnected structural islands.",
    )
    g.add_argument(
        "--graph-restore-band",
        type=float,
        default=0.35,
        help="Distance in metres from the estimated shell where graph-connected movable labels may be restored.",
    )
    g.add_argument(
        "--graph-prune-band",
        type=float,
        default=0.65,
        help="Structural vertices farther than this from the shell are pruned as inward appendages unless protected.",
    )
    g.add_argument(
        "--graph-min-seed-vertices",
        type=int,
        default=8,
        help="Minimum protected shell seeds needed to keep a structural graph component.",
    )
    g.add_argument(
        "--graph-min-seed-fraction",
        type=float,
        default=0.002,
        help="Protected seed fraction sufficient to keep a structural graph component.",
    )
    g.add_argument(
        "--graph-min-component-vertices",
        type=int,
        default=48,
        help="Minimum near-shell structural component size kept even without protected seeds.",
    )

    return p


def main() -> None:
    global OUT_DIR
    args = build_parser().parse_args()
    scan_path = Path(args.scan).expanduser().resolve()
    args.scan_path = scan_path
    OUT_DIR = Path(args.output_dir).expanduser().resolve()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{SEP}")
    print(f"  Loading and pre-processing scan: {scan_path.name}")
    print(SEP)
    raw_mesh, aligned_mesh = clean_and_align_meshes(scan_path)
    log.info(
        "Aligned mesh: %d verts, %d tris",
        len(np.asarray(aligned_mesh.vertices)),
        len(np.asarray(aligned_mesh.triangles)),
    )

    backends = ["mask3d", "dino_sam3"] if args.backend == "both" else [args.backend]

    for backend in backends:
        print(f"\n{SEP}")
        print(f"  Backend: {backend.upper()}")
        print(SEP)

        if backend == "dino_sam2" and args.protection_sweep:
            _run_dino_sam2_protection_sweep(aligned_mesh, args)
            continue

        if backend == "mask3d":
            if not args.mask3d_checkpoint:
                log.error("--mask3d-checkpoint is required for the mask3d backend.")
                sys.exit(1)
            result = run_mask3d(aligned_mesh, args)
        elif backend == "dino_sam2":
            for flag, name in [
                (args.dino_config,     "--dino-config"),
                (args.dino_checkpoint, "--dino-checkpoint"),
                (args.sam2_checkpoint, "--sam2-checkpoint"),
            ]:
                if not flag:
                    log.error("%s is required for the dino_sam2 backend.", name)
                    sys.exit(1)
            result = run_dino_sam2(aligned_mesh, args)
        else:
            for flag, name in [
                (args.dino_config,     "--dino-config"),
                (args.dino_checkpoint, "--dino-checkpoint"),
            ]:
                if not flag:
                    log.error("%s is required for the dino_sam3 backend.", name)
                    sys.exit(1)
            if args.sam3_checkpoint and not Path(args.sam3_checkpoint).exists():
                log.error(
                    "SAM3 checkpoint not found at %s. Download requires access to the gated Hugging Face repo facebook/sam3.",
                    args.sam3_checkpoint,
                )
                sys.exit(1)
            result = run_dino_sam3(aligned_mesh, args)

        print(f"\n  Results — {backend}:")
        print(f"    Total vertices : {result.n_total:,}")
        print(f"    Removed        : {result.n_removed:,}  ({result.removal_fraction*100:.1f}%)")
        print(f"    Label breakdown:")
        for lbl, idxs in sorted(result.label_map.items(), key=lambda x: -len(x[1])):
            pct = 100.0 * len(idxs) / max(1, result.n_total)
            print(f"      {lbl:<28} {len(idxs):>7,}  ({pct:.1f}%)")

        # ── Write outputs ────────────────────────────────────────────────
        _write_result_outputs(aligned_mesh, result, _make_output_tag(backend, args))

    print(f"\n{SEP}")
    print(f"  Done. Files written to: {OUT_DIR}")
    print(f"\n  Open the viewer:")
    print(f"    cd {PROJECT_ROOT / 'server_room_phone'}")
    print(f"    python -m http.server 8788")
    print(f"    http://localhost:8788/pipeline_viewer.html?vis={OUT_DIR.name}")
    print(SEP)


if __name__ == "__main__":
    main()
