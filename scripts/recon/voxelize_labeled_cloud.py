#!/usr/bin/env python3
"""Adapter: labeled Pi3/SAM3 point cloud -> metric voxel grid (room shell + solid object fill).

Bridges the new CV output (a recon_web run's per-instance colored ``labeled.ply`` +
``labeled.legend.json``) to the existing METRIC voxelizer. The voxelizer can't ingest
the cloud directly (it's up-to-scale, instance labels are "server rack N", and its
DBSCAN would re-merge our flush per-rack instances), so this adapter:

  1. decodes each point to its instance via the legend palette (nearest colour),
  2. scales the up-to-scale cloud to METRES per axis (see the recovery tiers below),
  3. shifts the floor to z=0,
  4. builds a clean virtual room shell (cuboid) as the background,
  5. stamps each per-rack / AC instance as a SOLID canonical box via the voxelizer's
     own `_stamp_detected_priors` (fed OUR instances -> bypasses the merging DBSCAN,
     fills hollow interiors with the 42U / AC prior),
  6. writes the int8 voxel grid (.npy) + a colour-by-label voxel-centre PLY to view.

Recovery is TIERED, CV first; each tier is the explicit fallback for the one before, and the DEFAULT is
the most general (no room knowledge supplied):

  * MAIN PATH (generalising CV) -- the default. Recover scale, room shell, and layout from the labeled
    cloud ALONE: the rack cuboid + the detected walls. See ``_rectify_warp`` (unbend a warped recon) and
    ``_fit_axis_scales`` (per-axis scale), then the detected-wall room shell. No room dimensions are given,
    so this path applies to ANY scanned room.
  * FALLBACK (prior-pinned) -- ``--scale-anchor room``. Only for captures too degraded for the main path:
    pin the scale and shell to a SUPPLIED ``--room-dims`` spec (``_fit_room_box_scale``).
  * FALLBACK -- ``--reference``. The CV registers the room, then instances from a supplied placements
    file are stamped instead of being derived from the cloud.

Usage (halo env):
    python scripts/recon/voxelize_labeled_cloud.py --run tools/recon_web/runs/pi3_chest32_final
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from engine.core.config import (
    AC_UNIT_DIMENSIONS, COOLING_AC_VENT, OBSTACLE_WALL, RACK_BODY,
    RACK_DIMENSIONS, RACK_EXHAUST, RACK_INTAKE, SPACE_EMPTY, VOXEL_SIZE,
)
from engine.core.data_types import Coordinate, RackFacing, RackPlacement
from engine.vision.voxelizer import (
    _build_layout_grid, _stamp_box, _stamp_detected_priors, _stamp_rack,
)

# Object handling falls into two priors:
#   PRIOR_FILL  — important objects with a known canonical size (rack/AC) -> stamped to spec.
#   background  — everything else static and wall-mounted: stamped flush to the nearest wall at
#                 its own (known per-class or measured) thickness — NOT extruded to the shell wall
#                 (the shell sits ~0.4 m behind the real wall, which over-thickens a thin cabinet).
PRIOR_FILL = {"server rack", "ac_unit"}      # important objects -> canonical prior dimensions
STRUCTURAL = {"wall", "floor", "ceiling"}    # the room shell
SPECIAL = {"ups"}                            # rack-adjacent -> own placement logic (not wall-relative)
CLUTTER = {"trash can", "chair", "cardboard box"}   # movable -> not part of the static twin
BG_VOX = {"fire hose": OBSTACLE_WALL}        # known background label -> voxel id (default OBSTACLE_WALL)
BG_DIMS = {"fire hose": (0.80, 0.30, 0.80)}  # known background label -> (W,D,H) m; else measured
CABINET_DEPTH_MIN = 0.40                     # floor on measured depth (only the front face is captured)
REAR_CLEARANCE = 0.50                        # rear service clearance (m) behind each rack row to the
#                                              wall — the datacentre layout prior used to RECOVER the
#                                              occluded room depth from the rack footprint (see below)
CEILING_CLEARANCE = 0.30                      # overhead clearance (m) above the racks (cable tray /
#                                              lighting / plenum) — recovers the room HEIGHT when a
#                                              chest-height capture under-reads the ceiling
UPS_VOX = 8                                  # ad-hoc voxel id for the UPS (no config id yet)
POWER_VOX = 9                                # electrical distribution cabinet (the green panel)
NETRACK_VOX = 10                             # standalone network rack (taller than a 42U server rack)
NET_RACK_HEIGHT = 2.20                       # network rack height prior (m); width = rack_w, depth TBD
# voxel-id -> RGB for the viewer
VOX_COLOR = {
    OBSTACLE_WALL:   (0.62, 0.62, 0.66),   # room shell / fire-hose cabinet (obstacle)
    RACK_BODY:       (0.10, 0.85, 0.35),   # rack body (solid 42U)
    RACK_INTAKE:     (0.20, 0.55, 1.00),   # front / cold-aisle intake
    RACK_EXHAUST:    (0.95, 0.35, 0.10),   # rear / hot-aisle exhaust
    COOLING_AC_VENT: (0.95, 0.15, 0.85),   # AC
    UPS_VOX:         (0.95, 0.85, 0.10),   # UPS cabinet
    POWER_VOX:       (0.42, 0.45, 0.28),   # military-green distribution cabinet (muted olive)
    NETRACK_VOX:     (0.20, 0.95, 0.85),   # network rack (teal, distinct from the green server racks)
}


def base_label(name: str) -> str:
    """'server rack 7' -> 'server rack'; 'ac_unit 1' -> 'ac_unit'."""
    return re.sub(r"\s+\d+$", "", name).strip()


def apply_placements(grid: np.ndarray, placements: list, origin: np.ndarray) -> None:
    """Stamp the editable instance manifest into a grid. Shared by the voxelizer (initial bake)
    and the re-stamp tool / editor so an edited manifest reproduces the room exactly. Racks carry
    their _stamp_rack params (pos/facing/rack_type); every other instance is an axis-aligned box
    given by its world centre + (w,d,h) dims."""
    for pl in placements:
        if pl.get("kind") == "rack":
            _stamp_rack(grid, RackPlacement(position=Coordinate(*pl["pos"]),
                        facing=RackFacing[pl["facing"]], rack_type=pl["rack_type"]), origin)
        else:
            cx, cy, cz = pl["center"]; w, d, h = pl["dims"]
            vw = max(1, round(w / VOXEL_SIZE)); vd = max(1, round(d / VOXEL_SIZE))
            # ROUND the box MIN corner to the grid (robust to float drift like 0.7999.. AND to odd
            # widths), then hand _stamp_box the matching centre index so the stamped voxels are exactly
            # the editor's box -- no 1-voxel drift between the voxel result and the editor.
            ixmin = int(round((cx - w / 2.0 - origin[0]) / VOXEL_SIZE))
            iymin = int(round((cy - d / 2.0 - origin[1]) / VOXEL_SIZE))
            izmin = int(round((cz - h / 2.0 - origin[2]) / VOXEL_SIZE))
            _stamp_box(grid, ixmin + vw // 2, iymin + vd // 2, izmin, int(pl["vox_id"]), (w, d, h))


def separate_wall_boxes(placements: list, ext: np.ndarray) -> None:
    """Push wall-mounted boxes that share a wall apart ALONG that wall so they don't overlap — the
    detected fire-hose / power-cabinet boxes can land overlapping. Racks and the rack-adjacent UPS
    are left alone (their positions are set by their own logic)."""
    boxes = [p for p in placements if p.get("kind") == "box" and p["name"] != "ups"]
    for p in boxes:
        cx, cy = p["center"][0], p["center"][1]
        dist = {"y0": cy, "ymax": ext[1] - cy, "x0": cx, "xmax": ext[0] - cx}
        p["_wall"] = min(dist, key=dist.get)
    for wall in {p["_wall"] for p in boxes}:
        ax = 0 if wall in ("y0", "ymax") else 1                 # axis that runs ALONG this wall
        grp = sorted((p for p in boxes if p["_wall"] == wall), key=lambda p: p["center"][ax])
        for i in range(1, len(grp)):
            prev_hi = grp[i - 1]["center"][ax] + grp[i - 1]["dims"][ax] / 2.0
            cur_lo = grp[i]["center"][ax] - grp[i]["dims"][ax] / 2.0
            if cur_lo < prev_hi:
                grp[i]["center"][ax] += prev_hi - cur_lo        # shift to touch, no overlap
    for p in boxes:
        del p["_wall"]


def _rectify_warp(P, pt_base):
    """MAIN PATH (generalising CV) -- step 1 of the default recovery, runs for every multi-view capture.

    Unbend a warped Pi3 recon using the CEILING outline as straightening rails (rows->+X frame out). Uses
    no room dimensions, only the cloud's own geometry, so it applies to any room.

    A point cloud is hollow and the racks occlude the FLOOR and the WALLS, so those outlines pinch to the
    aisle and look like a severe wedge -- but the CEILING is above the racks and sees the room's TRUE
    rectangular footprint. So: rotate to the room's principal axes, then for each cross-section fit the
    ceiling's two rails (perpendicular low/high edge as a smooth quadratic of the along coordinate) and map
    that band to a CONSTANT width -- this straightens the bend and normalises the width on both X and Y.
    Finally remove the floor tilt so the floor is level. On a rigid recon the rails are already straight,
    so it is ~identity. Returns the rotated, rectified cloud.
    """
    S = P[np.isin(pt_base, np.array(["ceiling", "floor"], dtype=object))]
    c = S[:, :2].mean(0)
    _, _, vt = np.linalg.svd(S[:, :2] - c, full_matrices=False)
    th = -np.arctan2(vt[0, 1], vt[0, 0]); cs, sn = np.cos(th), np.sin(th)
    Q = (P[:, :2] - c) @ np.array([[cs, sn], [-sn, cs]])
    X = Q[:, 0].astype(float); Y = Q[:, 1].astype(float); Z = P[:, 2].astype(float)
    ceil = pt_base == "ceiling"

    def straighten(a, pv, mask, nb=20):
        """Return a map pv -> constant-width band from `mask`'s low/high rails (quadratic in `a`)."""
        av = a[mask]; pvv = pv[mask]
        if len(av) < 120:
            return None
        a0, a1 = float(np.percentile(av, 1)), float(np.percentile(av, 99))
        ae = np.linspace(a0, a1, nb + 1); ac = 0.5 * (ae[:-1] + ae[1:])
        lo = np.full(nb, np.nan); hi = np.full(nb, np.nan)
        for i in range(nb):
            m = (av >= ae[i]) & (av < ae[i + 1])
            if int(m.sum()) >= 40:
                lo[i] = np.percentile(pvv[m], 4); hi[i] = np.percentile(pvv[m], 96)
        g = ~np.isnan(lo)
        if int(g.sum()) < 3:
            return None
        plo = np.polyfit(ac[g], lo[g], 2); phi = np.polyfit(ac[g], hi[g], 2)
        wm = float(np.mean(hi[g] - lo[g]))

        def fmap(a_all, pv_all):
            ac_ = np.clip(a_all, a0, a1)                  # clamp so the quadratic never extrapolates wild
            lo_ = np.polyval(plo, ac_); hi_ = np.polyval(phi, ac_)
            return (pv_all - lo_) / np.clip(hi_ - lo_, 0.3, None) * wm
        return fmap

    fy = straighten(X, Y, ceil)
    if fy is not None:
        Y = fy(X, Y)                                      # straighten Y rails along X
    fx = straighten(Y, X, ceil)
    if fx is not None:
        X = fx(Y, X)                                      # straighten X rails along the rectified Y
    fl = pt_base == "floor"
    if int(fl.sum()) >= 100:                              # remove floor tilt -> level floor
        A = np.c_[X[fl], Y[fl], np.ones(int(fl.sum()))]
        coef, *_ = np.linalg.lstsq(A, Z[fl], rcond=None)
        Z = Z - (X * coef[0] + Y * coef[1])
    return np.column_stack([X, Y, Z])


def _fit_room_box_scale(pt_base, P, room_dims):
    """FALLBACK (prior-pinned; degraded captures only) -- scale-anchor=room, OFF by default.

    Per-axis metre-per-unit from a SUPPLIED room-dimension spec, in the rows->+X frame. Used only when the
    MAIN PATH CV recovery (_fit_axis_scales) is unreliable on a degraded capture; it relies on the supplied
    dimensions instead of recovering the scale from the cloud.

    ``room_dims`` (= length X, depth Y, height Z) is treated as known, so the room box -- not the noisy,
    occluded racks -- is the ruler:
      sz = room height / floor-to-ceiling gap   (floor and ceiling are big horizontal surfaces);
      sx = room length / CEILING footprint X extent;
      sy = room depth  / CEILING footprint Y extent.
    The footprint comes from the CEILING, not the floor: the racks occlude the floor (and the walls), so
    the floor footprint pinches to the aisle and under-reads the room; the ceiling is above the racks and
    sees the true rectangle. Falls back to the floor, then all points, if the ceiling is sparse.
    """
    Rx, Ry, Rz = (float(d) for d in room_dims)
    fl = P[pt_base == "floor"]; ce = P[pt_base == "ceiling"]
    floor_z = float(np.percentile(fl[:, 2], 50)); ceil_z = float(np.percentile(ce[:, 2], 50))
    sz = Rz / max(ceil_z - floor_z, 1e-6)
    fp = ce if len(ce) >= 100 else (fl if len(fl) >= 100 else P)   # CEILING footprint (unoccluded)
    fx = float(np.percentile(fp[:, 0], 99) - np.percentile(fp[:, 0], 1)) or 1e-6
    fy = float(np.percentile(fp[:, 1], 99) - np.percentile(fp[:, 1], 1)) or 1e-6
    sx = Rx / fx; sy = Ry / fy
    print(f"room-box per-axis scale: sx={sx:.3f} ({Rx:.1f}/{fx:.3f}u ceiling X)  "
          f"sy={sy:.3f} ({Ry:.1f}/{fy:.3f}u ceiling Y)  sz={sz:.3f} ({Rz:.1f}/{ceil_z - floor_z:.3f}u floor->ceil)")
    return sx, sy, sz


def _fit_axis_scales(R, R_names, pt_base, P, args):
    """MAIN PATH (generalising CV) -- step 2 of the default recovery, scale-anchor=rack (the default).

    Per-axis metre-per-unit (sx, sy, sz) recovered from the rack SURFACE by CV alone, in the rows->+X
    frame -- NO room dimensions supplied, so it generalises to any room. The room-spec
    fallback (_fit_room_box_scale) is only taken when this is too unreliable on a degraded capture.

    A Pi3 recon is up-to-scale AND anisotropic, and a point cloud is hollow (only the visible SURFACES
    return points), so each axis is recovered from the prior-shaped surface it actually captures -- never
    from a filled volume:

      sx = rack_w / per-rack PITCH
           The front faces tile the row at one rack width. The pitch is the MEDIAN gap between adjacent
           rack-instance centres -- a per-rack quantity, so it is immune both to occluded END racks (the
           raw row extent truncates there and over-scales the room -> "too long") and to SAM3 split/merge
           (the median gap survives a few bad instances). Falls back to (n_std * rack_w)/row-extent only
           when a row has too few instances to form a pitch.
      sz = rack height / captured body height
           The full height is seen head-on, so the front-face Z span is a clean surface measurement.
      sy = aisle / front-to-front row spacing
           The filled body DEPTH is occluded (only the front shell returns), so each row's captured Y
           centre sits on its FRONT face and the two are one aisle apart. The room depth itself comes
           from the datacentre layout prior later, not from these points.
    """
    rack_w, _rack_d, rack_h = RACK_DIMENSIONS[args.rack_type]
    floor_u = (float(np.percentile(P[pt_base == "floor"][:, 2], 50)) if (pt_base == "floor").any()
               else float(np.percentile(R[:, 2], 1)))
    X, Y, Z = R[:, 0], R[:, 1], R[:, 2] - floor_u
    Hz = float(np.percentile(Z, 99) - np.percentile(Z, 1)) or 1e-6
    sz = rack_h / Hz
    # split into two rows at the depth-density valley near the middle
    yh, ye = np.histogram(Y, bins=60); ycb = 0.5 * (ye[:-1] + ye[1:]); q = max(1, len(yh) // 3)
    ysplit = float(ycb[q + int(np.argmin(yh[q:2 * q]))]) if 2 * q < len(yh) else float(np.median(Y))
    rows = [m for m in (Y < ysplit, Y >= ysplit) if int(m.sum()) >= 50]
    n_std = args.racks_per_row if args.racks_per_row > 0 else 6      # standard racks per row (layout prior)

    def edge(a):
        return float(np.percentile(a, 99.7) - np.percentile(a, 0.3)) or 1e-6

    def pitch(mask):
        """Median centre-to-centre gap of the row's rack instances (one rack width), or 0 if too few."""
        names = R_names[mask]; xs = X[mask]
        cents = sorted(float(xs[names == n].mean()) for n in np.unique(names) if (names == n).sum() >= 30)
        gaps = np.diff(cents)
        return float(np.median(gaps)) if len(gaps) >= 2 else 0.0

    pitches = [p for p in (pitch(m) for m in rows) if p > 1e-6]
    if pitches:
        pmed = float(np.median(pitches))
        sx = rack_w / pmed
        sx_src = f"rack_w {rack_w:.2f}/{pmed:.3f}u pitch over {len(pitches)} row(s)"
    else:                                                   # no instances -> truncation-prone extent
        widths = [edge(X[m]) for m in rows] or [edge(X)]
        sx = n_std * rack_w / min(widths)
        sx_src = f"{n_std}*{rack_w:.2f}/{min(widths):.3f}u row extent (no instances)"
    if len(rows) == 2:
        d_ff = abs(float(np.median(Y[rows[0]])) - float(np.median(Y[rows[1]]))) or 1e-6
        sy = args.aisle / d_ff
    else:
        d_ff = 0.0
        sy = sx
    print(f"per-axis scale: sx={sx:.3f} ({sx_src})  "
          f"sy={sy:.3f} (aisle {args.aisle:.1f}/{round(d_ff, 3)}u front-to-front)  "
          f"sz={sz:.3f} (rack {rack_h:.2f}/{Hz:.3f}u body)")
    return sx, sy, sz


def voxelize_labeled(run, *, rack_type="42U_real", y0_pad=0.0, face_depth=70,
                     aisle=1.1, room_depth=0.0, metric=False, racks_per_row=0, ups_at=None,
                     net_rack_depth=0.75, room_dims=(7.4, 3.9, 2.4), scale_anchor="rack",
                     rectify=True, reference=None):
    """Importable entry point (mirrors the CLI). Loads ``<run>/labeled.ply`` (+ the ``point_labels.npz``
    sidecar), writes ``voxel_grid.npy`` / ``voxel.ply`` / ``placements.json`` / ``voxel_empty*`` into
    ``<run>``, and returns ``(grid, placements, origin)`` so the backend can use them directly.
    ``metric=True`` skips the scale anchor for inputs already in metres (LAS/LAZ). MAIN PATH (the default)
    ``scale_anchor="rack"`` recovers the scale + room from the CV alone (the rack cuboid + the detected
    walls), so it generalises to any room. The ``scale_anchor="room"`` FALLBACK pins the scale to a supplied
    ``room_dims`` spec only when the CV is too degraded; ``reference`` is a last-resort fallback that places
    instances from a supplied layout file."""
    return _voxelize(argparse.Namespace(run=Path(run), rack_type=rack_type, y0_pad=y0_pad,
                                        face_depth=face_depth, aisle=aisle, room_depth=room_depth,
                                        metric=metric, racks_per_row=racks_per_row, ups_at=ups_at,
                                        net_rack_depth=net_rack_depth, room_dims=tuple(room_dims),
                                        scale_anchor=scale_anchor, rectify=rectify, reference=reference))


def _finalize_twin(args, placements, origin, ext, shape):
    """Shared tail (4f + write): voxel-align the boxes, REBUILD the grid from the final manifest, and
    write placements.json + voxel_grid.npy / voxel.ply + the movables-removed empty room. Shared by the
    cloud-derived path and the --reference path so both emit identical artifacts from a manifest."""
    separate_wall_boxes(placements, ext)
    for pl in placements:
        if pl.get("kind") == "box":
            pl["dims"] = [max(1, round(v / VOXEL_SIZE)) * VOXEL_SIZE for v in pl["dims"]]
            pl["center"] = [round((c - dd / 2.0) / VOXEL_SIZE) * VOXEL_SIZE + dd / 2.0
                            for c, dd in zip(pl["center"], pl["dims"])]
    grid = _build_layout_grid(shape, np.int8)
    apply_placements(grid, placements, origin)
    manifest = {"voxel_size": VOXEL_SIZE, "shape": [int(s) for s in shape],
                "origin": [float(o) for o in origin], "ext": [float(e) for e in ext],
                "rack_type": args.rack_type,
                "vox_color": {str(k): list(v) for k, v in VOX_COLOR.items()},
                "instances": placements}
    (args.run / "placements.json").write_text(json.dumps(manifest, indent=2))
    print(f"manifest -> placements.json ({len(placements)} instances)")
    np.save(args.run / "voxel_grid.npy", grid)
    occ = np.argwhere(grid != SPACE_EMPTY)
    cents = (occ + 0.5) * VOXEL_SIZE + origin
    rgb = np.array([VOX_COLOR.get(int(grid[x, y, z]), (0.4, 0.4, 0.42)) for x, y, z in occ])
    trimesh.PointCloud(vertices=cents.astype(np.float32),
                       colors=(rgb * 255).astype(np.uint8)).export(args.run / "voxel.ply")
    counts = {int(v): int((grid == v).sum()) for v in np.unique(grid) if v != SPACE_EMPTY}
    print(f"grid -> {args.run/'voxel_grid.npy'} ; voxel.ply ({len(occ):,} occupied voxels)")
    print(f"voxel counts by id: {counts}")
    MOVABLE = (RACK_BODY, RACK_INTAKE, RACK_EXHAUST, COOLING_AC_VENT, NETRACK_VOX)
    empty = grid.copy()
    empty[np.isin(empty, MOVABLE)] = SPACE_EMPTY
    np.save(args.run / "voxel_empty_grid.npy", empty)
    occ_e = np.argwhere(empty != SPACE_EMPTY)
    rgb_e = np.array([VOX_COLOR.get(int(empty[x, y, z]), (0.4, 0.4, 0.42)) for x, y, z in occ_e])
    trimesh.PointCloud(vertices=((occ_e + 0.5) * VOXEL_SIZE + origin).astype(np.float32),
                       colors=(rgb_e * 255).astype(np.uint8)).export(args.run / "voxel_empty.ply")
    print(f"empty room (movables removed) -> voxel_empty_grid.npy ; voxel_empty.ply "
          f"({len(occ_e):,} voxels, ids {sorted(int(v) for v in np.unique(empty) if v)})")
    return grid, placements, origin


def _voxelize(args):
    pc = trimesh.load(args.run / "labeled.ply", process=False)
    P = np.asarray(pc.vertices, np.float64)

    # 1. per-point instance name — prefer the label sidecar (colours can collide); else decode.
    sidecar = args.run / "point_labels.npz"
    if sidecar.exists():
        pt_name = np.load(sidecar, allow_pickle=False)["names"].astype(object)
        print(f"using label sidecar ({len(pt_name)} points)")
    else:
        pal = json.loads((args.run / "labeled.legend.json").read_text())["palette"]
        C = np.asarray(pc.colors)[:, :3].astype(np.float64) / 255.0
        names = list(pal); cols = np.array([pal[n] for n in names], np.float64)
        pt_name = np.array(names, dtype=object)[cKDTree(cols).query(C, k=1)[1]]
        print("WARNING: no sidecar — decoding by colour (may merge collided instances)")
    # Optional network-rack/UPS EXEMPLAR location (gravity-frame centroid from a SAM3 box prompt). It is
    # appended as one extra point so it rides through the SAME scale/Manhattan/shift transforms as the
    # cloud; after the racks are placed we read its room-frame position and RE-TAG the nearest placed
    # rack as the UPS (no carving -> the scale anchor is untouched).
    ux_idx = None
    if getattr(args, "ups_at", None):
        ua = np.array([[float(t) for t in str(args.ups_at).split(",")]], np.float64)
        P = np.vstack([P, ua]); pt_name = np.append(pt_name, np.array(["unknown"], dtype=object))
        ux_idx = len(P) - 1
    pt_base = np.array([base_label(str(n)) for n in pt_name], dtype=object)

    # 2. ROTATE the rack rows to +X, then SCALE PER AXIS. A Pi3 recon is UP-TO-SCALE *and* ANISOTROPIC,
    #    so a single scalar cannot recover the room. MAIN PATH (scale-anchor=rack, _fit_axis_scales, the
    #    default): recover the per-axis scale from the CV alone -- sz from the rack body height, sx from the
    #    per-rack pitch, sy from the row-to-row spacing -- so the pipeline GENERALISES to any room with no
    #    dimensions given. The FALLBACK (scale-anchor=room, _fit_room_box_scale) instead pins the scale to a
    #    supplied room-dimension spec, ONLY for the degraded captures where the CV cannot recover the room on
    #    its own. An already-metric input (LAS/LAZ, or metric=True) keeps 1.0 on every axis.
    rmask = pt_base == "server rack"
    if (not getattr(args, "metric", False)) and getattr(args, "rectify", True) \
            and (pt_base == "ceiling").any() and (pt_base == "floor").any():
        P = _rectify_warp(P, pt_base)                       # unbend the warp + rotate rows -> +X
        print("rectified: ceiling-rail unbend + floor levelled (rows -> X)")
    elif rmask.sum() >= 50:                                  # align rack rows to +X (needs rack points)
        rxy0 = P[rmask][:, :2]
        _, _, rvt = np.linalg.svd(rxy0 - rxy0.mean(0), full_matrices=False)
        th = -np.arctan2(rvt[0][1], rvt[0][0]); c_, s_ = np.cos(th), np.sin(th)
        P[:, :2] = P[:, :2] @ np.array([[c_, s_], [-s_, c_]])
        print(f"Manhattan yaw: rotated {np.degrees(th):+.1f} deg (rows -> X)")
    else:
        print("Manhattan yaw: skipped (no rack rows to align to)")
    if getattr(args, "metric", False):
        sx = sy = sz = 1.0
        print("metric input: per-axis scale = 1.0 (cloud already in metres)")
    elif rmask.sum() < 50:
        raise ValueError("no 'server rack' points to anchor metric scale on; pass metric=True if the "
                         "input is already in metres")
    elif args.scale_anchor == "room" and (pt_base == "floor").any() and (pt_base == "ceiling").any():
        sx, sy, sz = _fit_room_box_scale(pt_base, P, args.room_dims)   # FALLBACK: pin to the room-dim spec
    else:
        sx, sy, sz = _fit_axis_scales(P[rmask], pt_name[rmask], pt_base, P, args)   # MAIN PATH: CV from the rack cuboid
    Pm = P * np.array([sx, sy, sz])
    fz = (np.percentile(Pm[pt_base == "floor"][:, 2], 50) if (pt_base == "floor").any() else Pm[:, 2].min())
    Pm[:, 2] -= fz

    # 3. fit the room shell to the DETECTED wall/ceiling planes, NOT the raw bounding box. The bbox
    #    is inflated by stray points beyond the walls (e.g. y0 here: bbox 0.00 vs real wall ~0.33),
    #    so flush objects landed in a phantom gap and any surface-to-wall extrude grew through the
    #    true wall. Take the OUTERMOST consistent plane per side: a robust percentile of wall points,
    #    widened to enclose every rack so nothing pokes through; floor at z=0, ceiling at its plane.
    rk = Pm[pt_base == "server rack"]
    # 3. Room shell. MAIN PATH (the default): fit the shell to the DETECTED wall / ceiling planes (the else
    #    branch below), recovering the occluded depth and height from the datacentre cross-section -- a
    #    pure-CV room that generalises. The FALLBACK (scale-anchor=room) instead sets the shell to the
    #    supplied room-dimension spec exactly, ONLY for degraded captures where the detected walls cannot be
    #    trusted; it only shifts the footprint to the origin. The metric (LAS/LAZ) path uses the wall fit too.
    spec_room = (not getattr(args, "metric", False)) and args.scale_anchor == "room" \
        and (pt_base == "floor").any() and (pt_base == "ceiling").any()
    if spec_room:
        foot_xy = (Pm[pt_base == "ceiling"][:, :2] if (pt_base == "ceiling").sum() >= 100  # ceiling = unoccluded
                   else Pm[pt_base == "floor"][:, :2] if (pt_base == "floor").sum() >= 100
                   else Pm[:, :2])
        lo = np.array([float(np.percentile(foot_xy[:, 0], 1)), float(np.percentile(foot_xy[:, 1], 1))])
        Pm[:, 0] -= lo[0]; Pm[:, 1] -= lo[1]
        origin = np.zeros(3)
        ext = np.array(args.room_dims, float)
        shape = tuple(int(round(e / VOXEL_SIZE)) + 1 for e in ext)
        print(f"room (SPEC cuboid {tuple(round(float(d), 2) for d in args.room_dims)}): "
              f"{ext[0]:.2f} x {ext[1]:.2f} x {ext[2]:.2f} m -> grid {shape}")
    else:
        # Fit the shell to the DETECTED wall/ceiling planes (occlusion-robust: the OUTERMOST well-
        # supported wall plane per side, widened to enclose every rack), then recover the occluded room
        # depth and height from the datacentre cross-section priors.
        wxy = Pm[pt_base == "wall"][:, :2]

        def wall_plane(axis, hi_side):
            """Outermost well-supported wall plane on a side (farthest 0.1 m bin still >=25% of peak)."""
            w = wxy[:, axis]
            side = w[w > np.median(w)] if hi_side else w[w < np.median(w)]
            if len(side) < 50:
                return float(w.max() if hi_side else w.min())
            h, ed = np.histogram(side, bins=np.arange(side.min(), side.max() + 0.1, 0.1))
            keep = np.where(h >= 0.25 * h.max())[0]
            i = keep[-1] if hi_side else keep[0]
            return float(ed[i] + 0.05)

        ext_pts = rk[:, :2] if len(rk) else Pm[pt_base != "ceiling"][:, :2]
        if len(wxy) >= 50:
            lo = np.array([min(wall_plane(0, False), ext_pts[:, 0].min()), min(wall_plane(1, False), ext_pts[:, 1].min())])
            hi = np.array([max(wall_plane(0, True), ext_pts[:, 0].max()), max(wall_plane(1, True), ext_pts[:, 1].max())])
        else:
            lo = ext_pts.min(0); hi = ext_pts.max(0)
        ceil_z = (float(np.percentile(Pm[pt_base == "ceiling"][:, 2], 50)) if (pt_base == "ceiling").any()
                  else float(Pm[:, 2].max()))
        Pm[:, 0] -= lo[0]; Pm[:, 1] -= lo[1]
        origin = np.zeros(3)
        # Room DEPTH (Y) recovered from the canonical two-row cross-section (depth occluded behind the
        # fronts): 2*(rack_depth + REAR_CLEARANCE) + aisle. --room-depth wins if larger.
        det_x, det_y = hi[0] - lo[0], hi[1] - lo[1]
        ext_y = det_y + args.y0_pad
        rack_depth_prior = RACK_DIMENSIONS[args.rack_type][1]
        prior_depth = 2.0 * (rack_depth_prior + REAR_CLEARANCE) + args.aisle
        use_prior = bool(len(rk)) and not getattr(args, "metric", False)
        target_depth = max(args.room_depth, prior_depth) if use_prior else args.room_depth
        if target_depth > det_y:
            wy = Pm[pt_base == "wall"][:, 1]
            lo_sup = int((wy < det_y / 3.0).sum()); hi_sup = int((wy > 2.0 * det_y / 3.0).sum())
            if lo_sup < hi_sup:                          # low-Y wall occluded -> grow there: push all +pad
                Pm[:, 1] += target_depth - det_y
            ext_y = target_depth
            src = "rack-prior" if use_prior and target_depth == prior_depth else "--room-depth"
            print(f"room depth: detected {det_y:.2f} m -> recovered {target_depth:.2f} m [{src}: "
                  f"2*(rack_depth {rack_depth_prior:.1f} + rear {REAR_CLEARANCE:.1f}) + aisle {args.aisle:.1f}] "
                  f"on the {'low-Y' if lo_sup < hi_sup else 'high-Y'} (occluded) wall")
        rack_top = float(np.percentile(rk[:, 2], 99)) if len(rk) else 0.0
        ext_z = max(ceil_z, rack_top)
        # Height (Z) recovered from the rack height prior + overhead clearance when the ceiling is under-read.
        rack_h_prior = RACK_DIMENSIONS[args.rack_type][2]
        if use_prior and ext_z < rack_h_prior:
            ext_z = rack_h_prior + CEILING_CLEARANCE
            print(f"room height: detected ceiling {max(ceil_z, rack_top):.2f} m < rack prior {rack_h_prior:.1f} m "
                  f"(occluded) -> recovered {ext_z:.2f} m [rack_height + overhead {CEILING_CLEARANCE:.1f}]")
        ext = np.array([det_x, ext_y, ext_z])
        shape = tuple(int(round(e / VOXEL_SIZE)) + 1 for e in ext)   # +1 = far-wall voxel at the plane
        print(f"room (fit to detected walls): {ext[0]:.2f} x {ext[1]:.2f} x {ext[2]:.2f} m -> grid {shape}")
    grid = _build_layout_grid(shape, np.int8)

    # --reference: optional fallback (off by default). The MAIN PATH is the CV layout recovery below
    # (steps 4a-4f); this branch is taken only when a placements file is supplied. The cloud still
    # REGISTERS the room (pose + per-axis scale above), but the racks/AC/UPS/cabinets are stamped from the
    # supplied <reference>/placements.json rather than derived. The room is logged so a mismatch shows.
    if getattr(args, "reference", None):
        ref = json.loads((Path(args.reference) / "placements.json").read_text())
        reg = tuple(round(float(e), 2) for e in ext)
        ext = np.array(ref["ext"], float); shape = tuple(int(s) for s in ref["shape"])
        args.rack_type = ref.get("rack_type", args.rack_type)
        print(f"--reference {args.reference} (last-resort fallback): cloud registered to room {reg} m; "
              f"stamping {len(ref['instances'])} instances at "
              f"{tuple(round(float(e), 2) for e in ext)} m")
        return _finalize_twin(args, [dict(p) for p in ref["instances"]], np.zeros(3), ext, shape)

    placements: list = []     # editable per-instance manifest (drives the 3D editor + re-stamp)

    def stamp_wall_object(name, c_xy, base_z, dims_whd, vox_id, movable):
        """Anchor a box of KNOWN (prior) dims flush to the NEAREST wall (width along the wall, depth
        into the room), stamp it, and record a manifest entry."""
        W, D, H = dims_whd
        d = {"y0": c_xy[1], "ymax": ext[1] - c_xy[1], "x0": c_xy[0], "xmax": ext[0] - c_xy[0]}
        wall = min(d, key=d.get)
        if wall in ("y0", "ymax"):                       # wall runs along X
            cx = c_xy[0]; cy = D / 2 if wall == "y0" else ext[1] - D / 2; dims = (W, D, H)
        else:                                            # wall runs along Y
            cy = c_xy[1]; cx = D / 2 if wall == "x0" else ext[0] - D / 2; dims = (D, W, H)
        ix = int(round((cx - origin[0]) / VOXEL_SIZE)); iy = int(round((cy - origin[1]) / VOXEL_SIZE))
        iz = int(round((base_z - origin[2]) / VOXEL_SIZE))
        _stamp_box(grid, ix, iy, iz, vox_id, dims)
        placements.append({"name": name, "kind": "box",
                           "center": [float(cx), float(cy), float(base_z + dims[2] / 2.0)],
                           "dims": [float(x) for x in dims], "vox_id": int(vox_id), "movable": movable})
        return wall

    # 4. MAIN PATH (generalising CV) -- DERIVE the layout from the labeled cloud (steps 4a-4f). Every
    #    instance below comes from the cloud's own points. (The ``--reference`` fallback above replaces this
    #    whole block when a layout file is supplied instead.)
    # 4a. racks: cluster into 2 rows (now along X -> differ in Y), snap each row to ONE line,
    #     face the cold aisle (toward the other row); body extrudes the prior depth toward the wall.
    rack_names = [n for n in np.unique(pt_name)
                  if base_label(str(n)) == "server rack" and (pt_name == n).sum() >= 50]
    rcent = {n: Pm[pt_name == n].mean(0) for n in rack_names}
    split = np.median([rcent[n][1] for n in rack_names])
    rows = {n: int(rcent[n][1] > split) for n in rack_names}
    row_y = {r: float(np.mean([rcent[n][1] for n in rack_names if rows[n] == r])) for r in (0, 1)}
    # Aisle width comes from the PRIOR, not the points: each row's captured footprint is the body PLUS
    # the open doors swung into the aisle, so the two rows' points TOUCH and the real aisle is invisible.
    # CENTRE the two-row block in the room depth (the datacentre prior: facing rows sit mid-room with
    # the aisle between them and ~equal back gaps), with the face-to-face gap set to --aisle. This is
    # what reproduces the chest32_final layout from the cloud; row_y (cloud) only tells which row is the
    # low-Y vs high-Y one. _stamp_rack then extrudes each body OUTWARD (toward its wall) from the face.
    aisle_ctr = ext[1] / 2.0
    row_front = {r: aisle_ctr + (args.aisle / 2.0 if row_y[r] > row_y[1 - r] else -args.aisle / 2.0)
                 for r in (0, 1)}
    nr = na = nb = nu = 0
    rack_w, rack_depth = RACK_DIMENSIONS[args.rack_type][:2]
    row_xspan = {}                                           # stamped X extent per row (for the UPS abut)
    # Rack COUNT per row from the TRUE (height-anchored) scale: count = round(captured X-span / rack_w).
    # This is NO LONGER circular -- the old code anchored scale on `racks_per_row * rack_w` then "re-
    # derived" that same count, so it could only return the number it assumed. With a count-free height
    # anchor the span genuinely measures the count. Each row is counted INDEPENDENTLY because the rows
    # are intentionally UNEQUAL: the in-line network rack makes its row one cabinet longer, so the old
    # "both take the max" equal-rows prior would wrongly add a rack to the shorter row. The extra cabinet
    # in the longer row is re-tagged as the network rack below. --racks-per-row > 0 forces a fixed count.
    row_x = {r: np.concatenate([Pm[pt_name == n][:, 0] for n in rack_names if rows[n] == r])
             for r in (0, 1) if any(rows[n] == r for n in rack_names)}
    row_span = {r: (float(np.percentile(xs, 1)), float(np.percentile(xs, 99))) for r, xs in row_x.items()}
    k_per = {r: (args.racks_per_row if args.racks_per_row > 0 else max(1, int(round((hi - lo) / rack_w))))
             for r, (lo, hi) in row_span.items()}
    for r in (0, 1):
        if r not in row_span:
            continue
        k = k_per[r]                                        # per-row independent count
        # k racks at a uniform rack-width pitch, contiguous (touching), centred on the captured span.
        # Voxel-align the row origin so rack faces/ends fall on voxel lines -> the UPS abuts exactly.
        lo, hi = row_span[r]
        x0 = round(((lo + hi) / 2.0 - k * rack_w / 2.0) / VOXEL_SIZE) * VOXEL_SIZE
        facing = RackFacing["PLUS_Y" if row_y[1 - r] > row_y[r] else "MINUS_Y"]
        row_xspan[r] = (x0, x0 + k * rack_w)
        rh = RACK_DIMENSIONS[args.rack_type][2]
        for i in range(k):
            cx = x0 + (i + 0.5) * rack_w
            # The racks are CONSECUTIVE (each touching the next) -- gaps in the cloud come from
            # occlusion, not real space. Nudge the stamp position to a voxel CENTRE so _stamp_rack's
            # floor() is deterministic: a centre like 2.9 m gives 2.9/0.1 = 28.9999.. -> floor 28 (1
            # voxel short), leaving alternating gaps. Stamping at the voxel centre tiles them exactly.
            cxp = (round(cx / VOXEL_SIZE) + 0.5) * VOXEL_SIZE
            _stamp_rack(grid, RackPlacement(position=Coordinate(cxp, row_front[r], 0.0),
                        facing=facing, rack_type=args.rack_type), origin); nr += 1
            cyc = row_front[r] + (-rack_depth / 2.0 if facing == RackFacing.PLUS_Y else rack_depth / 2.0)
            placements.append({"name": f"server rack {nr}", "kind": "rack",
                               "pos": [float(cxp), float(row_front[r]), 0.0], "facing": facing.name,
                               "rack_type": args.rack_type, "vox_id": int(RACK_BODY), "movable": True,
                               "center": [float(cx), float(cyc), float(rh / 2.0)],
                               "dims": [float(rack_w), float(rack_depth), float(rh)]})
    # 4a'. NETWORK RACK: one cabinet is the taller standalone network rack (NET_RACK_HEIGHT ~2.2 m vs the
    #      1.95 m server racks). SAM3 labels it 'server rack' too, so it was stamped as a normal rack
    #      above; RE-TAG the correct one as its own labeled box (full 2.2 m height, teal) -- no carve, so
    #      the scale and the per-row count stay put. Per-column height barely separates it mid-row (~0.25 m
    #      vs recon noise), but it sits IN-LINE at a row END, and a tall-vs-short END comparison is robust:
    #      pick the end cabinet whose captured points reach the greatest height, and only when that height
    #      clears the 1.95/2.2 midpoint. --ups-at gives an explicit SAM3 exemplar centroid that overrides
    #      the geometry (use it when the network rack is hard to see).
    racks = [p for p in placements if p.get("kind") == "rack"]
    net_target = None
    if ux_idx is not None and racks:
        ux = Pm[ux_idx]
        net_target = min(racks, key=lambda p: (p["center"][0] - ux[0]) ** 2 + (p["center"][1] - ux[1]) ** 2)
    elif racks and len(rk):
        def cap_top(p):                                     # 95th-pct rack-point height under this stamp
            m = (np.abs(rk[:, 0] - p["center"][0]) <= rack_w / 2.0) & \
                (np.abs(rk[:, 1] - p["center"][1]) <= rack_depth)
            return float(np.percentile(rk[m, 2], 95)) if int(m.sum()) >= 20 else float("nan")
        top = {id(p): cap_top(p) for p in racks}
        valid = [t for t in top.values() if t == t]
        med = float(np.median(valid)) if valid else 0.0     # the room's typical (server) rack top
        ends = []
        for r in (0, 1):
            rr = sorted((p for p in racks if r in row_front and abs(p["pos"][1] - row_front[r]) < 1e-6),
                        key=lambda p: p["center"][0])
            if rr:
                ends += [rr[0], rr[-1]]                      # the two end cabinets of each row
        ends = [p for p in ends if top[id(p)] == top[id(p)]]
        if ends:
            cand = max(ends, key=lambda p: top[id(p)])
            # The network rack is ~0.25 m taller than the 1.95 m server racks; flag the tallest END
            # cabinet when it clears the room's median rack top by a clear margin. A RELATIVE test (vs
            # the median) is robust to the residual scale error -- an absolute 2.075 m threshold sat
            # right inside the recon noise and missed it.
            print(f"  network-rack scan: median rack top {med:.2f} m, tallest end {top[id(cand)]:.2f} m")
            if top[id(cand)] - med >= 0.12:
                net_target = cand
    if net_target is not None:
        nd = args.net_rack_depth if getattr(args, "net_rack_depth", 0) else net_target["dims"][1]
        cx, cy = net_target["center"][0], net_target["center"][1]
        net_target.clear()
        net_target.update({"name": "network rack", "kind": "box",
                           "center": [float(cx), float(cy), float(NET_RACK_HEIGHT / 2.0)],
                           "dims": [float(rack_w), float(nd), float(NET_RACK_HEIGHT)],
                           "vox_id": int(NETRACK_VOX), "movable": True})
        nu += 1
        print(f"  network rack: re-tagged end rack @ ({cx:.2f}, {cy:.2f}) -> "
              f"H={NET_RACK_HEIGHT:.2f} m W={rack_w:.2f} m D={nd:.2f} m (teal)")
    # 4b. AC: floor-standing CRAC against a wall, face (supply) toward the room. REJECT instances that
    #     do NOT stand on the floor -- a ceiling-height slab mislabeled ac_unit (e.g. a vent at z~2 m)
    #     is not a CRAC, so the real far unit is kept and the spurious near-ceiling one is dropped.
    for n in np.unique(pt_name):
        q = Pm[pt_name == n]
        if base_label(str(n)) == "ac_unit" and len(q) >= 50 and float(q[:, 2].min()) < 0.5:
            stamp_wall_object(str(n), q[:, :2].mean(0), 0.0,
                              AC_UNIT_DIMENSIONS, COOLING_AC_VENT, True); na += 1
    # 4c. background objects (static, wall-mounted): every SAM3-labeled instance that is not a
    #     prior object / shell / clutter / rack-adjacent is stamped FLUSH to the nearest wall at
    #     its own thickness — known per-class dims (BG_DIMS) where we have them, else the measured
    #     surface (width along the wall, captured depth floored, captured height). Today that is
    #     the fire-hose cabinet; re-segmentation adds more classes for free.
    for n in np.unique(pt_name):
        b = base_label(str(n))
        if (b in PRIOR_FILL or b in STRUCTURAL or b in SPECIAL or b in CLUTTER
                or b == "unknown" or (pt_name == n).sum() < 50):
            continue
        q = Pm[pt_name == n]
        if b in BG_DIMS:
            dims = BG_DIMS[b]; base_z = float(q[:, 2].min())
        else:                                            # measured: wider horizontal extent = along wall
            ex, ey = float(q[:, 0].ptp()), float(q[:, 1].ptp())
            dims = (max(ex, ey), max(min(ex, ey), CABINET_DEPTH_MIN), float(np.percentile(q[:, 2], 98)))
            base_z = 0.0 if q[:, 2].min() < 0.5 else float(q[:, 2].min())
        wall = stamp_wall_object(str(n), q[:, :2].mean(0), base_z, dims,
                                 BG_VOX.get(b, OBSTACLE_WALL), False); nb += 1
        print(f"  {b}: {dims[0]:.2f}(W) x {dims[1]:.2f}(D) x {dims[2]:.2f}(H) m, flush to {wall} wall")
    # 4d. UPS vs power cabinet from the "ups" CLASS, split by LOCATION. SAM3 conflates tall floor-
    #     standing cabinets under "ups", so a "ups" cluster is EITHER the in-row UPS or an electrical /
    #     power cabinet elsewhere. Route each TALL, FLOOR-STANDING "ups" cluster: one whose X falls
    #     within a rack row's span (+/- a cabinet width) is the UPS -> abut that row's end (touching,
    #     back-aligned); one AWAY from the rows is a power cabinet -> flush to the nearest wall (id 9).
    #     Short / airborne clusters are dropped as fragments. (The old "dominant cluster" guard wrongly
    #     stamped an across-the-room cabinet as the UPS beside the racks, because it never checked X.)
    ups_depth = 0.8 * rack_depth
    rh = RACK_DIMENSIONS[args.rack_type][2]
    npw = 0
    ups_names = [n for n in np.unique(pt_name)
                 if base_label(str(n)) == "ups" and (pt_name == n).sum() >= 50]
    ups_done = False     # --ups-at now pins the NETWORK RACK (4a'), not the UPS; the 'ups' class below
    #                      is handled on its own (a real in-row UPS or an across-the-room power cabinet)
    for n in sorted(ups_names, key=lambda nn: -int((pt_name == nn).sum())):
        q = Pm[pt_name == n]
        uh = float(np.percentile(q[:, 2], 98))                     # captured top
        if uh < 0.6 * rh or float(q[:, 2].min()) >= 0.5:           # short / airborne -> fragment, skip
            continue
        uw = max(float(q[:, 0].ptp()), 0.4)
        uxc, uyc = float(q[:, 0].mean()), float(q[:, 1].mean())
        r = min((0, 1), key=lambda rr: abs(uyc - row_y[rr])) if row_y else None
        in_row = (r is not None and r in row_xspan
                  and row_xspan[r][0] - uw <= uxc <= row_xspan[r][1] + uw)
        if in_row and not ups_done:                                # tall in-row floor unit = the UPS
            x_lo, x_hi = row_xspan[r]
            cx = (x_hi + uw / 2.0) if abs(uxc - x_hi) <= abs(uxc - x_lo) else (x_lo - uw / 2.0)
            faces_plus = row_y[1 - r] > row_y[r]
            cyc_row = row_front[r] + (-rack_depth / 2.0 if faces_plus else rack_depth / 2.0)
            rack_back = cyc_row + (-rack_depth / 2.0 if faces_plus else rack_depth / 2.0)
            cy = rack_back + (ups_depth / 2.0 if faces_plus else -ups_depth / 2.0)
            cx = min(max(cx, uw / 2.0), ext[0] - uw / 2.0)
            cy = min(max(cy, ups_depth / 2.0), ext[1] - ups_depth / 2.0)
            ix = int(round((cx - origin[0]) / VOXEL_SIZE)); iy = int(round((cy - origin[1]) / VOXEL_SIZE))
            _stamp_box(grid, ix, iy, 0, UPS_VOX, (uw, ups_depth, uh)); nu += 1
            placements.append({"name": "ups", "kind": "box",
                               "center": [float(cx), float(cy), float(uh / 2.0)],
                               "dims": [float(uw), float(ups_depth), float(uh)],
                               "vox_id": int(UPS_VOX), "movable": True})
            ups_done = True
            print(f"  UPS @ ({cx:.2f}, {cy:.2f}) abuts row {r} end; H={uh:.2f} m vs rack {rh:.2f} m")
        elif not in_row and uh >= 0.6 * rh:                        # tall + away from rows -> power cabinet
            cd = max(float(q[:, 1].ptp()), CABINET_DEPTH_MIN)
            wall = stamp_wall_object("power cabinet", q[:, :2].mean(0), 0.0,
                                     (uw, cd, uh), POWER_VOX, False); npw += 1
            print(f"  power cabinet (from 'ups') @ ({uxc:.2f}, {uyc:.2f}) {uw:.2f}x{cd:.2f}x{uh:.2f} m, "
                  f"flush to {wall} wall")
        # else: out-of-row short floor fragment -> clutter, skip
    # 4e. green electrical distribution cabinet — also recover any UNLABELED one (falls in "unknown"):
    #     its grey-green colour is too close to the wall to threshold, so recover it geometrically: the
    #     largest TALL, FLOOR-STANDING unknown cluster next to the fire-hose panel (they share a wall).
    fh = Pm[pt_base == "fire hose"]
    unk = Pm[pt_name.astype(str) == "unknown"]
    if len(fh) >= 50 and len(unk):
        import open3d as o3d   # heavy; only needed for this recovery, keep the module import lazy
        fc = fh.mean(0)
        near = unk[np.hypot(unk[:, 0] - fc[0], unk[:, 1] - fc[1]) < 2.0]
        lab = np.asarray(o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(near)).cluster_dbscan(eps=0.12, min_points=40))
        best = None
        for cl in range(lab.max() + 1):
            q = near[lab == cl]
            if len(q) < 800 or np.percentile(q[:, 2], 98) <= 1.6 or np.percentile(q[:, 2], 2) >= 0.3:
                continue                                              # require tall + floor-standing
            if best is None or len(q) > len(best):
                best = q
        if best is not None:
            cw = float(np.percentile(best[:, 0], 98) - np.percentile(best[:, 0], 2))   # width along wall (X)
            cd = max(float(np.percentile(best[:, 1], 98) - np.percentile(best[:, 1], 2)), CABINET_DEPTH_MIN)
            ch = float(np.percentile(best[:, 2], 98))
            wall = stamp_wall_object("power cabinet", best[:, :2].mean(0), 0.0,
                                     (cw, cd, ch), POWER_VOX, False); npw += 1
            print(f"  power cabinet: {cw:.2f}(W) x {cd:.2f}(D) x {ch:.2f}(H) m, flush to {wall} wall")
    print(f"stamped: {nr} racks (prior), {na} AC (prior), {nu} network-rack/UPS (rack-adjacent); "
          f"background (flush to wall): {nb} labeled + {npw} power cabinet")

    return _finalize_twin(args, placements, origin, ext, shape)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="recon_web run dir (labeled.ply + legend)")
    ap.add_argument("--rack-type", default="42U_real", choices=list(RACK_DIMENSIONS),
                    help="rack size prior; metric scale is anchored on this rack's HEIGHT. Default "
                         "42U_real = measured 0.60x0.90x1.95 m")
    ap.add_argument("--ups-at", default=None,
                    help="network-rack exemplar centroid 'x,y,z' (gravity frame, from a SAM3 box "
                         "prompt); the nearest placed rack is re-tagged as the network rack. When "
                         "omitted, the network rack is found geometrically as the tallest end cabinet")
    ap.add_argument("--racks-per-row", type=int, default=0,
                    help="force a fixed rack count per row; 0 (default) derives it from the recovered "
                         "scale as round(captured row span / rack_width), counted per row independently")
    ap.add_argument("--net-rack-depth", type=float, default=0.75,
                    help="network rack depth (m); measured 0.75 (network rack is 0.6 x 0.75 x 2.2 m)")
    ap.add_argument("--y0-pad", type=float, default=0.0,
                    help="back off the cabinet/UPS-side (y0) wall by this many metres -- a GT "
                         "correction for that wall being occluded behind the racks/cabinets and "
                         "under-detected; the y0 back-gap absorbs it, room Y grows by this amount")
    ap.add_argument("--face-depth", type=int, default=70,
                    help="(legacy, unused) old face-band percentile for the cold aisle; see --aisle")
    ap.add_argument("--aisle", type=float, default=1.1,
                    help="cold-aisle width PRIOR (m): face-to-face gap between the two rows. The cloud "
                         "can't show it (open doors fill the gap), so it comes from this prior. GT 1.1")
    ap.add_argument("--room-depth", type=float, default=0.0,
                    help="real room depth (Y, m) when the far wall is occluded and under-detected; the "
                         "shell is extended to it on the occluded side. 0 = use the detected span")
    ap.add_argument("--metric", action="store_true",
                    help="input is already in metres (LAS/LAZ): skip the rack-height scale anchor")
    ap.add_argument("--scale-anchor", choices=["rack", "room"], default="rack",
                    help="DEFAULT 'rack': recover scale + room from the CV alone (rack cuboid + detected "
                         "walls), so it generalises with no dimensions given. FALLBACK 'room': pin the "
                         "scale + shell to the --room-dims spec, for captures too degraded for the CV")
    ap.add_argument("--room-dims", type=lambda s: tuple(float(x) for x in s.split(",")),
                    default=(7.4, 3.9, 2.4),
                    help="room 'length,depth,height' (m) for the scale-anchor=room FALLBACK only "
                         "(rows->+X frame: X along the rows, Y across, Z floor-to-ceiling). Default 7.4,3.9,2.4")
    ap.add_argument("--rectify", action=argparse.BooleanOptionalAction, default=True,
                    help="unbend a warped recon using the ceiling outline as straightening rails before "
                         "scaling (rows -> +X). --no-rectify to disable. Skipped for --metric inputs")
    ap.add_argument("--reference", type=Path, default=None,
                    help="Fallback: a run dir whose placements.json supplies the layout. Bypasses CV layout "
                         "recovery -- the cloud registers the room and the supplied layout is stamped")
    _voxelize(ap.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
