#!/usr/bin/env python3
"""Adapter: labeled Pi3/SAM3 point cloud -> metric voxel grid (room shell + solid object fill).

Bridges the new CV output (a recon_web run's per-instance colored ``labeled.ply`` +
``labeled.legend.json``) to the existing METRIC voxelizer. The voxelizer can't ingest
the cloud directly (it's up-to-scale, instance labels are "server rack N", and its
DBSCAN would re-merge our flush per-rack instances), so this adapter:

  1. decodes each point to its instance via the legend palette (nearest colour),
  2. scales the up-to-scale cloud to METRES, anchoring on the known 42U rack width,
  3. shifts the floor to z=0,
  4. builds a clean virtual room shell (cuboid) as the background,
  5. stamps each per-rack / AC instance as a SOLID canonical box via the voxelizer's
     own `_stamp_detected_priors` (fed OUR instances -> bypasses the merging DBSCAN,
     fills hollow interiors with the 42U / AC prior),
  6. writes the int8 voxel grid (.npy) + a colour-by-label voxel-centre PLY to view.

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
    AC_UNIT_DIMENSIONS, COOLING_AC_VENT, DEFAULT_RACK_TYPE, OBSTACLE_WALL, RACK_BODY,
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
REAR_CLEARANCE = 0.40                        # rear service clearance (m) behind each rack row to the
#                                              wall — the datacentre layout prior used to RECOVER the
#                                              occluded room depth from the rack footprint (see below)
CEILING_CLEARANCE = 0.30                      # overhead clearance (m) above the racks (cable tray /
#                                              lighting / plenum) — recovers the room HEIGHT when a
#                                              chest-height capture under-reads the ceiling
UPS_VOX = 8                                  # ad-hoc voxel id for the UPS (no config id yet)
POWER_VOX = 9                                # electrical distribution cabinet (the green panel)
# voxel-id -> RGB for the viewer
VOX_COLOR = {
    OBSTACLE_WALL:   (0.62, 0.62, 0.66),   # room shell / fire-hose cabinet (obstacle)
    RACK_BODY:       (0.10, 0.85, 0.35),   # rack body (solid 42U)
    RACK_INTAKE:     (0.20, 0.55, 1.00),   # front / cold-aisle intake
    RACK_EXHAUST:    (0.95, 0.35, 0.10),   # rear / hot-aisle exhaust
    COOLING_AC_VENT: (0.95, 0.15, 0.85),   # AC
    UPS_VOX:         (0.95, 0.85, 0.10),   # UPS cabinet
    POWER_VOX:       (0.42, 0.45, 0.28),   # military-green distribution cabinet (muted olive)
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


def voxelize_labeled(run, *, rack_type=DEFAULT_RACK_TYPE, y0_pad=0.0, face_depth=70,
                     aisle=1.1, room_depth=0.0, metric=False, racks_per_row=6, ups_at=None):
    """Importable entry point (mirrors the CLI). Loads ``<run>/labeled.ply`` (+ the ``point_labels.npz``
    sidecar), writes ``voxel_grid.npy`` / ``voxel.ply`` / ``placements.json`` / ``voxel_empty*`` into
    ``<run>``, and returns ``(grid, placements, origin)`` so the backend can use them directly.
    ``metric=True`` skips the rack-width scale anchor for inputs already in metres (LAS/LAZ)."""
    return _voxelize(argparse.Namespace(run=Path(run), rack_type=rack_type, y0_pad=y0_pad,
                                        face_depth=face_depth, aisle=aisle, room_depth=room_depth,
                                        metric=metric, racks_per_row=racks_per_row, ups_at=ups_at))


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

    # 2. scale to metres. A multi-view recon is UP-TO-SCALE, so anchor on the known 42U rack width.
    #    An already-metric input (LAS/LAZ, or any caller passing metric=True) keeps scale 1.0.
    rmask = pt_base == "server rack"
    if getattr(args, "metric", False):
        s = 1.0
        print("metric input: scale = 1.0 (cloud already in metres)")
    elif rmask.sum() < 50:
        raise ValueError("no 'server rack' points to anchor metric scale on; pass metric=True if the "
                         "input is already in metres")
    else:
        rack_w_m = RACK_DIMENSIONS[args.rack_type][0]
        R = P[rmask]
        ctr = R[:, :2].mean(0)
        _, _, vt = np.linalg.svd(R[:, :2] - ctr, full_matrices=False)
        axis, perp = vt[0], vt[1]
        along = (R[:, :2] - ctr) @ axis          # coordinate ALONG the rows
        cross = (R[:, :2] - ctr) @ perp          # ACROSS the rows -> splits the two rows
        # Anchor metric scale on the rack ROW WIDTH, not the per-instance width. A rack row is a cuboid;
        # its full length is seen straight-on from the aisle, so the along-row span is robust to how
        # SAM3 split the row into instances (the per-instance width was noisy -- 0.32 u raw vs 0.24 u
        # curated -> a wrong scale and a 5.6 m room). The body DEPTH is occluded and the TOPS are clipped
        # in a chest-height capture, so width is the only cuboid length that survives. The row holds
        # `racks_per_row` 42U racks, so the row width = racks_per_row * rack_width. Average the two rows
        # (split across-row at the median); 1-99 pct trims flyers.
        split = np.median(cross)
        rws = [float(np.percentile(along[m], 99) - np.percentile(along[m], 1))
               for m in (cross < split, cross >= split) if int(m.sum()) >= 50]
        row_w_cloud = float(np.mean(rws)) if rws else float(np.percentile(along, 99) - np.percentile(along, 1))
        row_w_m = args.racks_per_row * rack_w_m
        s = row_w_m / row_w_cloud
        print(f"rack-row width = {row_w_cloud:.3f} u over {len(rws)} row(s)  ->  scale = {s:.3f} m/u  "
              f"(anchored on {args.racks_per_row}x{rack_w_m} m = {row_w_m:.2f} m row, {args.rack_type})")

    Pm = P * s
    # --- Manhattan yaw: rotate about Z so the rack rows align to +X (axis-aligned room) ---
    sname = pt_name.astype(str)
    rxy = Pm[np.char.startswith(sname, "server rack")][:, :2]
    if len(rxy) >= 50:                                       # align rack rows to +X (needs rack points)
        _, _, rvt = np.linalg.svd(rxy - rxy.mean(0), full_matrices=False)
        th = -np.arctan2(rvt[0][1], rvt[0][0])
        c_, s_ = np.cos(th), np.sin(th)
        Pm[:, :2] = Pm[:, :2] @ np.array([[c_, s_], [-s_, c_]])
        print(f"Manhattan yaw: rotated {np.degrees(th):+.1f} deg (rows -> X)")
    else:
        print("Manhattan yaw: skipped (no rack rows to align to)")
    fz = (np.percentile(Pm[pt_base == "floor"][:, 2], 50) if (pt_base == "floor").any() else Pm[:, 2].min())
    Pm[:, 2] -= fz

    # 3. fit the room shell to the DETECTED wall/ceiling planes, NOT the raw bounding box. The bbox
    #    is inflated by stray points beyond the walls (e.g. y0 here: bbox 0.00 vs real wall ~0.33),
    #    so flush objects landed in a phantom gap and any surface-to-wall extrude grew through the
    #    true wall. Take the OUTERMOST consistent plane per side: a robust percentile of wall points,
    #    widened to enclose every rack so nothing pokes through; floor at z=0, ceiling at its plane.
    rk = Pm[pt_base == "server rack"]
    wxy = Pm[pt_base == "wall"][:, :2]

    def wall_plane(axis, hi_side):
        """Occlusion-robust wall = the OUTERMOST well-supported plane of the wall points on that
        side. A rack occludes the wall behind it, so the only returns there are nearer rack backs
        that drag a 'densest'/percentile estimate inward; the true wall is the farthest 0.1 m bin
        still holding >=25% of the peak count, and the sparse occlusion 'hole' between it and the
        racks is expected (the rack backs themselves were never captured)."""
        w = wxy[:, axis]
        side = w[w > np.median(w)] if hi_side else w[w < np.median(w)]
        if len(side) < 50:
            return float(w.max() if hi_side else w.min())
        h, ed = np.histogram(side, bins=np.arange(side.min(), side.max() + 0.1, 0.1))
        keep = np.where(h >= 0.25 * h.max())[0]
        i = keep[-1] if hi_side else keep[0]
        return float(ed[i] + 0.05)

    # enclose the racks if present, else fall back to all non-ceiling points (shell-only scans)
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

    # Room DEPTH (Y): the far wall, the far row's body and the rear service gap are all OCCLUDED behind
    # the rack fronts, so the cloud under-reads Y -- it captures the 2-row rack block but not the
    # clearances that bracket it. RECOVER the true depth deterministically from the canonical two-row
    # datacentre cross-section using rack PRIORS (not the occluded points):
    #     rear_clearance + rack_depth + aisle + rack_depth + rear_clearance
    #   = 2*(rack_depth + REAR_CLEARANCE) + aisle
    # With the 42U prior (depth 1.0 m) + aisle 1.1 m + 0.4 m rear clearance this yields 3.9 m, matching
    # the GT chest32 room, with NO depth input. The detected rack block is already 2*1.0+1.1 = 3.1 m, so
    # the recovery is purely the two occluded rear gaps. We then extend the shell to that depth on the
    # occluded side (the Y half with fewer wall returns) and centre the rows in it (aisle_ctr below).
    # Applies only to the multi-view occluded case with a real rack layout: a metric scan (LAS/LAZ) sees
    # the true walls, and a rack-less shell has no cross-section to anchor on. --room-depth still wins if
    # a larger value is given explicitly.
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
    # Height (Z): a chest-height capture under-reads the ceiling -- here the detected ceiling fell BELOW
    # the rack tops (1.93 m < the 2.0 m 42U prior), which is physically impossible. Recover from the
    # rack HEIGHT prior + overhead clearance, same robustness as the depth prior. Only fires when the
    # ceiling is under-read (occluded); a validly-detected ceiling above the racks (e.g. chest32's 2.28 m)
    # is kept. Multi-view rack case only (a metric scan sees the true ceiling).
    rack_h_prior = RACK_DIMENSIONS[args.rack_type][2]
    if use_prior and ext_z < rack_h_prior:
        ext_z = rack_h_prior + CEILING_CLEARANCE
        print(f"room height: detected ceiling {max(ceil_z, rack_top):.2f} m < rack prior {rack_h_prior:.1f} m "
              f"(occluded) -> recovered {ext_z:.2f} m [rack_height + overhead {CEILING_CLEARANCE:.1f}]")
    ext = np.array([det_x, ext_y, ext_z])
    shape = tuple(int(round(e / VOXEL_SIZE)) + 1 for e in ext)   # +1 = the far-wall voxel at the plane
    grid = _build_layout_grid(shape, np.int8)
    print(f"room (fit to detected walls): {ext[0]:.2f} x {ext[1]:.2f} x {ext[2]:.2f} m -> grid {shape}")

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
    # Rack COUNT per row comes from the 42U WIDTH PRIOR, NOT the (noisy) segmentation instance
    # count: count = round(captured row X-span / rack_w). The two rows face each other across the
    # aisle and are equal-length, so BOTH take the LARGER count (the equal-rows prior) -- an occluded
    # end on one row cannot drop a rack. GT chest32 = 6 racks/row -> 12 (instance count gave 6+5=11).
    row_x = {r: np.concatenate([Pm[pt_name == n][:, 0] for n in rack_names if rows[n] == r])
             for r in (0, 1) if any(rows[n] == r for n in rack_names)}
    row_span = {r: (float(np.percentile(xs, 1)), float(np.percentile(xs, 99))) for r, xs in row_x.items()}
    k_row = max((max(1, int((hi - lo) / rack_w + 0.5)) for lo, hi in row_span.values()), default=0)
    for r in (0, 1):
        if r not in row_span:
            continue
        k = k_row                                           # equal rows (paired across the aisle)
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
    # 4a'. EXEMPLAR re-tag: the network rack (a SAM3 box exemplar, given as --ups-at) is one of the
    #      placed racks. Find the placed rack nearest the exemplar's room-frame position and RE-TAG it
    #      as the network rack / UPS (yellow solid box) -- no carve, so the scale + the 6+6 stay put.
    if ux_idx is not None:
        ux = Pm[ux_idx]
        racks = [p for p in placements if p.get("kind") == "rack"]
        if racks:
            tgt = min(racks, key=lambda p: (p["center"][0] - ux[0]) ** 2 + (p["center"][1] - ux[1]) ** 2)
            tgt["kind"] = "box"; tgt["vox_id"] = int(UPS_VOX); tgt["name"] = "ups"
            for f in ("pos", "facing", "rack_type"):
                tgt.pop(f, None)
            nu += 1
            print(f"  network rack/UPS: re-tagged placed rack @ "
                  f"({tgt['center'][0]:.2f}, {tgt['center'][1]:.2f}) [exemplar @ ({ux[0]:.2f}, {ux[1]:.2f})]")
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
    ups_done = ux_idx is not None      # an exemplar (--ups-at) is the authoritative UPS -> 4d won't add one
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
    print(f"stamped: {nr} racks (prior), {na} AC (prior), {nu} UPS (rack-adjacent); "
          f"background (flush to wall): {nb} labeled + {npw} power cabinet")

    # 4f. resolve overlaps among wall-mounted boxes, VOXEL-ALIGN every box (whole-voxel dims, min
    #     corner on a voxel line) so the editor's continuous box and the stamped voxels are bit-
    #     identical, then REBUILD the grid from the (final) manifest so the saved grid and
    #     placements.json are guaranteed identical (and the UPS abuts the rack row with no gap).
    separate_wall_boxes(placements, ext)
    for pl in placements:
        if pl.get("kind") == "box":
            pl["dims"] = [max(1, round(v / VOXEL_SIZE)) * VOXEL_SIZE for v in pl["dims"]]
            pl["center"] = [round((c - dd / 2.0) / VOXEL_SIZE) * VOXEL_SIZE + dd / 2.0
                            for c, dd in zip(pl["center"], pl["dims"])]
    grid = _build_layout_grid(shape, np.int8)
    apply_placements(grid, placements, origin)

    # 5. write the editable per-instance manifest (room shell + each instance) — drives the 3D
    #    editor and the re-stamp tool. The shell (walls/floor/ceiling) is rebuilt from `shape`.
    manifest = {"voxel_size": VOXEL_SIZE, "shape": [int(s) for s in shape],
                "origin": [float(o) for o in origin], "ext": [float(e) for e in ext],
                "rack_type": args.rack_type,
                "vox_color": {str(k): list(v) for k, v in VOX_COLOR.items()},
                "instances": placements}
    (args.run / "placements.json").write_text(json.dumps(manifest, indent=2))
    print(f"manifest -> placements.json ({len(placements)} instances)")

    # write grid + viewable voxel-centre PLY
    np.save(args.run / "voxel_grid.npy", grid)
    occ = np.argwhere(grid != SPACE_EMPTY)
    cents = (occ + 0.5) * VOXEL_SIZE + origin
    rgb = np.array([VOX_COLOR.get(int(grid[x, y, z]), (0.4, 0.4, 0.42)) for x, y, z in occ])
    trimesh.PointCloud(vertices=cents.astype(np.float32),
                       colors=(rgb * 255).astype(np.uint8)).export(args.run / "voxel.ply")
    counts = {int(v): int((grid == v).sum()) for v in np.unique(grid) if v != SPACE_EMPTY}
    print(f"grid -> {args.run/'voxel_grid.npy'} ; voxel.ply ({len(occ):,} occupied voxels)")
    print(f"voxel counts by id: {counts}")

    # 6. "remove all movables" -> empty room: strip the RL-placed equipment (racks + AC), keep the
    #    wall shell + fixed infrastructure (fire-hose box, UPS, power cabinet) as obstacles. This is
    #    the starting room for the RL env / the demo's before-and-after.
    MOVABLE = (RACK_BODY, RACK_INTAKE, RACK_EXHAUST, COOLING_AC_VENT)
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="recon_web run dir (labeled.ply + legend)")
    ap.add_argument("--rack-type", default=DEFAULT_RACK_TYPE, choices=list(RACK_DIMENSIONS))
    ap.add_argument("--ups-at", default=None,
                    help="network-rack/UPS exemplar centroid 'x,y,z' (gravity frame, from a SAM3 box "
                         "prompt); the nearest placed rack is re-tagged as the UPS without rescaling")
    ap.add_argument("--racks-per-row", type=int, default=6,
                    help="racks per row (GT layout prior); the metric scale is anchored on the row "
                         "WIDTH = racks_per_row * rack_width, robust to SAM3's per-instance split. GT 6")
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
                    help="input is already in metres (LAS/LAZ): skip the rack-width scale anchor")
    _voxelize(ap.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
