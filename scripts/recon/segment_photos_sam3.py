#!/usr/bin/env python3
"""Label a reconstruction by running SAM3 on the REAL photos and lifting the
2D masks to 3D via the SfM per-view pointmaps (no mesh render, no reprojection).

Pipeline:
  - SAM3 concept segmentation on each source photo (where SAM3 is near-perfect),
    for the server-room object classes -> per-pixel object masks.
  - Each pixel already has a 3D world point (dumped by run_mast3r --dump-views),
    so mask pixels lift straight to 3D points (after the shared gravity_R so they
    match the displayed cloud).
  - Voxel-majority vote fuses overlapping views into one label per region.
  - Object points are split into INSTANCES by 3D euclidean clustering.
  - Wall / floor / ceiling fill the unlabeled remainder from geometry (the cloud
    is gravity-oriented, so +Z is up: horizontal normals -> floor/ceiling by
    height, vertical normals -> wall).

Outputs: <labeled.ply> (per-instance vivid point cloud) + <labeled>.legend.json.

Usage (halo env):
    python scripts/recon/segment_photos_sam3.py --recon recon.ply \
        --views recon.views --up recon.up.npy --out labeled.ply \
        [--voxel 0.04] [--conf 1.5] [--sam3-thr 0.5]
"""
import argparse
import colorsys
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _recon_io import gravity_R
from engine.vision.multiview_common import GROUNDING_TARGETS
from engine.vision.segmentor_sam3_concept import segment_view_sam3_concept
from engine.vision.segmentor_dino_sam import (
    _clone_sam3_image_state, _load_sam3, _sam3_masks_to_numpy, _sam3_scores_to_numpy,
)
from engine.vision.rack_instancing import (
    associate_blobs, assign_points_to_instances, is_door, obb_metrics,
    split_geometric, standard_width,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("sam3_photo")

STRUCT_COLOR = {"floor": (1.00, 0.55, 0.05), "ceiling": (0.97, 0.86, 0.10),
                "wall": (0.15, 0.55, 1.00), "unknown": (0.30, 0.30, 0.33)}

# NOTE (2026-06-09): naming every metal cabinet as its own SAM3 text concept OVER-FIRES — "electrical
# distribution cabinet" grabbed ~280k pts (> all racks) and collapsed the server racks to 8 fragments,
# because racks / network rack / electrical cabinets are the same visual class to SAM3. Visually-
# distinct objects (fire-hose cabinet) get their own concept; the visually-ambiguous cabinets are
# pinned with a SAM3 BOX exemplar instead (see scripts/recon/relabel_exemplar.py + voxelizer --ups-at).
EXTRA_TARGETS = (("fire hose cabinet", "fire hose"), ("fire hose reel", "fire hose"),
                 ("UPS battery cabinet", "ups"), ("uninterruptible power supply unit", "ups"))
PHOTO_TARGETS = GROUNDING_TARGETS + EXTRA_TARGETS


def instance_color(k: int) -> tuple:
    h = (0.61803398875 * (k + 1)) % 1.0          # golden-ratio hue spread
    return colorsys.hsv_to_rgb(h, 0.68, 0.97)


def resize_mask(m: np.ndarray, hw: tuple) -> np.ndarray:
    H, W = hw
    if m.shape == (H, W):
        return m
    im = Image.fromarray(m.astype(np.uint8) * 255).resize((W, H), Image.NEAREST)
    return np.asarray(im) > 127


def lift_object_points(views, proc, targets, conf_thr, sam3_thr, R, thr_override=None):
    """Run SAM3 on each photo, lift masked pixels to gravity-frame 3D points."""
    labels = sorted({canon for _, canon in targets})
    code = {l: i for i, l in enumerate(labels)}
    pts, lab = [], []
    for vi, vf in enumerate(views):
        d = np.load(vf, allow_pickle=True)
        pts3d, conf = d["pts3d"], d["conf"]            # (H,W,3) world, (H,W)
        H, W = conf.shape
        rgb = np.asarray(Image.open(str(d["img_path"])).convert("RGB"), np.uint8)
        masks = segment_view_sam3_concept(proc, rgb, targets, sam3_thr, thr_override)  # {canon:(Hf,Wf)}
        kept = 0
        for canon, m_full in masks.items():
            sel = resize_mask(m_full, (H, W)) & (conf > conf_thr)
            if not sel.any():
                continue
            p = pts3d[sel].reshape(-1, 3)
            pts.append(p); lab.append(np.full(len(p), code[canon], np.int32))
            kept += len(p)
        log.info("  view %d/%d: %d object px lifted (%s)", vi + 1, len(views), kept,
                 ",".join(masks) or "none")
    if not pts:
        return np.empty((0, 3)), np.empty(0, np.int32), labels
    P = np.concatenate(pts) @ R.T                       # -> gravity frame
    return P.astype(np.float64), np.concatenate(lab), labels


def voxel_vote(recon_pts, P, L, n_labels, v):
    """Majority object-label per voxel, then look up each recon point's voxel.
    Returns an int label per recon point (n_labels = 'unknown')."""
    UNK = n_labels
    if len(P) == 0:
        return np.full(len(recon_pts), UNK, np.int32)
    vobj = np.floor(P / v).astype(np.int64)
    vrec = np.floor(recon_pts / v).astype(np.int64)
    mn = np.minimum(vobj.min(0), vrec.min(0))
    sh = (vobj - mn); sr = (vrec - mn)
    span = max(int(max(sh.max(), sr.max())) + 1, 1)
    b = int(np.ceil(np.log2(span))) + 1                 # bits per axis
    def pack(a): return (a[:, 0] << (2 * b)) | (a[:, 1] << b) | a[:, 2]
    ko, kr = pack(sh), pack(sr)
    uniq, inv = np.unique(ko, return_inverse=True)
    counts = np.zeros((len(uniq), n_labels), np.int64)
    np.add.at(counts, (inv, L), 1)
    winner = counts.argmax(1)
    order = np.argsort(uniq); us, ws = uniq[order], winner[order]
    pos = np.clip(np.searchsorted(us, kr), 0, len(us) - 1)
    hit = us[pos] == kr
    return np.where(hit, ws[pos], UNK).astype(np.int32)


def fill_structural(recon_pts, normals, rec_label, UNK):
    """Assign floor/ceiling/wall to still-unknown points from geometry (+Z=up)."""
    z = recon_pts[:, 2]
    zlo, zhi = np.percentile(z, 2), np.percentile(z, 98)
    mid = 0.5 * (zlo + zhi)
    nz = np.abs(normals[:, 2])
    u = rec_label == UNK
    horiz, vert = u & (nz > 0.85), u & (nz < 0.35)
    out = {"floor": horiz & (z < mid), "ceiling": horiz & (z >= mid), "wall": vert}
    return out


def cluster_subset(pts, eps, min_pts, voxel):
    """Memory-safe euclidean instance clustering for dense clouds.

    open3d's cluster_dbscan materializes every in-radius neighbor, so eps=0.20
    on a ~1 cm-spacing cloud (e.g. Pi3's dense output) allocates billions of
    pairs and OOM-kills the process — which is exactly why the Pi3 segment died
    at this stage while the sparser MASt3R clouds clustered fine. Instead,
    cluster a voxel-downsampled copy (bounded neighbor count), then propagate
    each full-res point's label from its nearest representative. Returns an int
    label per input point (-1 = noise), same convention as cluster_dbscan."""
    if len(pts) == 0:
        return np.empty(0, np.int32)
    sub = o3d.geometry.PointCloud()
    sub.points = o3d.utility.Vector3dVector(pts)
    ds = sub.voxel_down_sample(voxel)
    dpts = np.asarray(ds.points)
    if len(dpts) < 3:
        return np.zeros(len(pts), np.int32)              # one tiny blob, let caller size-filter
    dmin = max(3, int(round(min_pts * len(dpts) / len(pts))))  # scale min-pts to the downsample
    dl = np.asarray(ds.cluster_dbscan(eps=eps, min_points=dmin))
    nn = cKDTree(dpts).query(pts, k=1, workers=-1)[1]
    return dl[nn].astype(np.int32)


def lift_rack_blobs(views, proc, conf_thr, rack_thr, R):
    """SAM3 'server rack' INSTANCE masks per view, each lifted to a 3D world blob.

    Unlike the union lift, this keeps SAM3's per-image instances separate so they
    can be associated across views into per-rack instances. Returns (blobs,
    view_ids): blobs[i] is an (Mi,3) gravity-frame point set, view_ids[i] its view.
    """
    blobs, vids = [], []
    for vi, vf in enumerate(views):
        d = np.load(vf, allow_pickle=True)
        pts3d, conf = d["pts3d"], d["conf"]; H, W = conf.shape
        rgb = np.asarray(Image.open(str(d["img_path"])).convert("RGB"), np.uint8)
        base = proc.set_image(Image.fromarray(rgb))
        state = _clone_sam3_image_state(base)
        try:
            state = proc.set_text_prompt("server rack", state)
        except Exception as exc:  # noqa: BLE001
            log.warning("  view %d rack prompt failed: %s", vi + 1, exc)
            continue
        masks = _sam3_masks_to_numpy(state, rgb.shape[:2])
        if len(masks) == 0:
            continue
        scores = _sam3_scores_to_numpy(state, len(masks))
        kept = 0
        for mi in range(len(masks)):
            if mi < len(scores) and scores[mi] < rack_thr:
                continue
            sel = resize_mask(masks[mi], (H, W)) & (conf > conf_thr)
            if sel.sum() < 30:
                continue
            blobs.append((pts3d[sel].reshape(-1, 3) @ R.T).astype(np.float64))
            vids.append(vi); kept += 1
        log.info("  rack view %d/%d: %d instances", vi + 1, len(views), kept)
    return blobs, vids


def rack_instances(rpts, method, blobs_fn, eps, min_pts, voxel, overlap_thr):
    """Assign a per-rack instance id to the voted rack-class points.

    Shared first step (both methods): DBSCAN the rack points into components and
    drop door/slab + sub-size components via the 42U thickness prior. Then split
    the surviving real-rack region into instances —
      * ``geometric``: equal-extent bins at the 42U standard pitch (0.30*height);
      * ``sam3``: associate SAM3's per-view instance blobs across views, with the
        42U pitch only rescuing a blob SAM3 saw as one but is >=2 racks wide.
    Returns (inst per rpts point, n_inst); -1 = rejected/unassigned.
    """
    inst = np.full(len(rpts), -1, np.int32)
    comp = cluster_subset(rpts, eps, min_pts, voxel)
    real_comps, heights = [], []
    for c in range(comp.max() + 1) if comp.max() >= 0 else []:
        cidx = np.where(comp == c)[0]
        if len(cidx) < min_pts or is_door(rpts[cidx]):
            continue
        real_comps.append(cidx); heights.append(obb_metrics(rpts[cidx])["height"])
    sw = standard_width(heights)
    log.info("  rack: %d component(s), %d real after door/slab reject, std_width=%.3f",
             comp.max() + 1 if comp.max() >= 0 else 0, len(real_comps), sw)
    k = 0
    if method == "geometric":
        for cidx in real_comps:
            sub = split_geometric(rpts[cidx], sw)
            for s in range(sub.max() + 1):
                sel = cidx[sub == s]
                if len(sel) < min_pts:
                    continue
                inst[sel] = k; k += 1
    else:  # sam3
        real_all = np.concatenate(real_comps) if real_comps else np.empty(0, int)
        if len(real_all):
            blobs, vids = blobs_fn()
            g = associate_blobs(blobs, vids, voxel, overlap_thr) if blobs else np.empty(0, int)
            ipts, ilab, kk = [], [], 0
            for gid in range(g.max() + 1) if len(g) else []:
                bpts = np.concatenate([blobs[i] for i in range(len(blobs)) if g[i] == gid])
                if len(bpts) < 30 or is_door(bpts):
                    continue
                mm = obb_metrics(bpts)
                nsplit = max(1, int(round(mm["rowlen"] / sw))) if sw > 1e-6 else 1
                sub = np.zeros(len(bpts), np.int32) if nsplit == 1 else split_geometric(bpts, sw, mm)
                for s in range(sub.max() + 1):
                    ipts.append(bpts[sub == s]); ilab.append(np.full(int((sub == s).sum()), kk, np.int32))
                    kk += 1
            log.info("  rack(sam3): %d blobs -> %d associated instances", len(blobs), kk)
            if kk:
                assigned = assign_points_to_instances(
                    rpts[real_all], np.concatenate(ipts), np.concatenate(ilab), voxel, voxel * 3)
                for a in range(assigned.max() + 1) if (len(assigned) and assigned.max() >= 0) else []:
                    sel = real_all[assigned == a]
                    if len(sel) < min_pts:
                        continue
                    inst[sel] = k; k += 1
    return inst, k


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon", type=Path, required=True, help="gravity-oriented RGB cloud")
    ap.add_argument("--views", type=Path, required=True, help="<recon>.views dir of per-view npz")
    ap.add_argument("--up", type=Path, required=True, help="<recon>.up.npy gravity estimate")
    ap.add_argument("--out", type=Path, required=True, help="labeled.ply to write")
    ap.add_argument("--voxel", type=float, default=0.04, help="vote voxel size (recon units)")
    ap.add_argument("--conf", type=float, default=1.5, help="min SfM per-pixel confidence to lift")
    ap.add_argument("--sam3-thr", type=float, default=0.4, help="SAM3 mask score threshold")
    ap.add_argument("--rack-thr", type=float, default=0.55,
                    help="higher SAM3 score required for 'server rack' (it over-fires on AC units / "
                         "fire-hose boxes); set == --sam3-thr to disable the per-class override")
    ap.add_argument("--ac-thr", type=float, default=0.55,
                    help="higher SAM3 score required for 'ac_unit' — server racks + AC units are the "
                         "crucial classes, so demand clean high-confidence masks for both")
    ap.add_argument("--cluster-eps", type=float, default=0.20, help="instance DBSCAN eps")
    ap.add_argument("--cluster-min", type=int, default=250, help="instance min points")
    ap.add_argument("--rack-instancing", choices=["dbscan", "geometric", "sam3"], default="geometric",
                    help="per-rack instance method: dbscan (euclidean, merges a touching row); "
                         "geometric (door-reject + 42U pitch split); sam3 (SAM3 per-image rack "
                         "instances associated across views, 42U as door-reject + merged-pair rescue)")
    ap.add_argument("--rack-overlap", type=float, default=0.20,
                    help="sam3 method: min voxel overlap-coefficient to merge two views' rack blobs")
    ap.add_argument("--cache-lift", action="store_true",
                    help="cache/reuse the SAM3 lift (P,L) next to the views, to iterate vote/cluster cheaply")
    args = ap.parse_args()

    R = gravity_R(np.load(str(args.up)))
    pc = o3d.io.read_point_cloud(str(args.recon))
    recon_pts = np.asarray(pc.points)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=args.voxel * 4, max_nn=30))
    normals = np.asarray(pc.normals)
    log.info("recon cloud: %d points", len(recon_pts))

    views = sorted(args.views.glob("view_*.npz"))
    cache = args.views.parent / "lift_cache.npz"
    proc = None
    if args.cache_lift and cache.exists():
        z = np.load(cache, allow_pickle=False)
        P, L, labels = z["P"], z["L"], [str(x) for x in z["labels"]]
        log.info("loaded cached lift (%d object points) from %s", len(P), cache.name)
    else:
        log.info("loading SAM3 + lifting %d photo views", len(views))
        proc = _load_sam3(None, "cuda", args.sam3_thr)
        thr_override = {"server rack": args.rack_thr, "ac_unit": args.ac_thr}
        P, L, labels = lift_object_points(views, proc, PHOTO_TARGETS, args.conf, args.sam3_thr, R,
                                          thr_override)
        if args.cache_lift:
            np.savez_compressed(cache, P=P, L=L, labels=np.array(labels))
            log.info("cached lift -> %s", cache.name)
    n_obj = len(labels); UNK = n_obj
    log.info("lifted %d object points across %d classes: %s", len(P), n_obj, labels)

    rec_label = voxel_vote(recon_pts, P, L, n_obj, args.voxel)

    # per-rack instance ids (geometric / sam3 methods only; dbscan handled inline)
    rack_inst = None
    if args.rack_instancing != "dbscan" and "server rack" in labels:
        rci = labels.index("server rack")
        ridx = np.where(rec_label == rci)[0]
        if len(ridx) >= args.cluster_min:
            if args.rack_instancing == "sam3" and proc is None:
                proc = _load_sam3(None, "cuda", args.sam3_thr)
            blobs_fn = (lambda: lift_rack_blobs(views, proc, args.conf, args.rack_thr, R))
            ri, n_ri = rack_instances(recon_pts[ridx], args.rack_instancing, blobs_fn,
                                      args.cluster_eps, args.cluster_min, args.voxel, args.rack_overlap)
            rack_inst = (ridx, ri)
            rec_label[ridx[ri < 0]] = UNK               # door / rejected racks -> geometry
            log.info("rack-instancing=%s -> %d instances", args.rack_instancing, n_ri)

    # per-instance split of each object class via euclidean clustering
    colors = np.tile(np.array(STRUCT_COLOR["unknown"]), (len(recon_pts), 1))
    pt_names = np.full(len(recon_pts), "unknown", dtype="<U24")   # exact per-point instance id (voxelizer sidecar)
    legend_counts, legend_palette = {}, {}
    inst_k = 0

    def emit(canon, sel):
        nonlocal inst_k
        if len(sel) < args.cluster_min:
            return
        col = instance_color(inst_k); inst_k += 1
        colors[sel] = col
        name = f"{canon} {sum(1 for k in legend_counts if k.startswith(canon)) + 1}"
        pt_names[sel] = name
        legend_counts[name] = int(len(sel)); legend_palette[name] = [round(x, 3) for x in col]

    for ci, canon in enumerate(labels):
        m = rec_label == ci
        if m.sum() < args.cluster_min:
            rec_label[m] = UNK                          # too sparse -> let geometry decide
            continue
        idx = np.where(m)[0]
        if canon == "server rack" and rack_inst is not None:
            ridx, ri = rack_inst                         # ids on the original rack points
            li = ri[np.searchsorted(ridx, idx)]          # idx ⊆ ridx (both sorted) -> per-point id
            for cl in range(li.max() + 1) if len(li) and li.max() >= 0 else []:
                emit(canon, idx[li == cl])
            continue
        lab = cluster_subset(recon_pts[m], args.cluster_eps, args.cluster_min, args.voxel)
        for cl in range(lab.max() + 1):
            emit(canon, idx[lab == cl])
        # points that clustered as noise (-1) -> unknown, geometry may catch them
        rec_label[idx[lab < 0]] = UNK

    # structural shell on the remainder
    struct = fill_structural(recon_pts, normals, rec_label, UNK)
    for name, sel in struct.items():
        if sel.sum() == 0:
            continue
        colors[sel] = STRUCT_COLOR[name]
        pt_names[sel] = name
        legend_counts[name] = int(sel.sum())
        legend_palette[name] = [round(x, 3) for x in STRUCT_COLOR[name]]
    n_unknown = int(len(recon_pts) - sum(legend_counts.values()))
    if n_unknown > 0:
        legend_counts["unknown"] = n_unknown
        legend_palette["unknown"] = list(STRUCT_COLOR["unknown"])

    import trimesh
    trimesh.PointCloud(vertices=recon_pts.astype(np.float32),
                       colors=(colors * 255).astype(np.uint8)).export(str(args.out))
    np.savez(args.out.parent / "point_labels.npz", names=pt_names)   # exact instance ids -> voxelizer (no colour merge)
    counts = dict(sorted(legend_counts.items(), key=lambda x: -x[1]))
    legend = {"backend": "sam3_photo", "label_counts": counts, "palette": legend_palette,
              "n_instances": inst_k}
    Path(args.out.with_suffix("").as_posix() + ".legend.json").write_text(json.dumps(legend, indent=2))
    log.info("wrote %s (%d instances); counts: %s", args.out, inst_k, counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
