#!/usr/bin/env python3
"""MASt3R reconstruction -> colored PLY, in three modes:

  --mode sfm       : MASt3R-SfM. Content-based (retrieval) pairing -- frames that
                     SEE the same surface are paired regardless of capture order,
                     giving loop closure a sliding window misses. Most robust for
                     cramped rooms / revisited viewpoints. The similarity matrix is
                     built from MASt3R's own encoder features, so it needs no extra
                     weights. Then native sparse_global_alignment. Scales to all views.
  --mode sparse_ga : native sparse_global_alignment, sliding-window scene graph.
                     Scales to many views. Order-dependent pairing.
  --mode dense     : DUSt3R-style dense PointCloudOptimizer. O(N^2) pairs +
                     per-pixel optimization -> only feasible for a small subset.

Usage:
    python scripts/recon/run_mast3r.py IMG_DIR OUT.ply --mode sfm [--frames 0] [--anchors 0] [--knn 2]
    python scripts/recon/run_mast3r.py IMG_DIR OUT.ply --mode sparse_ga [--frames 0] [--winsize 5]
    python scripts/recon/run_mast3r.py IMG_DIR OUT.ply --mode dense --frames 10
"""
import argparse
import glob
import os
import tempfile
from pathlib import Path

import numpy as np
import torch

from _recon_io import add_vendor_paths, even_subsample, clean_and_write_ply, save_cam_up

add_vendor_paths("mast3r")
import mast3r.utils.path_to_dust3r  # noqa: F401  (puts vendored dust3r on path)
from mast3r.model import AsymmetricMASt3R
from dust3r.utils.image import load_images

WEIGHTS = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"


@torch.no_grad()
def encoder_sim_mat(model, imgs, device):
    """Build an NxN image-similarity matrix from MASt3R's own ViT encoder, so
    retrieval pairing works with no extra weights.  Each image -> mean-pooled
    patch-token descriptor (the encoder already sees scene content); cosine
    similarity ranks which views look at the same surface."""
    descs = []
    for im in imgs:
        x = im["img"].to(device)
        ts = torch.as_tensor(im["true_shape"]).to(device)
        feat, _, _ = model._encode_image(x, ts)        # (1, Npatch, C)
        d = feat.mean(dim=1).float()                    # (1, C) global descriptor
        descs.append(torch.nn.functional.normalize(d, dim=-1))
    D = torch.cat(descs, dim=0)                         # (N, C)
    sim = (D @ D.T).cpu().numpy()
    return np.clip((sim + 1.0) / 2.0, 0.0, 1.0)         # -> [0,1] for FPS distance


def _solve_and_write(model, files, pairs, out, device, conf, dump_views=False):
    """Shared sparse-GA solve + confidence-gated colored PLY export + cam-up.

    If ``dump_views``, also save per-view geometry next to the PLY (in
    ``<stem>.views/``) so the real-photo SAM3 labeler can lift 2D masks to 3D:
    each view -> pts3d (H,W,3, world frame), conf (H,W), and its source image
    path.  These are in the SAME frame as the raw cloud (before gravity_orient),
    so the labeler applies the shared gravity_R to match the displayed cloud."""
    from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
    with tempfile.TemporaryDirectory() as cache:
        scene = sparse_global_alignment(
            files, pairs, cache, model,
            lr1=0.07, niter1=300, lr2=0.014, niter2=300,
            device=device, opt_depth=True, shared_intrinsics=True,
            matching_conf_thr=5.0)
        pts3d, _, confs = scene.get_dense_pts3d(clean_depth=True)
        save_cam_up(scene.get_im_poses().detach().cpu().numpy()[:, :3, :3], out)
        rgb = scene.imgs
        if dump_views:
            vdir = Path(str(out).replace(".ply", ".views"))
            vdir.mkdir(parents=True, exist_ok=True)
            for i, (p, c, im) in enumerate(zip(pts3d, confs, rgb)):
                H, W = np.asarray(im).shape[:2]
                np.savez_compressed(
                    vdir / f"view_{i:03d}.npz",
                    pts3d=p.reshape(H, W, 3).detach().cpu().numpy().astype(np.float32),
                    conf=c.reshape(H, W).detach().cpu().numpy().astype(np.float32),
                    img_path=str(scene.img_paths[i]))
            print(f"  dumped {len(pts3d)} view geometries -> {vdir}")
        P = np.concatenate([p.reshape(-1, 3).detach().cpu().numpy() for p in pts3d])
        C = np.concatenate([c.reshape(-1, 3) for c in rgb])
        M = np.concatenate([(cf.reshape(-1) > conf).detach().cpu().numpy() for cf in confs])
    print(f"  conf>{conf} keeps {M.sum():,}/{M.size:,} points")
    clean_and_write_ply(out, P[M], C[M])


def run_sparse_ga(model, files, out, winsize, device, cyclic=False, conf=1.5, dump_views=False,
                  complete=False):
    from mast3r.image_pairs import make_pairs

    imgs = load_images(files, size=512, verbose=False)
    # swin-N assumes file order = capture order; use "complete" (all-pairs) for
    # UNORDERED sets so pairing is order-independent (like Pi3's joint attention).
    graph = "complete" if complete else (f"swin-{winsize}" if cyclic else f"swin-{winsize}-noncyclic")
    pairs = make_pairs(imgs, scene_graph=graph, prefilter=None, symmetrize=True)
    print(f"sparse-GA: {len(files)} imgs, graph={graph}, {len(pairs)} pairs, conf>{conf}")
    _solve_and_write(model, files, pairs, out, device, conf, dump_views=dump_views)


def run_sfm(model, files, out, device, anchors=0, knn=2, winsize=7, conf=1.5, dump_views=False):
    """MASt3R-SfM, HYBRID graph: sequential swin backbone UNION content-based
    retrieval links.  Pure retrieval failed on repetitive scenes -- the offline
    encoder descriptor is near-uniform (can't tell co-visible from unrelated
    frames), so FPS/kNN drop ~half the real adjacency overlaps and the solver
    fuses sub-maps at wrong poses ("entangled slices").  The swin backbone
    guarantees every adjacent overlap is paired; retrieval only ADDS long-range
    loop closures on top, and weak/false links are down-weighted at matching."""
    from mast3r.image_pairs import make_pairs

    imgs = load_images(files, size=512, verbose=False)
    n = len(imgs)
    sw = make_pairs(imgs, scene_graph=f"swin-{winsize}", prefilter=None, symmetrize=True)
    Na = anchors if anchors > 0 else min(n, max(8, n // 3))   # FPS-selected key views
    sim = encoder_sim_mat(model, imgs, device)
    rg = make_pairs(imgs, scene_graph=f"retrieval-{Na}-{knn}", prefilter=None,
                    symmetrize=True, sim_mat=sim)
    seen, pairs = set(), []          # union; dedup on the DIRECTED key so both
    for a, b in list(sw) + list(rg):  # (a,b) and (b,a) survive -- sparse-GA's
        key = (a["idx"], b["idx"])    # is_matching_ok is keyed per direction
        if key not in seen:
            seen.add(key); pairs.append((a, b))
    print(f"MASt3R-SfM hybrid: {n} imgs, swin-{winsize}({len(sw)//2}) U retrieval-{Na}-{knn}"
          f"({len(rg)//2}) = {len(pairs)} pairs, conf>{conf}")
    _solve_and_write(model, files, pairs, out, device, conf, dump_views=dump_views)


def run_dense(model, files, out, device):
    from dust3r.image_pairs import make_pairs
    from dust3r.inference import inference
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

    imgs = load_images(files, size=512, verbose=False)
    graph = "complete"
    pairs = make_pairs(imgs, scene_graph=graph, prefilter=None, symmetrize=True)
    print(f"dense GA: {len(files)} imgs, graph={graph}, {len(pairs)} pairs")

    output = inference(pairs, model, device, batch_size=1, verbose=False)
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer, verbose=True)
    scene.compute_global_alignment(init="mst", niter=300, schedule="cosine", lr=0.01)

    pts3d = [p.detach().cpu().numpy() for p in scene.get_pts3d()]
    rgb = scene.imgs
    masks = [m.detach().cpu().numpy() for m in scene.get_masks()]
    P = np.concatenate([p[m] for p, m in zip(pts3d, masks)]).reshape(-1, 3)
    C = np.concatenate([np.asarray(c)[m] for c, m in zip(rgb, masks)]).reshape(-1, 3)
    print(f"  mask keeps {len(P):,} points")
    clean_and_write_ply(out, P, C)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("img_dir", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--mode", choices=["sfm", "sparse_ga", "dense"], required=True)
    ap.add_argument("--frames", type=int, default=0, help="subsample to N frames (0 = all)")
    ap.add_argument("--winsize", type=int, default=5, help="sparse-GA sliding-window size")
    ap.add_argument("--cyclic", action="store_true", help="cyclic swin graph (ties sequence ends)")
    ap.add_argument("--complete", action="store_true",
                    help="sparse_ga: use a complete (all-pairs) graph — order-independent, for unordered sets")
    ap.add_argument("--anchors", type=int, default=0, help="sfm: # FPS key views (0 = auto)")
    ap.add_argument("--knn", type=int, default=2, help="sfm: local nearest-neighbor links per view")
    ap.add_argument("--conf", type=float, default=1.5, help="min point confidence")
    ap.add_argument("--dump-views", action="store_true",
                    help="save per-view pts3d/conf next to the PLY for real-photo SAM3 lifting")
    args = ap.parse_args()

    device = "cuda"
    files = sorted(glob.glob(os.path.join(args.img_dir, "*.jpg")))
    if args.frames:
        files = even_subsample(files, args.frames)

    model = AsymmetricMASt3R.from_pretrained(WEIGHTS).to(device).eval()

    if args.mode == "sfm":
        run_sfm(model, files, str(args.out), device, anchors=args.anchors, knn=args.knn,
                winsize=args.winsize, conf=args.conf, dump_views=args.dump_views)
    elif args.mode == "sparse_ga":
        run_sparse_ga(model, files, str(args.out), args.winsize, device,
                      cyclic=args.cyclic, conf=args.conf, dump_views=args.dump_views,
                      complete=args.complete)
    else:
        run_dense(model, files, str(args.out), device)

    print(f"peak VRAM {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
