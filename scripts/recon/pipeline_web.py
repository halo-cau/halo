#!/usr/bin/env python3
"""End-to-end recon+label pipeline for the drop-images web tool.

Stages (each heavy step is a subprocess so GPU memory frees between them):
    convert  any images -> upright RGB jpgs
    reconstruct  MASt3R sparse-GA (default) or VGGT  -> recon.ply (RGB cloud)
    mesh     Poisson surface for SAM3 rendering        -> mesh.ply
    segment  SAM3 concept multi-view labels            -> s3_seg_*_labels.ply
    label    remap labels onto the recon's SOURCE frame -> labeled.ply (+legend)

Writes status.json (state/step/pct/message/outputs) and pipeline.log so the
web backend can poll progress. Outputs land in --out.

Usage:
    python scripts/recon/pipeline_web.py --images DIR --out DIR \
        [--model sparse_ga|vggt] [--winsize 7] [--cyclic] [--conf 1.0] [--vote 0.30]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import pillow_heif

pillow_heif.register_heif_opener()

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PY = sys.executable
IMG_EXT = {".heic", ".heif", ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def run_env() -> dict:
    e = dict(os.environ)
    e.update(HF_HUB_OFFLINE="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
             PYOPENGL_PLATFORM="egl")
    return e


class Status:
    def __init__(self, out: Path):
        self.out = out
        self.d = {"state": "running", "step": "init", "pct": 0, "message": "",
                  "outputs": {}, "error": None, "started": time.time()}
        self.log = (out / "pipeline.log").open("a")

    def set(self, step=None, pct=None, message=None, **kw):
        if step is not None: self.d["step"] = step
        if pct is not None: self.d["pct"] = pct
        if message is not None:
            self.d["message"] = message
            self.log.write(f"[{time.strftime('%H:%M:%S')}] {message}\n"); self.log.flush()
        self.d.update(kw)
        (self.out / "status.json").write_text(json.dumps(self.d, indent=2))

    def fail(self, step, msg):
        self.d["state"] = "error"; self.d["error"] = msg
        self.set(step=step, message="ERROR: " + msg)

    def done(self, outputs):
        self.d["state"] = "done"; self.d["outputs"] = outputs
        self.set(step="done", pct=100, message="complete")


CONDA = os.path.expanduser("~/ENTER/bin/conda")  # for isolated-env runners


def sh(cmd, log_path: Path, extra_env: dict | None = None):
    """Run a subprocess, append stdout/stderr to the log, raise on failure."""
    env = run_env()
    if extra_env:
        env.update(extra_env)
    with log_path.open("a") as f:
        f.write("\n$ " + " ".join(str(c) for c in cmd) + "\n"); f.flush()
        r = subprocess.run(cmd, cwd=str(REPO), env=env, stdout=f,
                           stderr=subprocess.STDOUT)
    if r.returncode != 0:
        tail = "\n".join(log_path.read_text().splitlines()[-20:])
        raise RuntimeError(f"step failed (exit {r.returncode}):\n{tail}")


def convert(images: Path, jpg: Path) -> int:
    jpg.mkdir(parents=True, exist_ok=True)
    srcs = sorted(p for p in images.iterdir() if p.suffix.lower() in IMG_EXT)
    for i, s in enumerate(srcs):
        im = ImageOps.exif_transpose(Image.open(s)).convert("RGB")
        if max(im.size) > 1600:
            im.thumbnail((1600, 1600), Image.LANCZOS)
        im.save(jpg / f"{i:03d}_{s.stem}.jpg", "JPEG", quality=95)
    return len(srcs)


def gravity_orient(recon_ply: Path) -> bool:
    """Rotate the recon cloud so the camera-estimated gravity-up (<stem>.up.npy)
    points to +Z, before meshing/labeling.  Gives the floor/ceiling prior a
    correct vertical so the room isn't labeled or shown upside down.  No-op if
    no up estimate exists."""
    import trimesh
    from _recon_io import gravity_R
    up_npy = str(recon_ply).replace(".ply", ".up.npy")
    if not Path(up_npy).exists():
        return False
    R = gravity_R(np.load(up_npy))
    pc = trimesh.load(str(recon_ply), process=False)
    pc.vertices = np.asarray(pc.vertices) @ R.T
    pc.export(str(recon_ply))
    return True


def denoise(recon_ply: Path, nb_neighbors: int = 16, std_ratio: float = 2.0) -> int:
    """Statistical outlier removal: drop points whose neighborhood is far sparser
    than average — i.e. the scattered flyers from low-overlap / textureless
    regions, which are the main visible noise.  Colors preserved."""
    import open3d as o3d
    pc = o3d.io.read_point_cloud(str(recon_ply))
    n0 = len(pc.points)
    if n0 < 2000:
        return n0
    pc, _ = pc.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    o3d.io.write_point_cloud(str(recon_ply), pc)
    return n0 - len(pc.points)


def write_aligned_views(mesh_ply: Path, labels_ply: Path, labels_json: Path,
                        recon_out: Path, labeled_out: Path):
    """Write the recon (RGB) and label-stamped clouds in the SAME floor-aligned
    frame so they overlay exactly and sit right-side-up.  Both reuse the
    cleaner's 'manhattan' (aligned) mesh vertices; SAM3 labels were computed in
    that same frame, so a vivid palette maps onto them by index."""
    import open3d as o3d
    import trimesh
    sys.path.insert(0, str(REPO))
    from engine.vision.cleaner import clean_and_align_meshes_staged
    from engine.vision.segmentor_base import label_to_color

    VIVID = {"unknown": (0.30, 0.30, 0.33), "server rack": (0.10, 0.95, 0.35),
             "wall": (0.15, 0.55, 1.00), "floor": (1.00, 0.55, 0.05),
             "ceiling": (0.97, 0.86, 0.10), "ac_unit": (0.95, 0.15, 0.85),
             "cable tray": (0.10, 0.85, 0.95), "cardboard box": (0.85, 0.45, 0.20),
             "chair": (0.95, 0.45, 0.75), "trash can": (0.55, 0.30, 0.95),
             "object": (0.75, 0.75, 0.20)}
    fb = (0.55, 0.55, 0.58)

    staged = clean_and_align_meshes_staged(str(mesh_ply))
    manh = staged["manhattan"]                          # floor-down aligned frame
    mv = np.asarray(manh.vertices, dtype=np.float32)
    mrgb = np.asarray(manh.vertex_colors)
    rgb = (np.clip(mrgb, 0, 1) * 255).astype(np.uint8) if len(mrgb) == len(mv) \
        else np.full((len(mv), 3), 170, np.uint8)
    trimesh.PointCloud(vertices=mv, colors=rgb).export(str(recon_out))

    lab = o3d.io.read_triangle_mesh(str(labels_ply))    # same (aligned) frame & order
    lab_c = np.asarray(lab.vertex_colors)
    present = list(json.loads(labels_json.read_text()).get("label_counts", {})) or list(VIVID)
    muted = np.array([label_to_color(l) for l in present])
    n = min(len(lab_c), len(mv))                         # safety if counts drift
    idx = np.argmin(((lab_c[:n, None, :] - muted[None]) ** 2).sum(-1), axis=1)
    col = (np.array([VIVID.get(present[i], fb) for i in idx]) * 255).astype(np.uint8)
    trimesh.PointCloud(vertices=mv[:n], colors=col).export(str(labeled_out))

    counts = {present[i]: int((idx == i).sum()) for i in range(len(present))}
    counts = {k: v for k, v in sorted(counts.items(), key=lambda x: -x[1]) if v}
    legend = {"backend": "sam3_concept", "label_counts": counts,
              "palette": {k: [round(x, 3) for x in VIVID.get(k, fb)] for k in counts},
              "n_instances": len(counts)}
    Path(labeled_out.with_suffix("").as_posix() + ".legend.json").write_text(json.dumps(legend, indent=2))
    return counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, default=None,
                    help="directory of multi-view images (PRIMARY front); OR pass --scan")
    ap.add_argument("--scan", type=Path, default=None,
                    help="a single .obj/.ply/.las/.laz scan (SECONDARY geometry front)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", choices=["sfm", "sparse_ga", "vggt", "mapanything", "pi3"], default="pi3")
    ap.add_argument("--seg", choices=["photo", "render"], default="photo",
                    help="photo = SAM3 on real photos lifted via SfM pointmaps (MASt3R only); "
                         "render = legacy SAM3 on mesh renders")
    ap.add_argument("--winsize", type=int, default=7)
    ap.add_argument("--pi3-frames", type=int, default=20,
                    help="Pi3 frame cap (default 20 = 8GB ceiling at full res)")
    ap.add_argument("--pi3-pixel-limit", type=int, default=0,
                    help="Pi3 per-view pixel budget (0 = default 255000); lower it to fit more "
                         "frames in 8GB (coverage up, detail down)")
    ap.add_argument("--rack-instancing", choices=["dbscan", "geometric", "sam3"], default="geometric",
                    help="per-rack instance method for the photo segmenter (see segment_photos_sam3.py)")
    ap.add_argument("--cyclic", action="store_true")
    ap.add_argument("--sga-complete", action="store_true",
                    help="sparse_ga: complete (all-pairs) graph — order-independent for unordered sets")
    ap.add_argument("--conf", type=float, default=1.5)
    ap.add_argument("--vote", type=float, default=0.30)
    args = ap.parse_args()

    if (args.images is None) == (args.scan is None):
        print("provide exactly one of --images (multi-view) or --scan (geometry)", file=sys.stderr)
        return 2

    out = args.out; out.mkdir(parents=True, exist_ok=True)
    log = out / "pipeline.log"
    st = Status(out)
    try:
        if args.scan is not None:                        # SECONDARY front: one geometry scan -> labeled cloud
            st.set(step="label", pct=40, message=f"labeling geometry scan ({args.scan.suffix})")
            sys.path.insert(0, str(REPO))
            from engine.vision.twin import geometry_to_labeled_cloud
            geometry_to_labeled_cloud(args.scan, out)
            outputs = {"labeled": "labeled.ply", "legend": "labeled.legend.json"}
            st.set(step="voxelize", pct=90, message="semantic voxelization + editable manifest")
            cmd = [PY, "scripts/recon/voxelize_labeled_cloud.py", "--run", str(out)]
            if args.scan.suffix.lower() in (".las", ".laz"):
                cmd.append("--metric")                    # LAS/LAZ already in metres
            sh(cmd, log)
            outputs.update({"voxel": "voxel.ply", "manifest": "placements.json", "empty": "voxel_empty.ply"})
            try:    # standard two-panel figure (top-down plan + 3D semantic voxel grid); non-fatal
                sh([PY, "scripts/recon/visualize_twin.py", "--run", str(out)], log)
                outputs["view"] = "twin_view.png"
            except Exception:  # noqa: BLE001
                pass
            st.done(outputs)
            return 0

        st.set(step="convert", pct=5, message="converting images")
        jpg = out / "jpg"
        n = convert(args.images, jpg)
        if n < 2:
            st.fail("convert", f"need >=2 images, got {n}"); return 1
        st.set(message=f"{n} images ready", n_images=n, model=args.model)

        recon = out / "recon_raw.ply"  # raw solver output; aligned RGB view written later
        st.set(step="reconstruct", pct=15, message=f"reconstructing ({args.model}, {n} imgs)")
        if args.model == "vggt":
            frames = min(n, 40)  # 8GB ceiling
            sh([PY, "scripts/recon/run_vggt.py", str(jpg), str(recon), "--frames", str(frames)],
               log, extra_env={"HF_HUB_OFFLINE": "0"})  # weights re-fetch if cache wiped
        elif args.model == "pi3":
            # isolated env (torch 2.5.1). Peak VRAM tracks TOTAL tokens = frames*(pixel_limit/196),
            # not frame count alone: 20f @255k px = 6.80 GB is the proven 8GB ceiling. To cover more
            # of the capture, raise --pi3-frames and drop --pi3-pixel-limit so total tokens stay
            # under that ceiling (e.g. 49f @95k px ~= same token budget, lower per-view detail).
            frames = min(n, args.pi3_frames)
            cmd = [CONDA, "run", "--no-capture-output", "-n", "pi3", "python",
                   "scripts/recon/run_pi3.py", str(jpg), str(recon), "--frames", str(frames)]
            if args.pi3_pixel_limit:
                cmd += ["--pixel-limit", str(args.pi3_pixel_limit)]
            if args.seg == "photo": cmd.append("--dump-views")
            sh(cmd, log, extra_env={"HF_HUB_OFFLINE": "0"})
        elif args.model == "mapanything":
            # isolated env (uniception/torch clash with halo); allow weight fetch on first use.
            # ViT-g encoder is a ~5.4 GB fixed cost -> cap frames so 8 GB doesn't OOM
            # (12 frames measured 6.28 GB; ~0.07 GB/view marginal). Lift this on the 24/5090 box.
            frames = min(n, 28)
            sh([CONDA, "run", "--no-capture-output", "-n", "mapany", "python",
                "scripts/recon/run_mapanything.py", str(jpg), str(recon),
                "--apache", "--frames", str(frames), "--conf-pct", "10"],
               log, extra_env={"HF_HUB_OFFLINE": "0"})
        else:
            # MASt3R routes; dump per-view geometry when labeling from real photos
            mode = "sfm" if args.model == "sfm" else "sparse_ga"
            photo_seg = args.seg == "photo"
            cmd = [PY, "scripts/recon/run_mast3r.py", str(jpg), str(recon),
                   "--mode", mode, "--winsize", str(args.winsize), "--conf", str(args.conf)]
            if mode == "sfm": cmd += ["--knn", "2"]
            elif args.cyclic: cmd.append("--cyclic")
            if mode == "sparse_ga" and args.sga_complete: cmd.append("--complete")
            if photo_seg: cmd.append("--dump-views")
            sh(cmd, log, extra_env={"HF_HUB_OFFLINE": "0"})  # weights re-fetch if cache wiped

        used_up = gravity_orient(recon)
        removed = denoise(recon)
        st.set(message=f"oriented ({'cam-up' if used_up else 'RANSAC'}); denoised −{removed:,} flyers")

        views_dir = out / "recon_raw.views"
        if args.seg == "photo" and views_dir.exists():
            # real-photo SAM3 -> lift to 3D via SfM pointmaps (no mesh render)
            shutil.copyfile(recon, out / "recon.ply")          # gravity-oriented RGB display cloud
            st.set(step="segment", pct=60, message="SAM3 on real photos + 3D lift + per-instance vote")
            sh([PY, "scripts/recon/segment_photos_sam3.py",
                "--recon", str(recon), "--views", str(views_dir),
                "--up", str(recon).replace(".ply", ".up.npy"),
                "--out", str(out / "labeled.ply"), "--conf", str(args.conf),
                "--rack-instancing", args.rack_instancing], log)
            counts = json.loads((out / "labeled.legend.json").read_text()).get("label_counts", {})
            st.set(message=f"labels: {counts}", label_counts=counts)
        else:
            # fallback: render the mesh and run SAM3 on synthetic views
            mesh = out / "mesh.ply"
            st.set(step="mesh", pct=55, message="meshing cloud (Poisson)")
            sh([PY, "scripts/recon/mesh_cloud.py", str(recon), str(mesh), "--depth", "9"], log)
            st.set(step="segment", pct=65, message="SAM3 multi-view labeling (mesh render)")
            sh([PY, "scripts/segment_scan.py", "--backend", "sam3_concept",
                "--scan", str(mesh), "--output-dir", str(out), "--sam3-checkpoint", ""], log)
            st.set(step="label", pct=90, message="aligning + stamping labels")
            counts = write_aligned_views(mesh, out / "s3_seg_sam3_concept_labels.ply",
                                         out / "s3_seg_sam3_concept.json",
                                         out / "recon.ply", out / "labeled.ply")
            st.set(message=f"labels: {counts}", label_counts=counts)

        outputs = {"recon": "recon.ply", "labeled": "labeled.ply", "legend": "labeled.legend.json"}
        try:    # semantic voxelization -> editable manifest (non-fatal: recon+label stand on their own)
            st.set(step="voxelize", pct=95, message="semantic voxelization + editable manifest")
            sh([PY, "scripts/recon/voxelize_labeled_cloud.py", "--run", str(out)], log)
            outputs.update({"voxel": "voxel.ply", "manifest": "placements.json", "empty": "voxel_empty.ply"})
            sh([PY, "scripts/recon/visualize_twin.py", "--run", str(out)], log)   # standard 2-panel figure
            outputs["view"] = "twin_view.png"
            st.set(message="voxelized — open /editor to adjust instances")
        except Exception as ve:  # noqa: BLE001
            with log.open("a") as f:
                f.write(f"\n[voxelize] skipped (recon+label still available): {ve}\n")
        st.done(outputs)
        return 0
    except Exception as e:  # noqa: BLE001
        st.fail(st.d.get("step", "?"), str(e))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
