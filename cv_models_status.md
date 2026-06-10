# CV Models Tracker

Inventory of every model exercised (or queued) in the server-room CV pipeline,
and **what's been done vs. not**. Status only — this file deliberately does **not**
assess accuracy/quality. Living doc; update as runs happen.

_Last updated: 2026-06-04._

**Status legend**
- ✅ **done** — ran end-to-end / integrated / weights trained or downloaded
- 🟡 **partial** — scaffolded or cloned but the run/train step isn't finished
- ⬜ **not started**
- ⏸ **deferred** — blocked by a concrete constraint (e.g. GPU/env)

Pipeline has two model stages: **(A) reconstruction** (images → 3D point cloud)
and **(B) segmentation / labeling** of the cloud. Related context:
`mask3d_training_state.md` (Mask3D detail), `scripts/sonata/README.md` (Sonata).

---

## Stage A — Reconstruction (multi-view images → point cloud)

| Model | Where | Status | What was done | Not done / left |
|---|---|---|---|---|
| **MASt3R** | `opt/mast3r` | ✅ done | Ran on 3 phone photos → `room_mast3r.ply`, `room_mast3r_sparsega.ply` (+ labeled variants, `compare_mast3r_AB.png`). Selected as recon backbone (project decision). | Native `sparse_global_alignment` is the intended production path; runs so far used the borrowed dense optimizer. |
| **VGGT** | `opt/vggt` | ✅ done | Ran → `room_vggt.ply`, `room_vggt.png`. | — (benchmark comparison only) |
| **DUSt3R** | `opt/dust3r` | ✅ done | Ran → `room_dust3r.ply` (+ labeled). | — (baseline, superseded by MASt3R) |
| **AMB3R** | `opt/amb3r` | ⏸ deferred | Repo cloned. | Not run — needs an isolated env (its spconv/xformers pins conflict with `halo`). |
| **Pi3 (π³)** | `pi3` conda env (pip pkg; HF `yyfz233/Pi3`) | ✅ done | Two runs on the 49-img phone set: **(1)** 20 frames @ full res (574×434) → `recon.ply` (1.15M pts, 6.80 GB), 12 inst — recon_web `?job=a197b5252662`. **(2)** **all 49 frames** @ `pixel_limit=150000` (448×322), 7.25 GB → `recon.ply` (+36% occupied 5cm voxels, bbox long-axis +33% = missing fragments filled), 14 inst, ~8% unknown — recon_web `?job=pi3f49r150`. SAM3 runs on full-res photos either way (`pixel_limit` only affects Pi3 detail + lift sampling, not SAM3). | Optimal 8 GB fit is run (2). VRAM sweep proved peak ∝ total tokens (frames×patches), ≈ 5.75 GB + 4.0e-5·tokens; ceiling ~38–45k tokens, so full 574×434 can't reach even 35 frames. For full-res all-49, use the 24 GB/5090 box. |

---

## Reconstruction bake-off (50 pose-selected frames, 2026-06-05)
A 69-image retaken capture was pruned to the best **50** via
`scripts/recon/select_frames.py` (sharpness gate → low-res Pi3 scout for
poses+conf via `run_pi3.py --scout-out` → farthest-point select in pose space).
Same 50 frames fed to each backend (`pipeline_web.py` now passes HF online so
weights re-fetch after a cache wipe; `--pi3-frames/--pi3-pixel-limit/--rack-instancing` knobs added):

| backend | points | flyers | geometry (scale-inv RMS/diag) | note |
|---|---|---|---|---|
| Pi3 | 1.15M | 3.9% | wall ~2.2e-3, rack-front 2.83e-3 (crispest) | budget-capped @150k px |
| MASt3R sparse-GA | 1.17M | 4.0% | wall ~2.3e-3 | global pose opt; richest object detection |
| MASt3R SfM | 1.18M | 4.4% | wall ~1.3e-3 | graph mode |
| VGGT | 0.65M | 1.3% | wall 0.15e-3 | **incomplete** (40-frame cap, mesh-render labels → 1 rack) |

**Findings:** Pi3 / sparse-GA / SfM are a 3-way geometry tie (wall + rack-front
plane thickness within RANSAC noise) — Pi3's feed-forward pose does **not** ghost
the repetitive racks here. The "messy" Pi3 result was the rack-instance
**over-segmentation** (19 inst), not geometry; `?job=pi3sel50_geom` (14, balanced)
is the honest view. VGGT is sparser/less complete, not cleaner. Rack **tops
still unreconstructed** — pose analysis shows the capture is all near-horizontal
(median 12°, max 37° tilt); needs deliberate top-down shots. MapAnything OOM'd
at 28 frames on the higher-res set.

**Per-rack instancing** (`segment_photos_sam3.py --rack-instancing`,
`engine/vision/rack_instancing.py`): `geometric` (door-reject by 42U
thickness-ratio + pitch split, pose-robust) vs `sam3` (SAM3 per-image instances
associated across views). SAM3-association is defeated by Pi3's per-view pose
noise (blobs don't overlap → under-merge → over-segment); use `geometric` on Pi3,
`sam3` on the cleaner-pose MASt3R clouds. The 42U prior is applied as a
scale-invariant ratio (width:height ≈ 0.30) since Pi3 clouds are up-to-scale.

---

## Stage B — Segmentation / labeling

### B1. Pipeline backends wired in `engine/vision/segmentor_factory.py`
Selectable via `HALO_SEGMENTOR_BACKEND`. Default = `geometric`.

| Backend | Module | Status | What was done | Not done / left |
|---|---|---|---|---|
| `geometric` | `segmentor_geometric.py` | ✅ done | Default backend; percentile shell + DBSCAN. Ran across many scans (`pipeline_vis_*`). | — |
| `mask3d` | `segmentor_mask3d.py` (+ `opt/Mask3D`) | ✅ done (trained) | Finetuned on `las6_corrected`: checkpoints `halo_overfit_v3`, `halo_overfit_v4` (+ ScanNet200 base); 4 eval runs (`v1/v2/v3/smoke`). Inference via `scripts/predict_halo.py`. | Not productionized / not validated as the live backend. |
| `dino_sam3` | `segmentor_dino_sam.py` | ✅ done | GroundingDINO weights present (`opt/checkpoints/groundingdino`); render-mesh → 2D detect+SAM3 → backproject. | **2D+backprojection route — set aside (project decision 2026-06-01).** |
| `sam3_concept` | `segmentor_sam3_concept.py` | ✅ done | Ran → `server_room6_sam3_labeled.ply`. | Same backprojection route — set aside. |
| `dinov3` | `segmentor_dinov3.py` | ✅ done | Integrated; unsupervised feature clustering. | Set aside (backprojection family). |
| `dinov3_sam3` | `segmentor_dinov3.py` (`Dinov3Sam3Segmentor`) | ✅ done | Ran → `dinov3sam3_room_img.2.png`. | Set aside (backprojection family). |

### B2. Standalone 3D model experiments (run ad-hoc from `opt/`, not factory backends)

| Model | Where | Status | What was done | Not done / left |
|---|---|---|---|---|
| **Point-SAM** | `opt/point_sam` | ✅ done | Ran class-agnostic auto-mask instances → `pointsam_instances.npz` (70k pts, 17 inst), `server_room6_pointsam_{instances,masks,named}.ply`. | Not wired as a factory backend. Prompted (click/box) mode not exercised. |
| **OpenShape** | `opt/openshape` | ✅ done | Ran 3D-native CLIP naming on Point-SAM instances → `server_room6_openshape_named.ply`. | Not integrated. |
| **PointCLIP-lite** | (ad-hoc) | ✅ done | Tried (per project notes) — render instances → CLIP classify. | No repo/artifact retained; not integrated. |
| **Sonata / PTv3** | `opt/sonata`, `scripts/sonata/` | 🟡 partial | Scaffolded this session: data prepared (`data/sonata_las6/las6.npz`), env setup script, linear-probe finetune + predict scripts written & syntax-checked. | **Env build not run; finetune not run.** (`bash scripts/sonata/setup_sonata_env.sh` → finetune.) |

### B3. Naming / classification stage (assign class to instances)

| Method | Where | Status | What was done | Not done / left |
|---|---|---|---|---|
| **Geometry-priors namer** | `engine/vision/instance_namer.py` | ✅ done | Built this session; names instances wall/floor/ceiling/server_rack/object by orientation+height+position. Ran on Point-SAM instances → `pointsam_named.ply` (+ `scripts/name_pointsam_instances.py`, viewer `tools/ply_viewer.html`, unit tests). | No AC-unit class (no geometric signature). Not wired as a factory backend. |
| **OpenShape / PointCLIP** | see B2 | ✅ done | See B2. | — |

---

## Stage B — Shortlisted, NOT yet tried
From the 2026-06-01 native-3D survey (no 2D backprojection). All ⬜ not started.

| Model | Role | Note |
|---|---|---|
| **Segment3D** | class-agnostic 3D masks | Pretrained ckpt + MinkowskiEngine (already built); → geometry namer. |
| **UnScene3D** | class-agnostic 3D instances | 3D-only ckpt; runs on 8 GB (RTX 2080 tested). |
| **FreePoint** | class-agnostic 3D instances | Repo/weights maturity unverified. |
| **Concerto** | SSL features (PTv3) | Alternative to Sonata for the encoder. |
| **OneFormer3D** | supervised instance seg | ⏸ filtered — training needs 24–32 GB. |
| **OpenIns3D** | open-vocab instance seg | ⏸ filtered — needs 24 GB. |

---

## Input scans exercised (context, not models)
The geometric / Mask3D pipeline has been run on several captures (see
`server_room_phone/pipeline_vis_*`): `las6`, `lidar_laz`, `lidar_laz5`,
`lidar_obj`, `lidar_ply4`, `obj1`, `obj7`, plus `demo`. Labeled training data:
`data/mask3d_server_room/` (6-class `las6_corrected`).
