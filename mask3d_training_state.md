# Mask3D finetune — state of play

Status as of v3 training. Not used in the upcoming demo; left as work-in-progress
for the final presentation.

## What works end-to-end

The full training + inference pipeline executes on this machine:

* `opt/Mask3D/mask3d/main_instance_segmentation.py` runs the cvg/Mask3D trainer
  with our halo dataset config and saves checkpoints.
* `scripts/predict_halo.py` loads a trained checkpoint and produces a
  per-vertex colored PLY.
* The labelled dataset at `data/mask3d_server_room/train/las6_corrected.npy`
  (861,941 points, 6 classes, 13 rack instances + 1 AC + 2 box clutter + 3 shell)
  is in the workflow-spec format.

## Required patches applied to the cvg fork

Without these, training crashes or produces garbage. All patches live inside
`opt/Mask3D/` and persist across runs:

| File | Patch |
|---|---|
| `mask3d/models/__init__.py` | Fix unbound `mask3d` name → `mask3d_module` import |
| `mask3d/main_instance_segmentation.py` | `weights_save_path` → `default_root_dir` (PL ≥ 1.7), fallback `CSVLogger` so `LearningRateMonitor` callback survives without wandb |
| `mask3d/datasets/semseg.py` | Uncomment `_labels = _select_correct_labels(...)`; switch `yaml.load` → `yaml.safe_load`; add `halo_server_room` color_map branch |
| `mask3d/utils/utils.py` | All `torch.load(...)` → `torch.load(..., weights_only=False)` for PyTorch 2.6+ compat |
| `mask3d/__init__.py` | Add `halo` branch in `get_model()` (num_targets=7, num_labels=6, num_queries=100) |
| `mask3d/conf/data/datasets/halo_server_room.yaml` | New dataset config — `label_offset: 1`, `filter_out_classes: []`, `reps_per_epoch: ${data.reps_per_epoch}` in all three split blocks |
| `data/mask3d_server_room/Validation_database.yaml` | Point at `las6_corrected.npy` |
| `~/ENTER/envs/halo/.../detectron2/.../transform.py` | `Image.LINEAR` → `Image.BILINEAR` (Pillow 12 compat) |

## Reproducible training command

From `opt/Mask3D/mask3d/`:

```bash
OMP_NUM_THREADS=4 HYDRA_FULL_ERROR=1 python main_instance_segmentation.py \
  general.experiment_name=halo_overfit_v4 \
  general.save_dir=/home/ppco915/projects/Capstone/opt/Mask3D/checkpoints/halo_overfit_v4 \
  general.backbone_checkpoint=/home/ppco915/projects/Capstone/opt/Mask3D/checkpoints/scannet200/scannet200_benchmark.ckpt \
  general.num_targets=7 \
  general.reps_per_epoch=20 \
  data/datasets=halo_server_room \
  data.num_labels=6 \
  data.batch_size=1 \
  data.num_workers=0 \
  trainer.max_epochs=25 \
  trainer.check_val_every_n_epoch=25 \
  trainer.num_sanity_val_steps=0 \
  general.gpus=1
```

* `save_dir` MUST NOT already exist (it'll try to resume from a non-existent
  `last-epoch.ckpt`). Delete it before each fresh run.
* Exit code 2 at the end is a benign validation-AP-evaluator complaint; the
  checkpoint is saved correctly.

## Inference command

```bash
python scripts/predict_halo.py \
  --checkpoint opt/Mask3D/checkpoints/halo_overfit_v4/last-epoch.ckpt \
  --input  server_room_phone/pipeline_vis_lidar_laz/s3_manhattan.ply \
  --output server_room_phone/pipeline_vis_lidar_laz/s3_halo_pred.ply \
  --confidence-threshold 0.1 --mask-threshold 0.5
```

* `--confidence-threshold` gates on FOREGROUND argmax probability (BG is
  excluded), since the cvg/Mask3D fork emits very high BG prob on under-trained
  models.
* The checkpoint name must start with `halo_` so `mask3d.get_model()` picks the
  halo branch we added (which rebuilds the model with num_targets=7,
  num_labels=6, num_queries=100 matching the training config).
  `predict_halo.py` auto-renames via a `/tmp/halo_<filename>.ckpt` copy.

## v3 results — what trained, what didn't

After 25 epochs × 20 reps = 500 weight updates:

```
Loss: 124 → 43
Mask sigmoid: 0.117..0.428  (was 0.000..0.992 — masks LEARNED spatial regions)
74/100 queries produce at least one voxel with mask > 0.5
```

But:
```
Per-class max softmax prob (averaged across queries):
  wall          0.026
  floor         0.033
  ceiling       0.003
  server_rack   0.092   ← only class with non-trivial signal
  box_clutter   0.010
  ac_unit       0.011
  BACKGROUND    0.825   ← collapse: every query argmaxes to BG
```

Held-out lidar_laz inference (with `--confidence-threshold=0.1` to force
foreground argmax instead of BG): 90 instance proposals, 86.5% coverage,
48% labelled wall, 37% server_rack, 1% floor. Visually: ceiling regions are
mis-labelled server_rack; rack and AC regions are mis-labelled wall.

**Diagnosis: classification head and mask head haven't aligned.** The mask
head learned roughly where things are. The classification head defaults to
the most common class (wall) because background dominates and BG-vs-foreground
margin is small.

## Most likely fixes (for final-presentation effort)

In order of expected payoff:

1. **More total weight updates.** The cvg/Mask3D defaults assume 1000-epoch
   training on a 1000+ scene dataset. Our 500 updates is ~0.5% of that. Try
   `reps_per_epoch=50, max_epochs=40` (= 2,000 updates, ~3-4 h overnight) and
   re-check.

2. **Increase the classification loss weight.** In
   `mask3d/conf/loss/set_criterion_custom_weights_1.yaml` (or wherever the
   final set_criterion config resolves to), the dice+mask weights vs the CE
   weight may be unbalanced. Bumping `loss_ce` 2-5× should encourage the
   classification head to fight the background dominance.

3. **More labeled scans.** Even one re-scan of the same room labelled the same
   way (e.g. label `lidar_laz` or `obj1` with the same workflow) would give
   Mask3D two views to ground its class boundaries. Single-scan training is a
   degenerate case — every spatial region is uniquely identifiable, so the
   model has no incentive to learn semantics over location.

4. **Lower `num_queries`.** Currently 100 queries for ~17 expected instances
   (13 racks + 2 clutter + 1 AC + 3 shell). 100 queries spread weight updates
   thinly; 30-50 may converge faster.

5. **Tune `dbscan_eps` and `topk_per_image` for inference.** The current
   inference uses `use_dbscan: false`. Enabling DBSCAN may merge near-duplicate
   masks and improve coverage at the cost of inference complexity.

## What artifacts to keep

* `opt/Mask3D/checkpoints/halo_overfit_v3/last-epoch.ckpt` (476 MB) — best
  checkpoint so far. Loss 43, masks working, classes broken. Resume from this
  for v4 by passing `general.checkpoint=<path>` instead of (or in addition to)
  `general.backbone_checkpoint`.
* `data/mask3d_server_room/{train,instance_gt,previews,*.yaml}` — the labelled
  dataset.
* `scripts/predict_halo.py` — inference wrapper. Working.
* `scripts/force_manhattan_flatten.py`, `scripts/build_room_cuboid.py`,
  `scripts/force_rack_prior.py`, `scripts/split_class_instances.py` — the
  geometric pipeline (which IS demo-ready and will run instead of Mask3D for
  now).

## Demo storyline if final presentation requires Mask3D output

If Mask3D still doesn't classify cleanly by final presentation, the defensible
fallback is **Option B from the earlier triage**: run the model for masks,
then post-process with `force_rack_prior.py` to assign 42U-aligned classes.
The model demonstrates "AI finds the regions" while the geometric prior cleans
the labels. That's a stronger ML+domain story than relying on either alone.
