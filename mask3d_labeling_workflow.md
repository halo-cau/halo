# Mask3D Server-Room Labeling Workflow

This project is not trying to build a room-agnostic segmentation service. The finetuning target is the same server room under small changes: moved boxes, changed clutter, scan density differences, and minor alignment noise. Label quality should therefore prioritize stable room structure and server-rack preservation over broad indoor-scene coverage.

Canonical source for this dataset pass: use `room_6.las` only. In the current pipeline exports, that corresponds to `server_room_phone/pipeline_vis_las6/`. Do not use OBJ-derived scenes such as `pipeline_vis_obj*` for Mask3D labels or finetuning data; they are lower-density references/debug views and are not part of the chosen training source.

## 1. Label Set

Use the compact class set already defined in `data/mask3d_server_room/label_database.yaml`:

| ID | Class | Meaning |
| --- | --- | --- |
| 1 | `wall` | Fixed vertical room shell. Do not include rack faces, doors, boxes, or wall-adjacent equipment. |
| 2 | `floor` | Fixed walkable floor plane. Do not label rack bases or movable objects as floor. |
| 3 | `ceiling` | Fixed upper room shell. Include ceiling panels only when the scan clearly captures them. |
| 4 | `server_rack` | Rack cabinets or continuous rack banks that must be preserved during cleanup. |
| 5 | `box_clutter` | Movable objects such as boxes, carts, loose equipment, bags, or temporary clutter. |
| 6 | `ac_unit` | Fixed cooling equipment: wall/ceiling/floor AC units, visible supply/return faces, or cooling cabinets. |
| 255 | ignore | Ambiguous, noisy, unlabeled, or impossible-to-classify points. |

For this capstone, `server_rack` and `ac_unit` are the most important learned equipment classes. If there is a choice between perfect clutter labeling and cleaner rack/AC labeling, spend time on the fixed equipment.

## 2. Instance Policy

Mask3D needs both semantic labels and instance labels. Use this policy consistently:

| Class | Instance rule |
| --- | --- |
| `wall` | One structural instance is acceptable. |
| `floor` | One structural instance is acceptable. |
| `ceiling` | One structural instance is acceptable. |
| `server_rack` | Prefer one instance per physically separate rack bank or separated cabinet group. Individual cabinets are optional if the scan does not separate them clearly. |
| `ac_unit` | One instance per physically separate AC/cooling unit. If only the vent face is visible, label that visible cooling surface as one AC instance. |
| `box_clutter` | One instance per movable object when visible; one grouped clutter instance is acceptable for small piles. |
| ignore | Instance ID `-1`. |

The current `las6` seed labels use two `server_rack` instances. That granularity is enough for the demo goal because the downstream task mainly needs rack/AC preservation and movable-object separation.

## 3. Mask3D Processed Format

Each processed scene is stored as a NumPy array with shape `(N, 12)`:

| Column | Meaning |
| --- | --- |
| 0-2 | XYZ coordinates in meters after the pipeline alignment/Manhattan step. |
| 3-5 | RGB color in `[0, 255]`. |
| 6-8 | Normal vector. LAS samples may currently use zeros. |
| 9 | Segment/superpoint ID. This can be generated automatically and does not need manual editing. |
| 10 | Semantic class ID, for example `4` for `server_rack`. |
| 11 | Instance ID, or `-1` for ignored points. |

The matching `instance_gt/*.txt` file has one integer per point:

```text
semantic_id * 1000 + instance_id + 1
```

Ignored points should be encoded as `0`.

Examples from the current `las6` seed sample:

| Encoded value | Meaning |
| --- | --- |
| `1001` | wall, instance 0 |
| `2002` | floor, instance 1 |
| `3003` | ceiling, instance 2 |
| `4004` | server rack, instance 3 |
| `4005` | server rack, instance 4 |
| `6006` | AC unit, instance 5 |
| `0` | ignored point |

## 4. Labeling Workflow

Use the existing pipeline output as the starting point, then manually correct it.

1. Generate or reuse an aligned scene.
   - Required source for this pass: `server_room_phone/pipeline_vis_las6/s3_manhattan.ply`, generated from `room_6.las`.
   - Do not mix in OBJ-derived point clouds or meshes for this Mask3D dataset.

2. Seed structural labels automatically.
   - Use the geometry label output for wall, floor, and ceiling.
   - Use rack candidates from geometry and DINO/SAM2 as suggestions, not final truth.

3. Manually correct rack regions.
   - Inspect the colored preview PLY.
   - Correct any rack surfaces mislabeled as wall, unknown, cabinet, box, or clutter.
   - Keep rack labels stable across scans even when boxes or people move nearby.

4. Manually correct AC regions.
   - Label the visible AC/cooling unit body, grille, or supply/return face as `ac_unit`.
   - Do not merge the AC into `wall` or `ceiling` just because it is mounted on them.
   - Keep AC labels stable across scans; AC is fixed infrastructure and should be preserved during cleanup.

5. Manually mark obvious movable clutter.
   - Label boxes and temporary objects as `box_clutter`.
   - If clutter boundaries are unclear, group the points rather than spending too much time on perfect object splits.

6. Send ambiguous points to ignore.
   - Use ignore for sparse noise, mirror/glass artifacts, severe occlusion, and points that are not needed for the demo objective.

7. Export a preview PLY after each scene.
   - Review class colors before adding the scene to training.
   - Check especially for rack and AC points incorrectly swallowed by wall/floor/ceiling.

8. Write the processed Mask3D files.
   - `data/mask3d_server_room/train/<scene>.npy`
   - `data/mask3d_server_room/instance_gt/train/<scene>.txt`
   - Add an entry to `train_database.yaml`.
   - Keep one real scan out for validation when possible and place it in `Validation_database.yaml`.

## 4a. Interactive Correction Tool

Open the browser label tool from the same static server as the pipeline viewer:

```text
http://127.0.0.1:8790/label_tool.html?vis=pipeline_vis_las6
```

By default, the tool loads:

```text
server_room_phone/<vis>/s3_manhattan.ply
server_room_phone/<vis>/s3_seg_lidar_geometry_labels.ply
```

For the current dataset, keep `<vis>` as `pipeline_vis_las6`.

Use it as a region-correction editor:

1. Turn on selection mode.
2. Increase or decrease `Point size` if the scene looks too sparse or too thick.
3. Use `Lasso` and drag a freehand outline around visible points. Use `Rect` only for quick box-shaped selections.
4. Choose `server_rack`, `ac_unit`, `box_clutter`, or another class.
5. Enter the instance ID.
6. Assign the selected points.
7. Shift-drag to add another lasso/rectangle selection before assigning.
8. Download the override JSON.

You can also set the initial visual density from the URL, for example `&size=0.055`.

The override JSON stores point indices and label decisions. It is not the final training file. Apply it with:

```bash
/home/ppco915/ENTER/envs/halo/bin/python scripts/apply_mask3d_label_overrides.py \
   --scene las6_corrected \
   --source-ply server_room_phone/pipeline_vis_las6/s3_manhattan.ply \
   --seed-label-ply server_room_phone/pipeline_vis_las6/s3_seg_lidar_geometry_labels.ply \
   --overrides server_room_phone/label_overrides/label_overrides_pipeline_vis_las6.json
```

This produces the Mask3D `.npy`, `instance_gt` text file, database entry, preview PLY, and summary JSON.

## 5. Quality Checklist

Before using a labeled scene for finetuning, verify:

- The point count in `<scene>.npy` matches `<scene>.txt` exactly.
- Semantic IDs are only `1`, `2`, `3`, `4`, `5`, `6`, or `255`.
- Ignored points have instance ID `-1` and encoded GT value `0`.
- Every non-ignored point has a non-negative instance ID.
- Server racks are not merged into `wall` just because they are vertical and rectangular.
- AC units are not merged into `wall` or `ceiling` just because they are mounted on the room shell.
- Floor and ceiling labels do not include boxes or rack bases.
- Each physically separate rack bank has a stable instance ID within the scene.
- Each physically separate AC/cooling unit has a stable instance ID within the scene.
- The validation scene is not just the same file as the training scene, unless no other scan exists yet.

## 6. Practical Priority

For this capstone, label effort should be spent in this order:

1. Correct `server_rack` labels.
2. Correct `ac_unit` labels.
3. Correct wall, floor, and ceiling shell labels.
4. Mark major boxes/clutter.
5. Ignore ambiguous/noisy points.
6. Refine small clutter instances only if time remains.

Augmentation should be applied only after this base labeling pass is stable.