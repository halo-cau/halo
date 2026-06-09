# Sonata (PTv3) finetune on `las6_corrected`

Native-3D semantic segmentation that emits the required vocab — **wall, floor,
ceiling, server_rack, box_clutter, ac_unit** — by finetuning a tiny linear head
on the frozen, self-supervised **Sonata / Point Transformer V3** encoder. Chosen
because it's the most data-efficient option for our single labeled scan and runs
on 8 GB. No 2D rendering / backprojection anywhere.

## Pipeline
1. **Data** (run in `halo`): `python scripts/sonata/prepare_las6_for_sonata.py`
   → `data/sonata_las6/las6.npz` (+ `meta.json`). Decodes the 6 semantic classes,
   computes real normals (the source npy's are zero — PTv3 needs them), and makes
   a class-stratified 80/20 split. **Already run.**
2. **Env** (one-time, heavy): `bash scripts/sonata/setup_sonata_env.sh`
   → isolated `sonata` conda env (torch 2.5 / cu124 / spconv-cu124). Several GB +
   ~15-25 min. Flash-attn is skipped on purpose.
3. **Finetune**: `conda run -n sonata python scripts/sonata/finetune_sonata_las6.py`
   → `data/sonata_las6/sonata_head_las6.pth`. Linear probe by default; prints val
   mIoU every 500 iters. First run downloads the Sonata weights from HF.
4. **Predict + view**:
   ```
   conda run -n sonata python scripts/sonata/predict_sonata_las6.py --ckpt data/sonata_las6/sonata_head_las6.pth
   python -m http.server 8011        # from repo root
   # open http://localhost:8011/tools/ply_viewer.html?ply=/data/sonata_las6/pred_las6.ply
   ```

## Knobs (8 GB)
- `--mode linear` (default, encoder frozen, lowest VRAM) vs `--mode full` (unfreeze; more VRAM/overfit risk).
- If CUDA OOM: lower `--crop 60000` and/or `--patch 512` (or `256`). Patch only changes attention grouping, not weights.
- `--enable-flash` only if you later install FlashAttention (not required).

## Caveats — read before trusting the numbers
- **Isolated env on purpose.** Do NOT install this into `halo` (torch 2.11/cu13, no spconv) — it would break Mask3D/Point-SAM/OpenShape, the same trap as AMB3R.
- **Single scene → val mIoU measures *separability*, not generalization.** The train/val split shares the same room. The real test is `--input` on a *different* scan or the MVS cloud (`server_room6_rgb.ply`) — expect a domain drop (LiDAR-trained → photo-reconstructed).
- **AC is the weak class:** one labeled instance, ~28k pts. The loss upweights AC ×8.7 (inverse-freq), but a single instance caps how well any model learns it. More AC labels in another scan is the highest-leverage improvement.
- **Weights license:** Sonata weights are CC-BY-NC 4.0 (fine for the capstone; matters only if commercialized — then train PTv3 from scratch / ImageNet-free init via Pointcept).

## Files
- `prepare_las6_for_sonata.py` — npy → Sonata `{coord,color,normal,segment}` + split (runs in `halo`).
- `setup_sonata_env.sh` — build the isolated `sonata` conda env (no flash-attn).
- `finetune_sonata_las6.py` — linear/full finetune of the seg head on the frozen encoder.
- `predict_sonata_las6.py` — inference → class-colored PLY + legend (+ val IoU); works on the MVS cloud too.
- `_sonata_common.py` — shared encoder-load / unpool / inference / PLY-export helpers.
