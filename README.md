# HALO

Heat-Aware Layout Optimizer — Capstone 2026

## Formatter & Linter

### Python (backend, engine)

[Ruff](https://docs.astral.sh/ruff/)를 사용합니다. 설정 파일은 프로젝트 루트의 `ruff.toml`입니다.

```bash
# 설치
pip install ruff
# 또는
conda install -c conda-forge ruff

# 린트
ruff check .

# 린트 + 자동 수정
ruff check --fix .

# 포맷팅
ruff format .
```

VS Code 사용 시 Ruff 확장을 설치하면 저장할 때 자동으로 포맷 및 import 정리가 적용됩니다.

### Frontend

[Biome](https://biomejs.dev/)를 사용합니다. 설정 파일은 `frontend/biome.json`입니다.

```bash
cd frontend

# 린트
pnpm lint

# 린트 + 자동 수정
pnpm lint:fix

# 포맷팅
pnpm format
```

## Vision / AI Dependencies

The server-room scan cleanup stack has a separate dependency file at
`requirements-vision-ai.txt`. It pins the currently validated CUDA 13 / PyTorch
2.11 environment for Mask3D, GroundingDINO, SAM2/SAM3, and headless 3D
rendering.

Install from the project root:

```bash
python -m pip install -r requirements-vision-ai.txt
```

For DINO/SAM rendering on the headless server, run segmentation commands with:

```bash
PYOPENGL_PLATFORM=egl
```

## Segmentation Backends

The CV pipeline labels each mesh vertex before voxelization. Pick the backend
with the `HALO_SEGMENTOR_BACKEND` env var (read by
`engine/vision/segmentor_factory.py`); `scripts/segment_scan.py --backend <name>`
exposes the same set for ad-hoc PLY/JSON exports to the Three.js viewer.

| Backend | Models | Notes |
|---|---|---|
| `geometric` (default) | none | Percentile cuboid shell + DBSCAN clusters. No weights; safe in CI. |
| `mask3d` | cvg/Mask3D | One 3D forward pass. Needs a checkpoint (`HALO_MASK3D_CHECKPOINT`). |
| `dino_sam3` | GroundingDINO + SAM3 | Open-vocab boxes → SAM3 masks, multi-view → 3D vertex labels. |
| `sam3_concept` | SAM3 | SAM3 prompted directly with class names (no GroundingDINO). |
| `dinov3` | DINOv3 | Unsupervised k-means feature clusters — *unnamed* groups for eyeballing. |
| `dinov3_sam3` | SAM3 + DINOv3 | SAM3 seeds prototypes; DINOv3 labels every patch by nearest prototype. |
| `none` | none | Skip segmentation; voxelize the cleaned mesh directly. |

The SAM3 / DINOv3 weights are gated on Hugging Face — accept the licenses and
export `HF_TOKEN` (or run `huggingface-cli login`) before first use. Relevant
env vars: `HALO_SAM3_CHECKPOINT` (empty ⇒ download from HF), `HALO_DINOV3_MODEL`
(empty ⇒ library default), `HALO_DINO_CONFIG` / `HALO_DINO_CHECKPOINT`
(GroundingDINO).

For `sam3_concept` and `dinov3_sam3`, add `--no-rack-prior` to
`scripts/segment_scan.py` to keep the raw model labels (including AC units and
boxes) instead of letting the 3D rack-geometry prior relabel interior verticals
as `server rack`.

## Structural Room-Shell Protection

Segmentation can now protect a Manhattan cuboid room prior before deleting
vertices.  The default mode is `balanced`: floor, ceiling, and outer wall
vertices near the estimated shell are preserved unless repeated movable-object
votes override them.

Available levels for `scripts/segment_scan.py`:

- `off`: pure AI labels, no shell protection.
- `light`: weak protection for already-clean segmentation.
- `balanced`: default protection for dirty phone scans.
- `strong`: shell prior is authoritative for detected wall/floor/ceiling bands.
- `shell`: reserved for strict cuboid-shell reconstruction.

Key controls:

- `--protection-level`
- `--protection-tolerance`
- `--protection-outer-percentile`
- `--protection-normal-cos`
- `--protection-min-votes`
