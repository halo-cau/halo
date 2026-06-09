# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

HALO (Heat-Aware Layout Optimizer), Capstone 2026. Pipeline: scan a server room → build a labeled
voxel digital twin → reinforcement-learning optimization of equipment layout for ASHRAE TC 9.9 thermal
compliance → 3D visualization. Repo `github.com/halo-cau/halo`. The team is bilingual; commit messages
and many comments are Korean.

## Working rules specific to this repo

- **Do not delete obsolete or unused code.** Even code that looks dead (for example the RL serving layer
  `backend/app/api/inference.py` + `backend/app/core/rl_service.py`, which the `feat/cv` branch currently
  removes) must be preserved during integration. Reconcile by keeping both sides, not by dropping one.
- **Do not add a `Co-Authored-By: Claude` trailer** to commit messages.
- **Heavy CV/recon code is not import-safe in the API process.** The GPU pipeline (Pi3 + SAM3) and its
  deps live in the `halo` conda env and are invoked as a subprocess (see `HALO_PY` below), never imported
  into FastAPI. Keep that boundary.

## Commands

### Python (backend + engine)
- Lint: `ruff check .` — autofix: `ruff check --fix .` — format: `ruff format .` (config `ruff.toml`;
  line length 88, target py311, `src = ["backend", "engine"]`).
- Tests: `pytest` (config `pytest.ini`: `pythonpath = . backend`, `testpaths = tests`). The `backend`
  entry on the path is what makes `from app...` imports resolve.
  - Single file: `pytest tests/test_voxelizer.py`
  - Single test: `pytest tests/test_voxelizer.py::test_name` or by keyword `pytest -k voxel`
- Backend API: from `backend/`, `uvicorn app.main:app --reload --port 8000`. Install deps with
  `pip install -r backend/requirements.txt`.

### Frontend (`frontend/`, pnpm + Vite + Biome)
- `pnpm dev` (Vite dev server, proxies `/api` → `http://127.0.0.1:8000`) — `pnpm build` → `frontend/dist/`
  which `backend/app/main.py` serves at `/`.
- Lint `pnpm lint` / autofix `pnpm lint:fix` / format `pnpm format` (Biome, config `frontend/biome.json`).

### Vision / recon (the `halo` conda env)
- `pip install -r requirements-vision-ai.txt` pins the validated CUDA 13 / Torch 2.11 stack (Mask3D,
  GroundingDINO, SAM2/SAM3, headless render). Recon backbones (Pi3, MASt3R, VGGT, DUSt3R) live under `opt/`.
- Headless rendering / EGL: prefix commands with `PYOPENGL_PLATFORM=egl`.
- Ad-hoc segmentation to a viewer PLY: `python scripts/segment_scan.py --backend <name>`.

## Three-layer architecture

The codebase is a pure-compute **engine**, a thin **backend** that wraps it over HTTP, and a **frontend**.
The engine never imports web code; the backend imports the engine.

### `engine/` — pure compute, no web
- `engine/core/` — `config.py` (global constants: `VOXEL_SIZE` 0.1 m, `GRID_SHAPE`, int8 `SEMANTIC_*`
  tags 0–7, RANSAC/Manhattan/heat params, `RACK_DIMENSIONS`), `data_types.py` (frozen dataclasses that are
  the cross-module contracts: `Coordinate`, `RackPlacement`, `CoolingUnit`, `ScanMetadata`,
  `ComponentInstance`, `PipelineResult`), `exceptions.py`.
- `engine/vision/` — the CV pipeline. `pipeline.py` orchestrates **clean → segment (mesh vertices) →
  voxelize+stamp (voxel grid) → pad**. Two distinct labeling stages: segmentation labels mesh *vertices*
  from an AI model; stamping labels *voxels* from user `ScanMetadata`. `cleaner.py` (SOR denoise, RANSAC
  floor, Manhattan rectify), `voxelizer.py`, `structural_priors.py`, `instance_namer.py` (geometry-only
  floor/ceiling/wall/rack/object namer), `rack_instancing.py`, `exporter.py`.
- `engine/vision/twin.py` — **unified dispatcher; read its module docstring first.** ANY input converges on
  ONE labeled cloud (`labeled.ply` + `point_labels.npz`) then a SHARED voxelize tail
  (`scripts/recon/voxelize_labeled_cloud.py::voxelize_labeled`). Two fronts: PRIMARY multi-view (≥2 images
  → Pi3 recon + SAM3 seg, run by `scripts/recon/pipeline_web.py`) and SECONDARY geometry (one
  `.obj/.ply/.las/.laz` → clean → DBSCAN → `instance_namer`). The labeled-cloud seam is why one voxelizer
  serves every input type.
- `engine/rl/` — `datacenter.py` is the **canonical** `DataCenterEnv` (gymnasium; 42U rack footprint,
  MaskablePPO-compatible action mask, the shared `_ashrae_reward_from_metrics`). `datacenter_env.py` is an
  older alternate env — kept, not canonical. `thermal_bridge.py` converts a voxel grid ↔ RL state and runs
  the 3D solve. `train.py` (writes `engine/rl/model.zip` + `vecnormalize.pkl`), `test.py`, `inference.py`.
- `engine/thermal/` — `solver.py` (steady-state advection-diffusion thermal field on the voxel grid:
  prescribed velocity from fan/AC specs, upwind advection + buoyancy, sub-grid diffusion, rack source /
  AC sink terms), `metrics.py` (ASHRAE TC 9.9: per-rack intake/exhaust/ΔT compliance, RCI/SHI/RHI/RTI,
  vertical profile) consumed by the RL reward.

The **RL reward is two-tier**: dense per-step position shaping plus an end-of-episode ASHRAE score from the
full 3D thermal solve. `_ashrae_reward_from_metrics` in `datacenter.py` is the single shared formula so the
fast-2D and authoritative-3D paths stay comparable.

### `backend/` — FastAPI over the engine (`backend/app/`)
- `main.py` wires routers under `/api/v1`, mounts vendored three.js at `/vendor`, and serves built
  `frontend/dist/` at `/` (mounted last so API routes win). A no-cache middleware forces revalidation.
- `api/endpoints.py` — `POST /process-scan`. `api/visualize.py` — `POST /visualize`, `GET /visualize/demo`
  (returns base64 GLB meshes + thermal). `api/twin.py` — the async **job model** for the unified twin:
  `POST /twin` (enqueue, returns `job_id`), `GET /twin/{id}` (poll `status.json`),
  `GET /twin/{id}/artifact/{name}`, `POST /twin/{id}/restamp` (re-voxelize from edited `placements.json`).
  A single daemon worker thread serializes GPU jobs so the one GPU never thrashes.
- `api/inference.py` + `core/rl_service.py` — RL serving (`POST /inference`, loads `model.zip` +
  `vecnormalize.pkl` via `MaskablePPO`). **Present on `feat/split-view`, removed on `feat/cv`** — see
  Integration state.

### `frontend/` — Vite multi-page, three.js + ECharts
- Pages (each an entry in `vite.config.ts`): `index.html`, `scan.html`, `components.html`,
  `dashboard.html`. `src/components/` holds composable three.js scene objects (`room`, `airflow`,
  `heatmap`, `zones`, `equipment/*`); `src/lib/ashrae.ts`, `src/data/sceneGraphs.ts`.

## Integration state (branches)

Active multi-author branches converging toward `master`:
- `master` — integration target; latest has the presentation deck + batch optimization.
- `feat/cv` (current) — the vision/twin pipeline; **deletes** the RL backend serving files and wires
  `twin_router` in `main.py`.
- `feat/split-view` (newest remote push) — RL + backend RL wiring + a split-view frontend; **adds**
  `inference.py` / `rl_service.py` and wires `inference_router`, plus large `datacenter.py` and
  `main.ts` rewrites.
- `feat/visualization` — PR #1, OPEN (visualize endpoint + viewer). `feat/rl` — PR #3, MERGED.

Known reconciliation points when merging `feat/cv` and `feat/split-view`:
- **`main.py` routers:** the merged app needs BOTH `twin_router` (CV) AND `inference_router` (RL); do not
  drop either.
- **`engine/core/config.py` diverges:** `feat/cv` uses `MAX_ROOM_DIMENSIONS (20,20,6)` / `GRID_SHAPE
  (200,200,60)`; `feat/split-view` uses `(10,10,5)` / `(100,100,50)`. This changes both the voxel grid and
  the RL observation shape — choose deliberately, it is not a trivial conflict.

## Status / design docs to consult

- `cv_models_status.md` — living tracker of every reconstruction and segmentation model (status only).
- `mask3d_training_state.md`, `mask3d_labeling_workflow.md` — Mask3D finetune reproduction + the labeling
  workflow. `README.md` — formatter/linter usage, the segmentation-backend table, and the structural
  room-shell protection levels.

## Key environment variables

- `HALO_SEGMENTOR_BACKEND` — pipeline backend (`geometric` default; also `mask3d`, `dino_sam3`,
  `sam3_concept`, `dinov3`, `dinov3_sam3`, `none`); read by `engine/vision/segmentor_factory.py`.
- Weights: `HALO_MASK3D_CHECKPOINT`, `HALO_SAM3_CHECKPOINT`, `HALO_DINOV3_MODEL`, `HALO_DINO_CONFIG` /
  `HALO_DINO_CHECKPOINT`; `HF_TOKEN` for HF-gated SAM3/DINOv3.
- `HALO_PY` — interpreter for the GPU subprocess (default `/home/ppco915/ENTER/envs/halo/bin/python`).
  `TWIN_RUNS` — job/artifact directory (default `tools/recon_web/runs`).
- `PYOPENGL_PLATFORM=egl`, `HF_HUB_OFFLINE`, `PYTORCH_CUDA_ALLOC_CONF` are set for the subprocess.
