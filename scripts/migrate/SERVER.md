# GPU-server migration runbook (shared box: `jyc@user-wrx90`)

Move the VRAM-blocked CV models to the 4090/5090 box. This is a **shared research
machine** — the runbook is built around three hard rules:

- **Never touch a GPU in use.** All four cards run other researchers' (and your own)
  training. Run a card **only after the owner says it's free**, and the runner
  re-checks and refuses any card that isn't idle.
- **Stay entirely inside `/home/jyc`.** No `sudo`, nothing created outside `$HOME`
  (so **no `/home/ppco915` symlink** — the hard-coded paths are *rewritten* instead).
- **Don't fill the shared disk.** It's 97% full (~54 GB free, one partition). Build
  one env at a time, caches stay in `$HOME`, and a disk guard aborts below 25 GB.

Full rationale + GPU-compat evidence: `/home/ppco915/.claude/plans/radiant-launching-snowglobe.md`.

## Current server facts (from preflight)

- Host `user-wrx90`, account `jyc` (shared), `HOME=/home/jyc`.
- GPUs: 0=5090(32G), 1=4090(24G), 2=4090(24G), 3=5090(32G). Driver 590 / CUDA 13.1.
- **All four ~21 GB used right now = active research. Do not touch.**
- No `conda`, no `nvcc` on PATH (`/usr/local/cuda-12.0` exists but mismatches cu130).
- Disk: `/` 97% used, ~54 GB free, **no scratch partition**.

## Auth & connection (already working)

`~/.ssh/config` has `Host gpubox gpu-server` with the key + `ControlMaster`. One
master serves everything:

```bash
ssh -fN gpubox            # open the shared master (key auth, no password)
ssh -O check gpubox       # verify it's up
ssh gpubox 'whoami'       # rides the master — no new auth
```

## Order of operations (GPU-free steps first; GPU steps only when you say go)

These steps touch **no GPU** and stay in `/home/jyc` — safe to do while research runs:

```bash
# 1. land tracked code via git (inside jyc), over the master
ssh gpubox 'git clone https://github.com/halo-cau/halo.git /home/jyc/projects/Capstone \
            && cd /home/jyc/projects/Capstone && git checkout feat/cv'
#    (private repo -> may prompt for a GitHub PAT; that's separate from SSH)

# 2. add the gitignored blobs (WSL side; dry-run first)
bash scripts/migrate/rsync_to_server.sh             # manifest
DRY_RUN=0 bash scripts/migrate/rsync_to_server.sh   # ~2.5-3 GB into /home/jyc

# 3. bootstrap conda + rewrite paths + build ONE env (no GPU; disk-guarded)
ssh gpubox 'cd /home/jyc/projects/Capstone && ONLY=pi3 bash scripts/migrate/setup_server_envs.sh'
#    fix_paths.sh runs automatically; build sonata/amb3r/halo the same way, one at a time.
```

**Then STOP and wait.** GPU runs happen only when you tell me a specific card is free.

```bash
# 4. ONLY after you confirm a 4090 is free (say card 1):
ssh gpubox 'cd /home/jyc/projects/Capstone && I_CONFIRM_GPUS_FREE=1 GPU=1 JOBS=pi3 \
            tmux new -d -s mig "bash scripts/migrate/run_all_blocked.sh"'
#    the runner ABORTS if GPU 1 isn't actually idle. JOBS: pi3|mask3d|sonata|amb3r|all

# 5. collect results in one pull (WSL side)
bash scripts/migrate/rsync_from_server.sh
```

## The GPU gate (why it's safe)

`run_all_blocked.sh` cannot touch a busy card:

- With no `JOBS`, it only **prints** GPU status and exits.
- It refuses to run unless you pass `GPU=<idx>` **and** `I_CONFIRM_GPUS_FREE=1`.
- It then re-queries `nvidia-smi` and **aborts if the chosen GPU has >1000 MiB used**.
- It warns if you point a legacy/cu121 job at a 5090 (would fail `no kernel image`).

## Device rule (which card per model)

- **4090 (Ada sm_89), cards 1 & 2 — everything blocked.** MinkowskiEngine/spconv have
  no Blackwell kernels, so Mask3D / Sonata / AMB3R / Pi3 must use a 4090.
- **5090 (Blackwell sm_120), cards 0 & 3 — only pure-PyTorch cu128+ jobs** (MASt3R/
  VGGT/etc.), and only if one is free. Never put a legacy/cu121 env on a 5090.

## Disk discipline

- Build **one env at a time**: `ONLY=pi3` (smallest) → `sonata` → `amb3r` → `halo`
  (heaviest, ~10-12 GB; build last, only when training Mask3D).
- `setup_server_envs.sh` aborts below `MIN_FREE_GB=25` and runs `conda clean -afy`
  after each build. Caches forced under `$HOME` (`CONDA_PKGS_DIRS`, `PIP_CACHE_DIR`,
  `HF_HOME`).
- Remove an env you're done with: `conda env remove -n <name>`.

## Path rewrite (replaces the symlink)

`fix_paths.sh` (called by setup) rewrites `/home/ppco915/projects/Capstone` →
`/home/jyc/projects/Capstone` in the 4 runtime files that embed it:
`data/mask3d_server_room/{train_database,Validation_database}.yaml`,
`data/mask3d_server_room/las6_label_summary.json`, and
`opt/Mask3D/mask3d/conf/data/datasets/halo_server_room.yaml`.

## Jobs & what each unblocks

| `JOBS=` | env | card | unblocks |
|---|---|---|---|
| `pi3` | pi3 (cu121) | 4090 | full 49 frames (was 20-cap on 8 GB) |
| `mask3d` | halo (cu130+ME) | 4090 | train reps=50/epochs=40 (v3→v4 fix) + predict |
| `sonata` | sonata (cu124) | 4090 | env build + finetune never run before |
| `amb3r` | amb3r (cu118) | 4090 | first-ever run (entrypoint auto-detected) |

Tune: `IMG_DIR`, `MASK3D_INPUT`, `REPS`, `EPOCHS`, `IDLE_MIB`.

## Troubleshooting

- **`no kernel image is available`** → a cu121/cu124/cu118 env on a 5090. Use a 4090.
- **disk-guard abort** → free space < 25 GB; remove an unused env or lower
  `MIN_FREE_GB` deliberately.
- **MinkowskiEngine build fails** → needs in-env `cuda-nvcc` (setup installs it) and
  `TORCH_CUDA_ARCH_LIST=8.9`; never `12.0`.
- **Mask3D train rc=2** → benign (AP evaluator); checkpoint is saved (runner treats 0/2 as PASS).
- **runner says REFUSING / ABORT** → that's the safety gate doing its job; only run a
  card the owner confirmed free.
