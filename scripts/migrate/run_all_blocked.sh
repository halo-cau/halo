#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_all_blocked.sh — run VRAM-blocked CV jobs, GATED so it can NEVER touch a
# GPU that is running someone's research.
#
# HARD SAFETY (shared box):
#   * Refuses to run unless you pass GPU=<idx> AND I_CONFIRM_GPUS_FREE=1.
#   * Re-checks nvidia-smi and ABORTS if the chosen GPU has >IDLE_MIB used.
#   * Only run a card the owner has told you is free.
#   * Legacy models (mask3d/sonata/amb3r) need a 4090 (sm_89); Pi3 too (cu121).
#     Sending them to a 5090 will fail with "no kernel image" — pick a 4090.
#
# Usage (one job, one freed card):
#   I_CONFIRM_GPUS_FREE=1 GPU=1 JOBS=pi3    bash scripts/migrate/run_all_blocked.sh
#   I_CONFIRM_GPUS_FREE=1 GPU=1 JOBS=mask3d bash scripts/migrate/run_all_blocked.sh
#   JOBS in: pi3 | mask3d | sonata | amb3r | all   (default: none -> just shows status)
# ---------------------------------------------------------------------------
set -uo pipefail   # not -e: jobs are failure-isolated

REPO="${REPO:-/home/jyc/projects/Capstone}"
OUT="${OUT:-${REPO}/migrate_out}"
JOBS="${JOBS:-none}"
GPU="${GPU:-}"
IDLE_MIB="${IDLE_MIB:-1000}"        # a card with more than this used is "in use"
REPS="${REPS:-50}"; EPOCHS="${EPOCHS:-40}"
IMG_DIR="${IMG_DIR:-${REPO}/server_room_phone/my_room_images}"
MASK3D_INPUT="${MASK3D_INPUT:-${REPO}/server_room_phone/pipeline_vis_lidar_laz/s3_manhattan.ply}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" HYDRA_FULL_ERROR=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# ---------------- GPU SAFETY GATE (cannot be bypassed accidentally) --------
gpu_status() { nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv; }
if [[ "$JOBS" == "none" ]]; then
  echo "Nothing selected. Current GPU state (do NOT touch cards in use):"; gpu_status
  echo "When the OWNER says a card is free: I_CONFIRM_GPUS_FREE=1 GPU=<idx> JOBS=pi3 bash $0"
  exit 0
fi
if [[ "${I_CONFIRM_GPUS_FREE:-0}" != "1" || -z "$GPU" ]]; then
  echo "REFUSING: GPU jobs are gated on a shared box." >&2
  echo "  Pass GPU=<idx> and I_CONFIRM_GPUS_FREE=1, and only for a card the owner freed." >&2
  gpu_status; exit 1
fi
used="$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -dc '0-9')"
if [[ -z "$used" ]]; then echo "ABORT: GPU $GPU not found." >&2; gpu_status; exit 1; fi
if (( used > IDLE_MIB )); then
  echo "ABORT: GPU $GPU has ${used} MiB used (> ${IDLE_MIB}) — it's IN USE. Not touching it." >&2
  gpu_status; exit 1
fi
name="$(nvidia-smi -i "$GPU" --query-gpu=name --format=csv,noheader)"
echo ">>> GPU $GPU ('${name}') verified idle (${used} MiB used). Proceeding with JOBS=${JOBS}."
case "$name" in *5090*) echo "!! WARNING: GPU $GPU is a 5090 (Blackwell). Legacy/cu121 jobs will fail here — use a 4090.";; esac
export CUDA_VISIBLE_DEVICES="$GPU"

# ---------------- runner plumbing ------------------------------------------
mkdir -p "$OUT"; STATUS="${OUT}/STATUS.txt"; : > "$STATUS"
source "$(conda info --base)/etc/profile.d/conda.sh"
have_env() { conda env list | awk '{print $1}' | grep -qx "$1"; }
want() { [[ "$JOBS" == "all" || "$JOBS" == "$1" ]]; }
note() { echo "[$(date '+%F %T')] $*" | tee -a "$STATUS"; }
checkfile() { if [[ -s "$2" ]]; then note "  OK   $1 -> $2 ($(du -h "$2"|cut -f1))"; elif [[ -e "$2" ]]; then note "  EMPTY $1 -> $2"; else note "  MISSING $1 -> $2"; fi; }
run() { local n="$1" e="$2"; shift 2
  have_env "$e" || { note "SKIP ${n}: env '${e}' missing"; return 0; }
  note "START ${n} (env=${e})"
  if "$@" > "${OUT}/${n}.log" 2>&1; then note "PASS ${n}"; else note "FAIL ${n} rc=$? (log ${n}.log)"; fi
}

note "=== run_all_blocked | GPU=${GPU} (${name}) | JOBS=${JOBS} ==="

# 1) Mask3D train (rc 2 from the AP evaluator is benign) + predict -- 4090
if want mask3d && have_env halo; then
  SAVE="${REPO}/opt/Mask3D/checkpoints/halo_migrate"; rm -rf "$SAVE"
  note "START mask3d_train (reps=${REPS}, epochs=${EPOCHS})"
  ( cd "${REPO}/opt/Mask3D/mask3d" && conda run -n halo python main_instance_segmentation.py \
      general.experiment_name=halo_migrate general.save_dir="${SAVE}" \
      general.backbone_checkpoint="${REPO}/opt/Mask3D/checkpoints/scannet200/scannet200_benchmark.ckpt" \
      general.num_targets=7 general.reps_per_epoch="${REPS}" data/datasets=halo_server_room \
      data.num_labels=6 data.batch_size=1 data.num_workers=0 trainer.max_epochs="${EPOCHS}" \
      trainer.check_val_every_n_epoch="${EPOCHS}" trainer.num_sanity_val_steps=0 general.gpus=1 \
    ) > "${OUT}/mask3d_train.log" 2>&1
  rc=$?; { [[ $rc -eq 0 || $rc -eq 2 ]] && note "PASS mask3d_train (rc=${rc})"; } || note "FAIL mask3d_train rc=${rc}"
  checkfile mask3d_ckpt "${SAVE}/last-epoch.ckpt"
  CKPT="${SAVE}/last-epoch.ckpt"; [[ -s "$CKPT" ]] || CKPT="${REPO}/opt/Mask3D/checkpoints/halo_finetune_v4/last-epoch.ckpt"
  run mask3d_predict halo conda run -n halo python "${REPO}/scripts/predict_halo.py" \
      --checkpoint "$CKPT" --input "$MASK3D_INPUT" --output "${OUT}/mask3d_pred.ply" \
      --confidence-threshold 0.1 --mask-threshold 0.5
  checkfile mask3d_pred "${OUT}/mask3d_pred.ply"
fi

# 2) Pi3 full-frame recon -- 4090
if want pi3; then
  if [[ -d "$IMG_DIR" ]]; then
    run pi3_recon pi3 conda run -n pi3 python "${REPO}/scripts/recon/run_pi3.py" "$IMG_DIR" "${OUT}/pi3_recon.ply" --frames 0
    checkfile pi3_recon "${OUT}/pi3_recon.ply"
  else note "SKIP pi3_recon: IMG_DIR not found ('${IMG_DIR}') — set IMG_DIR"; fi
fi

# 3) Sonata finetune + predict -- 4090
if want sonata; then
  HEAD="${REPO}/data/sonata_las6/sonata_head_las6.pth"
  run sonata_finetune sonata conda run -n sonata python "${REPO}/scripts/sonata/finetune_sonata_las6.py" --out "$HEAD"
  checkfile sonata_head "$HEAD"
  [[ -s "$HEAD" ]] && { run sonata_predict sonata conda run -n sonata python "${REPO}/scripts/sonata/predict_sonata_las6.py" --ckpt "$HEAD" --out "${OUT}/sonata_pred.ply"; checkfile sonata_pred "${OUT}/sonata_pred.ply"; }
fi

# 4) AMB3R recon (first run; entrypoint auto-detected) -- 4090
if want amb3r && have_env amb3r; then
  ENTRY="$(ls "${REPO}"/opt/amb3r/{demo,infer,run,reconstruct,main}*.py 2>/dev/null | head -1 || true)"
  if [[ -n "$ENTRY" ]]; then note "AMB3R entrypoint: ${ENTRY} (verify args vs opt/amb3r/README)"; run amb3r_recon amb3r conda run -n amb3r python "$ENTRY" --help
  else note "SKIP amb3r_recon: no entrypoint found in opt/amb3r — add the command after reading its README"; fi
fi

note "=== finished — review ${STATUS}, then (on WSL) rsync_from_server.sh ==="
