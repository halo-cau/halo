#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# rsync_repo_to_server.sh — no-auth path: copy the WHOLE working tree (tracked
# code + gitignored blobs + your uncommitted work) WSL -> gpubox, with
# regenerable/vendored cruft excluded. No GitHub/PAT — rides the SSH master.
#
# Replaces git-clone + blob-rsync with one transfer into /home/jyc (inside the
# account, no sudo). `.git` is included so the server stays a real repo.
#
# Usage (WSL):
#   bash scripts/migrate/rsync_repo_to_server.sh            # dry-run (manifest)
#   DRY_RUN=0 bash scripts/migrate/rsync_repo_to_server.sh  # actually transfer
# ---------------------------------------------------------------------------
set -euo pipefail

HOST="${HOST:-gpubox}"
REPO="${REPO:-/home/ppco915/projects/Capstone}"            # LOCAL (WSL)
REMOTE_REPO="${REMOTE_REPO:-/home/jyc/projects/Capstone}"  # SERVER (inside jyc)
DRY_RUN="${DRY_RUN:-1}"

cd "$REPO"
ssh -O check "$HOST" 2>/dev/null || echo "!! no SSH master for ${HOST}; open one: ssh -fN ${HOST}" >&2

EXCLUDES=(
  --exclude 'tools/recon_web'            # 3 GB regenerable web-viewer cache
  --exclude 'node_modules'
  --exclude '__pycache__'
  --exclude '*.pyc'
  --exclude '*.egg-info'
  --exclude 'opt/MinkowskiEngine/build'  # force a clean rebuild on the server
  --exclude 'opt/Mask3D/mask3d/saved'    # hydra run logs (noise)
  --exclude 'opt/*/.git'                 # vendored git histories (patches are in-tree)
  --exclude 'migrate_out'                # results dir (pulled separately)
  --exclude '.DS_Store'
)

OPTS=(-aH --info=progress2 --human-readable "${EXCLUDES[@]}")
[[ "$DRY_RUN" == "1" ]] && OPTS+=(--dry-run) && echo ">>> DRY RUN (set DRY_RUN=0 to transfer)"

echo ">>> rsync ${REPO}/  ->  ${HOST}:${REMOTE_REPO}/"
rsync "${OPTS[@]}" "${REPO}/" "${HOST}:${REMOTE_REPO}/"

echo ">>> done.${DRY_RUN:+ (dry-run — nothing written)}"
[[ "$DRY_RUN" == "1" ]] && echo ">>> review, then: DRY_RUN=0 bash $0"
exit 0
