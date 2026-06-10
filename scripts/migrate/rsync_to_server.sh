#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# rsync_to_server.sh — push the GITIGNORED blobs the server can't get from git.
#
# Run ON WSL, AFTER you've `git clone`d the repo to the same path on the server.
# Tracked code comes from git; this adds only what git can't carry:
#   - opt/                  (vendored repos + Mask3D patches + trained ckpts)
#   - data/{mask3d,sonata}  (datasets)
#   - curated raw scans     (inputs the runs consume; artifacts are regenerable)
#
# Rides the multiplexed SSH master (Host 'gpubox' in ~/.ssh/config) → no extra
# auth. SAFETY: dry-run by default; inspect the manifest, then DRY_RUN=0 to send.
#
# Usage:
#   bash scripts/migrate/rsync_to_server.sh           # dry-run (manifest only)
#   DRY_RUN=0 bash scripts/migrate/rsync_to_server.sh # actually transfer
# ---------------------------------------------------------------------------
set -euo pipefail

HOST="${HOST:-gpubox}"
REPO="${REPO:-/home/ppco915/projects/Capstone}"            # LOCAL (WSL) path
REMOTE_REPO="${REMOTE_REPO:-/home/jyc/projects/Capstone}"  # SERVER path (inside jyc)
DRY_RUN="${DRY_RUN:-1}"

cd "$REPO"

# Warn (don't fail) if the SSH master isn't open — one connection for everything.
if ! ssh -O check "$HOST" 2>/dev/null; then
  echo "!! No SSH master for '${HOST}'. Open one first to avoid repeat auth:" >&2
  echo "     ssh -fN ${HOST}" >&2
fi

EXCLUDES=(
  --exclude '__pycache__/' --exclude '*.pyc' --exclude '.git/'
  --exclude '*.egg-info/'  --exclude 'node_modules/' --exclude '.DS_Store'
  --exclude 'opt/MinkowskiEngine/build/'   # force a clean rebuild on the server
  --exclude 'opt/Mask3D/mask3d/saved/'     # hydra run logs — noise
)

# Gitignored blobs (full) + curated raw inputs. -R preserves the relative paths
# under $REMOTE_REPO. Globs expand from $REPO (we cd'd above).
SRCS=(
  opt
  data/mask3d_server_room
  data/sonata_las6
  server_room_phone/lidar
  server_room_phone/my_room_images
  server_room_phone/server_room_images
  server_room_phone/server_room_images_1
  server_room_phone/pipeline_vis_lidar_laz/s3_manhattan.ply   # Mask3D predict input
)
# raw .las captures (may be several)
for f in server_room_phone/server_room_*.las; do [[ -e "$f" ]] && SRCS+=("$f"); done

RSYNC_OPTS=(-aHR --info=progress2 --human-readable "${EXCLUDES[@]}")
[[ "$DRY_RUN" == "1" ]] && RSYNC_OPTS+=(--dry-run) && echo ">>> DRY RUN (set DRY_RUN=0 to transfer)"

echo ">>> rsync ${#SRCS[@]} sources -> ${HOST}:${REMOTE_REPO}/"
rsync "${RSYNC_OPTS[@]}" "${SRCS[@]}" "${HOST}:${REMOTE_REPO}/"

echo ">>> done.${DRY_RUN:+ (dry-run — nothing written)}"
[[ "$DRY_RUN" == "1" ]] && echo ">>> review the file list above, then: DRY_RUN=0 bash $0"
