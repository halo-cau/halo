#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# rsync_from_server.sh — pull the batch results back to WSL in ONE transfer.
#
# Run ON WSL after run_all_blocked.sh finishes on the server. Rides the SSH
# master (Host 'gpubox'), so it costs no extra auth.
#
# Usage: bash scripts/migrate/rsync_from_server.sh
# ---------------------------------------------------------------------------
set -euo pipefail

HOST="${HOST:-gpubox}"
REPO="${REPO:-/home/ppco915/projects/Capstone}"            # LOCAL (WSL) path
REMOTE_REPO="${REMOTE_REPO:-/home/jyc/projects/Capstone}"  # SERVER path (inside jyc)

if ! ssh -O check "$HOST" 2>/dev/null; then
  echo "!! No SSH master for '${HOST}'. Open one first: ssh -fN ${HOST}" >&2
fi

mkdir -p "${REPO}/migrate_out"
echo ">>> pulling ${HOST}:${REMOTE_REPO}/migrate_out/ -> ${REPO}/migrate_out/"
rsync -aH --info=progress2 --human-readable \
  "${HOST}:${REMOTE_REPO}/migrate_out/" "${REPO}/migrate_out/"

echo ""
echo "=== STATUS.txt ==="
cat "${REPO}/migrate_out/STATUS.txt" 2>/dev/null || echo "(no STATUS.txt found)"
echo ""
echo ">>> View point clouds with tools/ply_viewer.html (serve repo, open ?ply=migrate_out/<file>.ply)"
