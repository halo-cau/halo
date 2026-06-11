#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# fix_paths.sh — rewrite absolute /home/ppco915 paths to the server's repo.
#
# Replaces the (forbidden) symlink approach: stays ENTIRELY inside the account,
# no sudo, nothing created outside $HOME. Run ON THE SERVER after git clone +
# rsync, before any Mask3D training. Idempotent.
#
# Usage: REPO=/home/jyc/projects/Capstone bash scripts/migrate/fix_paths.sh
# ---------------------------------------------------------------------------
set -euo pipefail

REPO="${REPO:-/home/jyc/projects/Capstone}"
OLD="/home/ppco915/projects/Capstone"
NEW="$REPO"

if [[ "$OLD" == "$NEW" ]]; then
  echo "paths already match ($NEW) — nothing to do"; exit 0
fi

# The 4 runtime files that embed the absolute path (found via git grep).
FILES=(
  "$REPO/data/mask3d_server_room/train_database.yaml"
  "$REPO/data/mask3d_server_room/Validation_database.yaml"
  "$REPO/data/mask3d_server_room/las6_label_summary.json"
  "$REPO/opt/Mask3D/mask3d/conf/data/datasets/halo_server_room.yaml"
)

for f in "${FILES[@]}"; do
  if [[ -f "$f" ]] && grep -q "$OLD" "$f"; then
    sed -i "s#${OLD}#${NEW}#g" "$f"; echo "rewrote $f"
  else
    echo "skip (absent or already clean): $f"
  fi
done

echo "--- remaining runtime references to ${OLD} (should be none) ---"
grep -rn "$OLD" "$REPO/data/mask3d_server_room" \
                "$REPO/opt/Mask3D/mask3d/conf" 2>/dev/null || echo "  none"
