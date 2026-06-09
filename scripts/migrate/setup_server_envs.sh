#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# setup_server_envs.sh — build CV conda envs, CONTAINED ENTIRELY IN /home/jyc.
#
# Shared-box safety rules baked in:
#   * Everything lives under $HOME (Miniconda, envs, caches). Nothing outside.
#   * No sudo. No system changes.
#   * Disk guard: aborts if free space is below MIN_FREE_GB (default 25).
#   * Package caches cleaned after builds to keep the shared disk lean.
#   * NO GPU is used here (pure CPU/disk) — safe to run while research trains.
#
# Build ONE env at a time on a near-full disk:
#   ONLY=pi3    bash scripts/migrate/setup_server_envs.sh
#   ONLY=sonata bash scripts/migrate/setup_server_envs.sh
#   ONLY=amb3r  bash scripts/migrate/setup_server_envs.sh
#   ONLY=halo   bash scripts/migrate/setup_server_envs.sh   # heaviest (Mask3D); build last
# FORCE=1 rebuilds. Default ONLY=none (does nothing but bootstrap+report).
# ---------------------------------------------------------------------------
set -euo pipefail

REPO="${REPO:-/home/jyc/projects/Capstone}"
MIGRATE="${REPO}/scripts/migrate"
FORCE="${FORCE:-0}"
ONLY="${ONLY:-none}"                  # none|halo|pi3|sonata|amb3r|all
ARCH="${TORCH_CUDA_ARCH_LIST:-8.9}"   # Ada (4090). NEVER 12.0 (ME has no Blackwell).
MIN_FREE_GB="${MIN_FREE_GB:-25}"
MINICONDA="${MINICONDA:-$HOME/miniconda3}"

# --- keep ALL caches inside the account ------------------------------------
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$HOME/.conda/pkgs}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$HOME/.cache/pip}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

log()  { printf '\n\033[1;36m== %s ==\033[0m\n' "$*"; }
warn() { printf '\033[1;33m!! %s\033[0m\n' "$*" >&2; }

# --- disk guard: never risk filling a shared 97%-full disk ------------------
free_gb="$(df -BG --output=avail "$HOME" | tail -1 | tr -dc '0-9')"
log "free space in \$HOME: ${free_gb}G (guard: ${MIN_FREE_GB}G)"
if (( free_gb < MIN_FREE_GB )); then
  warn "ABORT: ${free_gb}G free < ${MIN_FREE_GB}G guard. Free space or lower MIN_FREE_GB consciously."
  exit 1
fi

# --- conda bootstrap, INSIDE $HOME -----------------------------------------
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
elif [[ -d "$MINICONDA" ]]; then
  source "$MINICONDA/etc/profile.d/conda.sh"
else
  log "installing Miniconda to ${MINICONDA} (inside \$HOME)"
  curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o "$HOME/mc_installer.sh"
  bash "$HOME/mc_installer.sh" -b -p "$MINICONDA"
  rm -f "$HOME/mc_installer.sh"
  source "$MINICONDA/etc/profile.d/conda.sh"
fi
log "conda: $(conda --version) at $(conda info --base)"

env_exists() { conda env list | awk '{print $1}' | grep -qx "$1"; }
reclaim() { conda clean -afy >/dev/null 2>&1 || true; }

sanity() {
  conda run -n "$1" python - <<'PY' || warn "sanity import failed for the env above"
import torch
print("  torch", torch.__version__, "| cuda", torch.version.cuda)
print("  arch_list", torch.cuda.get_arch_list())
for m in ("MinkowskiEngine","spconv","sonata","xformers"):
    try: print("  ", m, __import__(m).__version__)
    except Exception: pass
PY
}

build_env() {
  local name="$1" fn="$2"
  if [[ "$ONLY" != "all" && "$ONLY" != "$name" ]]; then return 0; fi
  if env_exists "$name"; then
    if [[ "$FORCE" == "1" ]]; then log "removing '${name}' (FORCE=1)"; conda env remove -y -n "$name";
    else warn "env '${name}' exists — skipping (FORCE=1 to rebuild)"; sanity "$name" || true; return 0; fi
  fi
  log "building env '${name}'"
  if "$fn"; then reclaim; sanity "$name"; else
    warn "build of '${name}' FAILED — removing partial env"; conda env remove -y -n "$name" || true; reclaim; return 1
  fi
}

# --- pi3: torch 2.5.1 cu121 (Pi3 recon) — smallest; 4090 (sm_89) ----------
build_pi3() {
  conda create -y -n pi3 python=3.11
  conda run -n pi3 pip install --upgrade pip
  conda run -n pi3 pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
  grep -ivE '^(torch|torchvision|torchaudio)([=@ ]|$)' "${MIGRATE}/envs/pi3-pip-freeze.txt" > /tmp/pi3-req.txt
  conda run -n pi3 pip install -r /tmp/pi3-req.txt
}

# --- sonata: cu124 spconv (delegates to the repo's own setup) -- 4090 ------
build_sonata() { bash "${REPO}/scripts/sonata/setup_sonata_env.sh"; }

# --- amb3r: isolated, cu118 spconv/xformers -- 4090 ------------------------
build_amb3r() {
  conda create -y -n amb3r python=3.10
  conda run -n amb3r pip install --upgrade pip
  conda run -n amb3r pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
  conda run -n amb3r pip install -r "${REPO}/opt/amb3r/requirements.txt"
  [[ -f "${REPO}/opt/amb3r/setup.py" || -f "${REPO}/opt/amb3r/pyproject.toml" ]] \
    && conda run -n amb3r pip install -e "${REPO}/opt/amb3r" || true
}

# --- halo: torch 2.11 cu130 + detectron2 + MinkowskiEngine(source) ---------
# HEAVIEST (~10-12G) and needs a 4090 to run. Build only when you're about to
# train Mask3D and a 4090 is free. ME build needs nvcc matching cu130, installed
# in-env (no system touch).
build_halo() {
  conda create -y -n halo python=3.11
  conda run -n halo pip install --upgrade pip
  conda run -n halo pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130
  grep -ivE '^(torch|torchvision|torchaudio|minkowskiengine|detectron2)([=@ ]|$)' \
      "${MIGRATE}/envs/halo-pip-freeze.txt" > /tmp/halo-req.txt
  conda run -n halo pip install -r /tmp/halo-req.txt
  conda run -n halo pip install \
      'git+https://github.com/facebookresearch/detectron2.git@710e7795d0eeadf9def0e7ef957eea13532e34cf'
  # nvcc + cudart headers matching torch's CUDA major, in-env (no /usr/local touch)
  conda install -y -n halo -c nvidia cuda-nvcc cuda-cudart-dev || \
      warn "in-env cuda-nvcc install failed; ME build may need CUDA_HOME set manually"
  log "compiling MinkowskiEngine (TORCH_CUDA_ARCH_LIST=${ARCH})"
  TORCH_CUDA_ARCH_LIST="${ARCH}" conda run -n halo pip install -v --no-build-isolation "${REPO}/opt/MinkowskiEngine"
  patch_detectron2 halo
}
patch_detectron2() {
  local sp; sp="$(conda run -n "$1" python -c 'import detectron2,os;print(os.path.dirname(detectron2.__file__))' 2>/dev/null || true)"
  [[ -z "$sp" ]] && { warn "detectron2 not found in $1; skip patch"; return 0; }
  grep -rl 'Image\.LINEAR' "$sp" 2>/dev/null | while read -r f; do
    sed -i 's/Image\.LINEAR/Image.BILINEAR/g' "$f"; echo "  patched $f"
  done || true
}

# --- drive -----------------------------------------------------------------
log "REPO=${REPO} ONLY=${ONLY} FORCE=${FORCE} ARCH=${ARCH}"
log "fixing absolute paths (/home/ppco915 -> ${REPO})"
REPO="$REPO" bash "${MIGRATE}/fix_paths.sh" || warn "fix_paths reported issues — check above"

rc=0
build_env pi3    build_pi3    || rc=1
build_env sonata build_sonata || rc=1
build_env amb3r  build_amb3r  || rc=1
build_env halo   build_halo   || rc=1
reclaim

if [[ "$ONLY" == "none" ]]; then
  warn "ONLY=none → bootstrapped conda + fixed paths only. Pick an env, e.g.: ONLY=pi3 bash $0"
fi
log "done (rc=${rc}). GPU runs are gated — see run_all_blocked.sh; run ONLY when a card is free."
exit "$rc"
