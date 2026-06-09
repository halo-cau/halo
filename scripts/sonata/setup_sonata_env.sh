#!/usr/bin/env bash
# Create an ISOLATED conda env for Sonata (PTv3) finetuning.
#
# Why isolated: Sonata needs torch 2.5 + cu124 + spconv-cu124. The `halo` env is
# torch 2.11 / CUDA 13 with no spconv; installing Sonata's stack into `halo`
# would downgrade torch and break Mask3D/Point-SAM/OpenShape (same trap noted for
# AMB3R). So we build a fresh env named `sonata` and never touch `halo`.
#
# FlashAttention is intentionally SKIPPED (its from-source build is slow/fragile).
# All our configs pass enable_flash=False, which Sonata supports.
#
# Usage:  bash scripts/sonata/setup_sonata_env.sh
set -euo pipefail

ENV=sonata
REPO="$(cd "$(dirname "$0")/../../opt/sonata" && pwd)"
echo "Sonata repo: $REPO"

if conda env list | grep -qE "^\s*${ENV}\s"; then
  echo "conda env '${ENV}' already exists — reusing. (delete with: conda env remove -n ${ENV})"
else
  conda create -y -n "${ENV}" python=3.10
fi

run() { conda run -n "${ENV}" "$@"; }

echo "== torch 2.5.0 + cu124 =="
run pip install --upgrade pip
run pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu124

echo "== sparse-conv + scatter (cu124 wheels) =="
run pip install spconv-cu124
run pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.0+cu124.html

echo "== misc deps (NO flash-attn — optional; configs use enable_flash=False) =="
run pip install "numpy==1.26.4" timm huggingface_hub addict scipy psutil open3d

echo "== sonata package (editable) =="
run pip install -e "${REPO}"

echo "== sanity =="
run python - <<'PY'
import torch, sonata, spconv, torch_scatter
print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
print("sonata import OK; spconv", spconv.__version__)
PY
echo "DONE. First 'sonata.load(\"sonata\", repo_id=\"facebook/sonata\")' will download weights to ~/.cache/sonata (CC-BY-NC)."
