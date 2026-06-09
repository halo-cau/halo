# Frozen env specs

Deterministic captures of the WSL envs so the server rebuild isn't "whatever
resolves today". Consumed by `../setup_server_envs.sh`.

- `halo-pip-freeze.txt` — full `pip freeze` of the `halo` env (Python 3.11.15).
- `pi3-pip-freeze.txt` — full `pip freeze` of the `pi3` env (Python 3.11.15).

**Why these aren't `pip install -r`'d wholesale:** three packages need special
handling and are filtered out by the setup script:

- `torch` / `torchvision` — installed first from the matching CUDA index
  (`halo`: cu130 → torch 2.11.0 / tv 0.26.0; `pi3`: cu121 → torch 2.5.1 / tv 0.20.1).
- `MinkowskiEngine==0.5.4` — built from `opt/MinkowskiEngine` source with
  `TORCH_CUDA_ARCH_LIST=8.9` (Ada). No PyPI wheel; no Blackwell support.
- `detectron2` — installed from the exact pinned git commit, then patched
  (`Image.LINEAR → Image.BILINEAR`, Pillow-12).

`sonata` and `amb3r` are not frozen here — they build from their own pinned
sources (`scripts/sonata/setup_sonata_env.sh`, `opt/amb3r/requirements.txt`).
