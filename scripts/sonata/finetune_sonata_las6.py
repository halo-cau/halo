"""Finetune a linear seg head on the frozen Sonata PTv3 encoder, on las6_corrected.

Default = LINEAR PROBE (encoder frozen): the most data-efficient, lowest-memory
option, and the right call for a small labeled set — it tests whether Sonata's
features linearly separate wall/floor/ceiling/server_rack/box_clutter/ac_unit.
``--mode full`` unfreezes the encoder (more capacity, but needs more training
data and VRAM — lower --crop/--patch if it OOMs on 8 GB).

Run in the `sonata` env:
    conda run -n sonata python scripts/sonata/finetune_sonata_las6.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import sonata
from _sonata_common import (  # noqa: E402  (same dir)
    CLASS_NAMES,
    feature_dim,
    load_encoder,
    per_class_iou,
    predict_labels,
    to_device,
    unpool_to_full,
)

_ROOT = Path(__file__).resolve().parents[2]
_DATA = _ROOT / "data" / "sonata_las6" / "las6.npz"
_META = _ROOT / "data" / "sonata_las6" / "meta.json"


class SegHead(nn.Module):
    def __init__(self, backbone_out_channels, num_classes):
        super().__init__()
        self.seg_head = nn.Linear(backbone_out_channels, num_classes)

    def forward(self, x):
        return self.seg_head(x)


def make_train_transform(grid_size: float, crop: int):
    """Augment → grid-sample (train) → crop to `crop` pts → collect feat+segment."""
    cfg = [
        dict(type="CenterShift", apply_z=True),
        dict(type="RandomRotate", angle=[-1, 1], axis="z", p=0.5),
        dict(type="RandomScale", scale=[0.95, 1.05]),
        dict(type="RandomFlip", p=0.5),
        dict(type="RandomJitter", sigma=0.005, clip=0.02),
        dict(type="ChromaticJitter", p=0.95, std=0.02),
        dict(type="GridSample", grid_size=grid_size, hash_type="fnv", mode="train",
             return_grid_coord=True, return_inverse=False),
        dict(type="SphereCrop", point_max=crop, mode="random"),
        dict(type="NormalizeColor"),
        dict(type="ToTensor"),
        dict(type="Collect", keys=("coord", "grid_coord", "color", "segment"),
             feat_keys=("coord", "color", "normal")),
    ]
    return sonata.transform.Compose(cfg)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=_DATA)
    ap.add_argument("--out", type=Path, default=_ROOT / "data" / "sonata_las6" / "sonata_head_las6.pth")
    ap.add_argument("--mode", choices=["linear", "full"], default="linear")
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--grid", type=float, default=0.02)
    ap.add_argument("--crop", type=int, default=100000, help="max pts/step (lower if OOM)")
    ap.add_argument("--patch", type=int, default=1024, help="PTv3 attn patch (lower to 512/256 if OOM)")
    ap.add_argument("--enable-flash", action="store_true")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    torch.manual_seed(0); np.random.seed(0)
    d = np.load(args.data)
    full = {"coord": d["coord"].astype(np.float32), "color": d["color"].astype(np.float32),
            "normal": d["normal"].astype(np.float32), "segment": d["segment"].astype(np.int64)}
    split = d["split"]
    train = {k: v[split == 0].copy() for k, v in full.items()}  # train on the train split
    meta = json.loads(_META.read_text()) if _META.exists() else {}
    w = meta.get("class_weights_inv_freq", {n: 1.0 for n in CLASS_NAMES})
    class_w = torch.tensor([w[n] for n in CLASS_NAMES], dtype=torch.float32, device=args.device)
    print(f"train pts={len(train['coord']):,} | class weights={[round(float(x),2) for x in class_w]}")

    model = load_encoder(enable_flash=args.enable_flash, patch=args.patch, device=args.device)
    C = feature_dim(model, train, args.device)
    head = SegHead(C, len(CLASS_NAMES)).to(args.device)
    print(f"backbone_out_channels={C} → SegHead({C}, {len(CLASS_NAMES)})  mode={args.mode}")

    if args.mode == "linear":
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        params = list(head.parameters())
    else:
        model.train()
        params = list(model.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.iters)
    criterion = nn.CrossEntropyLoss(weight=class_w, ignore_index=-1)
    transform = make_train_transform(args.grid, args.crop)

    running = 0.0
    for it in range(1, args.iters + 1):
        pt = transform({k: v.copy() for k, v in train.items()})
        seg = pt.pop("segment").reshape(-1).to(args.device)
        pt = to_device(pt, args.device)
        enc_ctx = torch.no_grad() if args.mode == "linear" else torch.enable_grad()
        with enc_ctx:
            out = unpool_to_full(model(pt))
            feat = out.feat
        if args.mode == "linear":
            feat = feat.detach()
        logits = head(feat)
        loss = criterion(logits, seg)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step(); sched.step()
        running += float(loss.item())
        if it % 100 == 0:
            print(f"  iter {it:>5}/{args.iters}  loss={running/100:.4f}  lr={sched.get_last_lr()[0]:.2e}")
            running = 0.0
        if args.eval_every and (it % args.eval_every == 0 or it == args.iters):
            pred = predict_labels(model, head, full, args.device)
            val = split == 1
            iou = per_class_iou(pred[val], full["segment"][val])
            print(f"  [val@{it}] mIoU={iou['mIoU']}  " +
                  " ".join(f"{n}={iou[n]}" for n in CLASS_NAMES))
            model.eval() if args.mode == "linear" else model.train()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "config": {"backbone_out_channels": C, "num_classes": len(CLASS_NAMES)},
        "state_dict": head.state_dict(),
        "class_names": CLASS_NAMES,
        "mode": args.mode,
        "encoder": {"repo_id": "facebook/sonata", "enable_flash": args.enable_flash, "patch": args.patch},
    }, args.out)
    print(f"\nSaved head → {args.out}")
    print(f"Predict + view:\n  conda run -n sonata python scripts/sonata/predict_sonata_las6.py --ckpt {args.out}")


if __name__ == "__main__":
    main()
