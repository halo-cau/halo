#!/usr/bin/env python3
"""Convert a directory of HEIC photos to upright RGB JPGs.

EXIF orientation is baked in (exif_transpose) so downstream recon models see
upright images. Output filenames preserve the source stem and sort order.

Usage:
    python scripts/recon/convert_heic.py SRC_DIR DST_DIR [--max-side N] [--quality Q]
"""
import argparse
import sys
from pathlib import Path

from PIL import Image, ImageOps
import pillow_heif

pillow_heif.register_heif_opener()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--max-side", type=int, default=0,
                    help="downscale so longest side <= N (0 = keep original)")
    ap.add_argument("--quality", type=int, default=95)
    args = ap.parse_args()

    args.dst.mkdir(parents=True, exist_ok=True)
    heics = sorted(p for p in args.src.iterdir()
                   if p.suffix.lower() in (".heic", ".heif"))
    if not heics:
        print(f"no HEIC files in {args.src}", file=sys.stderr)
        return 1

    for i, src in enumerate(heics):
        img = Image.open(src)
        img = ImageOps.exif_transpose(img).convert("RGB")
        if args.max_side and max(img.size) > args.max_side:
            img.thumbnail((args.max_side, args.max_side), Image.LANCZOS)
        out = args.dst / f"{src.stem}.jpg"
        img.save(out, "JPEG", quality=args.quality)
        print(f"[{i+1}/{len(heics)}] {src.name} -> {out.name} {img.size}")

    print(f"done: {len(heics)} images -> {args.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
