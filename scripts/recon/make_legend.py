#!/usr/bin/env python3
"""Emit a <ply>.legend.json for tools/ply_viewer.html from a segment_scan
label-count JSON, using the project's canonical label->color map so the legend
swatches match the baked vertex colors.

Usage:
    python scripts/recon/make_legend.py --json s3_seg_sam3_concept.json --ply OUT.ply
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from engine.vision.segmentor_base import label_to_color


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, required=True, help="segment_scan label json")
    ap.add_argument("--ply", type=Path, required=True, help="labeled ply the legend pairs with")
    args = ap.parse_args()

    data = json.loads(args.json.read_text())
    counts = data.get("label_counts", {})
    palette = {lbl: [round(float(c), 4) for c in label_to_color(lbl)] for lbl in counts}

    legend = {
        "backend": data.get("backend", "sam3_concept"),
        "label_counts": counts,
        "palette": palette,
        "n_instances": len(counts),
    }
    out = args.ply.with_suffix("").as_posix() + ".legend.json"
    Path(out).write_text(json.dumps(legend, indent=2))
    print(f"wrote {out}  ({len(counts)} labels: {', '.join(counts)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
