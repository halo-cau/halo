#!/usr/bin/env python3
"""Convert a segment_scan label-colored MESH into a vivid, unlit-friendly
POINT CLOUD (float32 xyz + uchar rgb) + viewer legend.

Why: the browser viewer renders points unlit, so high-contrast label colors
"stamp" clearly; and the float32+uchar point format is the one the vendored
three.js PLYLoader reads reliably (Open3D double-precision meshes render dark
under lit materials). Recovers per-vertex labels by matching the mesh's baked
label_to_color values, then restamps a vivid palette.

Usage:
    python scripts/recon/labels_to_vivid_points.py --mesh LABELS.ply --json LABELS.json --out PTS.ply
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from engine.vision.segmentor_base import label_to_color

VIVID = {
    "unknown":       (0.30, 0.30, 0.33),  # neutral gray (structure, recedes)
    "server rack":   (0.10, 0.95, 0.35),  # green
    "wall":          (0.15, 0.55, 1.00),  # blue
    "floor":         (1.00, 0.55, 0.05),  # orange
    "ceiling":       (0.97, 0.86, 0.10),  # yellow
    "ac_unit":       (0.95, 0.15, 0.85),  # magenta
    "cable tray":    (0.10, 0.85, 0.95),  # cyan
    "cardboard box": (0.85, 0.45, 0.20),  # brown
    "chair":         (0.95, 0.45, 0.75),  # pink
    "trash can":     (0.55, 0.30, 0.95),  # purple
    "object":        (0.75, 0.75, 0.20),
}
_FALLBACK = (0.55, 0.55, 0.58)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", type=Path, required=True)
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    labels = list(json.loads(args.json.read_text()).get("label_counts", {}))
    if not labels:
        print("no labels in json", file=sys.stderr)
        return 1
    muted = np.array([label_to_color(l) for l in labels])

    m = o3d.io.read_triangle_mesh(str(args.mesh))
    v = np.asarray(m.vertices, dtype=np.float32)
    c = np.asarray(m.vertex_colors)
    idx = np.argmin(((c[:, None, :] - muted[None]) ** 2).sum(-1), axis=1)
    vivid = np.array([VIVID.get(labels[i], _FALLBACK) for i in idx])
    col = (np.clip(vivid, 0, 1) * 255).astype(np.uint8)

    trimesh.PointCloud(vertices=v, colors=col).export(str(args.out))
    counts = {labels[i]: int((idx == i).sum()) for i in range(len(labels))}
    counts = {k: v for k, v in sorted(counts.items(), key=lambda x: -x[1]) if v}
    legend = {"backend": "sam3_concept (vivid points)", "label_counts": counts,
              "palette": {k: [round(x, 3) for x in VIVID.get(k, _FALLBACK)] for k in counts},
              "n_instances": len(counts)}
    Path(args.out.with_suffix("").as_posix() + ".legend.json").write_text(json.dumps(legend, indent=2))
    print(f"wrote {args.out} ({len(v):,} pts) + legend; counts: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
