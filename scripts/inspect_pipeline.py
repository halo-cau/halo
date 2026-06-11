"""Step-by-step CV pipeline inspector.

Runs each stage on a real OBJ scan and reports statistics + saves
cross-section PNG visualisations to server_room_phone/pipeline_stages/.

Usage:
    conda run -n halo python scripts/inspect_pipeline.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import open3d as o3d
import trimesh
from scipy.ndimage import binary_closing

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.core.config import (
    CLOSING_ITERATIONS,
    GRID_SHAPE,
    OBSTACLE_WALL,
    RANSAC_DISTANCE_THRESHOLD,
    RANSAC_NUM_ITERATIONS,
    RANSAC_NUM_POINTS,
    SPACE_EMPTY,
    SOR_NB_NEIGHBORS,
    SOR_STD_RATIO,
    VOXEL_SIZE,
)
from engine.vision.cleaner import _align_floor_to_z0

# ── Paths ─────────────────────────────────────────────────────────────────────
OBJ_PATH = Path(__file__).resolve().parents[1] / "server_room_phone" / "textured_output.obj"
OUT_DIR  = OBJ_PATH.parent / "pipeline_stages"
OUT_DIR.mkdir(exist_ok=True)

SEP = "─" * 60

# ── Helpers ───────────────────────────────────────────────────────────────────

def _header(stage: int, name: str) -> None:
    print(f"\n{SEP}")
    print(f"  Stage {stage}: {name}")
    print(SEP)


def _mesh_stats(label: str, mesh: o3d.geometry.TriangleMesh) -> None:
    verts = np.asarray(mesh.vertices)
    print(f"  {label}")
    print(f"    Vertices  : {len(verts):,}")
    print(f"    Triangles : {len(mesh.triangles):,}")
    if len(verts):
        bb_min = verts.min(axis=0)
        bb_max = verts.max(axis=0)
        extents = bb_max - bb_min
        print(f"    BBox min  : [{bb_min[0]:.3f}, {bb_min[1]:.3f}, {bb_min[2]:.3f}] m")
        print(f"    BBox max  : [{bb_max[0]:.3f}, {bb_max[1]:.3f}, {bb_max[2]:.3f}] m")
        print(f"    Extents   : {extents[0]:.2f} × {extents[1]:.2f} × {extents[2]:.2f} m")


def _save_z_histogram(mesh: o3d.geometry.TriangleMesh, filename: str, title: str) -> None:
    verts = np.asarray(mesh.vertices)
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.hist(verts[:, 2], bins=80, color="steelblue", edgecolor="none", alpha=0.8)
    ax.axvline(0.0, color="red", linewidth=1.2, linestyle="--", label="Z = 0")
    ax.set_xlabel("Z (metres)")
    ax.set_ylabel("Vertex count")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    path = OUT_DIR / filename
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"    Saved → {path}")


# Semantic label colour map for cross-section plots
_LABEL_COLORS = {
    0: (0.95, 0.95, 0.95),   # empty — near-white
    1: (0.25, 0.25, 0.25),   # wall — dark grey
    2: (0.20, 0.55, 0.85),   # cooling AC vent — blue
    3: (0.95, 0.40, 0.15),   # rack body — orange
    4: (0.15, 0.80, 0.35),   # rack intake — green
    5: (0.90, 0.20, 0.20),   # rack exhaust — red
    6: (0.80, 0.70, 0.20),   # workspace — gold
    7: (0.70, 0.15, 0.70),   # legacy server heat — purple
}

def _label_to_rgb(slice_2d: np.ndarray) -> np.ndarray:
    h, w = slice_2d.shape
    img = np.ones((h, w, 3), dtype=np.float32)
    for label, color in _LABEL_COLORS.items():
        mask = slice_2d == label
        img[mask] = color
    return img


def _save_grid_slices(grid: np.ndarray, filename_prefix: str, title: str) -> None:
    """Save XY (floor), XZ (side), and YZ (front) cross-sections."""
    gx, gy, gz = grid.shape
    mid_z = max(gz // 4, 0)          # one-quarter height ≈ rack mid-level
    mid_y = gy // 2
    mid_x = gx // 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title, fontsize=12)

    def _add_legend(ax):
        legend_labels = {
            "Empty": _LABEL_COLORS[0],
            "Wall": _LABEL_COLORS[1],
            "Cooling": _LABEL_COLORS[2],
            "Rack body": _LABEL_COLORS[3],
            "Rack intake": _LABEL_COLORS[4],
            "Rack exhaust": _LABEL_COLORS[5],
            "Workspace": _LABEL_COLORS[6],
            "Heat": _LABEL_COLORS[7],
        }
        patches = [
            plt.Rectangle((0, 0), 1, 1, color=c, label=l)
            for l, c in legend_labels.items()
            if np.any(grid == list(_LABEL_COLORS.keys())[list(_LABEL_COLORS.values()).index(c)])
        ]
        if patches:
            ax.legend(handles=patches, fontsize=6, loc="lower right")

    # XY plan (top-down) at z = mid_z
    xy = grid[:, :, mid_z]
    axes[0].imshow(_label_to_rgb(xy.T), origin="lower", aspect="equal")
    axes[0].set_title(f"XY plan (z={mid_z}, ≈{mid_z*VOXEL_SIZE:.1f}m)")
    axes[0].set_xlabel("X voxels"); axes[0].set_ylabel("Y voxels")

    # XZ elevation (side view) at y = mid_y
    xz = grid[:, mid_y, :]
    axes[1].imshow(_label_to_rgb(xz.T), origin="lower", aspect="equal")
    axes[1].set_title(f"XZ elevation (y={mid_y})")
    axes[1].set_xlabel("X voxels"); axes[1].set_ylabel("Z voxels")

    # YZ elevation (front view) at x = mid_x
    yz = grid[mid_x, :, :]
    axes[2].imshow(_label_to_rgb(yz.T), origin="lower", aspect="equal")
    axes[2].set_title(f"YZ elevation (x={mid_x})")
    axes[2].set_xlabel("Y voxels"); axes[2].set_ylabel("Z voxels")

    fig.tight_layout()
    path = OUT_DIR / f"{filename_prefix}_slices.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"    Saved → {path}")


def _grid_stats(label: str, grid: np.ndarray) -> None:
    unique, counts = np.unique(grid, return_counts=True)
    label_names = {
        0: "empty", 1: "wall", 2: "cooling", 3: "rack_body",
        4: "rack_intake", 5: "rack_exhaust", 6: "workspace", 7: "heat",
    }
    total = grid.size
    print(f"  {label} — shape {grid.shape}")
    for u, c in zip(unique, counts):
        pct = 100.0 * c / total
        print(f"    label {u:2d} ({label_names.get(int(u), '?'):12s}): {c:>10,}  ({pct:.2f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# Stage 0 — Raw mesh
# ══════════════════════════════════════════════════════════════════════════════
_header(0, "Raw mesh (OBJ load)")

mesh = o3d.io.read_triangle_mesh(str(OBJ_PATH))
mesh.compute_vertex_normals()
raw_mesh = o3d.geometry.TriangleMesh(mesh)  # deep copy
_mesh_stats("Raw", mesh)
_save_z_histogram(mesh, "s0_raw_z_hist.png", "Stage 0 — Raw: Z distribution of vertices")

# ══════════════════════════════════════════════════════════════════════════════
# Stage 1 — Statistical Outlier Removal (SOR)
# ══════════════════════════════════════════════════════════════════════════════
_header(1, f"SOR  (nb_neighbors={SOR_NB_NEIGHBORS}, std_ratio={SOR_STD_RATIO})")

pcd = o3d.geometry.PointCloud()
pcd.points = mesh.vertices
pcd.normals = mesh.vertex_normals

_, inlier_idx = pcd.remove_statistical_outlier(
    nb_neighbors=SOR_NB_NEIGHBORS,
    std_ratio=SOR_STD_RATIO,
)
n_before = len(np.asarray(mesh.vertices))
n_removed = n_before - len(inlier_idx)

mesh_sor = mesh.select_by_index(inlier_idx)
mesh_sor.compute_vertex_normals()

print(f"  Removed outlier vertices : {n_removed:,} / {n_before:,} ({100.*n_removed/n_before:.1f}%)")
print(f"  Remaining vertices       : {len(inlier_idx):,}")
_mesh_stats("After SOR", mesh_sor)
_save_z_histogram(mesh_sor, "s1_sor_z_hist.png", "Stage 1 — After SOR: Z distribution of vertices")

# ══════════════════════════════════════════════════════════════════════════════
# Stage 2 — RANSAC floor detection + alignment
# ══════════════════════════════════════════════════════════════════════════════
_header(2, f"RANSAC floor detection  (dist_thresh={RANSAC_DISTANCE_THRESHOLD}, iters={RANSAC_NUM_ITERATIONS})")

pcd_clean = o3d.geometry.PointCloud()
pcd_clean.points = mesh_sor.vertices
pcd_clean.normals = mesh_sor.vertex_normals

best_plane    = None
best_z_comp   = 0.0
remaining     = pcd_clean
max_attempts  = 5
attempt_log   = []

for attempt in range(max_attempts):
    if len(remaining.points) < RANSAC_NUM_POINTS:
        break
    plane_model, inlier_idx_r = remaining.segment_plane(
        distance_threshold=RANSAC_DISTANCE_THRESHOLD,
        ransac_n=RANSAC_NUM_POINTS,
        num_iterations=RANSAC_NUM_ITERATIONS,
    )
    normal = np.array(plane_model[:3])
    normal /= np.linalg.norm(normal)
    z_comp = abs(normal[2])
    attempt_log.append((attempt + 1, plane_model, z_comp, len(inlier_idx_r)))

    if z_comp > best_z_comp:
        best_z_comp  = z_comp
        best_plane   = np.asarray(plane_model)

    if z_comp > 0.9:
        break
    remaining = remaining.select_by_index(inlier_idx_r, invert=True)

print(f"  RANSAC attempts:")
for a, pm, zc, ni in attempt_log:
    print(f"    attempt {a}: plane=({pm[0]:.3f},{pm[1]:.3f},{pm[2]:.3f},{pm[3]:.3f})"
          f"  |n_z|={zc:.3f}  inliers={ni:,}")

a, b, c, d = best_plane
print(f"\n  Best floor plane : {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")
print(f"  Floor normal     : ({a:.3f}, {b:.3f}, {c:.3f})  |n_z| = {best_z_comp:.4f}")

mesh_aligned = _align_floor_to_z0(mesh_sor, best_plane)
verts_aligned = np.asarray(mesh_aligned.vertices)

print(f"\n  After alignment:")
print(f"    Z range : [{verts_aligned[:,2].min():.4f}, {verts_aligned[:,2].max():.4f}] m")
print(f"    Z mean  : {verts_aligned[:,2].mean():.4f} m")
print(f"    Z std   : {verts_aligned[:,2].std():.4f} m")

_mesh_stats("After floor alignment", mesh_aligned)
_save_z_histogram(mesh_aligned, "s2_aligned_z_hist.png",
                  "Stage 2 — After RANSAC alignment: Z=0 should be the floor")

# Compare pre/post Z distributions side-by-side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Stage 2 — Z distribution before vs after floor alignment")
verts_raw = np.asarray(mesh_sor.vertices)
ax1.hist(verts_raw[:, 2], bins=80, color="coral",    edgecolor="none", alpha=0.8)
ax1.set_title("Before alignment"); ax1.set_xlabel("Z (m)")
ax2.hist(verts_aligned[:, 2], bins=80, color="steelblue", edgecolor="none", alpha=0.8)
ax2.axvline(0.0, color="red", linewidth=1.5, linestyle="--", label="Z=0 (floor)")
ax2.set_title("After alignment"); ax2.set_xlabel("Z (m)"); ax2.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "s2_alignment_compare.png", dpi=130)
plt.close(fig)
print(f"    Saved → {OUT_DIR / 's2_alignment_compare.png'}")

# ══════════════════════════════════════════════════════════════════════════════
# Stage 3 — Surface voxelization
# ══════════════════════════════════════════════════════════════════════════════
_header(3, f"Surface voxelization  (voxel_size={VOXEL_SIZE} m)")

import tempfile, os
tmp = tempfile.NamedTemporaryFile(suffix=".ply", delete=False)
tmp.close()
o3d.io.write_triangle_mesh(tmp.name, mesh_aligned)

mesh_tri = trimesh.load(tmp.name, force="mesh")
os.unlink(tmp.name)

voxel_grid = mesh_tri.voxelized(pitch=VOXEL_SIZE)
grid_raw = voxel_grid.matrix.astype(np.int8)
origin   = voxel_grid.transform[:3, 3]

print(f"  Grid shape  : {grid_raw.shape}  (x×y×z voxels)")
print(f"  Origin      : [{origin[0]:.3f}, {origin[1]:.3f}, {origin[2]:.3f}] m")
room_m = np.array(grid_raw.shape) * VOXEL_SIZE
print(f"  Room size   : {room_m[0]:.1f} × {room_m[1]:.1f} × {room_m[2]:.1f} m")
_grid_stats("Voxel counts", grid_raw)
_save_grid_slices(grid_raw, "s3_voxel", "Stage 3 — Surface voxelization (grey=wall, white=air)")

# ══════════════════════════════════════════════════════════════════════════════
# Stage 4 — Morphological closing
# ══════════════════════════════════════════════════════════════════════════════
_header(4, f"Morphological closing  (iterations={CLOSING_ITERATIONS})")

wall_mask  = grid_raw == OBSTACLE_WALL
closed     = binary_closing(wall_mask, iterations=CLOSING_ITERATIONS)
grid_closed = grid_raw.copy()
new_wall_mask = closed & (grid_raw == SPACE_EMPTY)
grid_closed[new_wall_mask] = OBSTACLE_WALL

n_new = int(new_wall_mask.sum())
print(f"  New wall voxels sealed : {n_new:,}")
print(f"  % of total volume      : {100.*n_new/grid_raw.size:.3f}%")
_grid_stats("After closing", grid_closed)
_save_grid_slices(grid_closed, "s4_closed", "Stage 4 — After morphological closing")

# ══════════════════════════════════════════════════════════════════════════════
# Stage 5 — Pad to fixed shape
# ══════════════════════════════════════════════════════════════════════════════
_header(5, f"Pad to fixed shape  {GRID_SHAPE}")

gx, gy, gz = grid_closed.shape
fx, fy, fz = GRID_SHAPE

if gx > fx or gy > fy or gz > fz:
    print(f"  WARNING: room grid {grid_closed.shape} exceeds fixed shape {GRID_SHAPE}!")
    print(f"  Skipping pad stage.")
    padded = None
else:
    ox = (fx - gx) // 2
    oy = (fy - gy) // 2
    oz = 0  # floor at bottom
    padded = np.zeros(GRID_SHAPE, dtype=np.int8)
    padded[ox:ox+gx, oy:oy+gy, oz:oz+gz] = grid_closed
    print(f"  Room grid     : {grid_closed.shape}")
    print(f"  Fixed shape   : {GRID_SHAPE}")
    print(f"  Offset (x,y,z): ({ox}, {oy}, {oz})")
    _grid_stats("Padded grid", padded)
    _save_grid_slices(padded, "s5_padded", f"Stage 5 — Padded to {GRID_SHAPE}")

# ══════════════════════════════════════════════════════════════════════════════
# Summary figure — wall voxel count at each stage
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  Summary")
print(SEP)

stages   = ["S0\nRaw", "S1\nSOR", "S2\nAligned", "S3\nVoxelized", "S4\nClosed"]
v_counts = [
    len(np.asarray(raw_mesh.vertices)),
    len(np.asarray(mesh_sor.vertices)),
    len(np.asarray(mesh_aligned.vertices)),
    int(np.sum(grid_raw   == OBSTACLE_WALL)),
    int(np.sum(grid_closed == OBSTACLE_WALL)),
]
fig, ax = plt.subplots(figsize=(9, 4))
bars = ax.bar(stages, v_counts, color=["#aaa", "#6ba3d6", "#4a90d9", "#e07b30", "#c0392b"])
ax.set_ylabel("Vertex / wall-voxel count")
ax.set_title("Pipeline progress — element count at each stage")
for bar, val in zip(bars, v_counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.01,
            f"{val:,}", ha="center", va="bottom", fontsize=8)
fig.tight_layout()
fig.savefig(OUT_DIR / "summary_counts.png", dpi=130)
plt.close(fig)
print(f"  Saved → {OUT_DIR / 'summary_counts.png'}")
print(f"\nAll outputs in: {OUT_DIR}")
