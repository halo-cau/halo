"""Interactive-style 3D visualisation of every CV pipeline stage.

Renders genuine 3D perspective views (not 2D cross-sections) using
matplotlib's 3D scatter/surface API, so no display or GPU is required.

Each stage is saved as a PNG in the chosen output directory.

Usage:
    conda run -n halo python scripts/visualize_pipeline.py \\
        --scan server_room_phone/server_room_6.las \\
        --out-dir server_room_phone/pipeline_vis_demo
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
import numpy as np
import open3d as o3d
import trimesh
from scipy.ndimage import binary_closing

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.core.config import (
    CLOSING_ITERATIONS,
    OBSTACLE_WALL,
    RANSAC_DISTANCE_THRESHOLD,
    RANSAC_NUM_ITERATIONS,
    RANSAC_NUM_POINTS,
    SPACE_EMPTY,
    SOR_NB_NEIGHBORS,
    SOR_STD_RATIO,
    VOXEL_SIZE,
)
from engine.vision.cleaner import _align_floor_to_z0, _find_floor_plane, _load_scan_as_vertex_mesh

# ── Config ────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_SCAN = _PROJECT_ROOT / "server_room_phone" / "textured_output.obj"

_argparser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_argparser.add_argument("--scan", type=Path, default=_DEFAULT_SCAN,
                        help="Path to a mesh scan (.obj/.ply with triangles). Default: textured_output.obj. "
                             "Note: LAS/LAZ point clouds lack faces and break the surface-voxelization stage.")
_argparser.add_argument("--out-dir", type=Path, default=None,
                        help="Directory for PNG artifacts. Default: <scan parent>/pipeline_vis.")
_args, _ = _argparser.parse_known_args()

OBJ_PATH = _args.scan.expanduser().resolve()
if not OBJ_PATH.exists():
    raise SystemExit(f"Scan not found: {OBJ_PATH}")
OUT_DIR = (_args.out_dir if _args.out_dir is not None else OBJ_PATH.parent / "pipeline_vis").expanduser().resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_PTS   = 25_000   # downsample cap for scatter plots (keeps renders fast)
ELEV, AZIM = 25, -55  # default 3D camera

# ── Helpers ───────────────────────────────────────────────────────────────────

def _downsample(pts: np.ndarray, max_pts: int = MAX_PTS) -> np.ndarray:
    if len(pts) <= max_pts:
        return pts
    idx = np.random.choice(len(pts), max_pts, replace=False)
    return pts[idx]


def _normals_to_rgb(normals: np.ndarray) -> np.ndarray:
    """Map XYZ normal directions to RGB so surface orientation is readable."""
    n = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-9)
    return np.clip(n * 0.5 + 0.5, 0, 1)


def _scatter3(ax, pts, colors=None, size=0.4, alpha=0.25, label=None):
    s = _downsample(pts)
    if colors is not None:
        c = colors[np.random.choice(len(pts), len(s), replace=False)] if len(colors) == len(pts) else colors
    else:
        c = "steelblue"
    ax.scatter(s[:, 0], s[:, 1], s[:, 2], c=c, s=size, alpha=alpha,
               linewidths=0, label=label, rasterized=True)


def _style_ax(ax, title, xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)"):
    ax.set_title(title, fontsize=9, pad=4)
    ax.set_xlabel(xlabel, fontsize=7, labelpad=2)
    ax.set_ylabel(ylabel, fontsize=7, labelpad=2)
    ax.set_zlabel(zlabel, fontsize=7, labelpad=2)
    ax.tick_params(labelsize=6)
    ax.view_init(elev=ELEV, azim=AZIM)


def _save(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


def _plane_mesh(plane_model, pts, margin=0.3):
    """Return (XX, YY, ZZ) arrays for the detected RANSAC plane.

    Parameterizes the two axes orthogonal to the plane normal's dominant
    component and solves for the third, so a plane whose normal is ±X or
    ±Y renders correctly (not flattened to a Z=0 sheet).
    """
    a, b, c, d = plane_model
    coeffs = np.array([a, b, c], dtype=np.float64)
    dom = int(np.argmax(np.abs(coeffs)))
    # parameter axes (the two non-dominant coords)
    p1, p2 = [i for i in range(3) if i != dom]
    lo1, hi1 = pts[:, p1].min() - margin, pts[:, p1].max() + margin
    lo2, hi2 = pts[:, p2].min() - margin, pts[:, p2].max() + margin
    G1, G2 = np.meshgrid(np.linspace(lo1, hi1, 30), np.linspace(lo2, hi2, 30))
    # ax + by + cz + d = 0  →  dom = -(other_terms + d) / coeffs[dom]
    G_dom = -(coeffs[p1] * G1 + coeffs[p2] * G2 + d) / coeffs[dom]
    out = [None, None, None]
    out[p1] = G1
    out[p2] = G2
    out[dom] = G_dom
    return out[0], out[1], out[2]


# ══════════════════════════════════════════════════════════════════════════════
# Load raw mesh
# ══════════════════════════════════════════════════════════════════════════════
print(f"Loading mesh from {OBJ_PATH.name} …")
# Use the cleaner's universal loader so LAS/LAZ point-cloud inputs are
# converted to vertex-only TriangleMesh containers consistently with the
# production pipeline.
mesh = _load_scan_as_vertex_mesh(OBJ_PATH)
mesh.compute_vertex_normals()

raw_verts   = np.asarray(mesh.vertices)
raw_normals = np.asarray(mesh.vertex_normals)

# ══════════════════════════════════════════════════════════════════════════════
# Stage 0 — Raw mesh: 3-panel (front / side / isometric)
# ══════════════════════════════════════════════════════════════════════════════
print("\nStage 0 — Raw mesh …")

rgb = _normals_to_rgb(raw_normals)

fig = plt.figure(figsize=(15, 5))
fig.suptitle("Stage 0 — Raw mesh  (colored by surface normal)", fontsize=11)

for col, (elev, azim, lbl) in enumerate([(20, -60, "Isometric"), (90, -90, "Top-down (XY)"), (0, -90, "Front (XZ)")]):
    ax = fig.add_subplot(1, 3, col + 1, projection="3d")
    _scatter3(ax, raw_verts, colors=rgb, size=0.3, alpha=0.3)
    ax.view_init(elev=elev, azim=azim)
    _style_ax(ax, lbl)

fig.tight_layout()
_save(fig, "s0_raw_mesh.png")

# ══════════════════════════════════════════════════════════════════════════════
# Stage 1 — SOR: show kept (blue) vs removed (red) vertices
# ══════════════════════════════════════════════════════════════════════════════
print("Stage 1 — SOR …")

pcd = o3d.geometry.PointCloud()
pcd.points  = mesh.vertices
pcd.normals = mesh.vertex_normals

_, inlier_idx = pcd.remove_statistical_outlier(nb_neighbors=SOR_NB_NEIGHBORS, std_ratio=SOR_STD_RATIO)
inlier_set  = set(inlier_idx)
all_idx     = np.arange(len(raw_verts))
outlier_idx = np.array([i for i in all_idx if i not in inlier_set])

inlier_pts  = raw_verts[inlier_idx]
outlier_pts = raw_verts[outlier_idx] if len(outlier_idx) else np.empty((0, 3))

print(f"  Outliers: {len(outlier_idx):,}  Inliers: {len(inlier_pts):,}")

fig = plt.figure(figsize=(15, 5))
fig.suptitle(
    f"Stage 1 — SOR cleaning  (blue = kept {len(inlier_pts):,} | red = removed {len(outlier_idx):,})",
    fontsize=11,
)
for col, (elev, azim, lbl) in enumerate([(20, -60, "Isometric"), (90, -90, "Top-down"), (0, -90, "Front")]):
    ax = fig.add_subplot(1, 3, col + 1, projection="3d")
    _scatter3(ax, inlier_pts,  colors="steelblue", size=0.3, alpha=0.20, label="Kept")
    if len(outlier_pts):
        _scatter3(ax, outlier_pts, colors="red",       size=3.0, alpha=0.7,  label="Outlier")
    ax.view_init(elev=elev, azim=azim)
    _style_ax(ax, lbl)
    if col == 0:
        ax.legend(markerscale=6, fontsize=7, loc="upper left")

fig.tight_layout()
_save(fig, "s1_sor_cleaning.png")

# Build cleaned mesh for next stages.  Snapshot vertices/normals as copies so
# they aren't silently rotated when `_align_floor_to_z0` mutates `mesh_sor`
# in place (numpy views into Open3D buffers would otherwise track the rotation
# and the "before alignment" panel would render post-alignment coords).
mesh_sor = mesh.select_by_index(inlier_idx)
mesh_sor.compute_vertex_normals()
sor_verts   = np.array(mesh_sor.vertices)
sor_normals = np.array(mesh_sor.vertex_normals)

# ══════════════════════════════════════════════════════════════════════════════
# Stage 2a — RANSAC plane detection: defer to the production cleaner so the
# visualization shows the plane the live pipeline actually picks.
# ══════════════════════════════════════════════════════════════════════════════
print("Stage 2 — RANSAC floor detection …")

best_plane, up_axis = _find_floor_plane(mesh_sor)
a, b, c, d = best_plane
axis_name = "XYZ"[up_axis]
n_up = abs(np.asarray(best_plane[:3]) / np.linalg.norm(best_plane[:3]))[up_axis]
print(f"  Inferred up-axis: {axis_name}   |n_{axis_name.lower()}|={n_up:.4f}")

# Render: point cloud + detected floor plane + plane normal annotation
fig = plt.figure(figsize=(10, 8))
fig.suptitle(
    f"Stage 2a — RANSAC floor detection\n"
    f"Best plane normal: ({a:.3f}, {b:.3f}, {c:.3f})  inferred up-axis: {axis_name}  |n_{axis_name.lower()}|={n_up:.3f}",
    fontsize=10,
)
ax = fig.add_subplot(1, 1, 1, projection="3d")
rgb_sor = _normals_to_rgb(sor_normals)
_scatter3(ax, sor_verts, colors=rgb_sor, size=0.3, alpha=0.18)

# Draw the detected floor plane as a translucent surface
XX, YY, ZZ = _plane_mesh(best_plane, sor_verts)
ax.plot_surface(XX, YY, ZZ, alpha=0.35, color="red", linewidth=0,
                antialiased=False, label="Detected plane")

# Draw plane normal arrow from centroid; orient it toward the inferred up axis.
centroid = sor_verts.mean(axis=0)
normal_v = np.array([a, b, c]) / np.linalg.norm([a, b, c])
if normal_v[up_axis] < 0:
    normal_v = -normal_v
arrow_len = float(np.ptp(sor_verts[:, up_axis]) * 0.3)
ax.quiver(*centroid, *(normal_v * arrow_len), color="orange", linewidth=2,
          arrow_length_ratio=0.15)

ax.view_init(elev=20, azim=-55)
_style_ax(ax, "Detected plane (red) + normal (orange)")
red_patch = mpatches.Patch(color="red", alpha=0.4, label="Detected floor plane")
ax.legend(handles=[red_patch], fontsize=8)
fig.tight_layout()
_save(fig, "s2a_ransac_plane.png")

# ── Stage 2b: before vs after alignment side-by-side ─────────────────────────
print("  Floor alignment …")

mesh_aligned = _align_floor_to_z0(mesh_sor, best_plane, up_axis=up_axis)
aligned_verts   = np.asarray(mesh_aligned.vertices)
aligned_normals = np.asarray(mesh_aligned.vertex_normals)
rgb_aligned = _normals_to_rgb(aligned_normals)

fig = plt.figure(figsize=(14, 6))
fig.suptitle("Stage 2b — Floor alignment: before (left) vs after (right, Z=0 is floor)", fontsize=11)

# Before
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
_scatter3(ax1, sor_verts, colors=rgb_sor, size=0.3, alpha=0.22)
# Mark the raw floor plane
XX, YY, ZZ = _plane_mesh(best_plane, sor_verts)
ax1.plot_surface(XX, YY, ZZ, alpha=0.3, color="red", linewidth=0)
ax1.view_init(elev=20, azim=-55)
_style_ax(ax1, f"Before  (floor plane in red)\nZ ∈ [{sor_verts[:,2].min():.2f}, {sor_verts[:,2].max():.2f}] m")

# After
ax2 = fig.add_subplot(1, 2, 2, projection="3d")
_scatter3(ax2, aligned_verts, colors=rgb_aligned, size=0.3, alpha=0.22)
# Z=0 floor reference plane
gx = np.linspace(aligned_verts[:,0].min(), aligned_verts[:,0].max(), 2)
gy = np.linspace(aligned_verts[:,1].min(), aligned_verts[:,1].max(), 2)
GX, GY = np.meshgrid(gx, gy)
GZ = np.zeros_like(GX)
ax2.plot_surface(GX, GY, GZ, alpha=0.15, color="green", linewidth=0)
ax2.view_init(elev=20, azim=-55)
_style_ax(ax2, f"After  (Z=0 floor in green)\nZ ∈ [{aligned_verts[:,2].min():.3f}, {aligned_verts[:,2].max():.2f}] m")

fig.tight_layout()
_save(fig, "s2b_floor_alignment.png")

# ══════════════════════════════════════════════════════════════════════════════
# Stage 3 — Surface voxelization
# ══════════════════════════════════════════════════════════════════════════════
print("Stage 3 — Voxelization …")

tmp = tempfile.NamedTemporaryFile(suffix=".ply", delete=False)
tmp.close()
o3d.io.write_triangle_mesh(tmp.name, mesh_aligned)
mesh_tri  = trimesh.load(tmp.name, force="mesh")
os.unlink(tmp.name)

voxel_grid = mesh_tri.voxelized(pitch=VOXEL_SIZE)
grid_raw   = voxel_grid.matrix.astype(np.int8)
origin     = voxel_grid.transform[:3, 3]

# Wall voxel centres in world space
wall_idx = np.argwhere(grid_raw == OBSTACLE_WALL)
wall_pts  = wall_idx * VOXEL_SIZE + origin

print(f"  Grid {grid_raw.shape}  wall voxels: {len(wall_pts):,}")

fig = plt.figure(figsize=(15, 5))
fig.suptitle(
    f"Stage 3 — Surface voxelization  ({VOXEL_SIZE*100:.0f} cm voxels)  "
    f"grid {grid_raw.shape}  |  {len(wall_pts):,} wall voxels",
    fontsize=10,
)

# Three views: isometric, top-down, front elevation
for col, (elev, azim, lbl) in enumerate([(22, -55, "Isometric"), (90, -90, "Top-down (XY plan)"), (0, -90, "Front elevation (XZ)")]):
    ax = fig.add_subplot(1, 3, col + 1, projection="3d")
    s = _downsample(wall_pts, MAX_PTS)
    # Color by height (Z) so structure depth is visible
    z_norm = (s[:, 2] - s[:, 2].min()) / (float(np.ptp(s[:, 2])) + 1e-6)
    ax.scatter(s[:, 0], s[:, 1], s[:, 2],
               c=z_norm, cmap="plasma", s=0.8, alpha=0.6,
               linewidths=0, rasterized=True)
    ax.view_init(elev=elev, azim=azim)
    _style_ax(ax, lbl)

fig.tight_layout()
_save(fig, "s3_voxelization.png")

# ══════════════════════════════════════════════════════════════════════════════
# Stage 4 — Morphological closing: show original walls + newly sealed voxels
# ══════════════════════════════════════════════════════════════════════════════
print("Stage 4 — Morphological closing …")

wall_mask   = grid_raw == OBSTACLE_WALL
closed      = binary_closing(wall_mask, iterations=CLOSING_ITERATIONS)
new_mask    = closed & (grid_raw == SPACE_EMPTY)

new_idx  = np.argwhere(new_mask)
new_pts  = new_idx * VOXEL_SIZE + origin

print(f"  New sealed voxels: {len(new_pts):,}")

fig = plt.figure(figsize=(15, 5))
fig.suptitle(
    f"Stage 4 — Morphological closing  "
    f"(grey = original wall | orange = {len(new_pts):,} newly sealed voxels)",
    fontsize=10,
)

orig_patch = mpatches.Patch(color="grey",   alpha=0.5, label="Original wall")
new_patch  = mpatches.Patch(color="orange", alpha=0.8, label="Newly sealed")

for col, (elev, azim, lbl) in enumerate([(22, -55, "Isometric"), (90, -90, "Top-down (XY plan)"), (0, -90, "Front elevation (XZ)")]):
    ax = fig.add_subplot(1, 3, col + 1, projection="3d")
    s_wall = _downsample(wall_pts, MAX_PTS)
    ax.scatter(s_wall[:, 0], s_wall[:, 1], s_wall[:, 2],
               c="grey", s=0.6, alpha=0.25, linewidths=0, rasterized=True)
    if len(new_pts):
        s_new = _downsample(new_pts, min(MAX_PTS, len(new_pts)))
        ax.scatter(s_new[:, 0], s_new[:, 1], s_new[:, 2],
                   c="orange", s=4.0, alpha=0.85, linewidths=0, rasterized=True)
    ax.view_init(elev=elev, azim=azim)
    _style_ax(ax, lbl)
    if col == 0:
        ax.legend(handles=[orig_patch, new_patch], fontsize=7, loc="upper left", markerscale=4)

fig.tight_layout()
_save(fig, "s4_morphological_closing.png")

# ══════════════════════════════════════════════════════════════════════════════
# Bonus: raw mesh vertex cloud vs cleaned aligned cloud — side-by-side comparison
# ══════════════════════════════════════════════════════════════════════════════
print("Bonus — raw vs clean comparison …")

fig = plt.figure(figsize=(14, 6))
fig.suptitle("Raw mesh vs cleaned + aligned  (isometric view, colored by normal)", fontsize=11)

ax1 = fig.add_subplot(1, 2, 1, projection="3d")
_scatter3(ax1, raw_verts, colors=_normals_to_rgb(raw_normals), size=0.3, alpha=0.22)
ax1.view_init(elev=22, azim=-55)
_style_ax(ax1, f"Raw  ({len(raw_verts):,} vertices)")

ax2 = fig.add_subplot(1, 2, 2, projection="3d")
_scatter3(ax2, aligned_verts, colors=rgb_aligned, size=0.3, alpha=0.22)
ax2.view_init(elev=22, azim=-55)
_style_ax(ax2, f"Cleaned + aligned  ({len(aligned_verts):,} vertices)\nZ=0 = floor")

fig.tight_layout()
_save(fig, "s_compare_raw_vs_clean.png")

print(f"\nAll images saved to: {OUT_DIR}")
