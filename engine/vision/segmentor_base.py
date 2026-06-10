"""Shared data types and abstract interface for 3D scene segmentation.

Both segmentation backends (Mask3D and Grounded-DINO+SAM2) return a
SegmentorResult and must implement BaseSegmentor.run().

Structural labels (wall, floor, ceiling), fixed server-room infrastructure,
and ambiguous/unlabeled vertices are kept in the cleaned mesh.  Only explicit
movable-object labels are stripped before voxelization.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import open3d as o3d


# ---------------------------------------------------------------------------
# Label taxonomy
# ---------------------------------------------------------------------------

#: Vertex-level labels produced by either backend.  String keys match
#: ScanNet-20 category names (Mask3D native) and Grounded-DINO prompt tokens.
STRUCTURAL_LABELS: frozenset[str] = frozenset({"wall", "floor", "ceiling"})

#: Fixed server-room assets that must remain as obstacles / thermal objects.
#: These are not flattened by wall/floor/ceiling priors, but they are preserved
#: in the cleaned geometry so server rack faces do not disappear.
STATIC_INFRASTRUCTURE_LABELS: frozenset[str] = frozenset({
    "server rack",
    "server cabinet",
    "rack cabinet",
    "network rack",
    "equipment rack",
    "cabinet",
    "ac_unit",
    "air conditioning unit",
    "AC unit",
    "cable tray",
})

#: Ambiguous labels are preserved so sparse / unlabeled scan regions remain
#: visible for review and available to the voxel closing stage.
AMBIGUOUS_LABELS: frozenset[str] = frozenset({"unknown", "object"})

#: Labels retained in the stripped mesh.
PRESERVED_LABELS: frozenset[str] = (
    STRUCTURAL_LABELS | STATIC_INFRASTRUCTURE_LABELS | AMBIGUOUS_LABELS
)

#: Everything else is movable / clutter.  This list is informational;
#: only these explicit movable classes should be removed by default.
MOVABLE_LABELS: tuple[str, ...] = (
    "bookshelf",
    "chair",
    "desk",
    "table",
    "door",
    "window",
    "curtain",
    "cardboard box",
    "otherfurniture",
    "fence",
    "monitor",
    "keyboard",
    "trash can",
    "fire extinguisher",
)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class SegmentorResult:
    """Output from a segmentation backend.

    Attributes
    ----------
    structural_mesh:
        Open3D TriangleMesh containing vertices classified as preserved
        scene geometry (wall / floor / ceiling / fixed infrastructure) plus
        ambiguous/unlabeled vertices.
        Ready for voxelization.
    vertex_labels:
        String label per vertex of the *original* mesh (before stripping).
        Length equals the number of vertices in the source mesh.
    label_colors:
        RGB float32 color per vertex for visualization (values in [0, 1]).
        Shape (N, 3).
    label_map:
        Dict mapping label string → list of vertex indices in the source mesh.
    n_removed:
        Number of vertices removed (explicit movable-object labels).
    backend:
        Name of the backend that produced this result ("mask3d" or "dino_sam2").
    """

    structural_mesh: o3d.geometry.TriangleMesh
    vertex_labels:   list[str]
    label_colors:    np.ndarray          # (N, 3) float32
    label_map:       dict[str, list[int]]
    n_removed:       int
    backend:         str
    extra:           dict = field(default_factory=dict)

    @property
    def n_total(self) -> int:
        return len(self.vertex_labels)

    @property
    def removal_fraction(self) -> float:
        return self.n_removed / max(1, self.n_total)


# ---------------------------------------------------------------------------
# Color palette (consistent across both backends for the viewer)
# ---------------------------------------------------------------------------

#: Deterministic per-label colors so both backends render identically.
LABEL_PALETTE: dict[str, tuple[float, float, float]] = {
    "wall":                    (0.216, 0.541, 0.867),  # #378ADD
    "floor":                   (0.706, 0.698, 0.663),  # #B4B2A9
    "ceiling":                 (0.910, 0.898, 0.867),  # #E8E5DD
    "server rack":             (0.114, 0.620, 0.459),  # #1D9E75
    "server cabinet":          (0.114, 0.620, 0.459),
    "rack cabinet":            (0.059, 0.431, 0.337),  # #0F6E56
    "network rack":            (0.059, 0.431, 0.337),
    "equipment rack":          (0.059, 0.431, 0.337),
    "cabinet":                 (0.114, 0.620, 0.459),
    "bookshelf":               (0.114, 0.620, 0.459),
    "ac_unit":                 (0.141, 0.663, 0.882),  # #24A9E1
    "air conditioning unit":   (0.141, 0.663, 0.882),
    "AC unit":                 (0.141, 0.663, 0.882),
    "chair":                   (0.910, 0.620, 0.310),  # #E89E4F
    "desk":                    (0.910, 0.620, 0.310),
    "table":                   (0.910, 0.620, 0.310),
    "door":                    (0.659, 0.420, 0.102),  # #A86B1A
    "window":                  (0.898, 0.941, 0.984),  # #E5F0FB
    "curtain":                 (0.886, 0.294, 0.290),  # #E24B4A
    "cardboard box":           (0.910, 0.620, 0.310),
    "otherfurniture":          (0.706, 0.698, 0.663),
    "fence":                   (0.373, 0.369, 0.353),  # #5F5E5A
    "cable tray":              (0.373, 0.369, 0.353),
    "object":                  (0.533, 0.529, 0.502),  # #888780
    "unknown":                 (0.373, 0.369, 0.353),
}

def label_to_color(label: str) -> tuple[float, float, float]:
    return LABEL_PALETTE.get(label, LABEL_PALETTE["unknown"])


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class BaseSegmentor(ABC):
    """Common interface for 3D segmentation backends."""

    @abstractmethod
    def run(
        self,
        mesh: o3d.geometry.TriangleMesh,
        source_path: Path,
    ) -> SegmentorResult:
        """Segment *mesh* and return a SegmentorResult.

        Parameters
        ----------
        mesh:
            Aligned Open3D TriangleMesh (after SOR + floor alignment).
        source_path:
            Path to the original scan file (OBJ/PLY).  Some backends use the
            raw colored texture; others only use the mesh geometry.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in filenames and logs."""


# ---------------------------------------------------------------------------
# Shared utility: strip non-structural vertices from a mesh
# ---------------------------------------------------------------------------

def strip_non_structural(
    mesh: o3d.geometry.TriangleMesh,
    vertex_labels: list[str],
) -> o3d.geometry.TriangleMesh:
    """Return a new mesh containing preserved + ambiguous vertices.

    Triangles where ANY vertex is an explicit movable object are removed.  This
    keeps unlabeled / ambiguous geometry in the mesh so the viewer and voxelizer
    can retain the full picture before destructive cleanup.
    """
    n = len(vertex_labels)
    keep = np.array(
        [lbl in PRESERVED_LABELS for lbl in vertex_labels],
        dtype=bool,
    )  # (N,)

    triangles = np.asarray(mesh.triangles)  # (T, 3)
    # Keep a triangle only if all three of its vertices are structural
    tri_keep = keep[triangles[:, 0]] & keep[triangles[:, 1]] & keep[triangles[:, 2]]

    old_to_new = np.full(n, -1, dtype=np.int64)
    old_to_new[keep] = np.arange(int(keep.sum()))

    new_tris = old_to_new[triangles[tri_keep]]  # re-map vertex indices
    # Sanity: discard any triangle with a -1 (should not happen after masking)
    valid = (new_tris >= 0).all(axis=1)
    new_tris = new_tris[valid]

    result = o3d.geometry.TriangleMesh()
    vertices = np.asarray(mesh.vertices)
    result.vertices = o3d.utility.Vector3dVector(vertices[keep])
    result.triangles = o3d.utility.Vector3iVector(new_tris)
    if mesh.has_vertex_colors():
        colors = np.asarray(mesh.vertex_colors)
        if len(colors) == n:
            result.vertex_colors = o3d.utility.Vector3dVector(colors[keep])
    result.compute_vertex_normals()
    return result
