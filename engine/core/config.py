"""Engine-level global constants for the HALO CV/RL pipeline."""

# --- Spatial Resolution ---
VOXEL_SIZE: float = 0.1  # 1 voxel = 0.1 m (10 cm)

# --- Grid Limits (meters) ---
MAX_ROOM_DIMENSIONS: tuple[float, float, float] = (20.0, 20.0, 4.0)

# --- Semantic Tags (np.int8 values) ---
SPACE_EMPTY: int = 0
OBSTACLE_WALL: int = 1
HEAT_LEGACY_SERVER: int = 2
COOLING_AC_VENT: int = 3
HUMAN_WORKSPACE: int = 4

SEMANTIC_LABELS: dict[int, str] = {
    SPACE_EMPTY: "empty",
    OBSTACLE_WALL: "wall",
    HEAT_LEGACY_SERVER: "legacy_server",
    COOLING_AC_VENT: "ac_vent",
    HUMAN_WORKSPACE: "human_workspace",
}

# --- Open3D Cleanup Parameters ---
SOR_NB_NEIGHBORS: int = 20
SOR_STD_RATIO: float = 2.0

RANSAC_DISTANCE_THRESHOLD: float = 0.02
RANSAC_NUM_POINTS: int = 3
RANSAC_NUM_ITERATIONS: int = 1000

# --- Morphological Closing ---
CLOSING_ITERATIONS: int = 2

# --- Gaussian Heat Injection ---
HEAT_RADIUS_VOXELS: int = 5
HEAT_SIGMA: float = 2.0
