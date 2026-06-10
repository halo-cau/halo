"""Engine-level global constants for the HALO CV/RL pipeline."""

# --- Spatial Resolution ---
VOXEL_SIZE: float = 0.1  # 1 voxel = 0.1 m (10 cm)

# --- Grid Limits (meters) ---
MAX_ROOM_DIMENSIONS: tuple[float, float, float] = (10.0, 10.0, 5.0)

# Maximum voxel grid that can represent any room within MAX_ROOM_DIMENSIONS.
# Derived from MAX_ROOM_DIMENSIONS / VOXEL_SIZE → (100, 100, 50).
# Actual scanned rooms are smaller; unused voxels are padded with SPACE_EMPTY.
GRID_SHAPE: tuple[int, int, int] = (100, 100, 50)

# --- Semantic Tags (np.int8 values) ---
SPACE_EMPTY: int = 0
OBSTACLE_WALL: int = 1
HEAT_LEGACY_SERVER: int = 2
COOLING_AC_VENT: int = 3
HUMAN_WORKSPACE: int = 4
RACK_BODY: int = 5
RACK_INTAKE: int = 6
RACK_EXHAUST: int = 7

SEMANTIC_LABELS: dict[int, str] = {
    SPACE_EMPTY: "empty",
    OBSTACLE_WALL: "wall",
    HEAT_LEGACY_SERVER: "legacy_server",
    COOLING_AC_VENT: "ac_vent",
    HUMAN_WORKSPACE: "human_workspace",
    RACK_BODY: "rack_body",
    RACK_INTAKE: "rack_intake",
    RACK_EXHAUST: "rack_exhaust",
}

# --- Open3D Cleanup Parameters ---
SOR_NB_NEIGHBORS: int = 20
SOR_STD_RATIO: float = 2.0

RANSAC_DISTANCE_THRESHOLD: float = 0.02
RANSAC_NUM_POINTS: int = 3
RANSAC_NUM_ITERATIONS: int = 1000

# --- Manhattan World Rectification (iterative RANSAC plane extraction) ---
# Plane normal must lie within this angle of a principal axis to qualify as a
# structural wall / floor / ceiling.
MANHATTAN_NORMAL_TOL_DEG: float = 25.0
# Maximum number of dominant planes extracted in one pass.
MANHATTAN_MAX_PLANES: int = 12
# RANSAC inlier distance: vertices within this distance of the fitted plane
# are treated as members of that structural surface. Tight (3 cm) so objects
# protruding from walls (AC units, door frames, racks) are NOT included.
MANHATTAN_PLANE_INLIER_DIST_M: float = 0.03
# A plane must contain at least this fraction of total vertices to be
# considered structural. Filters out small object faces (rack panels, cabinet
# sides).
MANHATTAN_MIN_PLANE_INLIER_FRAC: float = 0.01

# --- Morphological Closing ---
CLOSING_ITERATIONS: int = 2

# --- Manhattan World Rectification (iterative RANSAC plane extraction) ---
# Plane normal must lie within this angle of a principal axis to qualify as a
# structural wall / floor / ceiling.
MANHATTAN_NORMAL_TOL_DEG: float = 25.0
# Maximum number of dominant planes extracted in one pass.
MANHATTAN_MAX_PLANES: int = 12
# RANSAC inlier distance: vertices within this distance of the fitted plane are
# treated as members of that structural surface.  Tight (3 cm) so that objects
# protruding from walls (AC units, door frames, racks) are NOT included.
MANHATTAN_PLANE_INLIER_DIST_M: float = 0.03
# A plane must contain at least this fraction of total vertices to be considered
# structural.  Filters out small object faces (rack panels, cabinet sides).
MANHATTAN_MIN_PLANE_INLIER_FRAC: float = 0.01

# --- Gaussian Heat Injection ---
HEAT_RADIUS_VOXELS: int = 5
HEAT_SIGMA: float = 2.0

# --- Semantic Stamp Radius (AC vents, workspaces) ---
STAMP_RADIUS_VOXELS: int = 3

# --- Workspace (desk) dimensions in metres ---
WORKSPACE_DIMENSIONS: tuple[float, float, float] = (1.2, 0.6, 0.75)  # width, depth, height

# --- Canonical AC unit dimensions in metres ---
# Width × depth × height. Used as the prior when stamping detected AC units
# so the voxelized AC is a solid block, not a hollow point-cloud shell.
AC_UNIT_DIMENSIONS: tuple[float, float, float] = (1.5, 0.5, 2.0)

# --- ASHRAE Standard Rack Dimensions (metres) ---
# Based on EIA-310 / ASHRAE TC 9.9 standard 19" server racks.
# Each rack type is (width, depth, height).
RACK_DIMENSIONS: dict[str, tuple[float, float, float]] = {
    "42U": (0.60, 1.00, 2.00),   # 42U × 600mm wide × 1000mm deep
    "42U_deep": (0.60, 1.20, 2.00),  # 42U × 600mm wide × 1200mm deep
    "42U_wide": (0.75, 1.00, 2.00),  # 42U × 750mm wide × 1000mm deep
    "48U": (0.60, 1.00, 2.26),   # 48U × 600mm wide × 1000mm deep
    # Measured server-room prior (target room): 600mm W × 900mm D × 1950mm H. Used by the CV/twin
    # voxelizer to anchor metric scale on the rack HEIGHT and to stamp racks at the true size; kept
    # separate from "42U" so the trained RL footprint is not disturbed.
    "42U_real": (0.60, 0.90, 1.95),
}
DEFAULT_RACK_TYPE: str = "42U"
# Standalone NETWORK rack prior (taller than a 42U): width 600mm × depth 750mm × height 2200mm.
NETWORK_RACK_DIMENSIONS: tuple[float, float, float] = (0.60, 0.75, 2.20)
NETWORK_RACK_HEIGHT: float = NETWORK_RACK_DIMENSIONS[2]

# ASHRAE TC 9.9 recommended inlet temperature range (°C) for A1-class.
ASHRAE_INLET_TEMP_RANGE: tuple[float, float] = (18.0, 27.0)
ASHRAE_INLET_ALLOWABLE_RANGE: tuple[float, float] = (15.0, 35.0)
ASHRAE_RECOMMENDED_INLET: float = 22.0  # typical design target

# Rack thermal parameters
DEFAULT_RACK_POWER_KW: float = 5.0  # kW per rack (mid-range)
DEFAULT_RACK_AIRFLOW_CFM: float = 800.0  # CFM per rack (mid-range 42U)
INTAKE_DEPTH_VOXELS: int = 1  # 1-voxel slab at front face
EXHAUST_DEPTH_VOXELS: int = 1  # 1-voxel slab at rear face

# Cooling unit defaults
DEFAULT_AC_CAPACITY_KW: float = 10.0  # kW per cooling unit
DEFAULT_AC_SUPPLY_TEMP_C: float = 14.0  # °C supply air
DEFAULT_AC_AIRFLOW_CFM: float = 2000.0  # CFM per cooling unit

# Air thermodynamic constants
AIR_DENSITY_KG_M3: float = 1.2  # ρ at ~22 °C, sea level
AIR_CP_J_KG_K: float = 1006.0  # Cₚ of dry air
CFM_TO_M3_S: float = 0.000472  # 1 CFM ≈ 0.000472 m³/s
