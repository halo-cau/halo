"""Engine-level global constants for the HALO CV/RL pipeline."""

# --- Spatial Resolution ---
VOXEL_SIZE: float = 0.1  # 1 voxel = 0.1 m (10 cm)

# --- Grid Limits (meters) ---
MAX_ROOM_DIMENSIONS: tuple[float, float, float] = (20.0, 20.0, 6.0)

# Fixed observation shape for the RL environment.
# Derived from MAX_ROOM_DIMENSIONS / VOXEL_SIZE → (200, 200, 60).
GRID_SHAPE: tuple[int, int, int] = (200, 200, 60)

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

# --- Morphological Closing ---
CLOSING_ITERATIONS: int = 2

# --- Gaussian Heat Injection ---
HEAT_RADIUS_VOXELS: int = 5
HEAT_SIGMA: float = 2.0

# --- Semantic Stamp Radius (AC vents, workspaces) ---
STAMP_RADIUS_VOXELS: int = 3

# --- ASHRAE Standard Rack Dimensions (metres) ---
# Based on EIA-310 / ASHRAE TC 9.9 standard 19" server racks.
# Each rack type is (width, depth, height).
RACK_DIMENSIONS: dict[str, tuple[float, float, float]] = {
    "42U": (0.60, 1.00, 2.00),   # 42U × 600mm wide × 1000mm deep
    "42U_deep": (0.60, 1.20, 2.00),  # 42U × 600mm wide × 1200mm deep
    "42U_wide": (0.75, 1.00, 2.00),  # 42U × 750mm wide × 1000mm deep
    "48U": (0.60, 1.00, 2.26),   # 48U × 600mm wide × 1000mm deep
}
DEFAULT_RACK_TYPE: str = "42U"

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
