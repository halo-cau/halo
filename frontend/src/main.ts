import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { AirflowSystem } from "./components/airflow";
import { createEquipmentMesh } from "./components/equipment";
import { buildHeatmap } from "./components/heatmap";
import { buildRoom } from "./components/room";
import { buildZones } from "./components/zones";
import {
  allScenes,
  coolingEnergyBase,
  type Equipment,
  getLoadFactor,
  peakTempBase,
  type SceneGraph,
} from "./data/sceneGraphs";

// ===== State =====
let currentSceneIndex = 0;
let scene: THREE.Scene;
let camera: THREE.PerspectiveCamera;
let renderer: THREE.WebGLRenderer;
let controls: OrbitControls;
let furnitureGroup: THREE.Group;
let roomGroup: THREE.Group;
let heatmapGroup: THREE.Group;
let animating = false;
let heatmapVisible = false;
let airflowVisible = false;
let airflowGroup: THREE.Group;
let airflowSystem: AirflowSystem;
let zonesGroup: THREE.Group;
let zonesVisible = false;

// Simulation state
let simPlaying = false;
let simMinutes = 0; // 0~1440 (0:00 ~ 24:00)
let simSpeed = 1; // 1x, 2x, 4x, 8x
let lastSimTick = 0;

// ===== Init =====
function init() {
  const canvas = document.getElementById("canvas") as HTMLCanvasElement;
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setSize(canvas.clientWidth, canvas.clientHeight);
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf4f2ec);

  // Camera — 서버실 크기(12x9)에 맞춰 조정
  camera = new THREE.PerspectiveCamera(50, canvas.clientWidth / canvas.clientHeight, 0.1, 200);
  camera.position.set(16, 12, 16);
  camera.lookAt(6, 0, 4.5);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(6, 0, 4.5);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.maxPolarAngle = Math.PI / 2.1;

  // Lights — 서버실 형광등 느낌
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.35);
  scene.add(ambientLight);

  const dirLight = new THREE.DirectionalLight(0xe8f0ff, 1.0);
  dirLight.position.set(10, 16, 8);
  dirLight.castShadow = true;
  dirLight.shadow.mapSize.set(2048, 2048);
  dirLight.shadow.camera.left = -16;
  dirLight.shadow.camera.right = 16;
  dirLight.shadow.camera.top = 16;
  dirLight.shadow.camera.bottom = -16;
  scene.add(dirLight);

  // 형광등 (차가운 백색)
  const fluorescent1 = new THREE.PointLight(0xe0f0ff, 0.5, 18);
  fluorescent1.position.set(4, 3.2, 3);
  scene.add(fluorescent1);

  const fluorescent2 = new THREE.PointLight(0xe0f0ff, 0.5, 18);
  fluorescent2.position.set(8, 3.2, 6);
  scene.add(fluorescent2);

  roomGroup = new THREE.Group();
  furnitureGroup = new THREE.Group();
  heatmapGroup = new THREE.Group();
  airflowGroup = new THREE.Group();
  airflowSystem = new AirflowSystem(airflowGroup);
  zonesGroup = new THREE.Group();
  const grid = new THREE.GridHelper(18, 36, 0xe8e5dd, 0xf4f2ec);
  grid.position.set(6, -0.01, 4.5);
  scene.add(grid);

  scene.add(roomGroup);
  scene.add(furnitureGroup);
  scene.add(heatmapGroup);
  scene.add(airflowGroup);
  scene.add(zonesGroup);

  // Grid

  buildScene(allScenes[currentSceneIndex]);
  updateUI(allScenes[currentSceneIndex]);
  setupButtons();
  window.addEventListener("resize", onResize);
  animate();
}

function animate(now: number = 0) {
  requestAnimationFrame(animate);
  controls.update();

  // Simulation tick
  if (simPlaying && now - lastSimTick > 50) {
    lastSimTick = now;
    simMinutes += simSpeed;
    if (simMinutes >= 1440) simMinutes = 0;
    updateSimSlider();
    updateSimulation();
  }

  // Airflow streamlines update every frame (independent of simulation tick)
  if (airflowVisible && airflowSystem.isReady()) {
    const hour = simMinutes / 60;
    const loadFactor = getLoadFactor(hour);
    airflowSystem.update(allScenes[currentSceneIndex], loadFactor);
  }

  // Scan airflow particles (Step 4 thermal mode)
  if (scanAirflowActive && scanAirflowVisible) {
    updateScanAirflow();
  }

  renderer.render(scene, camera);
}

function onResize() {
  const canvas = renderer.domElement;
  const parent = canvas.parentElement!;
  const w = parent.clientWidth;
  const h = parent.clientHeight;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}

// ===== Simulation =====
function updateSimSlider() {
  const slider = document.getElementById("sim-slider") as HTMLInputElement;
  slider.value = String(simMinutes);
  const h = Math.floor(simMinutes / 60);
  const m = Math.floor(simMinutes % 60);
  document.getElementById("sim-clock")!.textContent =
    `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}`;
}

function updateSimulation() {
  const hour = simMinutes / 60;
  const loadFactor = getLoadFactor(hour);
  const sceneData = allScenes[currentSceneIndex];

  const maxHeat = buildHeatmap(heatmapGroup, sceneData, loadFactor);
  heatmapGroup.visible = heatmapVisible;

  if (airflowSystem.isReady()) {
    airflowSystem.update(sceneData, loadFactor);
  }

  updateEquipmentGlow(loadFactor);
  updateMonitorPanel(sceneData, loadFactor, maxHeat);
}

function updateEquipmentGlow(loadFactor: number) {
  furnitureGroup.children.forEach((child) => {
    if (!(child instanceof THREE.Group)) return;
    const { category, heatOutput } = child.userData;
    if (!category) return;

    child.traverse((obj) => {
      if (!(obj instanceof THREE.Mesh)) return;
      const mat = obj.material as THREE.MeshBasicMaterial | THREE.MeshStandardMaterial;

      // 서버 랙 상단 glow 업데이트
      if (
        category === "server_rack" &&
        mat.transparent &&
        "opacity" in mat &&
        mat.side === THREE.DoubleSide
      ) {
        const effectiveHeat = (heatOutput || 0) * loadFactor;
        mat.opacity = 0.05 + effectiveHeat * 0.02;
        if (effectiveHeat > 10) mat.color.setHex(0xff3300);
        else if (effectiveHeat > 5) mat.color.setHex(0xff8800);
        else mat.color.setHex(0xffcc00);
      }

      // 냉각 장치 glow 업데이트
      if (
        category === "cooling_unit" &&
        mat.transparent &&
        "opacity" in mat &&
        mat.side === THREE.DoubleSide
      ) {
        mat.opacity = 0.03 + 0.12 * loadFactor;
      }
    });
  });
}

function updateMonitorPanel(sceneData: SceneGraph, loadFactor: number, _maxHeat: number) {
  const loadEl = document.getElementById("mon-load");
  if (!loadEl) return; // right panel removed — nothing to update

  // Load
  const loadPct = Math.round(loadFactor * 100);
  loadEl.innerHTML = `${loadPct}<span class="mon-unit">%</span>`;
  const loadBar = document.getElementById("mon-load-bar")!;
  loadBar.style.width = `${loadPct}%`;
  if (loadPct >= 80) loadBar.style.backgroundColor = "#E24B4A";
  else if (loadPct >= 50) loadBar.style.backgroundColor = "#E89E4F";
  else loadBar.style.backgroundColor = "#1D9E75";

  // Max Temperature
  const basePeakTemp = peakTempBase[sceneData.id] || 35;
  const ambientTemp = 22;
  const currentMaxTemp = Math.round(ambientTemp + (basePeakTemp - ambientTemp) * loadFactor);
  const tempEl = document.getElementById("mon-temp")!;
  tempEl.innerHTML = `${currentMaxTemp}<span class="mon-unit">&deg;C</span>`;
  tempEl.className =
    "mon-value " +
    (currentMaxTemp >= 40 ? "temp-danger" : currentMaxTemp >= 32 ? "temp-warn" : "temp-ok");

  // Cooling Energy
  const baseCooling = coolingEnergyBase[sceneData.id] || 30;
  const currentCooling = Math.round(baseCooling * loadFactor * 10) / 10;
  document.getElementById("mon-cooling")!.innerHTML =
    `${currentCooling.toFixed(1)}<span class="mon-unit">kW</span>`;

  // PUE (Power Usage Effectiveness)
  // PUE = (IT Power + Cooling Power) / IT Power
  const totalServerPower =
    sceneData.furniture.filter((f) => f.heatOutput > 0).reduce((sum, f) => sum + f.heatOutput, 0) *
    loadFactor;
  const pue = totalServerPower > 0 ? (totalServerPower + currentCooling) / totalServerPower : 1.0;
  const pueEl = document.getElementById("mon-pue")!;
  pueEl.textContent = pue.toFixed(2);
  pueEl.style.color = pue <= 1.4 ? "#0F6E56" : pue <= 1.8 ? "#A86B1A" : "#A32D2D";
}

// ===== Scene Build with Animation =====
function buildScene(sceneData: SceneGraph, withAnimation = true) {
  furnitureGroup.clear();
  buildRoom(roomGroup, sceneData);
  buildHeatmap(heatmapGroup, sceneData);
  heatmapGroup.visible = heatmapVisible;
  airflowSystem.init(sceneData);
  airflowGroup.visible = airflowVisible;
  buildZones(zonesGroup, sceneData);
  zonesGroup.visible = zonesVisible;

  const equipCtx = { roomHeight: sceneData.room.dimensions[1] };

  if (withAnimation) {
    animating = true;
    let i = 0;
    const interval = setInterval(() => {
      if (i >= sceneData.furniture.length) {
        animating = false;
        clearInterval(interval);
        return;
      }

      const item = sceneData.furniture[i];
      const mesh = createEquipmentMesh(item, equipCtx);
      const targetY = mesh.position.y;
      mesh.position.y = 5;
      mesh.scale.set(0.01, 0.01, 0.01);
      furnitureGroup.add(mesh);

      const startTime = performance.now();
      const duration = 400;

      function animateDrop(now: number) {
        const elapsed = now - startTime;
        const t = Math.min(elapsed / duration, 1);
        const ease = 1 - (1 - t) ** 3;
        mesh.position.y = 5 + (targetY - 5) * ease;
        mesh.scale.setScalar(0.01 + 0.99 * ease);
        if (t < 1) requestAnimationFrame(animateDrop);
      }
      requestAnimationFrame(animateDrop);
      i++;
    }, 180);
  } else {
    for (const item of sceneData.furniture) {
      furnitureGroup.add(createEquipmentMesh(item, equipCtx));
    }
  }
}

// ===== UI =====
function updateUI(sceneData: SceneGraph) {
  const titleEl = document.getElementById("scene-title");
  if (titleEl) titleEl.textContent = sceneData.name;
  const descEl = document.getElementById("scene-desc");
  if (descEl) descEl.textContent = sceneData.description;

  document.querySelectorAll(".step").forEach((el, idx) => {
    el.classList.toggle("active", idx === currentSceneIndex);
  });
}

// ===== Chart =====
// ===== Buttons =====
function updateLegendVisibility() {
  document.getElementById("airflow-legend")!.style.display = zonesVisible ? "flex" : "none";
}

function setupButtons() {
  document.getElementById("btn-prev")?.addEventListener("click", () => {
    if (animating || currentSceneIndex === 0) return;
    currentSceneIndex--;
    switchToScene(currentSceneIndex);
  });

  document.getElementById("btn-next")?.addEventListener("click", () => {
    // Scenario steps + 1 Real-Scan step. Scan index = allScenes.length, so the
    // last reachable step index is allScenes.length (e.g. 2 for [random, rl]).
    if (animating || currentSceneIndex >= allScenes.length) return;
    currentSceneIndex++;
    switchToScene(currentSceneIndex);
  });

  document.querySelectorAll(".step").forEach((el, idx) => {
    el.addEventListener("click", () => {
      if (animating || idx === currentSceneIndex) return;
      currentSceneIndex = idx;
      switchToScene(currentSceneIndex);
    });
  });

  document.getElementById("btn-heatmap")?.addEventListener("click", () => {
    heatmapVisible = !heatmapVisible;
    heatmapGroup.visible = heatmapVisible;
    const btn = document.getElementById("btn-heatmap")!;
    btn.classList.toggle("active-toggle", heatmapVisible);
  });

  document.getElementById("btn-airflow")!.addEventListener("click", () => {
    airflowVisible = !airflowVisible;
    airflowGroup.visible = airflowVisible;
    const btn = document.getElementById("btn-airflow")!;
    btn.classList.toggle("active-toggle", airflowVisible);
    updateLegendVisibility();
  });

  document.getElementById("btn-zones")!.addEventListener("click", () => {
    zonesVisible = !zonesVisible;
    zonesGroup.visible = zonesVisible;
    const btn = document.getElementById("btn-zones")!;
    btn.classList.toggle("active-toggle", zonesVisible);
    updateLegendVisibility();
  });

  // Simulation controls
  document.getElementById("sim-play")?.addEventListener("click", () => {
    simPlaying = !simPlaying;
    const btn = document.getElementById("sim-play")!;
    btn.innerHTML = simPlaying ? "&#9646;&#9646;" : "&#9654;";
  });

  const simSlider = document.getElementById("sim-slider") as HTMLInputElement;
  simSlider.addEventListener("input", () => {
    simMinutes = Number(simSlider.value);
    updateSimSlider();
    updateSimulation();
  });

  document.getElementById("sim-speed")?.addEventListener("click", () => {
    const speeds = [1, 2, 4, 8];
    const idx = speeds.indexOf(simSpeed);
    simSpeed = speeds[(idx + 1) % speeds.length];
    document.getElementById("sim-speed")!.textContent = `${simSpeed}x`;
  });

  // 초기 모니터 업데이트
  updateSimulation();

  // === Step 4: Real Scan buttons ===
  setupScanButtons();
}

// ===== Scene Switching (supports Step 1-3 + Step 4 scan) =====
function switchToScene(idx: number) {
  const isScan = idx >= allScenes.length; // last step = Real Scan

  // Update step indicators
  document.querySelectorAll(".step").forEach((el, i) => {
    el.classList.toggle("active", i === idx);
  });

  // Toggle scan-specific UI
  const scanPanel = document.getElementById("scan-upload-panel")!;
  const voxelCtrl = document.getElementById("voxel-controls")!;
  const ashraeSection = document.getElementById("ashrae-section");
  const simBar = document.querySelector(".sim-bar") as HTMLElement;
  const viewportBtns = document.querySelector(".viewport-controls") as HTMLElement;

  if (isScan) {
    // Show scan UI, hide simulation controls
    if (!scanVoxelGroup.parent) {
      scene.add(scanVoxelGroup);
    }

    // Show upload panel only if no scan data loaded yet
    if (scanVoxelGroup.children.length === 0) {
      scanPanel.style.display = "block";
    }
    voxelCtrl.style.display = scanVoxelGroup.children.length > 0 ? "block" : "none";
    if (ashraeSection) ashraeSection.style.display = scanMetricsCache ? "block" : "none";
    simBar.style.display = "none";
    viewportBtns.style.display = "none";

    // Hide hardcoded scene stuff
    roomGroup.visible = false;
    furnitureGroup.visible = false;
    heatmapGroup.visible = false;
    airflowGroup.visible = false;
    zonesGroup.visible = false;

    // Update panel info
    const titleEl = document.getElementById("scene-title");
    if (titleEl) titleEl.textContent = "Real Room Scan";
    const descEl = document.getElementById("scene-desc");
    if (descEl) {
      descEl.textContent =
        "Upload a .obj scan or load the demo room to analyze thermal performance.";
    }
  } else {
    // Standard scene mode
    scanPanel.style.display = "none";
    voxelCtrl.style.display = "none";
    if (ashraeSection) ashraeSection.style.display = "none";
    simBar.style.display = "flex";
    viewportBtns.style.display = "flex";

    roomGroup.visible = true;
    furnitureGroup.visible = true;

    // Remove scan visuals
    scanVoxelGroup.clear();
    if (scanVoxelGroup.parent) scene.remove(scanVoxelGroup);

    buildScene(allScenes[idx]);
    updateUI(allScenes[idx]);
  }
}

// ===================================================================
// ===== STEP 4: REAL SCAN — Backend Integration + Voxel Renderer ====
// ===================================================================

const scanVoxelGroup = new THREE.Group();
let scanMetricsCache: ScanMetrics | null = null;
let currentScanMode: "semantic" | "thermal" = "semantic";
let scanAirflowVisible = false;

// ── Scan airflow particle system ──
const SCAN_PARTICLE_COUNT = 400;
const SCAN_TRAIL_LEN = 30;
const SCAN_SEG_COUNT = SCAN_PARTICLE_COUNT * (SCAN_TRAIL_LEN - 1);
const SCAN_VERT_COUNT = SCAN_SEG_COUNT * 2;
let scanHeads: Float32Array | null = null;
let scanVels: Float32Array | null = null;
let scanAges: Float32Array;
let scanMaxAges: Float32Array;
let scanTrailHist: Float32Array;
let scanTrailIdx: Int32Array;
let scanTrailPos: Float32Array;
let scanTrailCol: Float32Array;
let scanTrailGeo: THREE.BufferGeometry;
let scanTrailLines: THREE.LineSegments;
let scanAirflowActive = false;

// Cached scan equipment for airflow
let scanHeatSources: {
  x: number;
  y: number;
  z: number;
  heat: number;
  h: number;
  frontX: number;
  frontZ: number;
  backX: number;
  backZ: number;
  airflow_m3s: number;
}[] = [];
let scanCoolSources: { x: number; y: number; z: number; w: number; d: number }[] = [];
let scanObstacles: { x: number; y: number; z: number; hw: number; hh: number; hd: number }[] = [];
let scanRoomDims: [number, number, number] = [0, 0, 0]; // rw, rh, rd in Three.js coords
let scanRoomOrigin: [number, number, number] = [0, 0, 0];

// 3-D temperature lookup (flat Float32 from the solver grid)
let scanTempGrid: Float32Array | null = null;
let scanTempShape: [number, number, number] = [0, 0, 0];
let scanTempMin = 18;
let scanTempMax = 27;

// --- Types matching backend VisualizeResponse ---
interface VoxelData {
  shape: number[];
  voxel_size: number;
  origin: number[];
  positions: number[][];
  labels: number[];
}

interface ThermalData {
  positions: number[][];
  temperatures: number[];
  min_temp: number;
  max_temp: number;
}

interface RackMetrics {
  rack_index: number;
  intake_temp: number;
  exhaust_temp: number;
  delta_t: number;
  inlet_compliant: boolean;
  inlet_within_allowable: boolean;
}

interface RoomMetrics {
  rci_hi: number;
  rci_lo: number;
  shi: number;
  rhi: number;
  mean_intake: number;
  mean_exhaust: number;
  mean_return: number;
  vertical_profile: number[];
}

interface ScanMetrics {
  racks: RackMetrics[];
  room: RoomMetrics;
}

interface ScanEquipmentItem {
  id: string;
  category: string;
  label: string;
  position: number[]; // [x, y, z] centre-bottom, Z-up
  size: number[]; // [w, d, h] metres
  color: string;
  heat_output: number;
  facing: string | null;
}

interface VisualizeResponse {
  raw_glb: string;
  cleaned_glb: string;
  semantic_glb: string | null;
  voxel_grid: VoxelData | null;
  thermal: ThermalData | null;
  metrics: ScanMetrics | null;
  equipment: ScanEquipmentItem[] | null;
}

function tempToColorScan(t: number, _minT: number, _maxT: number): THREE.Color {
  // CFD-standard Jet colormap mapped to ASHRAE recommended range.
  // 18 °C (rec low) → blue side, 27 °C (rec high) → red side.
  const lo = 18,
    hi = 27;
  const n = Math.max(0, Math.min(1, (t - lo) / (hi - lo)));

  const stops = [
    [0.0, 0.0, 0.5],
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 1.0],
    [1.0, 1.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.5, 0.0, 0.0],
  ];
  const positions = [0.0, 0.125, 0.375, 0.625, 0.875, 1.0];
  let li = 0;
  for (let i = 1; i < positions.length; i++) {
    if (n <= positions[i]) {
      li = i - 1;
      break;
    }
    li = i;
  }
  const hi2 = Math.min(li + 1, positions.length - 1);
  const span = positions[hi2] - positions[li] || 1;
  const k = (n - positions[li]) / span;
  return new THREE.Color(
    stops[li][0] + (stops[hi2][0] - stops[li][0]) * k,
    stops[li][1] + (stops[hi2][1] - stops[li][1]) * k,
    stops[li][2] + (stops[hi2][2] - stops[li][2]) * k,
  );
}

function buildVoxelScene(resp: VisualizeResponse) {
  scanVoxelGroup.clear();
  scanMetricsCache = resp.metrics;

  const voxel = resp.voxel_grid;
  if (!voxel || voxel.positions.length === 0) return;

  const vs = voxel.voxel_size;
  const ox = voxel.origin[0];
  const oy = voxel.origin[1];
  const oz = voxel.origin[2];

  // Room dimensions from grid shape (Z-up → Y-up swap)
  const rw = voxel.shape[0] * vs; // X width
  const rh = voxel.shape[2] * vs; // Z height → Y in Three.js
  const rd = voxel.shape[1] * vs; // Y depth → Z in Three.js

  // ─── Build room shell (floor, walls, pillars) ───
  const roomSub = new THREE.Group();
  roomSub.name = "scanRoom";

  // Floor
  const floorGeo = new THREE.PlaneGeometry(rw, rd);
  const floorMat = new THREE.MeshStandardMaterial({ color: 0xe8e5dd, roughness: 0.85 });
  const floor = new THREE.Mesh(floorGeo, floorMat);
  floor.rotation.x = -Math.PI / 2;
  floor.position.set(ox + rw / 2, oz + 0.001, oy + rd / 2);
  floor.receiveShadow = true;
  roomSub.add(floor);

  // Tile grid on floor
  const tileMat = new THREE.LineBasicMaterial({ color: 0xd3d1c7 });
  const tileSize = 0.6;
  for (let x = 0; x <= rw; x += tileSize) {
    const g = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(ox + x, oz + 0.002, oy),
      new THREE.Vector3(ox + x, oz + 0.002, oy + rd),
    ]);
    roomSub.add(new THREE.Line(g, tileMat));
  }
  for (let z = 0; z <= rd; z += tileSize) {
    const g = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(ox, oz + 0.002, oy + z),
      new THREE.Vector3(ox + rw, oz + 0.002, oy + z),
    ]);
    roomSub.add(new THREE.Line(g, tileMat));
  }

  // Transparent walls
  const wallMat = new THREE.MeshStandardMaterial({
    color: 0xe8e5dd,
    transparent: true,
    opacity: 0.2,
    side: THREE.DoubleSide,
  });
  const northWall = new THREE.Mesh(new THREE.PlaneGeometry(rw, rh), wallMat.clone());
  northWall.position.set(ox + rw / 2, oz + rh / 2, oy);
  roomSub.add(northWall);

  const southWall = new THREE.Mesh(new THREE.PlaneGeometry(rw, rh), wallMat.clone());
  southWall.position.set(ox + rw / 2, oz + rh / 2, oy + rd);
  southWall.rotation.y = Math.PI;
  roomSub.add(southWall);

  const westWall = new THREE.Mesh(new THREE.PlaneGeometry(rd, rh), wallMat.clone());
  westWall.rotation.y = Math.PI / 2;
  westWall.position.set(ox, oz + rh / 2, oy + rd / 2);
  roomSub.add(westWall);

  const eastWall = new THREE.Mesh(new THREE.PlaneGeometry(rd, rh), wallMat.clone());
  eastWall.rotation.y = -Math.PI / 2;
  eastWall.position.set(ox + rw, oz + rh / 2, oy + rd / 2);
  roomSub.add(eastWall);

  // Floor outline
  const edgeMat = new THREE.LineBasicMaterial({ color: 0x888780, linewidth: 1 });
  const ol = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(ox, oz + 0.005, oy),
    new THREE.Vector3(ox + rw, oz + 0.005, oy),
    new THREE.Vector3(ox + rw, oz + 0.005, oy + rd),
    new THREE.Vector3(ox, oz + 0.005, oy + rd),
    new THREE.Vector3(ox, oz + 0.005, oy),
  ]);
  roomSub.add(new THREE.Line(ol, edgeMat));

  // Corner pillars
  const pillarGeo = new THREE.BoxGeometry(0.15, rh, 0.15);
  const pillarMat = new THREE.MeshStandardMaterial({ color: 0xb4b2a9 });
  for (const [px, pz] of [
    [ox, oy],
    [ox + rw, oy],
    [ox + rw, oy + rd],
    [ox, oy + rd],
  ]) {
    const p = new THREE.Mesh(pillarGeo, pillarMat);
    p.position.set(px, oz + rh / 2, pz);
    p.castShadow = true;
    roomSub.add(p);
  }

  scanVoxelGroup.add(roomSub);

  // ─── Build styled equipment meshes ───
  if (resp.equipment && resp.equipment.length > 0) {
    const equipGroup = new THREE.Group();
    equipGroup.name = "scanEquipGroup";
    const scanEquipCtx = { roomHeight: rh };

    for (const item of resp.equipment) {
      // Convert backend (Z-up) to Three.js (Y-up) Equipment
      const [bx, by, bz] = item.position;
      const [bw, bd, bh] = item.size;

      // Facing → rotation.y in Three.js
      let rotY = 0;
      if (item.facing === "+y")
        rotY = 0; // +Y (backend) → +Z (Three.js front)
      else if (item.facing === "-y") rotY = 180;
      else if (item.facing === "+x") rotY = 90;
      else if (item.facing === "-x") rotY = -90;

      const equipItem: Equipment = {
        id: item.id,
        category: item.category,
        label: item.label,
        position: [bx, bz, by], // Z-up→Y-up: X=X, Y=Z, Z=Y
        rotation: [0, rotY, 0],
        size: [bw, bh, bd], // [width, height, depth]
        color: item.color,
        heatOutput: item.heat_output,
        relations: [],
      };

      const mesh = createEquipmentMesh(equipItem, scanEquipCtx);
      equipGroup.add(mesh);
    }

    scanVoxelGroup.add(equipGroup);
  }

  // ─── Airflow direction arrows (visible in thermal mode) ───
  if (resp.equipment && resp.equipment.length > 0) {
    const arrowGroup = new THREE.Group();
    arrowGroup.name = "scanAirflowArrows";

    for (const item of resp.equipment) {
      // Convert backend Z-up → Three.js Y-up
      const [bx, by, bz] = item.position;
      const [bw, bd, bh] = item.size;
      const px = bx;
      const py = bz; // bottom Y
      const pz = by;
      const depth = bd; // rack depth along facing axis
      const height = bh;

      if (item.category === "server_rack" && item.facing) {
        // Front direction in Three.js (facing maps: +y→+Z, -y→-Z, +x→+X, -x→-X)
        let frontX = 0,
          frontZ = 0;
        if (item.facing === "+y") frontZ = 1;
        else if (item.facing === "-y") frontZ = -1;
        else if (item.facing === "+x") frontX = 1;
        else if (item.facing === "-x") frontX = -1;

        const backX = -frontX,
          backZ = -frontZ;

        // Intake arrows (cold blue) – at front face, pointing into rack
        const intakeDir = new THREE.Vector3(-frontX, 0, -frontZ);
        for (const yFrac of [0.25, 0.5, 0.75]) {
          const origin = new THREE.Vector3(
            px + frontX * (depth / 2 + 0.5),
            py + height * yFrac,
            pz + frontZ * (depth / 2 + 0.5),
          );
          arrowGroup.add(new THREE.ArrowHelper(intakeDir, origin, 0.7, 0x378add, 0.15, 0.1));
        }

        // Exhaust arrows (hot red) – at back face, pointing horizontally out
        const exhaustDir = new THREE.Vector3(backX, 0, backZ);
        for (const yFrac of [0.25, 0.5, 0.75]) {
          const origin = new THREE.Vector3(
            px + backX * (depth / 2 + 0.1),
            py + height * yFrac,
            pz + backZ * (depth / 2 + 0.1),
          );
          arrowGroup.add(new THREE.ArrowHelper(exhaustDir, origin, 0.7, 0xe24b4a, 0.15, 0.1));
        }
      } else if (item.category === "cooling_unit") {
        // AC arrows (calm blue) – cold air blows downward and outward at floor level
        arrowGroup.add(
          new THREE.ArrowHelper(
            new THREE.Vector3(0, -1, 0),
            new THREE.Vector3(px, py + 0.3, pz),
            0.8,
            0x378add,
            0.2,
            0.12,
          ),
        );
        const halfW = bw / 2 + 0.15;
        const halfD = bd / 2 + 0.15;
        for (const [dx, dz] of [
          [halfW, 0],
          [-halfW, 0],
          [0, halfD],
          [0, -halfD],
        ] as [number, number][]) {
          const dir = new THREE.Vector3(Math.sign(dx), -0.3, Math.sign(dz)).normalize();
          arrowGroup.add(
            new THREE.ArrowHelper(
              dir,
              new THREE.Vector3(px + dx, py + 0.2, pz + dz),
              0.6,
              0x378add,
              0.15,
              0.1,
            ),
          );
        }
      }
    }

    arrowGroup.visible = false; // shown only in thermal mode
    scanVoxelGroup.add(arrowGroup);
  }

  // ─── Thermal instanced mesh (all non-empty voxels, initially hidden) ───
  if (resp.thermal && resp.thermal.positions.length > 0) {
    const boxGeo = new THREE.BoxGeometry(vs * 1.6, vs * 1.6, vs * 0.95);
    const dummy = new THREE.Object3D();
    const tCount = resp.thermal.positions.length;
    const tMat = new THREE.MeshLambertMaterial({
      transparent: true,
      opacity: 0.35,
      depthWrite: false,
    });
    const tMesh = new THREE.InstancedMesh(boxGeo, tMat, tCount);
    tMesh.name = "thermalVoxels";
    tMesh.visible = false;

    for (let i = 0; i < tCount; i++) {
      const [ix, iy, iz] = resp.thermal.positions[i];
      dummy.position.set(ox + ix * vs + vs / 2, oz + iz * vs + vs / 2, oy + iy * vs + vs / 2);
      dummy.updateMatrix();
      tMesh.setMatrixAt(i, dummy.matrix);
      const c = tempToColorScan(
        resp.thermal.temperatures[i],
        resp.thermal.min_temp,
        resp.thermal.max_temp,
      );
      tMesh.setColorAt(i, c);
    }
    tMesh.instanceMatrix.needsUpdate = true;
    if (tMesh.instanceColor) tMesh.instanceColor.needsUpdate = true;
    scanVoxelGroup.add(tMesh);
  }

  // ─── Airflow particle system (physics-driven streamlines) ───
  initScanAirflow(resp);

  // Store voxel data for Z-slice filtering
  scanVoxelGroup.userData = { voxelData: voxel, thermalData: resp.thermal };

  // Camera framing
  const cx = ox + rw / 2;
  const cy = oz + rh / 2;
  const cz = oy + rd / 2;
  const maxDim = Math.max(rw, rd, rh);
  camera.position.set(cx + maxDim, cy + maxDim * 0.8, cz + maxDim);
  controls.target.set(cx, cy, cz);
  controls.update();

  // Update Z slider range
  const zSlider = document.getElementById("voxel-z-slider") as HTMLInputElement;
  zSlider.max = String(voxel.shape[2]);
  zSlider.value = zSlider.max;
  document.getElementById("voxel-z-value")!.textContent = "All";

  // Show voxel controls
  document.getElementById("voxel-controls")!.style.display = "block";
  document.getElementById("scan-upload-panel")!.style.display = "none";

  // Show ASHRAE section (only if right panel exists)
  if (resp.metrics) {
    renderASHRAEPanel(resp.metrics, resp.thermal);
    const sec = document.getElementById("ashrae-section");
    if (sec) sec.style.display = "block";
  }
}

function renderASHRAEPanel(metrics: ScanMetrics, thermal: ThermalData | null) {
  const grid = document.getElementById("ashrae-grid");
  if (!grid) return; // right panel removed

  const okColor = "#0F6E56";
  const warnColor = "#A86B1A";
  const dangerColor = "#A32D2D";
  const infoColor = "#0C447C";

  const rciHiColor =
    metrics.room.rci_hi >= 95 ? okColor : metrics.room.rci_hi >= 80 ? warnColor : dangerColor;
  const rciLoColor =
    metrics.room.rci_lo >= 95 ? okColor : metrics.room.rci_lo >= 80 ? warnColor : dangerColor;

  grid.innerHTML = `
    <div class="monitor-card">
      <div class="mon-label">RCI-Hi</div>
      <div class="mon-value" style="color:${rciHiColor}">${metrics.room.rci_hi.toFixed(1)}<span class="mon-unit">%</span></div>
    </div>
    <div class="monitor-card">
      <div class="mon-label">RCI-Lo</div>
      <div class="mon-value" style="color:${rciLoColor}">${metrics.room.rci_lo.toFixed(1)}<span class="mon-unit">%</span></div>
    </div>
    <div class="monitor-card">
      <div class="mon-label">SHI</div>
      <div class="mon-value">${metrics.room.shi.toFixed(3)}</div>
    </div>
    <div class="monitor-card">
      <div class="mon-label">RHI</div>
      <div class="mon-value">${metrics.room.rhi.toFixed(3)}</div>
    </div>
    <div class="monitor-card">
      <div class="mon-label">Mean Intake</div>
      <div class="mon-value" style="color:${infoColor}">${metrics.room.mean_intake.toFixed(1)}<span class="mon-unit">&deg;C</span></div>
    </div>
    <div class="monitor-card">
      <div class="mon-label">Mean Exhaust</div>
      <div class="mon-value" style="color:${warnColor}">${metrics.room.mean_exhaust.toFixed(1)}<span class="mon-unit">&deg;C</span></div>
    </div>
    ${
      thermal
        ? `
    <div class="monitor-card">
      <div class="mon-label">Min Temp</div>
      <div class="mon-value" style="color:${infoColor}">${thermal.min_temp.toFixed(1)}<span class="mon-unit">&deg;C</span></div>
    </div>
    <div class="monitor-card">
      <div class="mon-label">Max Temp</div>
      <div class="mon-value" style="color:${dangerColor}">${thermal.max_temp.toFixed(1)}<span class="mon-unit">&deg;C</span></div>
    </div>`
        : ""
    }
  `;

  // Per-rack details — restrained card style (warm off-white, hairline border).
  const racksDiv = document.getElementById("ashrae-racks");
  if (!racksDiv) return;
  if (metrics.racks.length > 0) {
    let html =
      '<div style="font-size:11px;color:#888780;margin-bottom:6px;">Per-Rack Details</div>';
    for (const r of metrics.racks) {
      const inColor = r.inlet_compliant
        ? okColor
        : r.inlet_within_allowable
          ? warnColor
          : dangerColor;
      const inTag = r.inlet_compliant ? "OK" : r.inlet_within_allowable ? "WARN" : "FAIL";
      const dtOk = r.delta_t <= 15;
      const dtWarn = r.delta_t <= 20;
      const exColor = dtOk ? okColor : dtWarn ? warnColor : dangerColor;
      const dtTag = dtOk ? "OK" : dtWarn ? "WARN" : "FAIL";
      html += `
        <div style="background:#FDFCF9;border:0.5px solid #E8E5DD;border-radius:6px;padding:8px 10px;margin-bottom:6px;font-size:11px;">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
            <span style="color:#2C2C2A;font-weight:500;">Rack ${r.rack_index}</span>
            <span style="display:flex;gap:8px;font-size:10px;">
              <span title="Intake: ASHRAE 18-27°C" style="color:${inColor};font-weight:500;">In ${inTag}</span>
              <span style="color:#888780;">·</span>
              <span title="ΔT: ${r.delta_t.toFixed(1)}°C" style="color:${exColor};font-weight:500;">ΔT ${dtTag}</span>
            </span>
          </div>
          <div style="display:flex;justify-content:space-between;font-variant-numeric:tabular-nums;color:#5F5E5A;">
            <span>In: <span style="color:${inColor};">${r.intake_temp.toFixed(1)}°C</span></span>
            <span>Out: <span style="color:${exColor};">${r.exhaust_temp.toFixed(1)}°C</span></span>
            <span>ΔT <span style="color:${exColor};">${r.delta_t.toFixed(1)}°C</span></span>
          </div>
        </div>`;
    }
    racksDiv.innerHTML = html;
  } else {
    racksDiv.innerHTML = "";
  }
}

// ── Scan airflow: build 3-D temp lookup + init particles ──

function buildScanTempLookup(resp: VisualizeResponse) {
  if (!resp.thermal || !resp.voxel_grid) {
    scanTempGrid = null;
    return;
  }
  const [sx, sy, sz] = resp.voxel_grid.shape;
  scanTempShape = [sx, sy, sz];
  scanTempMin = resp.thermal.min_temp;
  scanTempMax = resp.thermal.max_temp;

  // Fill with ambient (22 °C), then overwrite with solver values
  const grid = new Float32Array(sx * sy * sz).fill(22);
  for (let k = 0; k < resp.thermal.positions.length; k++) {
    const [ix, iy, iz] = resp.thermal.positions[k];
    grid[ix * sy * sz + iy * sz + iz] = resp.thermal.temperatures[k];
  }
  scanTempGrid = grid;
}

function sampleScanTemp(wx: number, wy: number, wz: number): number {
  // Three.js coords → backend voxel indices (Z-up swap)
  if (!scanTempGrid) return 22;
  const vs = 0.1;
  const [ox, oy, oz] = scanRoomOrigin;
  const ix = Math.floor((wx - ox) / vs);
  const iy = Math.floor((wz - oy) / vs); // Three.js Z = backend Y
  const iz = Math.floor((wy - oz) / vs); // Three.js Y = backend Z
  const [sx, sy, sz] = scanTempShape;
  if (ix < 0 || ix >= sx || iy < 0 || iy >= sy || iz < 0 || iz >= sz) return 22;
  return scanTempGrid[ix * sy * sz + iy * sz + iz];
}

function initScanAirflow(resp: VisualizeResponse) {
  // Remove old
  const old = scanVoxelGroup.getObjectByName("scanStreamlines");
  if (old) scanVoxelGroup.remove(old);

  if (!resp.equipment || resp.equipment.length === 0 || !resp.voxel_grid) {
    scanAirflowActive = false;
    return;
  }

  buildScanTempLookup(resp);

  const vs = resp.voxel_grid.voxel_size;
  const vox = resp.voxel_grid;
  const ox = vox.origin[0];
  const oy = vox.origin[1];
  const oz = vox.origin[2];
  scanRoomOrigin = [ox, oy, oz];

  // Room dims in Three.js coords (X=X, Y=Z_backend, Z=Y_backend)
  const rw = vox.shape[0] * vs;
  const rh = vox.shape[2] * vs;
  const rd = vox.shape[1] * vs;
  scanRoomDims = [rw, rh, rd];

  // Cache heat sources (rack positions → Three.js coords)
  scanHeatSources = [];
  scanCoolSources = [];
  scanObstacles = [];

  for (const item of resp.equipment) {
    const [bx, by, bz] = item.position;
    const [bw, bd, bh] = item.size;
    const px = bx,
      py = bz,
      pz = by;

    // All solid equipment acts as airflow obstacle
    scanObstacles.push({ x: px, y: py, z: pz, hw: bw / 2, hh: bh, hd: bd / 2 });

    if (item.category === "server_rack" && item.facing) {
      let frontX = 0,
        frontZ = 0;
      if (item.facing === "+y") frontZ = 1;
      else if (item.facing === "-y") frontZ = -1;
      else if (item.facing === "+x") frontX = 1;
      else if (item.facing === "-x") frontX = -1;

      scanHeatSources.push({
        x: px,
        y: py,
        z: pz,
        heat: item.heat_output / 1000,
        h: bh,
        frontX,
        frontZ,
        backX: -frontX,
        backZ: -frontZ,
        airflow_m3s: Math.max(500, item.heat_output * 100) * 0.000472,
      });
    } else if (item.category === "cooling_unit") {
      scanCoolSources.push({ x: px, y: py, z: pz, w: bw, d: bd });
    }
  }

  // Allocate buffers
  scanHeads = new Float32Array(SCAN_PARTICLE_COUNT * 3);
  scanVels = new Float32Array(SCAN_PARTICLE_COUNT * 3);
  scanAges = new Float32Array(SCAN_PARTICLE_COUNT);
  scanMaxAges = new Float32Array(SCAN_PARTICLE_COUNT);
  scanTrailHist = new Float32Array(SCAN_PARTICLE_COUNT * SCAN_TRAIL_LEN * 3);
  scanTrailIdx = new Int32Array(SCAN_PARTICLE_COUNT);
  scanTrailPos = new Float32Array(SCAN_VERT_COUNT * 3);
  scanTrailCol = new Float32Array(SCAN_VERT_COUNT * 3);

  for (let i = 0; i < SCAN_PARTICLE_COUNT; i++) {
    resetScanParticle(i);
    scanAges[i] = Math.random() * scanMaxAges[i] * 0.5;
    const hx = scanHeads[i * 3],
      hy = scanHeads[i * 3 + 1],
      hz = scanHeads[i * 3 + 2];
    for (let t = 0; t < SCAN_TRAIL_LEN; t++) {
      const off = (i * SCAN_TRAIL_LEN + t) * 3;
      scanTrailHist[off] = hx;
      scanTrailHist[off + 1] = hy;
      scanTrailHist[off + 2] = hz;
    }
  }

  scanTrailGeo = new THREE.BufferGeometry();
  scanTrailGeo.setAttribute("position", new THREE.BufferAttribute(scanTrailPos, 3));
  scanTrailGeo.setAttribute("color", new THREE.BufferAttribute(scanTrailCol, 3));

  const mat = new THREE.LineBasicMaterial({
    vertexColors: true,
    transparent: true,
    opacity: 0.7,
    depthWrite: false,
  });
  scanTrailLines = new THREE.LineSegments(scanTrailGeo, mat);
  scanTrailLines.name = "scanStreamlines";
  scanTrailLines.frustumCulled = false;
  scanTrailLines.visible = scanAirflowVisible;
  scanVoxelGroup.add(scanTrailLines);
  scanAirflowActive = true;
}

function resetScanParticle(i: number) {
  if (!scanHeads || !scanVels) return;
  const idx = i * 3;
  const [rw, rh, rd] = scanRoomDims;
  const [ox, oy, oz] = scanRoomOrigin;
  const roll = Math.random();

  if (roll < 0.45 && scanHeatSources.length > 0) {
    // Spawn at rack exhaust (back side)
    const src = scanHeatSources[Math.floor(Math.random() * scanHeatSources.length)];
    const backOff = 0.3 + Math.random() * 0.6;
    scanHeads[idx] = src.x + src.backX * backOff + (Math.random() - 0.5) * 0.25;
    scanHeads[idx + 1] = src.y + 0.3 + Math.random() * (src.h - 0.3);
    scanHeads[idx + 2] = src.z + src.backZ * backOff + (Math.random() - 0.5) * 0.25;
    const speed = 0.008 + src.heat * 0.004;
    scanVels[idx] = src.backX * speed + (Math.random() - 0.5) * 0.002;
    scanVels[idx + 1] = 0.006 + Math.random() * 0.008;
    scanVels[idx + 2] = src.backZ * speed + (Math.random() - 0.5) * 0.002;
  } else if (roll < 0.8 && scanCoolSources.length > 0) {
    // Spawn at cooling unit — cold air along floor
    const cool = scanCoolSources[Math.floor(Math.random() * scanCoolSources.length)];
    scanHeads[idx] = cool.x + (Math.random() - 0.5) * cool.w;
    scanHeads[idx + 1] = oz + 0.05 + Math.random() * 0.3;
    scanHeads[idx + 2] = cool.z + (Math.random() - 0.5) * cool.d;
    let driftX = (Math.random() - 0.5) * 0.01;
    let driftZ = (Math.random() - 0.5) * 0.01;
    if (scanHeatSources.length > 0) {
      const tgt = scanHeatSources[Math.floor(Math.random() * scanHeatSources.length)];
      const dx = tgt.x + tgt.frontX * 0.5 - cool.x;
      const dz = tgt.z + tgt.frontZ * 0.5 - cool.z;
      const d = Math.sqrt(dx * dx + dz * dz) + 0.1;
      driftX = (dx / d) * 0.015 + (Math.random() - 0.5) * 0.003;
      driftZ = (dz / d) * 0.015 + (Math.random() - 0.5) * 0.003;
    }
    scanVels[idx] = driftX;
    scanVels[idx + 1] = -0.001;
    scanVels[idx + 2] = driftZ;
  } else {
    // Spawn at ceiling — return air descending toward CRACs
    scanHeads[idx] = ox + Math.random() * rw;
    scanHeads[idx + 1] = oz + rh - 0.3 - Math.random() * 0.5;
    scanHeads[idx + 2] = oy + Math.random() * rd;
    let driftX = (Math.random() - 0.5) * 0.008;
    let driftZ = (Math.random() - 0.5) * 0.008;
    if (scanCoolSources.length > 0) {
      const cool = scanCoolSources[Math.floor(Math.random() * scanCoolSources.length)];
      const dx = cool.x - scanHeads[idx];
      const dz = cool.z - scanHeads[idx + 2];
      const d = Math.sqrt(dx * dx + dz * dz) + 0.1;
      driftX = (dx / d) * 0.012;
      driftZ = (dz / d) * 0.012;
    }
    scanVels[idx] = driftX;
    scanVels[idx + 1] = -0.004 - Math.random() * 0.003;
    scanVels[idx + 2] = driftZ;
  }

  // Reject spawn positions inside solid obstacles
  for (const obs of scanObstacles) {
    const dx = scanHeads[idx] - obs.x;
    const dz = scanHeads[idx + 2] - obs.z;
    const py2 = scanHeads[idx + 1];
    if (Math.abs(dx) < obs.hw && Math.abs(dz) < obs.hd && py2 >= obs.y && py2 <= obs.y + obs.hh) {
      // Shift particle above the obstacle
      scanHeads[idx + 1] = obs.y + obs.hh + 0.1;
      break;
    }
  }

  scanAges[i] = 0;
  scanMaxAges[i] = 180 + Math.random() * 250;
  scanTrailIdx[i] = 0;
  const hx = scanHeads[idx],
    hy = scanHeads[idx + 1],
    hz = scanHeads[idx + 2];
  for (let t = 0; t < SCAN_TRAIL_LEN; t++) {
    const off = (i * SCAN_TRAIL_LEN + t) * 3;
    scanTrailHist[off] = hx;
    scanTrailHist[off + 1] = hy;
    scanTrailHist[off + 2] = hz;
  }
}

function updateScanAirflow() {
  if (!scanAirflowActive || !scanHeads || !scanVels) return;
  const [rw, rh, rd] = scanRoomDims;
  const [ox, oy, oz] = scanRoomOrigin;
  const xMin = ox,
    xMax = ox + rw;
  const yMin = oz,
    yMax = oz + rh;
  const zMin = oy,
    zMax = oy + rd;

  for (let i = 0; i < SCAN_PARTICLE_COUNT; i++) {
    const idx = i * 3;
    scanAges[i]++;

    const px = scanHeads[idx];
    const py = scanHeads[idx + 1];
    const pz = scanHeads[idx + 2];

    // Respawn check
    if (
      scanAges[i] > scanMaxAges[i] ||
      py > yMax + 0.3 ||
      py < yMin - 0.1 ||
      px < xMin - 0.5 ||
      px > xMax + 0.5 ||
      pz < zMin - 0.5 ||
      pz > zMax + 0.5
    ) {
      resetScanParticle(i);
      continue;
    }

    let forceX = 0,
      forceY = 0,
      forceZ = 0;
    let heatInfluence = 0;

    // ─ Rack physics: directional front-to-back airflow ─
    for (const src of scanHeatSources) {
      const dx = px - src.x;
      const dz = pz - src.z;
      const distSq = dx * dx + dz * dz;
      const dist = Math.sqrt(distSq);

      if (dist < 3.5) {
        const strength = src.heat / (1 + distSq);
        heatInfluence += strength;
        const dotFront = dx * src.frontX + dz * src.frontZ;

        if (dist < 2.0 && py <= src.y + src.h && py >= src.y) {
          if (dotFront > 0) {
            // Front (cold aisle): air sucked into rack
            const pull = strength * 0.006;
            forceX -= (dx / (dist + 0.1)) * pull;
            forceZ -= (dz / (dist + 0.1)) * pull;
          } else {
            // Back (hot aisle): hot air pushed out horizontally + rises
            const push = strength * 0.008;
            forceX += src.backX * push;
            forceZ += src.backZ * push;
            forceY += strength * 0.012;
          }
        } else if (py > src.y + src.h && dist < 2.0) {
          // Above rack: exhaust plume rises
          forceY += strength * 0.008;
          forceX += (dx / (dist + 0.1)) * strength * 0.001;
          forceZ += (dz / (dist + 0.1)) * strength * 0.001;
        } else if (py < src.y + 0.5 && dist < 3.5) {
          // Floor level: cold air drawn toward front intake
          const toFX = src.x + src.frontX * 0.6 - px;
          const toFZ = src.z + src.frontZ * 0.6 - pz;
          const toFD = Math.sqrt(toFX * toFX + toFZ * toFZ) + 0.1;
          forceX += (toFX / toFD) * strength * 0.003;
          forceZ += (toFZ / toFD) * strength * 0.003;
        }
      }
    }

    // ─ Cooling units: cold air blows outward along floor, returns above ─
    for (const cool of scanCoolSources) {
      const dx = px - cool.x;
      const dz = pz - cool.z;
      const dist = Math.sqrt(dx * dx + dz * dz);

      if (py < yMin + 1.0 && dist < 4.0) {
        const blow = 0.006 / (1 + dist * 0.3);
        forceX += (dx / (dist + 0.1)) * blow;
        forceZ += (dz / (dist + 0.1)) * blow;
        forceY -= 0.002;
      }
      if (py > yMin + 1.5 && dist < 6.0) {
        const pull = 0.004 / (1 + dist * 0.3);
        forceX -= (dx / (dist + 0.1)) * pull;
        forceZ -= (dz / (dist + 0.1)) * pull;
        if (dist < 2.0) forceY -= pull * 1.5;
      }
    }

    // Buoyancy from heat
    if (heatInfluence > 0.2) forceY += heatInfluence * 0.004;

    // Ceiling barrier → push toward CRAC
    if (py > yMax - 0.5) {
      forceY -= 0.012;
      for (const cool of scanCoolSources) {
        const dx = cool.x - px;
        const dz = cool.z - pz;
        const d = Math.sqrt(dx * dx + dz * dz) + 0.1;
        forceX += (dx / d) * 0.004;
        forceZ += (dz / d) * 0.004;
      }
    }

    // Floor
    if (py < yMin + 0.1 && heatInfluence < 0.5) forceY += 0.001;

    // Wall bounce
    if (px < xMin + 0.3) forceX += 0.003;
    if (px > xMax - 0.3) forceX -= 0.003;
    if (pz < zMin + 0.3) forceZ += 0.003;
    if (pz > zMax - 0.3) forceZ -= 0.003;

    // ─ Solid obstacle deflection (desks, legacy servers, etc.) ─
    for (const obs of scanObstacles) {
      const dx = px - obs.x;
      const dz = pz - obs.z;
      const insideX = Math.abs(dx) < obs.hw + 0.15;
      const insideZ = Math.abs(dz) < obs.hd + 0.15;
      const insideY = py >= obs.y - 0.05 && py <= obs.y + obs.hh + 0.15;
      if (insideX && insideZ && insideY) {
        // Push particle out along the shortest escape axis
        const overlapX = obs.hw + 0.15 - Math.abs(dx);
        const overlapZ = obs.hd + 0.15 - Math.abs(dz);
        const overlapYlo = py - (obs.y - 0.05);
        const overlapYhi = obs.y + obs.hh + 0.15 - py;
        const overlapY = Math.min(overlapYlo, overlapYhi);
        if (overlapX < overlapZ && overlapX < overlapY) {
          forceX += Math.sign(dx) * 0.015;
          scanVels[idx] = Math.sign(dx) * Math.abs(scanVels[idx]) * 0.3;
        } else if (overlapZ < overlapY) {
          forceZ += Math.sign(dz) * 0.015;
          scanVels[idx + 2] = Math.sign(dz) * Math.abs(scanVels[idx + 2]) * 0.3;
        } else {
          forceY += (overlapYhi < overlapYlo ? -1 : 1) * 0.015;
          scanVels[idx + 1] =
            (overlapYhi < overlapYlo ? -1 : 1) * Math.abs(scanVels[idx + 1]) * 0.3;
        }
      }
    }

    // Damping + force integration
    const damping = 0.94;
    scanVels[idx] = scanVels[idx] * damping + forceX;
    scanVels[idx + 1] = scanVels[idx + 1] * damping + forceY;
    scanVels[idx + 2] = scanVels[idx + 2] * damping + forceZ;

    // Speed limit
    const speed = Math.sqrt(scanVels[idx] ** 2 + scanVels[idx + 1] ** 2 + scanVels[idx + 2] ** 2);
    if (speed > 0.06) {
      const s = 0.06 / speed;
      scanVels[idx] *= s;
      scanVels[idx + 1] *= s;
      scanVels[idx + 2] *= s;
    }

    // Turbulence
    scanVels[idx] += (Math.random() - 0.5) * 0.0008;
    scanVels[idx + 1] += (Math.random() - 0.5) * 0.0004;
    scanVels[idx + 2] += (Math.random() - 0.5) * 0.0008;

    // Integrate position
    scanHeads[idx] += scanVels[idx];
    scanHeads[idx + 1] += scanVels[idx + 1];
    scanHeads[idx + 2] += scanVels[idx + 2];

    // Clamp floor/ceiling
    if (scanHeads[idx + 1] < yMin + 0.02) {
      scanHeads[idx + 1] = yMin + 0.02;
      scanVels[idx + 1] = Math.abs(scanVels[idx + 1]) * 0.1;
    }
    if (scanHeads[idx + 1] > yMax - 0.05) {
      scanHeads[idx + 1] = yMax - 0.05;
      scanVels[idx + 1] = -Math.abs(scanVels[idx + 1]) * 0.3;
    }

    // Record trail
    const cursor = scanTrailIdx[i] % SCAN_TRAIL_LEN;
    const hi = (i * SCAN_TRAIL_LEN + cursor) * 3;
    scanTrailHist[hi] = scanHeads[idx];
    scanTrailHist[hi + 1] = scanHeads[idx + 1];
    scanTrailHist[hi + 2] = scanHeads[idx + 2];
    scanTrailIdx[i]++;

    // Build line segments from trail ring buffer
    const segBase = i * (SCAN_TRAIL_LEN - 1);
    const curCursor = scanTrailIdx[i];

    for (let s = 0; s < SCAN_TRAIL_LEN - 1; s++) {
      const vIdx = (segBase + s) * 2;
      const older = (curCursor + s) % SCAN_TRAIL_LEN;
      const newer = (curCursor + s + 1) % SCAN_TRAIL_LEN;
      const oldOff = (i * SCAN_TRAIL_LEN + older) * 3;
      const newOff = (i * SCAN_TRAIL_LEN + newer) * 3;

      scanTrailPos[vIdx * 3] = scanTrailHist[oldOff];
      scanTrailPos[vIdx * 3 + 1] = scanTrailHist[oldOff + 1];
      scanTrailPos[vIdx * 3 + 2] = scanTrailHist[oldOff + 2];
      scanTrailPos[(vIdx + 1) * 3] = scanTrailHist[newOff];
      scanTrailPos[(vIdx + 1) * 3 + 1] = scanTrailHist[newOff + 1];
      scanTrailPos[(vIdx + 1) * 3 + 2] = scanTrailHist[newOff + 2];

      // Color from solver temperature at this position
      const segT = sampleScanTemp(
        scanTrailHist[oldOff],
        scanTrailHist[oldOff + 1],
        scanTrailHist[oldOff + 2],
      );
      const c = tempToColorScan(segT, scanTempMin, scanTempMax);

      const ageFade = s / (SCAN_TRAIL_LEN - 1);
      const brightness = ageFade * ageFade * 0.85 + 0.15;

      scanTrailCol[vIdx * 3] = c.r * brightness;
      scanTrailCol[vIdx * 3 + 1] = c.g * brightness;
      scanTrailCol[vIdx * 3 + 2] = c.b * brightness;
      scanTrailCol[(vIdx + 1) * 3] = c.r * Math.min(brightness * 1.3, 1);
      scanTrailCol[(vIdx + 1) * 3 + 1] = c.g * Math.min(brightness * 1.3, 1);
      scanTrailCol[(vIdx + 1) * 3 + 2] = c.b * Math.min(brightness * 1.3, 1);
    }
  }

  scanTrailGeo.attributes.position.needsUpdate = true;
  scanTrailGeo.attributes.color.needsUpdate = true;
}

function setScanMode(mode: "semantic" | "thermal") {
  currentScanMode = mode;
  for (const child of scanVoxelGroup.children) {
    if (child.name === "scanRoom") {
      child.visible = true; // always show room shell
    } else if (child.name === "scanEquipGroup") {
      // Dim equipment when thermal or airflow is active
      const dim = mode === "thermal" || scanAirflowVisible;
      child.visible = true;
      child.traverse((obj) => {
        if (obj instanceof THREE.Mesh && obj.material instanceof THREE.MeshStandardMaterial) {
          obj.material.opacity = dim ? 0.25 : 1.0;
          obj.material.transparent = dim;
          obj.material.depthWrite = !dim;
        }
      });
    } else if (child.name === "scanAirflowArrows") {
      child.visible = scanAirflowVisible;
    } else if (child.name === "scanStreamlines") {
      child.visible = scanAirflowVisible;
    } else if (child instanceof THREE.InstancedMesh && child.name === "thermalVoxels") {
      child.visible = mode === "thermal";
    }
  }
  // Update button styling
  const semBtn = document.getElementById("vc-semantic")!;
  const therBtn = document.getElementById("vc-thermal")!;
  semBtn.style.borderColor = mode === "semantic" ? "#378ADD" : "#E8E5DD";
  semBtn.style.color = mode === "semantic" ? "#0C447C" : "#5F5E5A";
  therBtn.style.borderColor = mode === "thermal" ? "#E24B4A" : "#E8E5DD";
  therBtn.style.color = mode === "thermal" ? "#A32D2D" : "#5F5E5A";
  // Keep airflow button state in sync
  updateAirflowBtnStyle();
}

function toggleScanAirflow() {
  scanAirflowVisible = !scanAirflowVisible;
  for (const child of scanVoxelGroup.children) {
    if (child.name === "scanAirflowArrows" || child.name === "scanStreamlines") {
      child.visible = scanAirflowVisible;
    }
  }
  // Dim equipment when airflow or thermal is active
  const dimEquip = scanAirflowVisible || currentScanMode === "thermal";
  for (const child of scanVoxelGroup.children) {
    if (child.name === "scanEquipGroup") {
      child.traverse((obj) => {
        if (obj instanceof THREE.Mesh && obj.material instanceof THREE.MeshStandardMaterial) {
          obj.material.opacity = dimEquip ? 0.25 : 1.0;
          obj.material.transparent = dimEquip;
          obj.material.depthWrite = !dimEquip;
        }
      });
    }
  }
  updateAirflowBtnStyle();
}

function updateAirflowBtnStyle() {
  const btn = document.getElementById("vc-airflow");
  if (btn) {
    (btn as HTMLElement).style.borderColor = scanAirflowVisible ? "#1D9E75" : "#E8E5DD";
    (btn as HTMLElement).style.color = scanAirflowVisible ? "#0F6E56" : "#5F5E5A";
  }
}

function applyZSlice(maxZ: number) {
  const data = scanVoxelGroup.userData as
    | {
        voxelData: VoxelData;
        thermalData: ThermalData | null;
      }
    | undefined;
  if (!data?.voxelData) return;
  const totalZ = data.voxelData.shape[2];
  const showAll = maxZ >= totalZ;
  document.getElementById("voxel-z-value")!.textContent = showAll ? "All" : `≤ ${maxZ}`;

  // Z-slice only applies to thermal instanced mesh
  for (const child of scanVoxelGroup.children) {
    if (!(child instanceof THREE.InstancedMesh) || child.name !== "thermalVoxels") continue;
    const positions = data.thermalData?.positions;
    if (!positions) continue;

    const dummy = new THREE.Object3D();
    const mat = new THREE.Matrix4();
    for (let j = 0; j < positions.length; j++) {
      const iz = positions[j][2];
      child.getMatrixAt(j, mat);
      if (!showAll && iz >= maxZ) {
        // Scale to zero to hide
        dummy.position.setFromMatrixPosition(mat);
        dummy.scale.set(0, 0, 0);
        dummy.updateMatrix();
        child.setMatrixAt(j, dummy.matrix);
      } else {
        // Restore scale
        dummy.position.setFromMatrixPosition(mat);
        dummy.scale.set(1, 1, 1);
        dummy.updateMatrix();
        child.setMatrixAt(j, dummy.matrix);
      }
    }
    child.instanceMatrix.needsUpdate = true;
  }
}

function setScanStatus(msg: string, isError = false) {
  const el = document.getElementById("scan-status")!;
  el.textContent = msg;
  el.className = `scan-status${isError ? " error" : ""}`;
}

async function uploadScan() {
  const fileInput = document.getElementById("scan-obj-file") as HTMLInputElement;
  const metaInput = document.getElementById("scan-metadata") as HTMLTextAreaElement;
  const uploadBtn = document.getElementById("scan-upload-btn") as HTMLButtonElement;

  if (!fileInput.files?.length) {
    setScanStatus("Please select a .obj file.", true);
    return;
  }

  const file = fileInput.files[0];
  if (!file.name.toLowerCase().endsWith(".obj")) {
    setScanStatus("Only .obj files are supported.", true);
    return;
  }

  // Validate metadata JSON
  try {
    JSON.parse(metaInput.value);
  } catch {
    setScanStatus("Invalid JSON in metadata field.", true);
    return;
  }

  uploadBtn.disabled = true;
  setScanStatus("Uploading and processing…");

  try {
    const form = new FormData();
    form.append("file", file);
    form.append("metadata", metaInput.value);

    const res = await fetch("/api/v1/visualize", { method: "POST", body: form });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || res.statusText);
    }
    const data: VisualizeResponse = await res.json();
    buildVoxelScene(data);
    setScanStatus(`Done — ${data.voxel_grid?.positions.length ?? 0} voxels rendered.`);
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err);
    setScanStatus(`Error: ${msg}`, true);
  } finally {
    uploadBtn.disabled = false;
  }
}

async function loadDemo() {
  const demoBtn = document.getElementById("scan-demo-btn") as HTMLButtonElement;
  demoBtn.disabled = true;
  setScanStatus("Loading demo room…");

  try {
    const res = await fetch("/api/v1/visualize/demo");
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || res.statusText);
    }
    const data: VisualizeResponse = await res.json();
    buildVoxelScene(data);
    setScanStatus(`Demo loaded — ${data.voxel_grid?.positions.length ?? 0} voxels.`);
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err);
    setScanStatus(`Error: ${msg}`, true);
  } finally {
    demoBtn.disabled = false;
  }
}

function setupScanButtons() {
  document.getElementById("scan-upload-btn")?.addEventListener("click", uploadScan);
  document.getElementById("scan-demo-btn")?.addEventListener("click", loadDemo);
  document.getElementById("vc-semantic")?.addEventListener("click", () => setScanMode("semantic"));
  document.getElementById("vc-thermal")?.addEventListener("click", () => setScanMode("thermal"));
  document.getElementById("vc-airflow")?.addEventListener("click", () => toggleScanAirflow());

  const zSlider = document.getElementById("voxel-z-slider") as HTMLInputElement;
  zSlider.addEventListener("input", () => {
    applyZSlice(Number(zSlider.value));
  });
}

// ===== Start =====
init();
