import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import {
  allScenes,
  coolingEnergyBase,
  type Equipment,
  getLoadFactor,
  peakTempBase,
  type SceneGraph,
  type Score,
  trainingHistory,
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
let zonesGroup: THREE.Group;
let zonesVisible = false;

// Airflow streamline system
const STREAMLINE_COUNT = 500;
const TRAIL_LENGTH = 40; // points per streamline
const SEGMENT_COUNT = STREAMLINE_COUNT * (TRAIL_LENGTH - 1); // line segments
const VERTEX_COUNT = SEGMENT_COUNT * 2; // 2 vertices per segment

// Per-streamline state
let streamlineHeads: Float32Array; // current position (x,y,z per streamline)
let streamlineVelocities: Float32Array; // current velocity
let streamlineAges: Float32Array;
let streamlineMaxAges: Float32Array;
// Trail geometry buffers
let trailPositions: Float32Array;
let trailColors: Float32Array;
let trailGeometry: THREE.BufferGeometry;
let trailLines: THREE.LineSegments;

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
  scene.background = new THREE.Color(0x0a0e17);

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
  zonesGroup = new THREE.Group();
  const grid = new THREE.GridHelper(18, 36, 0x1a2744, 0x111a2a);
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
  drawChart();
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
  if (airflowVisible && streamlineHeads) {
    const hour = simMinutes / 60;
    const loadFactor = getLoadFactor(hour);
    updateAirflow(allScenes[currentSceneIndex], loadFactor);
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

// ===== Room Builder =====
function buildRoom(sceneData: SceneGraph) {
  roomGroup.clear();
  const [rw, rh, rd] = sceneData.room.dimensions;

  // Floor — 서버실 바닥 (raised floor tiles)
  const floorGeo = new THREE.PlaneGeometry(rw, rd);
  const floorMat = new THREE.MeshStandardMaterial({
    color: 0xd0d0d0,
    roughness: 0.7,
  });
  const floor = new THREE.Mesh(floorGeo, floorMat);
  floor.rotation.x = -Math.PI / 2;
  floor.position.set(rw / 2, 0, rd / 2);
  floor.receiveShadow = true;
  roomGroup.add(floor);

  // 바닥 타일 그리드 (raised floor 느낌)
  const tileSize = 0.6;
  const tileMat = new THREE.LineBasicMaterial({
    color: 0xb0b0b0,
    linewidth: 1,
  });
  for (let x = 0; x <= rw; x += tileSize) {
    const geo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(x, 0.002, 0),
      new THREE.Vector3(x, 0.002, rd),
    ]);
    roomGroup.add(new THREE.Line(geo, tileMat));
  }
  for (let z = 0; z <= rd; z += tileSize) {
    const geo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(0, 0.002, z),
      new THREE.Vector3(rw, 0.002, z),
    ]);
    roomGroup.add(new THREE.Line(geo, tileMat));
  }

  // Walls
  const wallMat = new THREE.MeshStandardMaterial({
    color: 0xe0e0e0,
    transparent: true,
    opacity: 0.2,
    side: THREE.DoubleSide,
  });

  const northWall = new THREE.Mesh(new THREE.PlaneGeometry(rw, rh), wallMat.clone());
  northWall.position.set(rw / 2, rh / 2, 0);
  roomGroup.add(northWall);

  const southWall = new THREE.Mesh(new THREE.PlaneGeometry(rw, rh), wallMat.clone());
  southWall.position.set(rw / 2, rh / 2, rd);
  southWall.rotation.y = Math.PI;
  roomGroup.add(southWall);

  const westWall = new THREE.Mesh(new THREE.PlaneGeometry(rd, rh), wallMat.clone());
  westWall.rotation.y = Math.PI / 2;
  westWall.position.set(0, rh / 2, rd / 2);
  roomGroup.add(westWall);

  const eastWall = new THREE.Mesh(new THREE.PlaneGeometry(rd, rh), wallMat.clone());
  eastWall.rotation.y = -Math.PI / 2;
  eastWall.position.set(rw, rh / 2, rd / 2);
  roomGroup.add(eastWall);

  // Floor outline
  const edgeMat = new THREE.LineBasicMaterial({
    color: 0x00bcd4,
    linewidth: 2,
  });
  const floorOutline = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(0, 0.005, 0),
    new THREE.Vector3(rw, 0.005, 0),
    new THREE.Vector3(rw, 0.005, rd),
    new THREE.Vector3(0, 0.005, rd),
    new THREE.Vector3(0, 0.005, 0),
  ]);
  roomGroup.add(new THREE.Line(floorOutline, edgeMat));

  // 수직 기둥 (코너)
  const pillarGeo = new THREE.BoxGeometry(0.15, rh, 0.15);
  const pillarMat = new THREE.MeshStandardMaterial({ color: 0x90a4ae });
  const corners = [
    [0, rh / 2, 0],
    [rw, rh / 2, 0],
    [rw, rh / 2, rd],
    [0, rh / 2, rd],
  ];
  for (const [cx, cy, cz] of corners) {
    const pillar = new THREE.Mesh(pillarGeo, pillarMat);
    pillar.position.set(cx, cy, cz);
    pillar.castShadow = true;
    roomGroup.add(pillar);
  }

  // Openings
  for (const opening of sceneData.room.openings) {
    const isDoor = opening.type === "door";
    const isVent = opening.type === "vent";
    const color = isDoor ? 0xff9800 : isVent ? 0x00bcd4 : 0x4caf50;
    const markerH = isDoor ? 2.4 : 0.8;

    const markerGeo = new THREE.BoxGeometry(opening.width, markerH, 0.08);
    const markerMat = new THREE.MeshStandardMaterial({
      color,
      transparent: true,
      opacity: isDoor ? 0.5 : 0.35,
    });
    const marker = new THREE.Mesh(markerGeo, markerMat);

    if (opening.wall === "north") {
      marker.position.set(opening.position[0], isDoor ? markerH / 2 : opening.position[1], 0);
    } else if (opening.wall === "south") {
      marker.position.set(opening.position[0], isDoor ? markerH / 2 : opening.position[1], rd);
    } else if (opening.wall === "east") {
      marker.rotation.y = Math.PI / 2;
      marker.position.set(rw, isDoor ? markerH / 2 : opening.position[1], opening.position[2]);
    } else {
      marker.rotation.y = Math.PI / 2;
      marker.position.set(0, isDoor ? markerH / 2 : opening.position[1], opening.position[2]);
    }
    roomGroup.add(marker);

    // Vent에 화살표 (공기 흐름 표시)
    if (isVent) {
      const arrowDir = new THREE.Vector3(0, 0, 1);
      if (opening.wall === "south") arrowDir.set(0, 0, -1);
      else if (opening.wall === "east") arrowDir.set(-1, 0, 0);
      else if (opening.wall === "west") arrowDir.set(1, 0, 0);

      const arrowOrigin = new THREE.Vector3(
        opening.position[0],
        opening.position[1],
        opening.wall === "north" ? 0.2 : opening.wall === "south" ? rd - 0.2 : opening.position[2],
      );
      if (opening.wall === "east")
        arrowOrigin.set(rw - 0.2, opening.position[1], opening.position[2]);
      if (opening.wall === "west") arrowOrigin.set(0.2, opening.position[1], opening.position[2]);

      const arrow = new THREE.ArrowHelper(arrowDir, arrowOrigin, 1.2, 0x00bcd4, 0.3, 0.15);
      roomGroup.add(arrow);
    }
  }
}

// ===== Heatmap (physics-based) =====
// References:
//   ASHRAE TC 9.9 — Thermal Guidelines for Data Processing Environments
//   Turbulent plane jet: Rajaratnam, "Turbulent Jets" (1976)
//   ΔT = Q / (ṁ · Cₚ)  where  ṁ = ρ · V̇
//
// Constants (from engine/core/config.py):
const AIR_RHO = 1.2;          // kg/m³ at ~22 °C sea level
const AIR_CP = 1006;           // J/(kg·K) — specific heat of dry air
const CFM_TO_M3S = 0.000472;  // 1 CFM → m³/s
const DEFAULT_CFM = 800;       // mid-range 42U rack airflow (CFM)
const AC_CFM = 2000;           // CRAC airflow (CFM)
const AC_SUPPLY_T = 14;        // °C — CRAC supply temperature
const AMBIENT_T = 22;          // °C — ASHRAE recommended inlet
// Turbulent plane-jet half-angle spreading rate (Rajaratnam 1976):
//   σ(x) = 0.11·x  →  Gaussian σ_angle ≈ 0.22 rad ≈ 12.6°
// For floor projection at rack mid-height we use a wider effective cone
// because buoyancy + ceiling return widens the floor footprint.
const PLUME_SIGMA_RAD = 0.35;  // ≈ 20° effective half-angle on floor
// Heatmap sample height: ASHRAE recommends measuring at 1/3 and 2/3
// rack height.  We use mid-height as a representative slice.
const HEATMAP_HEIGHT = 1.0;    // m — rack mid-height sample plane

function buildHeatmap(sceneData: SceneGraph, loadFactor = 1.0) {
  heatmapGroup.clear();
  const [rw, , rd] = sceneData.room.dimensions;
  const resolution = 0.5;

  // Build physics-based heat sources from rack exhaust
  // Exhaust spans 0 to rackH; sample plane is at HEATMAP_HEIGHT
  const heatSources = sceneData.furniture
    .filter((f) => f.heatOutput > 0)
    .map((f) => {
      const rotY = THREE.MathUtils.degToRad(f.rotation[1]);
      const backX = -Math.sin(rotY);
      const backZ = -Math.cos(rotY);
      const rackW = f.size[0] || 0.6;
      const rackH = f.size[1] || 2.0;
      const rackD = f.size[2] || 1.0;
      const halfDepth = rackD / 2;

      // ΔT_exhaust = Q / (ṁ · Cₚ)
      const Q_watts = f.heatOutput * 1000 * loadFactor;  // kW → W
      const m_dot = AIR_RHO * DEFAULT_CFM * CFM_TO_M3S;  // kg/s
      const deltaT = Q_watts / (m_dot * AIR_CP);         // °C rise

      // Vertical factor: fraction of exhaust energy at sample height.
      // Exhaust is roughly uniform over 0..rackH; at HEATMAP_HEIGHT
      // we get full exposure if within the rack span, reduced outside.
      const vertFrac = HEATMAP_HEIGHT <= rackH ? 1.0
        : Math.exp(-0.5 * ((HEATMAP_HEIGHT - rackH) / 0.5) ** 2);

      return {
        x: f.position[0] + backX * halfDepth,
        z: f.position[2] + backZ * halfDepth,
        backX,
        backZ,
        rackW,
        deltaT: deltaT * vertFrac,
      };
    });

  // Cooling sources: cold supply jets
  const coolingSources = sceneData.furniture
    .filter((f) => f.category === "cooling_unit")
    .map((cool) => {
      const coolQ = AIR_RHO * AC_CFM * CFM_TO_M3S * AIR_CP * (AMBIENT_T - AC_SUPPLY_T);
      return {
        x: cool.position[0],
        z: cool.position[2],
        coolingW: coolQ * loadFactor,  // Watts of cooling
      };
    });

  // Compute temperature field on floor grid
  let maxDeltaT = 0;

  // Pre-compute into a 2D array so we can normalize properly
  const nx = Math.ceil(rw / resolution);
  const nz = Math.ceil(rd / resolution);
  const field = new Float32Array(nx * nz);

  for (let xi = 0; xi < nx; xi++) {
    const x = resolution / 2 + xi * resolution;
    for (let zi = 0; zi < nz; zi++) {
      const z = resolution / 2 + zi * resolution;
      let localDeltaT = 0;

      // Contribution from each rack exhaust (Gaussian plume model)
      for (const src of heatSources) {
        const dx = x - src.x;
        const dz = z - src.z;
        const dist = Math.sqrt(dx * dx + dz * dz);

        if (dist < 0.01) {
          localDeltaT += src.deltaT;
          continue;
        }

        // Angle between exhaust direction and sample point
        const dot = (dx * src.backX + dz * src.backZ) / dist;
        // theta = angle from exhaust centreline (0 = directly behind)
        const theta = Math.acos(Math.min(1, Math.max(-1, dot)));

        // Gaussian angular decay: exp(-θ²/(2σ²))
        // This models the turbulent jet spreading rate
        const angularDecay = Math.exp(-(theta * theta) / (2 * PLUME_SIGMA_RAD * PLUME_SIGMA_RAD));

        // Axial decay: plane-jet centreline velocity ∝ 1/√x for x > L_core
        // Temperature excess follows same scaling (Rajaratnam 1976, Ch.2)
        // Potential core length ≈ 5×b₀ for plane jets (Albertson et al. 1950)
        const b0 = src.rackW / 2; // initial half-width of jet slot
        const coreLen = b0 * 5;   // potential core: no decay within ~5 slot widths
        const axialDecay = dist < coreLen
          ? 1.0                                // potential core (no decay)
          : Math.sqrt(coreLen / dist);         // far-field: 1/√r

        // Minimum recirculation: ~3% of ΔT reaches intake side
        // (hot ceiling return path contributes background warming)
        const recirc = 0.03;

        const contribution = src.deltaT * (angularDecay * axialDecay * (1 - recirc) + recirc);
        localDeltaT += contribution;
      }

      // Cooling effect: AC cold air delivered at floor, mixed at sample height.
      // Cold air rises and mixes turbulently; at HEATMAP_HEIGHT the cooling
      // effect is attenuated by vertical mixing fraction.
      for (const cool of coolingSources) {
        const dx = x - cool.x;
        const dz = z - cool.z;
        const dist = Math.sqrt(dx * dx + dz * dz);
        const r0 = 0.3;
        const m_dot_cool = AIR_RHO * AC_CFM * CFM_TO_M3S;
        const baseCoolingDT = cool.coolingW / (m_dot_cool * AIR_CP);
        const spatialDecay = dist < r0 ? 1.0 : r0 / dist;
        // Vertical mixing: cold supply at floor (y=0) loses intensity
        // rising to sample height. Approximate as exp(-h/L_mix)
        // L_mix ~ 1.5 m for typical perforated-tile raised-floor supply
        const vertMix = Math.exp(-HEATMAP_HEIGHT / 1.5);
        const cooling = baseCoolingDT * spatialDecay * vertMix;
        localDeltaT -= Math.min(cooling, localDeltaT);
      }

      localDeltaT = Math.max(0, localDeltaT);
      field[xi * nz + zi] = localDeltaT;
      if (localDeltaT > maxDeltaT) maxDeltaT = localDeltaT;
    }
  }

  // Render floor tiles — scale colour from 0 to max ΔT above ambient
  if (maxDeltaT < 0.1) maxDeltaT = 1; // avoid div-by-zero
  for (let xi = 0; xi < nx; xi++) {
    const x = resolution / 2 + xi * resolution;
    for (let zi = 0; zi < nz; zi++) {
      const z = resolution / 2 + zi * resolution;
      const dt = field[xi * nz + zi];
      const t = Math.min(dt / maxDeltaT, 1);

      // Colour ramp: Blue → Cyan → Green → Yellow → Red
      const color = new THREE.Color();
      if (t < 0.25) {
        color.setRGB(0, t * 4, 1);
      } else if (t < 0.5) {
        color.setRGB(0, 1, 1 - (t - 0.25) * 4);
      } else if (t < 0.75) {
        color.setRGB((t - 0.5) * 4, 1, 0);
      } else {
        color.setRGB(1, 1 - (t - 0.75) * 4, 0);
      }

      const tileGeo = new THREE.PlaneGeometry(resolution * 0.95, resolution * 0.95);
      const tileMat = new THREE.MeshBasicMaterial({
        color,
        transparent: true,
        opacity: 0.15 + 0.3 * loadFactor,
        side: THREE.DoubleSide,
      });
      const tile = new THREE.Mesh(tileGeo, tileMat);
      tile.rotation.x = -Math.PI / 2;
      tile.position.set(x, 0.01, z);
      heatmapGroup.add(tile);
    }
  }

  heatmapGroup.visible = heatmapVisible;
  return maxDeltaT;
}

// ===== Airflow Streamline System =====
// Each streamline is a head particle that leaves a trail of line segments behind it.
// This creates the CFD-like flow visualization seen in professional airflow analysis.

// Per-streamline trail history: ring buffer of past positions
let trailHistory: Float32Array; // STREAMLINE_COUNT * TRAIL_LENGTH * 3
let trailHistoryIdx: Int32Array; // write cursor per streamline

function makeTextSprite(text: string, color: string): THREE.Sprite {
  const canvas = document.createElement("canvas");
  canvas.width = 128;
  canvas.height = 48;
  const ctx = canvas.getContext("2d")!;
  ctx.font = "bold 28px monospace";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillStyle = color;
  ctx.globalAlpha = 0.25;
  const tw = ctx.measureText(text).width + 16;
  ctx.beginPath();
  ctx.roundRect((128 - tw) / 2, 6, tw, 36, 6);
  ctx.fill();
  ctx.globalAlpha = 0.9;
  ctx.fillStyle = color;
  ctx.fillText(text, 64, 24);
  const tex = new THREE.CanvasTexture(canvas);
  const mat = new THREE.SpriteMaterial({
    map: tex,
    transparent: true,
    depthWrite: false,
  });
  const sprite = new THREE.Sprite(mat);
  sprite.scale.set(1.2, 0.45, 1);
  return sprite;
}

function initAirflow(sceneData: SceneGraph) {
  airflowGroup.clear();

  streamlineHeads = new Float32Array(STREAMLINE_COUNT * 3);
  streamlineVelocities = new Float32Array(STREAMLINE_COUNT * 3);
  streamlineAges = new Float32Array(STREAMLINE_COUNT);
  streamlineMaxAges = new Float32Array(STREAMLINE_COUNT);
  trailHistory = new Float32Array(STREAMLINE_COUNT * TRAIL_LENGTH * 3);
  trailHistoryIdx = new Int32Array(STREAMLINE_COUNT);

  // Trail rendering: each streamline has (TRAIL_LENGTH-1) segments, each segment = 2 vertices
  trailPositions = new Float32Array(VERTEX_COUNT * 3);
  trailColors = new Float32Array(VERTEX_COUNT * 3);

  const coolingSources = sceneData.furniture.filter((f) => f.category === "cooling_unit");
  const [rw, , rd] = sceneData.room.dimensions;

  // Pre-cache heat sources so resetStreamline can use them
  cachedHeatSources = sceneData.furniture
    .filter((f) => f.heatOutput > 0)
    .map((f) => {
      const rotY = THREE.MathUtils.degToRad(f.rotation[1]);
      const frontX = Math.sin(rotY);
      const frontZ = Math.cos(rotY);
      return {
        x: f.position[0],
        z: f.position[2],
        heat: f.heatOutput,
        h: f.size[1],
        frontX,
        frontZ,
        backX: -frontX,
        backZ: -frontZ,
      };
    });

  for (let i = 0; i < STREAMLINE_COUNT; i++) {
    resetStreamline(i, coolingSources, rw, rd);
    // Stagger so trails build up gradually
    streamlineAges[i] = Math.random() * streamlineMaxAges[i] * 0.5;
    // Pre-fill trail history with spawn position
    const hx = streamlineHeads[i * 3];
    const hy = streamlineHeads[i * 3 + 1];
    const hz = streamlineHeads[i * 3 + 2];
    for (let t = 0; t < TRAIL_LENGTH; t++) {
      const ti = (i * TRAIL_LENGTH + t) * 3;
      trailHistory[ti] = hx;
      trailHistory[ti + 1] = hy;
      trailHistory[ti + 2] = hz;
    }
  }

  trailGeometry = new THREE.BufferGeometry();
  trailGeometry.setAttribute("position", new THREE.BufferAttribute(trailPositions, 3));
  trailGeometry.setAttribute("color", new THREE.BufferAttribute(trailColors, 3));

  const trailMat = new THREE.LineBasicMaterial({
    vertexColors: true,
    transparent: true,
    opacity: 0.85,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    linewidth: 1,
  });

  trailLines = new THREE.LineSegments(trailGeometry, trailMat);
  airflowGroup.add(trailLines);

  // === Cold / Hot zone indicators + Problem markers → zonesGroup ===
  zonesGroup.clear();
  const quality = sceneData.score.total;
  const serverRacks = sceneData.furniture.filter((f) => f.heatOutput > 0);

  for (const rack of serverRacks) {
    const rotY = THREE.MathUtils.degToRad(rack.rotation[1]);
    const frontX = Math.sin(rotY);
    const frontZ = Math.cos(rotY);
    const [rackW, rackH, _rackD] = rack.size;
    const zoneDepth = 1.2;
    const zoneW = rackW + 0.6;
    const zoneH = rackH + 0.3;

    // Cold zone (front)
    const coldGeo = new THREE.BoxGeometry(zoneW, zoneH, zoneDepth);
    const coldZone = new THREE.Mesh(
      coldGeo,
      new THREE.MeshBasicMaterial({
        color: 0x44aaff,
        transparent: true,
        opacity: 0.28,
        side: THREE.DoubleSide,
        depthWrite: false,
      }),
    );
    coldZone.position.set(
      rack.position[0] + frontX * (zoneDepth / 2 - 0.1),
      rackH / 2,
      rack.position[2] + frontZ * (zoneDepth / 2 - 0.1),
    );
    coldZone.rotation.y = rotY;
    zonesGroup.add(coldZone);

    const coldEdge = new THREE.LineSegments(
      new THREE.EdgesGeometry(coldGeo),
      new THREE.LineBasicMaterial({ color: 0x44aaff, transparent: true, opacity: 0.35 }),
    );
    coldEdge.position.copy(coldZone.position);
    coldEdge.rotation.copy(coldZone.rotation);
    zonesGroup.add(coldEdge);

    // Hot zone (back) — red/orange to indicate exhaust heat
    const hotGeo = new THREE.BoxGeometry(zoneW, zoneH + 0.5, zoneDepth);
    const hotZone = new THREE.Mesh(
      hotGeo,
      new THREE.MeshBasicMaterial({
        color: 0xff6633,
        transparent: true,
        opacity: 0.28,
        side: THREE.DoubleSide,
        depthWrite: false,
      }),
    );
    hotZone.position.set(
      rack.position[0] - frontX * (zoneDepth / 2 - 0.1),
      (rackH + 0.5) / 2,
      rack.position[2] - frontZ * (zoneDepth / 2 - 0.1),
    );
    hotZone.rotation.y = rotY;
    zonesGroup.add(hotZone);

    const hotEdge = new THREE.LineSegments(
      new THREE.EdgesGeometry(hotGeo),
      new THREE.LineBasicMaterial({ color: 0xff6633, transparent: true, opacity: 0.35 }),
    );
    hotEdge.position.copy(hotZone.position);
    hotEdge.rotation.copy(hotZone.rotation);
    zonesGroup.add(hotEdge);

    // Arrows
    const coldArrow = new THREE.ArrowHelper(
      new THREE.Vector3(-frontX, 0, -frontZ),
      new THREE.Vector3(
        rack.position[0] + frontX * (zoneDepth + 0.1),
        rackH * 0.5,
        rack.position[2] + frontZ * (zoneDepth + 0.1),
      ),
      0.7,
      0x44aaff,
      0.15,
      0.1,
    );
    zonesGroup.add(coldArrow);

    // Exhaust arrows — horizontal backward (matching real rack fan physics)
    for (const yFrac of [0.25, 0.5, 0.75]) {
      const hotArrow = new THREE.ArrowHelper(
        new THREE.Vector3(-frontX, 0, -frontZ),
        new THREE.Vector3(
          rack.position[0] - frontX * (zoneDepth * 0.15),
          rackH * yFrac,
          rack.position[2] - frontZ * (zoneDepth * 0.15),
        ),
        0.7,
        0xff3300,
        0.15,
        0.1,
      );
      zonesGroup.add(hotArrow);
    }
  }

  // === Problem markers for BAD layouts ===
  if (quality < 0.6) {
    for (let a = 0; a < serverRacks.length; a++) {
      const rackA = serverRacks[a];
      const rotA = THREE.MathUtils.degToRad(rackA.rotation[1]);
      const backAx = rackA.position[0] - Math.sin(rotA) * 1.5;
      const backAz = rackA.position[2] - Math.cos(rotA) * 1.5;

      for (let b = 0; b < serverRacks.length; b++) {
        if (a === b) continue;
        const rackB = serverRacks[b];
        const rotB = THREE.MathUtils.degToRad(rackB.rotation[1]);
        const frontBx = rackB.position[0] + Math.sin(rotB) * 1.0;
        const frontBz = rackB.position[2] + Math.cos(rotB) * 1.0;
        const dist = Math.sqrt((backAx - frontBx) ** 2 + (backAz - frontBz) ** 2);

        if (dist < 2.5) {
          const midX = (backAx + frontBx) / 2;
          const midZ = (backAz + frontBz) / 2;
          const overlapH = Math.max(rackA.size[1], rackB.size[1]) + 0.3;
          const spanX = Math.abs(backAx - frontBx) + 1.2;
          const spanZ = Math.abs(backAz - frontBz) + 1.2;
          const recircGeo = new THREE.BoxGeometry(spanX, overlapH, spanZ);
          const recircZone = new THREE.Mesh(
            recircGeo,
            new THREE.MeshBasicMaterial({
              color: 0xff1744,
              transparent: true,
              opacity: 0.12,
              side: THREE.DoubleSide,
              depthWrite: false,
            }),
          );
          recircZone.position.set(midX, overlapH / 2, midZ);
          zonesGroup.add(recircZone);

          const recircEdge = new THREE.LineSegments(
            new THREE.EdgesGeometry(recircGeo),
            new THREE.LineBasicMaterial({ color: 0xff1744, transparent: true, opacity: 0.4 }),
          );
          recircEdge.position.copy(recircZone.position);
          zonesGroup.add(recircEdge);
        }
      }
    }

    for (const cool of coolingSources) {
      let reachesAny = false;
      for (const rack of serverRacks) {
        const rotR = THREE.MathUtils.degToRad(rack.rotation[1]);
        const frontRx = rack.position[0] + Math.sin(rotR) * 0.5;
        const frontRz = rack.position[2] + Math.cos(rotR) * 0.5;
        const dist = Math.sqrt(
          (cool.position[0] - frontRx) ** 2 + (cool.position[2] - frontRz) ** 2,
        );
        if (dist < 4.0) {
          reachesAny = true;
          break;
        }
      }
      if (!reachesAny) {
        const bypassLabel = makeTextSprite("BYPASS", "#ffab00");
        bypassLabel.position.set(cool.position[0], cool.size[1] + 0.5, cool.position[2]);
        zonesGroup.add(bypassLabel);
      }
    }
  }

  zonesGroup.visible = zonesVisible;
  airflowGroup.visible = airflowVisible;
}

// Cached scene data for resetStreamline
let cachedHeatSources: {
  x: number;
  z: number;
  heat: number;
  h: number;
  frontX: number;
  frontZ: number;
  backX: number;
  backZ: number;
}[] = [];

function resetStreamline(i: number, coolingSources: Equipment[], rw: number, rd: number) {
  const idx = i * 3;
  const roll = Math.random();

  if (roll < 0.45 && cachedHeatSources.length > 0) {
    // === 45%: Spawn at SERVER RACK BACK — hot air exhaust inside hot zone ===
    const src = cachedHeatSources[Math.floor(Math.random() * cachedHeatSources.length)];
    // Spawn inside the hot zone (back side of rack, within zone volume)
    const backOff = 0.3 + Math.random() * 0.7;
    streamlineHeads[idx] = src.x + src.backX * backOff + (Math.random() - 0.5) * 0.3;
    streamlineHeads[idx + 1] = 0.3 + Math.random() * src.h;
    streamlineHeads[idx + 2] = src.z + src.backZ * backOff + (Math.random() - 0.5) * 0.3;

    // Initial velocity: outward from back + upward
    streamlineVelocities[idx] = src.backX * 0.012 + (Math.random() - 0.5) * 0.003;
    streamlineVelocities[idx + 1] = 0.01 + Math.random() * 0.015;
    streamlineVelocities[idx + 2] = src.backZ * 0.012 + (Math.random() - 0.5) * 0.003;
  } else if (roll < 0.8 && coolingSources.length > 0) {
    // === 35%: Spawn at COOLING UNIT bottom — cold air flowing along floor ===
    const cool = coolingSources[Math.floor(Math.random() * coolingSources.length)];
    streamlineHeads[idx] = cool.position[0] + (Math.random() - 0.5) * cool.size[0];
    streamlineHeads[idx + 1] = 0.05 + Math.random() * 0.3; // Near floor
    streamlineHeads[idx + 2] = cool.position[2] + (Math.random() - 0.5) * cool.size[2];

    // Initial velocity: toward the FRONT zone of a server rack (cold aisle supply)
    let bestDx = (Math.random() - 0.5) * 0.02;
    let bestDz = (Math.random() - 0.5) * 0.02;
    if (cachedHeatSources.length > 0) {
      const target = cachedHeatSources[Math.floor(Math.random() * cachedHeatSources.length)];
      // Aim toward the front zone center (where cold zone box is)
      const frontTargetX = target.x + target.frontX * 0.6;
      const frontTargetZ = target.z + target.frontZ * 0.6;
      const dx = frontTargetX - cool.position[0];
      const dz = frontTargetZ - cool.position[2];
      const dist = Math.sqrt(dx * dx + dz * dz) + 0.1;
      bestDx = (dx / dist) * 0.02 + (Math.random() - 0.5) * 0.003;
      bestDz = (dz / dist) * 0.02 + (Math.random() - 0.5) * 0.003;
    }
    streamlineVelocities[idx] = bestDx;
    streamlineVelocities[idx + 1] = -0.001 + Math.random() * 0.001;
    streamlineVelocities[idx + 2] = bestDz;
  } else {
    // === 20%: Spawn at CEILING near servers — return air path ===
    const [, roomH] = [rw, 3.5]; // room height
    if (cachedHeatSources.length > 0) {
      const src = cachedHeatSources[Math.floor(Math.random() * cachedHeatSources.length)];
      streamlineHeads[idx] = src.x + (Math.random() - 0.5) * 2;
      streamlineHeads[idx + 1] = roomH - 0.3 - Math.random() * 0.5;
      streamlineHeads[idx + 2] = src.z + (Math.random() - 0.5) * 2;
    } else {
      streamlineHeads[idx] = Math.random() * rw;
      streamlineHeads[idx + 1] = roomH - 0.3 - Math.random() * 0.5;
      streamlineHeads[idx + 2] = Math.random() * rd;
    }

    // Initial velocity: horizontal drift toward nearest CRAC
    let driftX = (Math.random() - 0.5) * 0.01;
    let driftZ = (Math.random() - 0.5) * 0.01;
    if (coolingSources.length > 0) {
      const cool = coolingSources[Math.floor(Math.random() * coolingSources.length)];
      const dx = cool.position[0] - streamlineHeads[idx];
      const dz = cool.position[2] - streamlineHeads[idx + 2];
      const dist = Math.sqrt(dx * dx + dz * dz) + 0.1;
      driftX = (dx / dist) * 0.015;
      driftZ = (dz / dist) * 0.015;
    }
    streamlineVelocities[idx] = driftX;
    streamlineVelocities[idx + 1] = -0.003 - Math.random() * 0.003; // Descending toward CRAC intake
    streamlineVelocities[idx + 2] = driftZ;
  }

  streamlineAges[i] = 0;
  streamlineMaxAges[i] = 200 + Math.random() * 300;
  trailHistoryIdx[i] = 0;

  // Fill trail history with current position (collapsed trail)
  const hx = streamlineHeads[idx];
  const hy = streamlineHeads[idx + 1];
  const hz = streamlineHeads[idx + 2];
  for (let t = 0; t < TRAIL_LENGTH; t++) {
    const ti = (i * TRAIL_LENGTH + t) * 3;
    trailHistory[ti] = hx;
    trailHistory[ti + 1] = hy;
    trailHistory[ti + 2] = hz;
  }
}

function computeTemperatureColor(temp: number): [number, number, number] {
  // Blue(cold) → Cyan → White → Yellow → Red(hot)
  if (temp < 0.2) {
    return [0.1, 0.2 + temp * 4, 1.0];
  } else if (temp < 0.4) {
    const t2 = (temp - 0.2) * 5;
    return [0.1 + t2 * 0.5, 0.6 + t2 * 0.4, 1.0];
  } else if (temp < 0.6) {
    const t2 = (temp - 0.4) * 5;
    return [0.6 + t2 * 0.4, 1.0, 1.0 - t2];
  } else if (temp < 0.8) {
    const t2 = (temp - 0.6) * 5;
    return [1.0, 1.0 - t2 * 0.5, 0];
  } else {
    const t2 = (temp - 0.8) * 5;
    return [1.0, Math.max(0.5 - t2 * 0.5, 0.1), 0];
  }
}

function updateAirflow(sceneData: SceneGraph, loadFactor: number) {
  if (!airflowVisible || !streamlineHeads) return;

  // === Scene quality → airflow behavior ===
  // Low quality (random, 0.15) → chaotic, turbulent, recirculation, mixed colors
  // High quality (RL, 0.91)   → laminar, clean front→back→up, separated colors
  const quality = sceneData.score.total;
  const chaos = 1 - quality; // 0.85 for random, 0.09 for RL

  // Turbulence strength: chaotic layouts have 10x more turbulence
  const turbulence = 0.0003 + chaos * 0.003;
  // Damping: good layouts have smoother flow
  const damping = 0.96 - chaos * 0.04; // 0.92 random, 0.96 RL
  // How strongly directional forces work (front/back distinction)
  const directionality = quality; // 0.15 random (barely works), 0.91 RL (very clean)

  // Cache heat sources with front/back direction
  cachedHeatSources = sceneData.furniture
    .filter((f) => f.heatOutput > 0)
    .map((f) => {
      const rotY = THREE.MathUtils.degToRad(f.rotation[1]);
      const frontX = Math.sin(rotY);
      const frontZ = Math.cos(rotY);
      return {
        x: f.position[0],
        z: f.position[2],
        heat: f.heatOutput * loadFactor,
        h: f.size[1],
        frontX,
        frontZ,
        backX: -frontX,
        backZ: -frontZ,
      };
    });

  const coolingSources = sceneData.furniture.filter((f) => f.category === "cooling_unit");
  const [rw, rh, rd] = sceneData.room.dimensions;

  for (let i = 0; i < STREAMLINE_COUNT; i++) {
    const idx = i * 3;
    streamlineAges[i]++;

    const px = streamlineHeads[idx];
    const py = streamlineHeads[idx + 1];
    const pz = streamlineHeads[idx + 2];

    // Respawn check
    if (
      streamlineAges[i] > streamlineMaxAges[i] ||
      py > rh + 0.3 ||
      py < -0.1 ||
      px < -0.5 ||
      px > rw + 0.5 ||
      pz < -0.5 ||
      pz > rd + 0.5
    ) {
      resetStreamline(i, coolingSources, rw, rd);
      continue;
    }

    // === Physics ===
    let heatInfluence = 0;
    let forceX = 0;
    let forceY = 0;
    let forceZ = 0;

    // --- Server racks: DIRECTIONAL airflow (front=intake, back=exhaust) ---
    for (const src of cachedHeatSources) {
      const dx = px - src.x;
      const dz = pz - src.z;
      const distSq = dx * dx + dz * dz;
      const dist = Math.sqrt(distSq);

      if (dist < 3.5) {
        const strength = src.heat / (1 + distSq);
        heatInfluence += strength;

        // Determine front vs back side using dot product
        const dotFront = dx * src.frontX + dz * src.frontZ;

        if (dist < 2.0 && py <= src.h) {
          if (dotFront > 0) {
            // === FRONT SIDE (cold aisle): cold air sucked IN ===
            // directionality controls how clean this works
            const pull = strength * (0.003 + directionality * 0.004);
            forceX -= (dx / (dist + 0.1)) * pull;
            forceZ -= (dz / (dist + 0.1)) * pull;
            if (py < 0.5) forceY -= 0.001;
          } else {
            // === BACK SIDE (hot aisle): hot air BLOWS OUT and RISES ===
            const push = strength * (0.004 + directionality * 0.005);
            forceX += src.backX * push;
            forceZ += src.backZ * push;
            forceY += strength * (0.008 + directionality * 0.008);
          }

          // BAD LAYOUT EFFECT: recirculation — hot air leaks to wrong side
          // In chaotic layouts, forces are partially random/reversed
          if (chaos > 0.5) {
            const leak = (chaos - 0.5) * 2 * strength * 0.004;
            forceX += (Math.random() - 0.5) * leak * 2;
            forceZ += (Math.random() - 0.5) * leak * 2;
            // Hot air pushed sideways instead of cleanly up
            forceY -= leak * 0.5;
          }
        } else if (py > src.h && dist < 2.0) {
          // Above rack: hot exhaust continues rising
          forceY += strength * (0.006 + directionality * 0.006);
          const push = strength * 0.001;
          forceX += (dx / (dist + 0.1)) * push;
          forceZ += (dz / (dist + 0.1)) * push;
        } else if (py < 0.5 && dist < 3.5) {
          // Floor-level: cold air drawn toward server FRONT side
          const toFrontX = src.x + src.frontX * 0.6 - px;
          const toFrontZ = src.z + src.frontZ * 0.6 - pz;
          const toFrontDist = Math.sqrt(toFrontX * toFrontX + toFrontZ * toFrontZ) + 0.1;
          const pull = strength * 0.002 * directionality;
          forceX += (toFrontX / toFrontDist) * pull;
          forceZ += (toFrontZ / toFrontDist) * pull;

          // BAD LAYOUT: cold air wanders randomly, doesn't reach servers (bypass)
          if (chaos > 0.4) {
            const bypass = (chaos - 0.4) * 0.005;
            forceX += (Math.random() - 0.5) * bypass;
            forceZ += (Math.random() - 0.5) * bypass;
          }
        }
      }
    }

    // --- Cooling units ---
    for (const cool of coolingSources) {
      const dx = px - cool.position[0];
      const dz = pz - cool.position[2];
      const dist = Math.sqrt(dx * dx + dz * dz);

      if (py < 1.0 && dist < 4.0) {
        // COLD AIR OUTPUT: push outward along floor
        const blowStrength = (0.008 * loadFactor) / (1 + dist * 0.3);
        forceX += (dx / (dist + 0.1)) * blowStrength;
        forceZ += (dz / (dist + 0.1)) * blowStrength;
        forceY -= 0.002;
      }

      if (py > 1.5 && dist < 6.0) {
        // RETURN AIR: pulled toward CRAC intake
        // Good layouts: clean return path. Bad layouts: weak, air wanders.
        const pullStrength = ((0.003 + directionality * 0.003) * loadFactor) / (1 + dist * 0.3);
        forceX -= (dx / (dist + 0.1)) * pullStrength;
        forceZ -= (dz / (dist + 0.1)) * pullStrength;
        if (dist < 2.0) {
          forceY -= pullStrength * 1.5;
        }
      }
    }

    // Buoyancy
    if (heatInfluence > 0.2) {
      forceY += heatInfluence * 0.005;
    }

    // CEILING: hard barrier
    if (py > rh - 0.5) {
      const penetration = (py - (rh - 0.5)) / 0.5;
      forceY -= 0.015 + penetration * 0.02;
      for (const cool of coolingSources) {
        const dx = cool.position[0] - px;
        const dz = cool.position[2] - pz;
        const dist = Math.sqrt(dx * dx + dz * dz) + 0.1;
        forceX += (dx / dist) * 0.005;
        forceZ += (dz / dist) * 0.005;
      }
    }

    // Floor
    if (py < 0.1 && heatInfluence < 0.5) {
      forceY += 0.001;
    }

    // Walls
    if (px < 0.3) forceX += 0.004;
    if (px > rw - 0.3) forceX -= 0.004;
    if (pz < 0.3) forceZ += 0.004;
    if (pz > rd - 0.3) forceZ -= 0.004;

    // Apply forces with quality-based damping
    streamlineVelocities[idx] = streamlineVelocities[idx] * damping + forceX;
    streamlineVelocities[idx + 1] = streamlineVelocities[idx + 1] * damping + forceY;
    streamlineVelocities[idx + 2] = streamlineVelocities[idx + 2] * damping + forceZ;

    // Speed limit
    const speed = Math.sqrt(
      streamlineVelocities[idx] ** 2 +
        streamlineVelocities[idx + 1] ** 2 +
        streamlineVelocities[idx + 2] ** 2,
    );
    const maxSpeed = 0.07 * (0.5 + loadFactor);
    if (speed > maxSpeed) {
      const s = maxSpeed / speed;
      streamlineVelocities[idx] *= s;
      streamlineVelocities[idx + 1] *= s;
      streamlineVelocities[idx + 2] *= s;
    }

    // Turbulence: BAD layouts = much more chaotic motion
    streamlineVelocities[idx] += (Math.random() - 0.5) * turbulence * 2;
    streamlineVelocities[idx + 1] += (Math.random() - 0.5) * turbulence;
    streamlineVelocities[idx + 2] += (Math.random() - 0.5) * turbulence * 2;

    // Update position
    streamlineHeads[idx] += streamlineVelocities[idx];
    streamlineHeads[idx + 1] += streamlineVelocities[idx + 1];
    streamlineHeads[idx + 2] += streamlineVelocities[idx + 2];

    // Floor clamp
    if (streamlineHeads[idx + 1] < 0.02) {
      streamlineHeads[idx + 1] = 0.02;
      streamlineVelocities[idx + 1] = Math.abs(streamlineVelocities[idx + 1]) * 0.1;
    }

    // Ceiling clamp
    if (streamlineHeads[idx + 1] > rh - 0.05) {
      streamlineHeads[idx + 1] = rh - 0.05;
      streamlineVelocities[idx + 1] = -Math.abs(streamlineVelocities[idx + 1]) * 0.3;
    }

    // Record head position into trail ring buffer
    const cursor = trailHistoryIdx[i] % TRAIL_LENGTH;
    const hi = (i * TRAIL_LENGTH + cursor) * 3;
    trailHistory[hi] = streamlineHeads[idx];
    trailHistory[hi + 1] = streamlineHeads[idx + 1];
    trailHistory[hi + 2] = streamlineHeads[idx + 2];
    trailHistoryIdx[i]++;

    // === Build line segments from trail history ===
    const segBase = i * (TRAIL_LENGTH - 1);
    const currentCursor = trailHistoryIdx[i];

    for (let s = 0; s < TRAIL_LENGTH - 1; s++) {
      const vIdx = (segBase + s) * 2;

      const older = (currentCursor + s) % TRAIL_LENGTH;
      const newer = (currentCursor + s + 1) % TRAIL_LENGTH;

      const oldOff = (i * TRAIL_LENGTH + older) * 3;
      const newOff = (i * TRAIL_LENGTH + newer) * 3;

      trailPositions[vIdx * 3] = trailHistory[oldOff];
      trailPositions[vIdx * 3 + 1] = trailHistory[oldOff + 1];
      trailPositions[vIdx * 3 + 2] = trailHistory[oldOff + 2];

      trailPositions[(vIdx + 1) * 3] = trailHistory[newOff];
      trailPositions[(vIdx + 1) * 3 + 1] = trailHistory[newOff + 1];
      trailPositions[(vIdx + 1) * 3 + 2] = trailHistory[newOff + 2];

      // Color per-segment: local temperature at segment position
      const segX = trailHistory[oldOff];
      const segY = trailHistory[oldOff + 1];
      const segZ = trailHistory[oldOff + 2];

      let localHeat = 0;
      for (const src of cachedHeatSources) {
        const sdx = segX - src.x;
        const sdz = segZ - src.z;
        const sd = sdx * sdx + sdz * sdz;
        localHeat += src.heat / (1 + sd);
      }
      const heightTemp = Math.min(segY / rh, 1) * 0.35;
      const segTemp = Math.min(localHeat / 8 + heightTemp, 1);
      const [cr, cg, cb] = computeTemperatureColor(segTemp);

      const ageFade = s / (TRAIL_LENGTH - 1);
      const brightness = ageFade * ageFade * 0.9 + 0.1;

      trailColors[vIdx * 3] = cr * brightness;
      trailColors[vIdx * 3 + 1] = cg * brightness;
      trailColors[vIdx * 3 + 2] = cb * brightness;

      trailColors[(vIdx + 1) * 3] = cr * Math.min(brightness * 1.3, 1);
      trailColors[(vIdx + 1) * 3 + 1] = cg * Math.min(brightness * 1.3, 1);
      trailColors[(vIdx + 1) * 3 + 2] = cb * Math.min(brightness * 1.3, 1);
    }
  }

  trailGeometry.attributes.position.needsUpdate = true;
  trailGeometry.attributes.color.needsUpdate = true;
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

  // 히트맵 업데이트
  const maxHeat = buildHeatmap(sceneData, loadFactor);

  // Airflow 업데이트
  updateAirflow(sceneData, loadFactor);

  // 서버 랙 LED glow 업데이트
  updateEquipmentGlow(loadFactor);

  // 모니터 패널 업데이트
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
  // Load
  const loadPct = Math.round(loadFactor * 100);
  const loadEl = document.getElementById("mon-load")!;
  loadEl.innerHTML = `${loadPct}<span class="mon-unit">%</span>`;
  const loadBar = document.getElementById("mon-load-bar")!;
  loadBar.style.width = `${loadPct}%`;
  if (loadPct >= 80) loadBar.style.backgroundColor = "#ff1744";
  else if (loadPct >= 50) loadBar.style.backgroundColor = "#ffab00";
  else loadBar.style.backgroundColor = "#00e676";

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
  pueEl.style.color = pue <= 1.4 ? "#00e676" : pue <= 1.8 ? "#ffab00" : "#ff1744";
}

// ===== Equipment Builder (서버실용) =====
function createEquipmentMesh(item: Equipment): THREE.Group {
  const group = new THREE.Group();
  const [w, h, d] = item.size;

  if (item.category === "server_rack") {
    // 서버 랙 — 42U 캐비닛
    const cabinetMat = new THREE.MeshStandardMaterial({
      color: item.color,
      roughness: 0.4,
      metalness: 0.6,
    });

    // 메인 본체
    const bodyGeo = new THREE.BoxGeometry(w, h, d);
    const body = new THREE.Mesh(bodyGeo, cabinetMat);
    body.position.y = h / 2;
    group.add(body);

    // 전면 패널 (서버 유닛들)
    const unitCount = 8;
    const unitHeight = (h - 0.2) / unitCount;
    for (let i = 0; i < unitCount; i++) {
      const unitGeo = new THREE.BoxGeometry(w * 0.85, unitHeight * 0.8, 0.02);
      const isActive = Math.random() > 0.2;
      const unitMat = new THREE.MeshStandardMaterial({
        color: isActive ? 0x263238 : 0x1a1a1a,
        roughness: 0.3,
        metalness: 0.5,
      });
      const unit = new THREE.Mesh(unitGeo, unitMat);
      unit.position.set(0, 0.1 + unitHeight * (i + 0.5), d / 2 + 0.011);
      group.add(unit);

      // LED 표시등
      if (isActive) {
        const ledGeo = new THREE.BoxGeometry(0.02, 0.02, 0.01);
        const ledColor =
          item.heatOutput > 10 ? 0xff1744 : item.heatOutput > 5 ? 0xffab00 : 0x00e676;
        const ledMat = new THREE.MeshStandardMaterial({
          color: ledColor,
          emissive: ledColor,
          emissiveIntensity: 1.5,
        });
        const led = new THREE.Mesh(ledGeo, ledMat);
        led.position.set(w * 0.35, 0.1 + unitHeight * (i + 0.5), d / 2 + 0.016);
        group.add(led);
      }
    }

    // 상단 프레임
    const topFrame = new THREE.Mesh(
      new THREE.BoxGeometry(w + 0.02, 0.03, d + 0.02),
      new THREE.MeshStandardMaterial({ color: 0x212121, metalness: 0.7 }),
    );
    topFrame.position.y = h;
    group.add(topFrame);

    // Heat glow — exhaust face (back, -Z in local space) + top
    if (item.heatOutput > 0) {
      const glowColor = item.heatOutput > 10 ? 0xff3300 : item.heatOutput > 5 ? 0xff8800 : 0xffcc00;
      const glowOpacity = 0.15 + item.heatOutput * 0.01;

      // Back-face exhaust glow (local -Z = back)
      const backGlowGeo = new THREE.PlaneGeometry(w * 0.85, h * 0.8);
      const backGlowMat = new THREE.MeshBasicMaterial({
        color: glowColor,
        transparent: true,
        opacity: glowOpacity * 1.5,
        side: THREE.DoubleSide,
      });
      const backGlow = new THREE.Mesh(backGlowGeo, backGlowMat);
      backGlow.position.set(0, h / 2, -d / 2 - 0.02);
      group.add(backGlow);

      // Top glow (smaller, secondary)
      const topGlowGeo = new THREE.PlaneGeometry(w * 0.5, d * 0.5);
      const topGlowMat = new THREE.MeshBasicMaterial({
        color: glowColor,
        transparent: true,
        opacity: glowOpacity * 0.7,
        side: THREE.DoubleSide,
      });
      const topGlow = new THREE.Mesh(topGlowGeo, topGlowMat);
      topGlow.rotation.x = -Math.PI / 2;
      topGlow.position.y = h + 0.05;
      group.add(topGlow);
    }
  } else if (item.category === "cooling_unit") {
    // 냉각 장치 (CRAC)
    const bodyMat = new THREE.MeshStandardMaterial({
      color: item.color,
      roughness: 0.3,
      metalness: 0.4,
    });

    // 메인 본체
    const bodyGeo = new THREE.BoxGeometry(w, h, d);
    const body = new THREE.Mesh(bodyGeo, bodyMat);
    body.position.y = h / 2;
    group.add(body);

    // 전면 그릴 (냉각 공기 배출)
    for (let i = 0; i < 6; i++) {
      const grillGeo = new THREE.BoxGeometry(w * 0.8, 0.02, 0.01);
      const grillMat = new THREE.MeshStandardMaterial({
        color: 0x004d40,
        metalness: 0.5,
      });
      const grill = new THREE.Mesh(grillGeo, grillMat);
      grill.position.set(0, h * 0.3 + i * h * 0.08, d / 2 + 0.01);
      group.add(grill);
    }

    // 상단 팬
    const fanGeo = new THREE.CylinderGeometry(w * 0.3, w * 0.3, 0.05, 24);
    const fanMat = new THREE.MeshStandardMaterial({
      color: 0x004d40,
      metalness: 0.6,
    });
    const fan = new THREE.Mesh(fanGeo, fanMat);
    fan.position.y = h + 0.025;
    group.add(fan);

    // 냉각 효과 표시 (파란 glow)
    const coolGlow = new THREE.Mesh(
      new THREE.PlaneGeometry(w * 1.5, d * 1.5),
      new THREE.MeshBasicMaterial({
        color: 0x00bcd4,
        transparent: true,
        opacity: 0.28,
        side: THREE.DoubleSide,
      }),
    );
    coolGlow.rotation.x = -Math.PI / 2;
    coolGlow.position.y = 0.005;
    group.add(coolGlow);
  } else if (item.category === "network_switch") {
    // 네트워크 스위치/코어 스위치
    const bodyMat = new THREE.MeshStandardMaterial({
      color: item.color,
      roughness: 0.3,
      metalness: 0.5,
    });

    // 캐비닛
    const bodyGeo = new THREE.BoxGeometry(w, h, d);
    const body = new THREE.Mesh(bodyGeo, bodyMat);
    body.position.y = h / 2;
    group.add(body);

    // 포트 패널
    for (let row = 0; row < 4; row++) {
      for (let col = 0; col < 6; col++) {
        const portGeo = new THREE.BoxGeometry(0.03, 0.03, 0.01);
        const portMat = new THREE.MeshStandardMaterial({
          color: 0x00e676,
          emissive: 0x00e676,
          emissiveIntensity: 0.8,
        });
        const port = new THREE.Mesh(portGeo, portMat);
        port.position.set(-w * 0.3 + col * w * 0.12, h * 0.4 + row * h * 0.12, d / 2 + 0.011);
        group.add(port);
      }
    }

    // 상단 표시
    const labelGeo = new THREE.BoxGeometry(w * 0.6, 0.02, 0.01);
    const labelMat = new THREE.MeshStandardMaterial({
      color: 0x42a5f5,
      emissive: 0x42a5f5,
      emissiveIntensity: 0.5,
    });
    const label = new THREE.Mesh(labelGeo, labelMat);
    label.position.set(0, h * 0.85, d / 2 + 0.011);
    group.add(label);
  } else if (item.category === "ups") {
    // UPS 전원 장치
    const bodyMat = new THREE.MeshStandardMaterial({
      color: item.color,
      roughness: 0.4,
      metalness: 0.3,
    });

    const bodyGeo = new THREE.BoxGeometry(w, h, d);
    const body = new THREE.Mesh(bodyGeo, bodyMat);
    body.position.y = h / 2;
    group.add(body);

    // 전원 표시 LED
    const ledGeo = new THREE.BoxGeometry(w * 0.3, 0.04, 0.01);
    const ledMat = new THREE.MeshStandardMaterial({
      color: 0x00e676,
      emissive: 0x00e676,
      emissiveIntensity: 1.0,
    });
    const led = new THREE.Mesh(ledGeo, ledMat);
    led.position.set(0, h * 0.8, d / 2 + 0.011);
    group.add(led);

    // 배터리 표시 (전면)
    const battGeo = new THREE.BoxGeometry(w * 0.5, h * 0.3, 0.01);
    const battMat = new THREE.MeshStandardMaterial({
      color: 0xbf360c,
      roughness: 0.5,
    });
    const batt = new THREE.Mesh(battGeo, battMat);
    batt.position.set(0, h * 0.4, d / 2 + 0.011);
    group.add(batt);
  } else if (item.category === "pdu") {
    // PDU 분전반
    const bodyMat = new THREE.MeshStandardMaterial({
      color: item.color,
      roughness: 0.4,
      metalness: 0.4,
    });

    const bodyGeo = new THREE.BoxGeometry(w, h, d);
    const body = new THREE.Mesh(bodyGeo, bodyMat);
    body.position.y = h / 2;
    group.add(body);

    // 소켓 표시
    for (let i = 0; i < 6; i++) {
      const socketGeo = new THREE.BoxGeometry(0.04, 0.04, 0.01);
      const socketMat = new THREE.MeshStandardMaterial({ color: 0x212121 });
      const socket = new THREE.Mesh(socketGeo, socketMat);
      socket.position.set(0, 0.3 + i * 0.25, d / 2 + 0.011);
      group.add(socket);
    }
  } else if (item.category === "monitoring") {
    // 환경 모니터링 장비
    const bodyMat = new THREE.MeshStandardMaterial({
      color: item.color,
      roughness: 0.3,
      metalness: 0.3,
    });

    // 본체
    const bodyGeo = new THREE.BoxGeometry(w, h * 0.7, d);
    const body = new THREE.Mesh(bodyGeo, bodyMat);
    body.position.y = h * 0.35;
    group.add(body);

    // 화면
    const screenGeo = new THREE.BoxGeometry(w * 0.8, h * 0.35, 0.02);
    const screenMat = new THREE.MeshStandardMaterial({
      color: 0x0d47a1,
      emissive: 0x1565c0,
      emissiveIntensity: 0.5,
      roughness: 0.1,
    });
    const screen = new THREE.Mesh(screenGeo, screenMat);
    screen.position.set(0, h * 0.85, d * 0.15);
    screen.rotation.x = -0.2;
    group.add(screen);

    // 안테나
    const antennaGeo = new THREE.CylinderGeometry(0.01, 0.01, h * 0.3, 6);
    const antenna = new THREE.Mesh(antennaGeo, new THREE.MeshStandardMaterial({ color: 0x424242 }));
    antenna.position.set(w * 0.3, h * 0.85 + h * 0.15, 0);
    group.add(antenna);
  } else if (item.category === "cable_tray") {
    // 케이블 트레이 (바닥 위 낮은 구조물)
    const trayMat = new THREE.MeshStandardMaterial({
      color: item.color,
      roughness: 0.5,
      metalness: 0.4,
      transparent: true,
      opacity: 0.7,
    });

    // 트레이 바닥
    const trayGeo = new THREE.BoxGeometry(w, 0.05, d);
    const tray = new THREE.Mesh(trayGeo, trayMat);
    tray.position.y = 0.15;
    group.add(tray);

    // 가드레일 양쪽
    const railGeo = new THREE.BoxGeometry(w, 0.08, 0.02);
    const railMat = new THREE.MeshStandardMaterial({
      color: 0xf9a825,
      metalness: 0.5,
    });
    const rail1 = new THREE.Mesh(railGeo, railMat);
    rail1.position.set(0, 0.19, d / 2);
    group.add(rail1);

    const rail2 = new THREE.Mesh(railGeo, railMat);
    rail2.position.set(0, 0.19, -d / 2);
    group.add(rail2);

    // 케이블 묶음 시뮬레이션
    const cableGeo = new THREE.CylinderGeometry(0.04, 0.04, w * 0.9, 8);
    const cableMat = new THREE.MeshStandardMaterial({
      color: 0x212121,
      roughness: 0.8,
    });
    const cable = new THREE.Mesh(cableGeo, cableMat);
    cable.rotation.z = Math.PI / 2;
    cable.position.y = 0.19;
    group.add(cable);
  } else {
    // Fallback
    const geo = new THREE.BoxGeometry(w, h, d);
    const mat = new THREE.MeshStandardMaterial({
      color: item.color,
      roughness: 0.5,
    });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.y = h / 2;
    group.add(mesh);
  }

  // Shadow
  group.traverse((child) => {
    if (child instanceof THREE.Mesh) {
      child.castShadow = true;
      child.receiveShadow = true;
    }
  });

  // Position and rotation
  group.position.set(...item.position);
  group.rotation.set(
    THREE.MathUtils.degToRad(item.rotation[0]),
    THREE.MathUtils.degToRad(item.rotation[1]),
    THREE.MathUtils.degToRad(item.rotation[2]),
  );

  group.userData = {
    id: item.id,
    label: item.label,
    category: item.category,
    heatOutput: item.heatOutput,
  };
  return group;
}

// ===== Relation Lines =====
function drawRelations(sceneData: SceneGraph) {
  const posMap = new Map<string, THREE.Vector3>();
  for (const f of sceneData.furniture) {
    posMap.set(f.id, new THREE.Vector3(f.position[0], 0.5, f.position[2]));
  }

  for (const f of sceneData.furniture) {
    for (const rel of f.relations) {
      const from = posMap.get(f.id);
      const to = posMap.get(rel.target);
      if (!from || !to) continue;

      const color =
        rel.type === "cooling_serves"
          ? 0x00bcd4
          : rel.type === "cable_connected"
            ? 0xfdd835
            : rel.type === "hot_aisle"
              ? 0xff5722
              : rel.type === "cold_aisle"
                ? 0x2196f3
                : rel.type === "adjacent_to"
                  ? 0x4caf50
                  : 0xff9800;

      const lineGeo = new THREE.BufferGeometry().setFromPoints([from, to]);
      const lineMat = new THREE.LineDashedMaterial({
        color,
        dashSize: 0.2,
        gapSize: 0.12,
        linewidth: 1,
      });
      const line = new THREE.Line(lineGeo, lineMat);
      line.computeLineDistances();
      furnitureGroup.add(line);
    }
  }
}

// ===== Scene Build with Animation =====
function buildScene(sceneData: SceneGraph, withAnimation = true) {
  furnitureGroup.clear();
  buildRoom(sceneData);
  buildHeatmap(sceneData);
  initAirflow(sceneData);

  if (withAnimation) {
    animating = true;
    let i = 0;
    const interval = setInterval(() => {
      if (i >= sceneData.furniture.length) {
        drawRelations(sceneData);
        animating = false;
        clearInterval(interval);
        return;
      }

      const item = sceneData.furniture[i];
      const mesh = createEquipmentMesh(item);
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
      furnitureGroup.add(createEquipmentMesh(item));
    }
    drawRelations(sceneData);
  }
}

// ===== UI =====
function updateUI(sceneData: SceneGraph) {
  document.getElementById("scene-title")!.textContent = sceneData.name;
  document.getElementById("scene-desc")!.textContent = sceneData.description;

  document.querySelectorAll(".step").forEach((el, idx) => {
    el.classList.toggle("active", idx === currentSceneIndex);
  });

  updateScoreBars(sceneData.score);
  updateSceneGraphPanel(sceneData);
}

function updateScoreBars(score: Score) {
  const metrics: (keyof Score)[] = [
    "total",
    "thermal",
    "cooling",
    "cable",
    "proximity",
    "constraint",
  ];
  for (const key of metrics) {
    const bar = document.getElementById(`bar-${key}`);
    const value = document.getElementById(`val-${key}`);
    if (bar && value) {
      const pct = Math.round(score[key] * 100);
      bar.style.width = `${pct}%`;
      value.textContent = `${pct}`;
      if (pct >= 80) bar.style.backgroundColor = "#00e676";
      else if (pct >= 50) bar.style.backgroundColor = "#ffab00";
      else bar.style.backgroundColor = "#ff1744";
    }
  }
}

function updateSceneGraphPanel(sceneData: SceneGraph) {
  const panel = document.getElementById("sg-content")!;
  let html = `<div class="sg-room">Room: ${sceneData.room.type} (${sceneData.room.dimensions.join(" x ")}m)</div>`;
  html += `<div class="sg-nodes">`;

  for (const f of sceneData.furniture) {
    const relStr = f.relations
      .map((r) => `<span class="rel rel-${r.type}">${r.type} &rarr; ${r.target}</span>`)
      .join("");
    const heatStr = f.heatOutput > 0 ? `<div class="sg-heat">heat: ${f.heatOutput} kW</div>` : "";
    html += `
      <div class="sg-node">
        <div class="sg-node-header">
          <span class="sg-dot" style="background:${f.color}"></span>
          <strong>${f.label}</strong>
          <span class="sg-id">${f.id}</span>
        </div>
        <div class="sg-pos">pos: [${f.position.map((v) => v.toFixed(1)).join(", ")}] rot: ${f.rotation[1]}°</div>
        ${heatStr}
        ${relStr ? `<div class="sg-rels">${relStr}</div>` : ""}
      </div>`;
  }

  html += `</div>`;
  panel.innerHTML = html;
}

// ===== Chart =====
function drawChart() {
  const canvas = document.getElementById("chart") as HTMLCanvasElement;
  const ctx = canvas.getContext("2d")!;
  const dpr = window.devicePixelRatio || 1;

  canvas.width = canvas.clientWidth * dpr;
  canvas.height = canvas.clientHeight * dpr;
  ctx.scale(dpr, dpr);

  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  const padding = { top: 20, right: 20, bottom: 35, left: 40 };
  const plotW = w - padding.left - padding.right;
  const plotH = h - padding.top - padding.bottom;

  ctx.clearRect(0, 0, w, h);

  ctx.fillStyle = "#0d1220";
  ctx.fillRect(0, 0, w, h);

  ctx.strokeStyle = "#1a2744";
  ctx.lineWidth = 0.5;
  for (let i = 0; i <= 5; i++) {
    const y = padding.top + (plotH / 5) * i;
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(w - padding.right, y);
    ctx.stroke();
  }

  ctx.fillStyle = "#aaa";
  ctx.font = "10px monospace";
  ctx.textAlign = "right";
  for (let i = 0; i <= 5; i++) {
    const y = padding.top + (plotH / 5) * i;
    ctx.fillText((1 - i / 5).toFixed(1), padding.left - 5, y + 3);
  }

  ctx.textAlign = "center";
  const eps = trainingHistory.episodes;
  const maxEp = eps[eps.length - 1];
  for (let i = 0; i < eps.length; i += 2) {
    const x = padding.left + (eps[i] / maxEp) * plotW;
    ctx.fillText(eps[i] >= 1000 ? `${eps[i] / 1000}k` : `${eps[i]}`, x, h - padding.bottom + 15);
  }

  ctx.fillText("Episodes", w / 2, h - 3);

  ctx.save();
  ctx.translate(10, h / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("Score", 0, 0);
  ctx.restore();

  const colors: Record<string, string> = {
    total: "#fff",
    thermal: "#ff5722",
    cooling: "#00bcd4",
    cable: "#fdd835",
    proximity: "#4caf50",
    constraint: "#ff9800",
  };

  for (const [key, color] of Object.entries(colors)) {
    const data = trainingHistory.scores[key as keyof typeof trainingHistory.scores];
    ctx.strokeStyle = color;
    ctx.lineWidth = key === "total" ? 2.5 : 1.5;
    ctx.beginPath();
    for (let i = 0; i < eps.length; i++) {
      const x = padding.left + (eps[i] / maxEp) * plotW;
      const y = padding.top + (1 - data[i]) * plotH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  const legendX = padding.left + 10;
  let legendY = padding.top + 10;
  ctx.font = "10px monospace";
  const labels: Record<string, string> = {
    total: "Total",
    thermal: "Thermal",
    cooling: "Cooling",
    cable: "Cable",
    proximity: "Proximity",
    constraint: "Constraint",
  };
  for (const [key, color] of Object.entries(colors)) {
    ctx.fillStyle = color;
    ctx.fillRect(legendX, legendY - 4, 12, 3);
    ctx.fillStyle = "#ccc";
    ctx.textAlign = "left";
    ctx.fillText(labels[key], legendX + 16, legendY);
    legendY += 14;
  }
}

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
    if (animating || currentSceneIndex >= 3) return;  // 4 steps total (0-3)
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

  document.getElementById("btn-relations")?.addEventListener("click", () => {
    const btn = document.getElementById("btn-relations")!;
    const showing = btn.dataset.showing === "true";
    furnitureGroup.children.forEach((child) => {
      if (child instanceof THREE.Line) child.visible = !showing;
    });
    btn.dataset.showing = showing ? "false" : "true";
    btn.textContent = showing ? "Show Relations" : "Hide Relations";
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
  const isScan = idx >= allScenes.length;  // index 3 = Step 4

  // Update step indicators
  document.querySelectorAll(".step").forEach((el, i) => {
    el.classList.toggle("active", i === idx);
  });

  // Toggle scan-specific UI
  const scanPanel = document.getElementById("scan-upload-panel")!;
  const voxelCtrl = document.getElementById("voxel-controls")!;
  const ashraeSection = document.getElementById("ashrae-section")!;
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
    ashraeSection.style.display = scanMetricsCache ? "block" : "none";
    simBar.style.display = "none";
    viewportBtns.style.display = "none";

    // Hide hardcoded scene stuff
    roomGroup.visible = false;
    furnitureGroup.visible = false;
    heatmapGroup.visible = false;
    airflowGroup.visible = false;
    zonesGroup.visible = false;

    // Update panel info
    document.getElementById("scene-title")!.textContent = "Real Room Scan";
    document.getElementById("scene-desc")!.textContent =
      "Upload a .obj scan or load the demo room to analyze thermal performance.";
  } else {
    // Standard scene mode
    scanPanel.style.display = "none";
    voxelCtrl.style.display = "none";
    ashraeSection.style.display = "none";
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
  x: number; y: number; z: number; heat: number; h: number;
  frontX: number; frontZ: number; backX: number; backZ: number;
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
  position: number[];   // [x, y, z] centre-bottom, Z-up
  size: number[];        // [w, d, h] metres
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

// --- Semantic label colors (match engine/core/config.py) ---
const LABEL_COLORS: Record<number, number> = {
  0: 0x1a2744,  // SPACE_EMPTY (dim, mostly hidden)
  1: 0xb0bec5,  // OBSTACLE_WALL
  2: 0xff9800,  // HEAT_LEGACY_SERVER
  3: 0x00bcd4,  // COOLING_AC_VENT
  4: 0x4caf50,  // HUMAN_WORKSPACE
  5: 0x37474f,  // RACK_BODY
  6: 0x2196f3,  // RACK_INTAKE
  7: 0xff5722,  // RACK_EXHAUST
};

function tempToColorScan(t: number, _minT: number, _maxT: number): THREE.Color {
  // Map to ASHRAE recommended range so thermal contrast is visible.
  // 18 °C (rec low) = pure blue, 27 °C (rec high) = pure red.
  // Below 18 = deep blue, above 27 = bright red/white (over-temp emphasis).
  const lo = 18, hi = 27;
  const n = Math.max(0, Math.min(1, (t - lo) / (hi - lo)));  // 0..1

  // Blue → Cyan → Green → Yellow → Red
  let r: number, g: number, b: number;
  if (n < 0.25) {
    r = 0; g = n * 4; b = 1;
  } else if (n < 0.5) {
    r = 0; g = 1; b = 1 - (n - 0.25) * 4;
  } else if (n < 0.75) {
    r = (n - 0.5) * 4; g = 1; b = 0;
  } else {
    r = 1; g = 1 - (n - 0.75) * 4; b = 0;
  }

  // Over-temp glow: above ASHRAE max push towards bright white-red
  if (t > hi) {
    const excess = Math.min((t - hi) / 15, 1);  // saturates 15 °C above
    r = r + (1 - r) * excess * 0.5;
    g = g * (1 - excess * 0.7);
    b = b + excess * 0.3;
  }

  return new THREE.Color(r, g, b);
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
  const rw = voxel.shape[0] * vs;  // X width
  const rh = voxel.shape[2] * vs;  // Z height → Y in Three.js
  const rd = voxel.shape[1] * vs;  // Y depth → Z in Three.js

  // ─── Build room shell (floor, walls, pillars) ───
  const roomSub = new THREE.Group();
  roomSub.name = "scanRoom";

  // Floor
  const floorGeo = new THREE.PlaneGeometry(rw, rd);
  const floorMat = new THREE.MeshStandardMaterial({ color: 0xd0d0d0, roughness: 0.7 });
  const floor = new THREE.Mesh(floorGeo, floorMat);
  floor.rotation.x = -Math.PI / 2;
  floor.position.set(ox + rw / 2, oz + 0.001, oy + rd / 2);
  floor.receiveShadow = true;
  roomSub.add(floor);

  // Tile grid on floor
  const tileMat = new THREE.LineBasicMaterial({ color: 0xb0b0b0 });
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
    color: 0xe0e0e0, transparent: true, opacity: 0.15, side: THREE.DoubleSide,
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
  const edgeMat = new THREE.LineBasicMaterial({ color: 0x00bcd4, linewidth: 2 });
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
  const pillarMat = new THREE.MeshStandardMaterial({ color: 0x90a4ae });
  for (const [px, pz] of [[ox, oy], [ox + rw, oy], [ox + rw, oy + rd], [ox, oy + rd]]) {
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

    for (const item of resp.equipment) {
      // Convert backend (Z-up) to Three.js (Y-up) Equipment
      const [bx, by, bz] = item.position;
      const [bw, bd, bh] = item.size;

      // Facing → rotation.y in Three.js
      let rotY = 0;
      if (item.facing === "+y") rotY = 0;        // +Y (backend) → +Z (Three.js front)
      else if (item.facing === "-y") rotY = 180;
      else if (item.facing === "+x") rotY = 90;
      else if (item.facing === "-x") rotY = -90;

      const equipItem: Equipment = {
        id: item.id,
        category: item.category,
        label: item.label,
        position: [bx, bz, by],             // Z-up→Y-up: X=X, Y=Z, Z=Y
        rotation: [0, rotY, 0],
        size: [bw, bh, bd],                  // [width, height, depth]
        color: item.color,
        heatOutput: item.heat_output,
        relations: [],
      };

      const mesh = createEquipmentMesh(equipItem);
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
      const depth = bd;  // rack depth along facing axis
      const height = bh;

      if (item.category === "server_rack" && item.facing) {
        // Front direction in Three.js (facing maps: +y→+Z, -y→-Z, +x→+X, -x→-X)
        let frontX = 0, frontZ = 0;
        if (item.facing === "+y") frontZ = 1;
        else if (item.facing === "-y") frontZ = -1;
        else if (item.facing === "+x") frontX = 1;
        else if (item.facing === "-x") frontX = -1;

        const backX = -frontX, backZ = -frontZ;

        // Intake arrows (cold blue) – at front face, pointing into rack
        const intakeDir = new THREE.Vector3(-frontX, 0, -frontZ);
        for (const yFrac of [0.25, 0.5, 0.75]) {
          const origin = new THREE.Vector3(
            px + frontX * (depth / 2 + 0.5),
            py + height * yFrac,
            pz + frontZ * (depth / 2 + 0.5),
          );
          arrowGroup.add(
            new THREE.ArrowHelper(intakeDir, origin, 0.7, 0x44aaff, 0.15, 0.1),
          );
        }

        // Exhaust arrows (hot red) – at back face, pointing horizontally out
        const exhaustDir = new THREE.Vector3(backX, 0, backZ);
        for (const yFrac of [0.25, 0.5, 0.75]) {
          const origin = new THREE.Vector3(
            px + backX * (depth / 2 + 0.1),
            py + height * yFrac,
            pz + backZ * (depth / 2 + 0.1),
          );
          arrowGroup.add(
            new THREE.ArrowHelper(exhaustDir, origin, 0.7, 0xff3300, 0.15, 0.1),
          );
        }
      } else if (item.category === "cooling_unit") {
        // AC arrows (cyan) – cold air blows downward and outward at floor level
        arrowGroup.add(
          new THREE.ArrowHelper(
            new THREE.Vector3(0, -1, 0),
            new THREE.Vector3(px, py + 0.3, pz),
            0.8, 0x00bcd4, 0.2, 0.12,
          ),
        );
        const halfW = bw / 2 + 0.15;
        const halfD = bd / 2 + 0.15;
        for (const [dx, dz] of [[halfW, 0], [-halfW, 0], [0, halfD], [0, -halfD]] as [number, number][]) {
          const dir = new THREE.Vector3(Math.sign(dx), -0.3, Math.sign(dz)).normalize();
          arrowGroup.add(
            new THREE.ArrowHelper(
              dir,
              new THREE.Vector3(px + dx, py + 0.2, pz + dz),
              0.6, 0x00bcd4, 0.15, 0.1,
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
      transparent: true, opacity: 0.35,
      depthWrite: false,
    });
    const tMesh = new THREE.InstancedMesh(boxGeo, tMat, tCount);
    tMesh.name = "thermalVoxels";
    tMesh.visible = false;

    for (let i = 0; i < tCount; i++) {
      const [ix, iy, iz] = resp.thermal.positions[i];
      dummy.position.set(
        ox + ix * vs + vs / 2,
        oz + iz * vs + vs / 2,
        oy + iy * vs + vs / 2,
      );
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

  // Show ASHRAE section
  if (resp.metrics) {
    renderASHRAEPanel(resp.metrics, resp.thermal);
    document.getElementById("ashrae-section")!.style.display = "block";
  }
}

function renderASHRAEPanel(metrics: ScanMetrics, thermal: ThermalData | null) {
  const grid = document.getElementById("ashrae-grid")!;
  const rciHiColor = metrics.room.rci_hi >= 95 ? "#00e676" : metrics.room.rci_hi >= 80 ? "#ffab00" : "#ff1744";
  const rciLoColor = metrics.room.rci_lo >= 95 ? "#00e676" : metrics.room.rci_lo >= 80 ? "#ffab00" : "#ff1744";

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
      <div class="mon-value" style="color:#ff9800">${metrics.room.shi.toFixed(3)}</div>
    </div>
    <div class="monitor-card">
      <div class="mon-label">RHI</div>
      <div class="mon-value" style="color:#00bcd4">${metrics.room.rhi.toFixed(3)}</div>
    </div>
    <div class="monitor-card">
      <div class="mon-label">Mean Intake</div>
      <div class="mon-value" style="color:#2196f3">${metrics.room.mean_intake.toFixed(1)}<span class="mon-unit">&deg;C</span></div>
    </div>
    <div class="monitor-card">
      <div class="mon-label">Mean Exhaust</div>
      <div class="mon-value" style="color:#ff5722">${metrics.room.mean_exhaust.toFixed(1)}<span class="mon-unit">&deg;C</span></div>
    </div>
    ${thermal ? `
    <div class="monitor-card">
      <div class="mon-label">Min Temp</div>
      <div class="mon-value" style="color:#2196f3">${thermal.min_temp.toFixed(1)}<span class="mon-unit">&deg;C</span></div>
    </div>
    <div class="monitor-card">
      <div class="mon-label">Max Temp</div>
      <div class="mon-value" style="color:#ff5722">${thermal.max_temp.toFixed(1)}<span class="mon-unit">&deg;C</span></div>
    </div>` : ""}
  `;

  // Per-rack details
  const racksDiv = document.getElementById("ashrae-racks")!;
  if (metrics.racks.length > 0) {
    let html = '<div style="font-size:11px;color:#888;margin-bottom:6px;">Per-Rack Details</div>';
    for (const r of metrics.racks) {
      // Intake icon: ASHRAE recommended 18-27 °C / allowable 15-35 °C
      const inIcon = r.inlet_compliant ? "✅" : r.inlet_within_allowable ? "⚠️" : "🔥";
      const inColor = r.inlet_compliant ? "#00e676" : r.inlet_within_allowable ? "#ffab00" : "#ff1744";
      // Exhaust / ΔT icon: ΔT ≤ 15 good, ≤ 20 warning, > 20 critical
      const dtOk = r.delta_t <= 15;
      const dtWarn = r.delta_t <= 20;
      const exIcon = dtOk ? "✅" : dtWarn ? "⚠️" : "🔥";
      const exColor = dtOk ? "#00e676" : dtWarn ? "#ffab00" : "#ff1744";
      html += `
        <div style="background:#141d30;border:1px solid #1a2744;border-radius:6px;padding:8px 10px;margin-bottom:6px;font-size:11px;">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
            <span style="color:#ccc;font-weight:600;">Rack ${r.rack_index}</span>
            <span style="display:flex;gap:8px;font-size:10px;">
              <span title="Intake: ASHRAE 18-27°C">${inIcon}<span style="color:${inColor};margin-left:2px;">In</span></span>
              <span title="Exhaust ΔT: ≤15°C good">∣</span>
              <span title="ΔT: ${r.delta_t.toFixed(1)}°C">${exIcon}<span style="color:${exColor};margin-left:2px;">ΔT</span></span>
            </span>
          </div>
          <div style="display:flex;justify-content:space-between;font-family:monospace;">
            <span style="color:${inColor};">In: ${r.intake_temp.toFixed(1)}°C</span>
            <span style="color:${exColor};">Out: ${r.exhaust_temp.toFixed(1)}°C</span>
            <span style="color:${exColor};">ΔT ${r.delta_t.toFixed(1)}°C</span>
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
  if (!resp.thermal || !resp.voxel_grid) { scanTempGrid = null; return; }
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
  const iy = Math.floor((wz - oy) / vs);  // Three.js Z = backend Y
  const iz = Math.floor((wy - oz) / vs);  // Three.js Y = backend Z
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
    const px = bx, py = bz, pz = by;

    // All solid equipment acts as airflow obstacle
    scanObstacles.push({ x: px, y: py, z: pz, hw: bw / 2, hh: bh, hd: bd / 2 });

    if (item.category === "server_rack" && item.facing) {
      let frontX = 0, frontZ = 0;
      if (item.facing === "+y") frontZ = 1;
      else if (item.facing === "-y") frontZ = -1;
      else if (item.facing === "+x") frontX = 1;
      else if (item.facing === "-x") frontX = -1;

      scanHeatSources.push({
        x: px, y: py, z: pz,
        heat: item.heat_output / 1000,
        h: bh,
        frontX, frontZ,
        backX: -frontX, backZ: -frontZ,
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
    const hx = scanHeads[i * 3], hy = scanHeads[i * 3 + 1], hz = scanHeads[i * 3 + 2];
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
    vertexColors: true, transparent: true, opacity: 0.7, depthWrite: false,
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
    scanHeads[idx]     = src.x + src.backX * backOff + (Math.random() - 0.5) * 0.25;
    scanHeads[idx + 1] = src.y + 0.3 + Math.random() * (src.h - 0.3);
    scanHeads[idx + 2] = src.z + src.backZ * backOff + (Math.random() - 0.5) * 0.25;
    const speed = 0.008 + src.heat * 0.004;
    scanVels[idx]     = src.backX * speed + (Math.random() - 0.5) * 0.002;
    scanVels[idx + 1] = 0.006 + Math.random() * 0.008;
    scanVels[idx + 2] = src.backZ * speed + (Math.random() - 0.5) * 0.002;

  } else if (roll < 0.80 && scanCoolSources.length > 0) {
    // Spawn at cooling unit — cold air along floor
    const cool = scanCoolSources[Math.floor(Math.random() * scanCoolSources.length)];
    scanHeads[idx]     = cool.x + (Math.random() - 0.5) * cool.w;
    scanHeads[idx + 1] = oz + 0.05 + Math.random() * 0.3;
    scanHeads[idx + 2] = cool.z + (Math.random() - 0.5) * cool.d;
    let driftX = (Math.random() - 0.5) * 0.01;
    let driftZ = (Math.random() - 0.5) * 0.01;
    if (scanHeatSources.length > 0) {
      const tgt = scanHeatSources[Math.floor(Math.random() * scanHeatSources.length)];
      const dx = (tgt.x + tgt.frontX * 0.5) - cool.x;
      const dz = (tgt.z + tgt.frontZ * 0.5) - cool.z;
      const d = Math.sqrt(dx * dx + dz * dz) + 0.1;
      driftX = (dx / d) * 0.015 + (Math.random() - 0.5) * 0.003;
      driftZ = (dz / d) * 0.015 + (Math.random() - 0.5) * 0.003;
    }
    scanVels[idx]     = driftX;
    scanVels[idx + 1] = -0.001;
    scanVels[idx + 2] = driftZ;

  } else {
    // Spawn at ceiling — return air descending toward CRACs
    scanHeads[idx]     = ox + Math.random() * rw;
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
    scanVels[idx]     = driftX;
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
  const hx = scanHeads[idx], hy = scanHeads[idx + 1], hz = scanHeads[idx + 2];
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
  const xMin = ox, xMax = ox + rw;
  const yMin = oz, yMax = oz + rh;
  const zMin = oy, zMax = oy + rd;

  for (let i = 0; i < SCAN_PARTICLE_COUNT; i++) {
    const idx = i * 3;
    scanAges[i]++;

    const px = scanHeads[idx];
    const py = scanHeads[idx + 1];
    const pz = scanHeads[idx + 2];

    // Respawn check
    if (scanAges[i] > scanMaxAges[i] || py > yMax + 0.3 || py < yMin - 0.1 ||
        px < xMin - 0.5 || px > xMax + 0.5 || pz < zMin - 0.5 || pz > zMax + 0.5) {
      resetScanParticle(i);
      continue;
    }

    let forceX = 0, forceY = 0, forceZ = 0;
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
        const overlapYhi = (obs.y + obs.hh + 0.15) - py;
        const overlapY = Math.min(overlapYlo, overlapYhi);
        if (overlapX < overlapZ && overlapX < overlapY) {
          forceX += Math.sign(dx) * 0.015;
          scanVels[idx] = Math.sign(dx) * Math.abs(scanVels[idx]) * 0.3;
        } else if (overlapZ < overlapY) {
          forceZ += Math.sign(dz) * 0.015;
          scanVels[idx + 2] = Math.sign(dz) * Math.abs(scanVels[idx + 2]) * 0.3;
        } else {
          forceY += (overlapYhi < overlapYlo ? -1 : 1) * 0.015;
          scanVels[idx + 1] = (overlapYhi < overlapYlo ? -1 : 1) * Math.abs(scanVels[idx + 1]) * 0.3;
        }
      }
    }

    // Damping + force integration
    const damping = 0.94;
    scanVels[idx]     = scanVels[idx] * damping + forceX;
    scanVels[idx + 1] = scanVels[idx + 1] * damping + forceY;
    scanVels[idx + 2] = scanVels[idx + 2] * damping + forceZ;

    // Speed limit
    const speed = Math.sqrt(scanVels[idx] ** 2 + scanVels[idx + 1] ** 2 + scanVels[idx + 2] ** 2);
    if (speed > 0.06) {
      const s = 0.06 / speed;
      scanVels[idx] *= s; scanVels[idx + 1] *= s; scanVels[idx + 2] *= s;
    }

    // Turbulence
    scanVels[idx]     += (Math.random() - 0.5) * 0.0008;
    scanVels[idx + 1] += (Math.random() - 0.5) * 0.0004;
    scanVels[idx + 2] += (Math.random() - 0.5) * 0.0008;

    // Integrate position
    scanHeads[idx]     += scanVels[idx];
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
    scanTrailHist[hi]     = scanHeads[idx];
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

      scanTrailPos[vIdx * 3]     = scanTrailHist[oldOff];
      scanTrailPos[vIdx * 3 + 1] = scanTrailHist[oldOff + 1];
      scanTrailPos[vIdx * 3 + 2] = scanTrailHist[oldOff + 2];
      scanTrailPos[(vIdx + 1) * 3]     = scanTrailHist[newOff];
      scanTrailPos[(vIdx + 1) * 3 + 1] = scanTrailHist[newOff + 1];
      scanTrailPos[(vIdx + 1) * 3 + 2] = scanTrailHist[newOff + 2];

      // Color from solver temperature at this position
      const segT = sampleScanTemp(scanTrailHist[oldOff], scanTrailHist[oldOff + 1], scanTrailHist[oldOff + 2]);
      const c = tempToColorScan(segT, scanTempMin, scanTempMax);

      const ageFade = s / (SCAN_TRAIL_LEN - 1);
      const brightness = ageFade * ageFade * 0.85 + 0.15;

      scanTrailCol[vIdx * 3]     = c.r * brightness;
      scanTrailCol[vIdx * 3 + 1] = c.g * brightness;
      scanTrailCol[vIdx * 3 + 2] = c.b * brightness;
      scanTrailCol[(vIdx + 1) * 3]     = c.r * Math.min(brightness * 1.3, 1);
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
      child.visible = true;  // always show room shell
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
  semBtn.style.borderColor = mode === "semantic" ? "#00bcd4" : "#1a2744";
  semBtn.style.color = mode === "semantic" ? "#00bcd4" : "#ccc";
  therBtn.style.borderColor = mode === "thermal" ? "#ff5722" : "#1a2744";
  therBtn.style.color = mode === "thermal" ? "#ff5722" : "#ccc";
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
    (btn as HTMLElement).style.borderColor = scanAirflowVisible ? "#4caf50" : "#1a2744";
    (btn as HTMLElement).style.color = scanAirflowVisible ? "#4caf50" : "#ccc";
  }
}

function applyZSlice(maxZ: number) {
  const data = scanVoxelGroup.userData as {
    voxelData: VoxelData; thermalData: ThermalData | null;
  } | undefined;
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
  el.className = "scan-status" + (isError ? " error" : "");
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
