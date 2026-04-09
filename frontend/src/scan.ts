/**
 * Scan Viewer — upload .obj, call /api/v1/visualize, render GLB in Three.js.
 */
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

// ── DOM refs ──────────────────────────────────────────────
const canvas = document.getElementById("viewer") as HTMLCanvasElement;
const fileInput = document.getElementById("obj-file") as HTMLInputElement;
const metaInput = document.getElementById("metadata") as HTMLTextAreaElement;
const uploadBtn = document.getElementById("upload-btn") as HTMLButtonElement;
const statusEl = document.getElementById("status") as HTMLDivElement;
const stageBtns = document.getElementById("stage-btns") as HTMLDivElement;
const legendEl = document.getElementById("legend") as HTMLDivElement;
const metricsPanel = document.getElementById("metrics-panel") as HTMLDivElement;
const demoBtn = document.getElementById("demo-btn") as HTMLButtonElement;

// ── Three.js setup ────────────────────────────────────────
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setClearColor(0x0a0e17);
renderer.shadowMap.enabled = true;

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 200);
// Z-up: camera looks down from +Z side
camera.up.set(0, 0, 1);
camera.position.set(8, 8, 6);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.08;

// Lights
scene.add(new THREE.AmbientLight(0xffffff, 0.6));
const dirLight = new THREE.DirectionalLight(0xffffff, 1.0);
dirLight.position.set(5, 7, 10);
dirLight.castShadow = true;
scene.add(dirLight);

// Grid helper — lies on XY plane (Z-up)
const grid = new THREE.GridHelper(20, 40, 0x1a2744, 0x111827);
grid.rotation.x = Math.PI / 2; // rotate from XZ-plane to XY-plane
scene.add(grid);

// Axes
const axes = new THREE.AxesHelper(3);
scene.add(axes);

// Resize handling
function resize() {
  const parent = canvas.parentElement!;
  const w = parent.clientWidth;
  const h = parent.clientHeight;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);
resize();

// Render loop
function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
animate();

// ── GLB storage ───────────────────────────────────────────
const glbLoader = new GLTFLoader();
let meshGroup: THREE.Group | null = null;

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

interface MetricsData {
  racks: RackMetrics[];
  room: RoomMetrics;
}

interface VisualizeResult {
  raw_glb: string;
  cleaned_glb: string;
  semantic_glb: string | null;
  voxel_grid: VoxelData | null;
  thermal: ThermalData | null;
  metrics: MetricsData | null;
}
let result: VisualizeResult | null = null;

// Semantic label → color mapping (matches legend)
const LABEL_COLORS: Record<number, THREE.Color> = {
  1: new THREE.Color(0xb4b4b4), // wall
  2: new THREE.Color(0xe63c3c), // legacy server (heat)
  3: new THREE.Color(0x3c8ce6), // AC vent (cooling)
  4: new THREE.Color(0x3cc864), // human workspace
  5: new THREE.Color(0x555555), // rack body
  6: new THREE.Color(0x00bcd4), // rack intake
  7: new THREE.Color(0xff9800), // rack exhaust
};

// ── Helpers ───────────────────────────────────────────────
function base64ToArrayBuffer(b64: string): ArrayBuffer {
  const binary = atob(b64);
  const buf = new ArrayBuffer(binary.length);
  const view = new Uint8Array(buf);
  for (let i = 0; i < binary.length; i++) {
    view[i] = binary.charCodeAt(i);
  }
  return buf;
}

function clearMesh() {
  if (meshGroup) {
    scene.remove(meshGroup);
    meshGroup.traverse((child) => {
      if (child instanceof THREE.Mesh) {
        child.geometry.dispose();
        if (Array.isArray(child.material)) {
          child.material.forEach((m) => m.dispose());
        } else {
          child.material.dispose();
        }
      }
    });
    meshGroup = null;
  }
}

function loadGlb(b64: string) {
  clearMesh();
  const buf = base64ToArrayBuffer(b64);

  glbLoader.parse(
    buf,
    "",
    (gltf) => {
      meshGroup = new THREE.Group();
      gltf.scene.traverse((child) => {
        if (child instanceof THREE.Mesh) {
          child.castShadow = true;
          child.receiveShadow = true;
          // If no vertex colors, apply a default material
          if (!child.geometry.attributes.color) {
            child.material = new THREE.MeshStandardMaterial({
              color: 0x88aacc,
              roughness: 0.6,
              metalness: 0.1,
              side: THREE.DoubleSide,
              transparent: true,
              opacity: 0.35,
              depthWrite: false,
            });
          } else {
            child.material = new THREE.MeshStandardMaterial({
              vertexColors: true,
              roughness: 0.5,
              metalness: 0.1,
              side: THREE.DoubleSide,
              transparent: true,
              opacity: 0.5,
              depthWrite: false,
            });
          }
        }
      });
      meshGroup!.add(gltf.scene);
      scene.add(meshGroup!);

      // Auto-fit camera
      const box = new THREE.Box3().setFromObject(meshGroup!);
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3());
      const maxDim = Math.max(size.x, size.y, size.z);
      camera.position.copy(
        center.clone().add(new THREE.Vector3(maxDim, maxDim * 0.8, maxDim))
      );
      controls.target.copy(center);
      controls.update();
    },
    (err) => {
      setStatus(`GLB parse error: ${err}`, true);
    }
  );
}

function setStatus(msg: string, isError = false) {
  statusEl.textContent = msg;
  statusEl.className = isError ? "status error" : "status success";
}

// ── Voxel grid renderer ──────────────────────────────────
function loadVoxelGrid(data: VoxelData) {
  clearMesh();
  meshGroup = new THREE.Group();

  const vs = data.voxel_size;
  const geo = new THREE.BoxGeometry(vs * 0.92, vs * 0.92, vs * 0.92); // slight gap between voxels
  const count = data.positions.length;
  if (count === 0) {
    setStatus("Voxel grid is empty", true);
    return;
  }

  const instanced = new THREE.InstancedMesh(
    geo,
    new THREE.MeshStandardMaterial({
      roughness: 0.6,
      metalness: 0.1,
      vertexColors: false,
      transparent: true,
      opacity: 0.25,
      depthWrite: false,
    }),
    count
  );

  const dummy = new THREE.Object3D();
  const color = new THREE.Color();

  for (let i = 0; i < count; i++) {
    const [ix, iy, iz] = data.positions[i];
    const label = data.labels[i];

    // Position: origin + index * voxel_size (center of voxel)
    dummy.position.set(
      data.origin[0] + (ix + 0.5) * vs,
      data.origin[1] + (iy + 0.5) * vs,
      data.origin[2] + (iz + 0.5) * vs
    );
    dummy.updateMatrix();
    instanced.setMatrixAt(i, dummy.matrix);

    color.copy(LABEL_COLORS[label] ?? new THREE.Color(0x888888));
    instanced.setColorAt(i, color);
  }

  instanced.instanceMatrix.needsUpdate = true;
  if (instanced.instanceColor) instanced.instanceColor.needsUpdate = true;
  instanced.castShadow = true;
  instanced.receiveShadow = true;

  meshGroup.add(instanced);
  scene.add(meshGroup);

  // Auto-fit camera
  const box = new THREE.Box3().setFromObject(meshGroup);
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const maxDim = Math.max(size.x, size.y, size.z);
  camera.position.copy(
    center.clone().add(new THREE.Vector3(maxDim, maxDim * 0.8, maxDim))
  );
  controls.target.copy(center);
  controls.update();
}

// ── Thermal heatmap renderer ─────────────────────────────
function tempToColor(t: number, minT: number, maxT: number): THREE.Color {
  // Map temperature to [0,1], then apply a power curve to emphasize
  // differences around ambient (the middle of the range).
  let frac = maxT > minT ? (t - minT) / (maxT - minT) : 0.5;
  frac = Math.max(0, Math.min(1, frac));
  // Slight gamma to spread out the mid-range
  frac = Math.pow(frac, 0.8);

  // 5-stop colour ramp: Blue → Cyan → Green → Yellow → Red
  const c = new THREE.Color();
  if (frac < 0.25) {
    c.setRGB(0, frac * 4, 1);                    // blue → cyan
  } else if (frac < 0.5) {
    c.setRGB(0, 1, 1 - (frac - 0.25) * 4);      // cyan → green
  } else if (frac < 0.75) {
    c.setRGB((frac - 0.5) * 4, 1, 0);            // green → yellow
  } else {
    c.setRGB(1, 1 - (frac - 0.75) * 4, 0);      // yellow → red
  }
  return c;
}

function loadThermalGrid(voxel: VoxelData, thermal: ThermalData) {
  clearMesh();
  meshGroup = new THREE.Group();

  const vs = voxel.voxel_size;
  const geo = new THREE.BoxGeometry(vs * 0.92, vs * 0.92, vs * 0.92);

  // Filter out wall voxels (label 1) — they block the interior and are
  // all at ambient anyway.  Build a map from thermal index → voxel label.
  const labelMap = new Map<string, number>();
  for (let i = 0; i < voxel.positions.length; i++) {
    const [ix, iy, iz] = voxel.positions[i];
    labelMap.set(`${ix},${iy},${iz}`, voxel.labels[i]);
  }

  const indices: number[] = [];
  for (let i = 0; i < thermal.positions.length; i++) {
    const [ix, iy, iz] = thermal.positions[i];
    const label = labelMap.get(`${ix},${iy},${iz}`) ?? 0;
    if (label !== 1) indices.push(i);  // skip walls
  }

  const count = indices.length;
  if (count === 0) {
    setStatus("Thermal data is empty", true);
    return;
  }

  const instanced = new THREE.InstancedMesh(
    geo,
    new THREE.MeshStandardMaterial({
      roughness: 0.5,
      metalness: 0.0,
      vertexColors: false,
      transparent: true,
      opacity: 0.15,
      depthWrite: false,
      side: THREE.DoubleSide,
    }),
    count
  );

  const dummy = new THREE.Object3D();
  const color = new THREE.Color();

  for (let j = 0; j < count; j++) {
    const i = indices[j];
    const [ix, iy, iz] = thermal.positions[i];
    const temp = thermal.temperatures[i];

    dummy.position.set(
      voxel.origin[0] + (ix + 0.5) * vs,
      voxel.origin[1] + (iy + 0.5) * vs,
      voxel.origin[2] + (iz + 0.5) * vs
    );
    dummy.updateMatrix();
    instanced.setMatrixAt(j, dummy.matrix);

    color.copy(tempToColor(temp, thermal.min_temp, thermal.max_temp));
    instanced.setColorAt(j, color);
  }

  instanced.instanceMatrix.needsUpdate = true;
  if (instanced.instanceColor) instanced.instanceColor.needsUpdate = true;
  instanced.castShadow = true;
  instanced.receiveShadow = true;

  meshGroup.add(instanced);
  scene.add(meshGroup);

  // Auto-fit camera
  const box = new THREE.Box3().setFromObject(meshGroup);
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const maxDim = Math.max(size.x, size.y, size.z);
  camera.position.copy(
    center.clone().add(new THREE.Vector3(maxDim, maxDim * 0.8, maxDim))
  );
  controls.target.copy(center);
  controls.update();
}

// ── ASHRAE metrics overlay renderer ──────────────────────
// Shows the thermal field but colors rack intake/exhaust voxels by
// ASHRAE compliance (green = compliant, yellow = allowable, red = violation).
// Also renders a metrics panel with the numbers.

const ASHRAE_REC_LO = 18;
const ASHRAE_REC_HI = 27;
const ASHRAE_ALLOW_LO = 15;
const ASHRAE_ALLOW_HI = 35;

function intakeComplianceColor(temp: number): THREE.Color {
  if (temp >= ASHRAE_REC_LO && temp <= ASHRAE_REC_HI) {
    return new THREE.Color(0x2ecc71); // green — recommended range
  } else if (temp >= ASHRAE_ALLOW_LO && temp <= ASHRAE_ALLOW_HI) {
    return new THREE.Color(0xf39c12); // amber — within allowable
  }
  return new THREE.Color(0xe74c3c); // red — violation
}

function loadMetricsOverlay(
  voxel: VoxelData,
  thermal: ThermalData,
  metrics: MetricsData
) {
  clearMesh();
  meshGroup = new THREE.Group();

  const vs = voxel.voxel_size;
  const geo = new THREE.BoxGeometry(vs * 0.92, vs * 0.92, vs * 0.92);

  // Build label map from voxel data
  const labelMap = new Map<string, number>();
  for (let i = 0; i < voxel.positions.length; i++) {
    const [ix, iy, iz] = voxel.positions[i];
    labelMap.set(`${ix},${iy},${iz}`, voxel.labels[i]);
  }

  // Filter: skip walls (1) and rack body (5) — show air + intake + exhaust + AC
  const indices: number[] = [];
  for (let i = 0; i < thermal.positions.length; i++) {
    const [ix, iy, iz] = thermal.positions[i];
    const label = labelMap.get(`${ix},${iy},${iz}`) ?? 0;
    if (label !== 1 && label !== 5) indices.push(i);
  }

  const count = indices.length;
  if (count === 0) return;

  const instanced = new THREE.InstancedMesh(
    geo,
    new THREE.MeshStandardMaterial({
      roughness: 0.5,
      metalness: 0.0,
      vertexColors: false,
      transparent: true,
      opacity: 0.15,
      depthWrite: false,
      side: THREE.DoubleSide,
    }),
    count
  );

  const dummy = new THREE.Object3D();
  const color = new THREE.Color();

  for (let j = 0; j < count; j++) {
    const i = indices[j];
    const [ix, iy, iz] = thermal.positions[i];
    const temp = thermal.temperatures[i];
    const label = labelMap.get(`${ix},${iy},${iz}`) ?? 0;

    dummy.position.set(
      voxel.origin[0] + (ix + 0.5) * vs,
      voxel.origin[1] + (iy + 0.5) * vs,
      voxel.origin[2] + (iz + 0.5) * vs
    );
    dummy.updateMatrix();
    instanced.setMatrixAt(j, dummy.matrix);

    // Rack intake (6) and exhaust (7): compliance coloring, fully opaque
    if (label === 6 || label === 7) {
      color.copy(intakeComplianceColor(temp));
    } else if (label === 3) {
      // AC vent — blue
      color.set(0x3c8ce6);
    } else {
      // Regular air: thermal heatmap
      color.copy(tempToColor(temp, thermal.min_temp, thermal.max_temp));
    }
    instanced.setColorAt(j, color);
  }

  instanced.instanceMatrix.needsUpdate = true;
  if (instanced.instanceColor) instanced.instanceColor.needsUpdate = true;

  meshGroup.add(instanced);
  scene.add(meshGroup);

  // Auto-fit camera
  const box = new THREE.Box3().setFromObject(meshGroup);
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const maxDim = Math.max(size.x, size.y, size.z);
  camera.position.copy(
    center.clone().add(new THREE.Vector3(maxDim, maxDim * 0.8, maxDim))
  );
  controls.target.copy(center);
  controls.update();

  // Populate metrics panel
  renderMetricsPanel(metrics);
}

function renderMetricsPanel(m: MetricsData) {
  const r = m.room;
  let html = `<h3>Room Metrics</h3>
    <div class="metric-row">
      <span class="metric-label">RCI High</span>
      <span class="metric-value ${r.rci_hi >= 95 ? "good" : r.rci_hi >= 80 ? "warn" : "bad"}">${r.rci_hi.toFixed(1)}%</span>
    </div>
    <div class="metric-row">
      <span class="metric-label">RCI Low</span>
      <span class="metric-value ${r.rci_lo >= 95 ? "good" : r.rci_lo >= 80 ? "warn" : "bad"}">${r.rci_lo.toFixed(1)}%</span>
    </div>
    <div class="metric-row">
      <span class="metric-label">SHI</span>
      <span class="metric-value">${r.shi.toFixed(3)}</span>
    </div>
    <div class="metric-row">
      <span class="metric-label">RHI</span>
      <span class="metric-value">${r.rhi.toFixed(3)}</span>
    </div>
    <div class="metric-row">
      <span class="metric-label">Mean Intake</span>
      <span class="metric-value">${r.mean_intake.toFixed(1)} °C</span>
    </div>
    <div class="metric-row">
      <span class="metric-label">Mean Exhaust</span>
      <span class="metric-value">${r.mean_exhaust.toFixed(1)} °C</span>
    </div>
    <div class="metric-row">
      <span class="metric-label">Mean Return</span>
      <span class="metric-value">${r.mean_return.toFixed(1)} °C</span>
    </div>`;

  html += `<h3 style="margin-top:12px">Per-Rack</h3>`;
  for (const rack of m.racks) {
    const status = rack.inlet_compliant ? "good" : rack.inlet_within_allowable ? "warn" : "bad";
    const icon = rack.inlet_compliant ? "✓" : rack.inlet_within_allowable ? "⚠" : "✗";
    html += `
      <div class="rack-card ${status}">
        <div class="rack-header">Rack ${rack.rack_index + 1} <span class="rack-status">${icon}</span></div>
        <div class="rack-detail">Intake: ${rack.intake_temp.toFixed(1)} °C</div>
        <div class="rack-detail">Exhaust: ${rack.exhaust_temp.toFixed(1)} °C</div>
        <div class="rack-detail">ΔT: ${rack.delta_t.toFixed(1)} °C</div>
      </div>`;
  }

  html += `<h3 style="margin-top:12px">Legend</h3>
    <div class="legend-item"><div class="legend-dot" style="background:#2ecc71"></div> Recommended (18–27 °C)</div>
    <div class="legend-item"><div class="legend-dot" style="background:#f39c12"></div> Allowable (15–35 °C)</div>
    <div class="legend-item"><div class="legend-dot" style="background:#e74c3c"></div> Violation</div>`;

  metricsPanel.innerHTML = html;
}

// ── Stage switching ───────────────────────────────────────
function setActiveStage(stage: string) {
  if (!result) return;

  if (stage === "voxel") {
    if (!result.voxel_grid) {
      setStatus("No voxel grid available", true);
      return;
    }
    loadVoxelGrid(result.voxel_grid);
  } else if (stage === "thermal") {
    if (!result.voxel_grid || !result.thermal) {
      setStatus("No thermal data available", true);
      return;
    }
    loadThermalGrid(result.voxel_grid, result.thermal);
  } else if (stage === "ashrae") {
    if (!result.voxel_grid || !result.thermal || !result.metrics) {
      setStatus("No metrics data available", true);
      return;
    }
    loadMetricsOverlay(result.voxel_grid, result.thermal, result.metrics);
  } else {
    const glbMap: Record<string, string | null> = {
      raw: result.raw_glb,
      cleaned: result.cleaned_glb,
      semantic: result.semantic_glb,
    };

    const b64 = glbMap[stage];
    if (!b64) {
      setStatus(`No ${stage} mesh available`, true);
      return;
    }

    loadGlb(b64);
  }

  // Update button state
  stageBtns.querySelectorAll(".stage-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.getAttribute("data-stage") === stage);
  });

  // Show legend for semantic, voxel, and thermal stages
  legendEl.style.display =
    stage === "semantic" || stage === "voxel" || stage === "thermal" ? "block" : "none";

  // Show metrics panel only on ASHRAE stage
  metricsPanel.style.display = stage === "ashrae" ? "block" : "none";
}

stageBtns.addEventListener("click", (e) => {
  const btn = (e.target as HTMLElement).closest(".stage-btn");
  if (btn) setActiveStage(btn.getAttribute("data-stage")!);
});

// ── Upload handler ────────────────────────────────────────
uploadBtn.addEventListener("click", async () => {
  const file = fileInput.files?.[0];
  if (!file) {
    setStatus("Please select an .obj file", true);
    return;
  }

  const metaText = metaInput.value.trim() || "{}";
  try {
    JSON.parse(metaText);
  } catch {
    setStatus("Invalid JSON in metadata field", true);
    return;
  }

  uploadBtn.disabled = true;
  setStatus("Processing scan...");

  const formData = new FormData();
  formData.append("file", file);
  formData.append("metadata", metaText);

  try {
    const resp = await fetch("/api/v1/visualize", {
      method: "POST",
      body: formData,
    });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(err.detail || `HTTP ${resp.status}`);
    }

    result = (await resp.json()) as VisualizeResult;
    setStatus("Scan processed — select a stage below", false);

    // Show stage buttons
    stageBtns.style.display = "flex";

    // Enable/disable semantic button
    const semBtn = stageBtns.querySelector(
      '[data-stage="semantic"]'
    ) as HTMLButtonElement;
    if (!result.semantic_glb) {
      semBtn.style.opacity = "0.3";
      semBtn.style.pointerEvents = "none";
    } else {
      semBtn.style.opacity = "1";
      semBtn.style.pointerEvents = "auto";
    }

    // Enable/disable voxel button
    const voxBtn = stageBtns.querySelector(
      '[data-stage="voxel"]'
    ) as HTMLButtonElement;
    if (!result.voxel_grid) {
      voxBtn.style.opacity = "0.3";
      voxBtn.style.pointerEvents = "none";
    } else {
      voxBtn.style.opacity = "1";
      voxBtn.style.pointerEvents = "auto";
    }

    // Enable/disable thermal button
    const thermBtn = stageBtns.querySelector(
      '[data-stage="thermal"]'
    ) as HTMLButtonElement;
    if (!result.thermal) {
      thermBtn.style.opacity = "0.3";
      thermBtn.style.pointerEvents = "none";
    } else {
      thermBtn.style.opacity = "1";
      thermBtn.style.pointerEvents = "auto";
    }

    // Load raw by default
    setActiveStage("raw");
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err);
    setStatus(`Error: ${msg}`, true);
  } finally {
    uploadBtn.disabled = false;
  }
});

// ── Demo handler ──────────────────────────────────────────
demoBtn.addEventListener("click", async () => {
  demoBtn.disabled = true;
  setStatus("Generating demo room...");

  try {
    const resp = await fetch("/api/v1/visualize/demo");

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(err.detail || `HTTP ${resp.status}`);
    }

    result = (await resp.json()) as VisualizeResult;
    setStatus("Demo loaded — select a stage below", false);

    stageBtns.style.display = "flex";

    const semBtn = stageBtns.querySelector(
      '[data-stage="semantic"]'
    ) as HTMLButtonElement;
    if (!result.semantic_glb) {
      semBtn.style.opacity = "0.3";
      semBtn.style.pointerEvents = "none";
    } else {
      semBtn.style.opacity = "1";
      semBtn.style.pointerEvents = "auto";
    }

    const voxBtn = stageBtns.querySelector(
      '[data-stage="voxel"]'
    ) as HTMLButtonElement;
    if (!result.voxel_grid) {
      voxBtn.style.opacity = "0.3";
      voxBtn.style.pointerEvents = "none";
    } else {
      voxBtn.style.opacity = "1";
      voxBtn.style.pointerEvents = "auto";
    }

    const thermBtn = stageBtns.querySelector(
      '[data-stage="thermal"]'
    ) as HTMLButtonElement;
    if (!result.thermal) {
      thermBtn.style.opacity = "0.3";
      thermBtn.style.pointerEvents = "none";
    } else {
      thermBtn.style.opacity = "1";
      thermBtn.style.pointerEvents = "auto";
    }

    const ashraeBtn = stageBtns.querySelector(
      '[data-stage="ashrae"]'
    ) as HTMLButtonElement;
    if (!result.metrics) {
      ashraeBtn.style.opacity = "0.3";
      ashraeBtn.style.pointerEvents = "none";
    } else {
      ashraeBtn.style.opacity = "1";
      ashraeBtn.style.pointerEvents = "auto";
    }

    setActiveStage("raw");
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err);
    setStatus(`Error: ${msg}`, true);
  } finally {
    demoBtn.disabled = false;
  }
});
