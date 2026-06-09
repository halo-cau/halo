/**
 * Scan Viewer — upload scan files, call /api/v1/visualize, render GLB in Three.js.
 */
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { PLYLoader } from "three/addons/loaders/PLYLoader.js";
import { clearJob, loadSavedJob, saveJob, type TwinStatus } from "./lib/twinJob";

// ── DOM refs ──────────────────────────────────────────────
const canvas = document.getElementById("viewer") as HTMLCanvasElement;
const fileInput = document.getElementById("obj-file") as HTMLInputElement;
const metaInput = document.getElementById("metadata") as HTMLTextAreaElement;
const uploadBtn = document.getElementById("upload-btn") as HTMLButtonElement;
const statusEl = document.getElementById("status") as HTMLDivElement;
const stageBtns = document.getElementById("stage-btns") as HTMLDivElement;
const legendEl = document.getElementById("legend") as HTMLDivElement;
const metricsPanel = document.getElementById("metrics-panel") as HTMLDivElement;
const editBtn = document.getElementById("edit-btn") as HTMLButtonElement;
const SUPPORTED_SCAN_EXTENSIONS = [".obj", ".ply", ".las", ".laz"];
const SUPPORTED_SCAN_EXTENSIONS_TEXT = ".obj, .ply, .las";

function isSupportedScanFile(file: File): boolean {
  const name = file.name.toLowerCase();
  return SUPPORTED_SCAN_EXTENSIONS.some((ext) => name.endsWith(ext));
}

// ── Three.js setup ────────────────────────────────────────
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setClearColor(0xf4f2ec);
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
const grid = new THREE.GridHelper(20, 40, 0xe8e5dd, 0xf4f2ec);
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
  layout_voxel_grid: VoxelData | null;
  thermal: ThermalData | null;
  metrics: MetricsData | null;
}
const result: VisualizeResult | null = null;

// Semantic label → color mapping (matches legend, design-system palette)
const LABEL_COLORS: Record<number, THREE.Color> = {
  1: new THREE.Color(0xb4b2a9), // wall — rack-gray
  2: new THREE.Color(0xe24b4a), // legacy server (heat) — danger
  3: new THREE.Color(0x378add), // AC vent (cooling) — info blue
  4: new THREE.Color(0x1d9e75), // human workspace — success green
  5: new THREE.Color(0x888780), // rack body — text-3 muted gray
  6: new THREE.Color(0x378add), // rack intake — calm blue
  7: new THREE.Color(0xe89e4f), // rack exhaust — warning amber
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
      if (child instanceof THREE.Mesh || child instanceof THREE.Points) {
        child.geometry.dispose();
        if (Array.isArray(child.material)) {
          for (const m of child.material) m.dispose();
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
        } else if (child instanceof THREE.Points) {
          child.material = new THREE.PointsMaterial({
            color: 0x88aacc,
            size: 0.04,
            sizeAttenuation: true,
            vertexColors: Boolean(child.geometry.attributes.color),
          });
        }
      });
      meshGroup!.add(gltf.scene);
      scene.add(meshGroup!);

      // Auto-fit camera
      const box = new THREE.Box3().setFromObject(meshGroup!);
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3());
      const maxDim = Math.max(size.x, size.y, size.z, 1);
      camera.position.copy(center.clone().add(new THREE.Vector3(maxDim, maxDim * 0.8, maxDim)));
      controls.target.copy(center);
      controls.update();
      resize();
    },
    (err) => {
      setStatus(`GLB parse error: ${err}`, true);
    },
  );
}

function setStatus(msg: string, isError = false) {
  statusEl.textContent = msg;
  statusEl.className = isError ? "status error" : "status success";
}

// ── Progress bar (visible job feedback, driven by the pipeline's pct) ──
const progressEl = document.createElement("div");
progressEl.id = "twin-progress";
progressEl.style.cssText = "display:none;margin-top:10px";
progressEl.innerHTML = `
  <div style="display:flex;justify-content:space-between;font:12px system-ui,-apple-system,sans-serif;color:#6b685f;margin-bottom:4px">
    <span class="tp-step">queued</span><span class="tp-pct">0%</span>
  </div>
  <div style="height:10px;border-radius:6px;background:#eee9df;overflow:hidden">
    <div class="tp-fill" style="height:100%;width:0;background:#378add;border-radius:6px;transition:width .35s ease"></div>
  </div>`;
statusEl.insertAdjacentElement("afterend", progressEl);
const tpStep = progressEl.querySelector(".tp-step") as HTMLElement;
const tpPct = progressEl.querySelector(".tp-pct") as HTMLElement;
const tpFill = progressEl.querySelector(".tp-fill") as HTMLElement;

// pct null hides the bar; otherwise fill to pct and label the current step.
function setProgress(pct: number | null, step?: string): void {
  if (pct === null) {
    progressEl.style.display = "none";
    return;
  }
  progressEl.style.display = "block";
  const p = Math.max(0, Math.min(100, pct));
  tpFill.style.width = `${p}%`;
  tpPct.textContent = `${Math.round(p)}%`;
  if (step) tpStep.textContent = step;
}

// ── Camera auto-fit (shared) ──────────────────────────────
function fitCamera(obj: THREE.Object3D) {
  const box = new THREE.Box3().setFromObject(obj);
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const maxDim = Math.max(size.x, size.y, size.z, 1);
  camera.position.copy(center.clone().add(new THREE.Vector3(maxDim, maxDim * 0.8, maxDim)));
  controls.target.copy(center);
  controls.update();
  resize();
}

// ── PLY renderer (twin stages: recon / SAM3 labels / voxel) ──
const plyLoader = new PLYLoader();
function loadPly(url: string) {
  clearMesh();
  setStatus(`loading ${url.split("/").pop()} …`);
  plyLoader.load(
    url,
    (geo) => {
      const hasColor = Boolean(geo.getAttribute("color"));
      const pts = new THREE.Points(
        geo,
        // screen-pixel points (NOT world-size) so the 0.1 m voxel grid reads as a solid room at scale,
        // exactly like the recon_web viewer the scenes were aligned in.
        new THREE.PointsMaterial({
          size: 2.5,
          sizeAttenuation: false,
          vertexColors: hasColor,
          color: hasColor ? 0xffffff : 0x88aacc,
        }),
      );
      meshGroup = new THREE.Group();
      meshGroup.add(pts);
      scene.add(meshGroup);
      fitCamera(meshGroup);
      setStatus("", false);
    },
    undefined,
    (err) => setStatus(`PLY load error: ${err}`, true),
  );
}

// ── Twin (multi-view CV pipeline) state ───────────────────
type ViewMode = "mesh" | "twin";
let mode: ViewMode = "mesh";
let twinJob: { id: string; outputs: Record<string, string> } | null = null;
// twin stage -> the outputs key whose artifact it renders
const TWIN_STAGE_KEY: Record<string, string> = {
  recon: "recon",
  semantic: "labeled",
  sam3class: "classes",
  voxel: "voxel",
};

function twinArtifactUrl(stage: string): string | null {
  const key = TWIN_STAGE_KEY[stage];
  const name = key && twinJob ? twinJob.outputs[key] : undefined;
  return name ? `/api/v1/twin/${twinJob!.id}/artifact/${name}` : null;
}

function enableTwinTabs() {
  if (!twinJob) return;
  stageBtns.style.display = "flex";
  editBtn.style.display = "block"; // the per-instance editor reads placements.json, which only twin jobs emit
  const out = twinJob.outputs;
  const avail: Record<string, boolean> = {
    recon: Boolean(out.recon),
    semantic: Boolean(out.labeled),
    sam3class: Boolean(out.classes),
    voxel: Boolean(out.voxel),
    layout: false, // Layout / Thermal / ASHRAE land with the thermal stage (not yet emitted by the twin)
    thermal: false,
    ashrae: false,
  };
  stageBtns.querySelectorAll(".stage-btn").forEach((b) => {
    const st = b.getAttribute("data-stage") ?? "";
    const ok = avail[st] ?? false;
    (b as HTMLElement).style.opacity = ok ? "1" : "0.3";
    (b as HTMLElement).style.pointerEvents = ok ? "auto" : "none";
  });
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
    count,
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
      data.origin[2] + (iz + 0.5) * vs,
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
  const maxDim = Math.max(size.x, size.y, size.z, 1);
  camera.position.copy(center.clone().add(new THREE.Vector3(maxDim, maxDim * 0.8, maxDim)));
  controls.target.copy(center);
  controls.update();
  resize();
}

// ── Thermal heatmap renderer ─────────────────────────────
function tempToColor(t: number, minT: number, maxT: number): THREE.Color {
  // CFD-standard Jet colormap (MATLAB Jet) for temperature voxels.
  // Dark blue (cold) → blue → cyan → green → yellow → red → dark red (hot).
  let frac = maxT > minT ? (t - minT) / (maxT - minT) : 0.5;
  frac = Math.max(0, Math.min(1, frac));

  const stops = [
    [0.0, 0.0, 0.5],
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 1.0],
    [1.0, 1.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.5, 0.0, 0.0],
  ];
  const positions = [0.0, 0.125, 0.375, 0.625, 0.875, 1.0];

  let lo = 0;
  for (let i = 1; i < positions.length; i++) {
    if (frac <= positions[i]) {
      lo = i - 1;
      break;
    }
    lo = i;
  }
  const hi = Math.min(lo + 1, positions.length - 1);
  const span = positions[hi] - positions[lo] || 1;
  const t01 = (frac - positions[lo]) / span;
  const r = stops[lo][0] + (stops[hi][0] - stops[lo][0]) * t01;
  const g = stops[lo][1] + (stops[hi][1] - stops[lo][1]) * t01;
  const b = stops[lo][2] + (stops[hi][2] - stops[lo][2]) * t01;
  return new THREE.Color(r, g, b);
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
    if (label !== 1) indices.push(i); // skip walls
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
    count,
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
      voxel.origin[2] + (iz + 0.5) * vs,
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
  const maxDim = Math.max(size.x, size.y, size.z, 1);
  camera.position.copy(center.clone().add(new THREE.Vector3(maxDim, maxDim * 0.8, maxDim)));
  controls.target.copy(center);
  controls.update();
  resize();
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
    return new THREE.Color(0x1d9e75); // deep green — recommended
  } else if (temp >= ASHRAE_ALLOW_LO && temp <= ASHRAE_ALLOW_HI) {
    return new THREE.Color(0xe89e4f); // amber — within allowable
  }
  return new THREE.Color(0xe24b4a); // muted red — violation
}

function loadMetricsOverlay(voxel: VoxelData, thermal: ThermalData, metrics: MetricsData) {
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
    count,
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
      voxel.origin[2] + (iz + 0.5) * vs,
    );
    dummy.updateMatrix();
    instanced.setMatrixAt(j, dummy.matrix);

    // Rack intake (6) and exhaust (7): compliance coloring, fully opaque
    if (label === 6 || label === 7) {
      color.copy(intakeComplianceColor(temp));
    } else if (label === 3) {
      // AC vent — calm blue
      color.set(0x378add);
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
  const maxDim = Math.max(size.x, size.y, size.z, 1);
  camera.position.copy(center.clone().add(new THREE.Vector3(maxDim, maxDim * 0.8, maxDim)));
  controls.target.copy(center);
  controls.update();
  resize();

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
    const tag = rack.inlet_compliant ? "OK" : rack.inlet_within_allowable ? "WARN" : "FAIL";
    html += `
      <div class="rack-card ${status}">
        <div class="rack-header">Rack ${rack.rack_index + 1} <span class="rack-status metric-value ${status}">${tag}</span></div>
        <div class="rack-detail">Intake: ${rack.intake_temp.toFixed(1)} °C</div>
        <div class="rack-detail">Exhaust: ${rack.exhaust_temp.toFixed(1)} °C</div>
        <div class="rack-detail">ΔT: ${rack.delta_t.toFixed(1)} °C</div>
      </div>`;
  }

  html += `<h3 style="margin-top:12px">Legend</h3>
    <div class="legend-item"><div class="legend-dot" style="background:#1D9E75"></div> Recommended (18–27 °C)</div>
    <div class="legend-item"><div class="legend-dot" style="background:#E89E4F"></div> Allowable (15–35 °C)</div>
    <div class="legend-item"><div class="legend-dot" style="background:#E24B4A"></div> Violation</div>`;

  metricsPanel.innerHTML = html;
}

// ── Stage switching ───────────────────────────────────────
function setActiveStage(stage: string) {
  if (mode === "twin") {
    const url = twinArtifactUrl(stage);
    if (!url) {
      setStatus(`No "${stage}" stage for this job`, true);
      return;
    }
    loadPly(url);
    stageBtns.querySelectorAll(".stage-btn").forEach((btn) => {
      btn.classList.toggle("active", btn.getAttribute("data-stage") === stage);
    });
    legendEl.style.display = "none";
    metricsPanel.style.display = "none";
    return;
  }

  if (!result) return;

  if (stage === "voxel") {
    if (!result.voxel_grid) {
      setStatus("No voxel grid available", true);
      return;
    }
    loadVoxelGrid(result.voxel_grid);
  } else if (stage === "layout") {
    if (!result.layout_voxel_grid) {
      setStatus("No layout grid available", true);
      return;
    }
    loadVoxelGrid(result.layout_voxel_grid);
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

  // Show legend for semantic, voxel, layout, and thermal stages
  legendEl.style.display =
    stage === "semantic" || stage === "voxel" || stage === "layout" || stage === "thermal"
      ? "block"
      : "none";

  // Show metrics panel only on ASHRAE stage
  metricsPanel.style.display = stage === "ashrae" ? "block" : "none";
}

stageBtns.addEventListener("click", (e) => {
  const btn = (e.target as HTMLElement).closest(".stage-btn");
  if (btn) setActiveStage(btn.getAttribute("data-stage")!);
});

// ── Background twin job: survives page navigation ─────────
// The GPU pipeline (Pi3 + SAM3) is multi-minute and runs server-side in an isolated subprocess (a single
// backend worker serializes jobs; progress is written to status.json on disk). So the browser does NOT
// need to stay on this page: the job id is persisted in localStorage (lib/twinJob), we poll in a
// resumable way, and reconnect on load. Submit images, leave for any other page, and the twin will be
// waiting when you return. The same persisted job also drives the cross-page progress badge
// (lib/twinIndicator) shown on every other page.

let polling = false;

// Poll a job to completion. Safe to call after an upload OR on page load to reconnect to a job that is
// already running (or done) on the server. Manages the upload button and never blocks navigation.
async function pollJob(jobId: string): Promise<void> {
  if (polling) return;
  polling = true;
  uploadBtn.disabled = true;
  try {
    let s: TwinStatus = { state: "queued", step: "queued" };
    while (s.state !== "done" && s.state !== "error") {
      const resp = await fetch(`/api/v1/twin/${jobId}`);
      if (resp.status === 404) {
        clearJob();
        setProgress(null);
        setStatus("Previous job is no longer on the server — upload again", true);
        return;
      }
      s = (await resp.json()) as TwinStatus;
      setProgress(s.pct ?? 0, s.step);
      setStatus(
        `${s.step} — ${s.message ?? ""} (${s.pct ?? 0}%) · runs in the background; you can switch pages`,
      );
      if (s.state === "done" || s.state === "error") break;
      await new Promise((res) => setTimeout(res, 2000));
    }
    if (s.state === "error") {
      setProgress(null);
      setStatus(`Error: ${s.error || s.message || "pipeline failed"}`, true);
      clearJob();
      return;
    }

    setProgress(100, "done");
    mode = "twin";
    twinJob = { id: jobId, outputs: s.outputs ?? {} };
    setStatus(
      s.precomputed
        ? "사전 계산된 샘플 — Pi3+SAM3 결과 사전 실행 (precomputed sample)"
        : "done — showing the voxel twin",
      false,
    );
    enableTwinTabs();
    setActiveStage("voxel");
  } finally {
    polling = false;
    uploadBtn.disabled = false;
  }
}

// On load, reconnect to the last submitted job (running or finished) so leaving and returning is seamless.
function resumeSavedJob(): void {
  const saved = loadSavedJob();
  if (!saved) return;
  setStatus("reconnecting to your reconstruction job…");
  void pollJob(saved.jobId);
}

// ── Submit the current file selection as an async twin job (multi-view images OR one geometry scan) ──
async function startTwinJob(): Promise<void> {
  const files = Array.from(fileInput.files ?? []);
  if (files.length === 0) {
    setStatus(
      "Select ≥2 images, or one .obj/.ply/.las scan — or drag the files onto this panel",
      true,
    );
    return;
  }

  uploadBtn.disabled = true;
  const fd = new FormData();
  for (const f of files) fd.append("files", f);
  setStatus("uploading…");
  setProgress(2, "uploading");

  try {
    const r = await fetch("/api/v1/twin", { method: "POST", body: fd });
    if (!r.ok) {
      const e = await r.json().catch(() => ({ detail: r.statusText }));
      throw new Error(e.detail || `HTTP ${r.status}`);
    }
    const { job_id, kind } = (await r.json()) as { job_id: string; kind: string };

    // Persist, then poll WITHOUT awaiting — the job is isolated on the server, so navigation is free.
    saveJob(job_id, kind);
    setStatus(
      "queued — reconstruction runs in the background. You can switch pages and come back.",
    );
    void pollJob(job_id);
  } catch (err: unknown) {
    setProgress(null);
    setStatus(`Error: ${err instanceof Error ? err.message : String(err)}`, true);
    uploadBtn.disabled = false;
  }
}

uploadBtn.addEventListener("click", () => void startTwinJob());

// Confirm the selection so the picker never looks like it did nothing.
fileInput.addEventListener("change", () => {
  const n = fileInput.files?.length ?? 0;
  if (n > 0) setStatus(`${n} file${n === 1 ? "" : "s"} ready — click “Process Scan”`, false);
});

// ── Drag-and-drop: drop image/scan files anywhere on the page to load + start the job ──
const dropPanel = document.querySelector(".panel") as HTMLElement | null;
function highlightDrop(on: boolean): void {
  if (dropPanel) dropPanel.style.outline = on ? "2px dashed #378add" : "";
}
document.addEventListener("dragover", (e) => {
  e.preventDefault(); // required so the page accepts the drop instead of opening the file
  highlightDrop(true);
});
document.addEventListener("dragleave", (e) => {
  if ((e as DragEvent).relatedTarget === null) highlightDrop(false); // pointer left the window
});
document.addEventListener("drop", (e) => {
  e.preventDefault();
  highlightDrop(false);
  const dropped = (e as DragEvent).dataTransfer?.files;
  if (!dropped || dropped.length === 0) return;
  const dt = new DataTransfer();
  for (const f of Array.from(dropped)) dt.items.add(f);
  fileInput.files = dt.files; // feed the same input the button reads, then process
  void startTwinJob();
});

resumeSavedJob();

// ── Edit handler (transition to the per-instance room editor for the current twin) ──
editBtn.addEventListener("click", () => {
  if (!twinJob) {
    setStatus("Load a twin first, then edit", true);
    return;
  }
  // same-origin editor page (served from dist); reads placements.json + posts to /api/v1/twin/{id}/restamp
  window.location.href = `./editor.html?job=${twinJob.id}`;
});
