import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import "./lib/twinIndicator";
import { AirflowSystem } from "./components/airflow";
import { createEquipmentMesh } from "./components/equipment";
import { buildHeatmap } from "./components/heatmap";
import { buildRoom } from "./components/room";
import { buildZones } from "./components/zones";
import { allScenes, getLoadFactor, type SceneGraph } from "./data/sceneGraphs";
import { computeAshraeMetrics } from "./lib/ashrae";

// ===== Shared simulation state =====
let simPlaying = false;
let simMinutes = 0;
let simSpeed = 1;
let lastSimTick = 0;

// ===== Shared toggle state =====
let heatmapVisible = false;
let airflowVisible = false;
let zonesVisible = false;

// ===== SceneViewer =====
class SceneViewer {
  renderer: THREE.WebGLRenderer;
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  controls: OrbitControls;
  furnitureGroup: THREE.Group;
  roomGroup: THREE.Group;
  heatmapGroup: THREE.Group;
  airflowGroup: THREE.Group;
  airflowSystem: AirflowSystem;
  zonesGroup: THREE.Group;
  sceneData: SceneGraph;
  tempReadout: HTMLElement | null;
  animating = false;

  constructor(canvas: HTMLCanvasElement, sceneData: SceneGraph, tempReadout: HTMLElement | null) {
    this.sceneData = sceneData;
    this.tempReadout = tempReadout;

    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0xf4f2ec);

    // Frame the camera to the actual room footprint (from room.dimensions) so the real
    // scanned room (≈7.5×3.9 m) fills the view instead of the old fixed 12×9 m assumption.
    const [rw, , rd] = this.sceneData.room.dimensions;
    const cx = rw / 2;
    const cz = rd / 2;
    const span = Math.max(rw, rd);

    this.camera = new THREE.PerspectiveCamera(50, 1, 0.1, 200);
    this.camera.position.set(cx + span * 1.0, span * 1.0, cz + span * 1.5);
    this.camera.lookAt(cx, 0, cz);

    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.target.set(cx, 0, cz);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;
    this.controls.maxPolarAngle = Math.PI / 2.1;

    const ambient = new THREE.AmbientLight(0xffffff, 0.35);
    this.scene.add(ambient);

    const dirLight = new THREE.DirectionalLight(0xe8f0ff, 1.0);
    dirLight.position.set(10, 16, 8);
    dirLight.castShadow = true;
    dirLight.shadow.mapSize.set(2048, 2048);
    dirLight.shadow.camera.left = -16;
    dirLight.shadow.camera.right = 16;
    dirLight.shadow.camera.top = 16;
    dirLight.shadow.camera.bottom = -16;
    this.scene.add(dirLight);

    const f1 = new THREE.PointLight(0xe0f0ff, 0.5, 18);
    f1.position.set(4, 3.2, 3);
    this.scene.add(f1);

    const f2 = new THREE.PointLight(0xe0f0ff, 0.5, 18);
    f2.position.set(8, 3.2, 6);
    this.scene.add(f2);

    this.roomGroup = new THREE.Group();
    this.furnitureGroup = new THREE.Group();
    this.heatmapGroup = new THREE.Group();
    this.airflowGroup = new THREE.Group();
    this.airflowSystem = new AirflowSystem(this.airflowGroup);
    this.zonesGroup = new THREE.Group();

    const gridSpan = Math.ceil(span) + 4;
    const grid = new THREE.GridHelper(gridSpan, gridSpan * 2, 0xe8e5dd, 0xf4f2ec);
    grid.position.set(cx, -0.01, cz);
    this.scene.add(grid);

    this.scene.add(this.roomGroup);
    this.scene.add(this.furnitureGroup);
    this.scene.add(this.heatmapGroup);
    this.scene.add(this.airflowGroup);
    this.scene.add(this.zonesGroup);

    this.onResize();
    this.buildScene();
  }

  buildScene() {
    this.furnitureGroup.clear();
    buildRoom(this.roomGroup, this.sceneData);
    buildHeatmap(this.heatmapGroup, this.sceneData);
    this.updateTempReadout(1.0);
    this.heatmapGroup.visible = heatmapVisible;
    this.airflowSystem.init(this.sceneData);
    this.airflowGroup.visible = airflowVisible;
    buildZones(this.zonesGroup, this.sceneData);
    this.zonesGroup.visible = zonesVisible;

    const equipCtx = { roomHeight: this.sceneData.room.dimensions[1] };
    this.animating = true;
    let i = 0;
    const furniture = this.furnitureGroup;
    const interval = setInterval(() => {
      if (i >= this.sceneData.furniture.length) {
        this.animating = false;
        clearInterval(interval);
        return;
      }
      const item = this.sceneData.furniture[i];
      const mesh = createEquipmentMesh(item, equipCtx);
      const targetY = mesh.position.y;
      mesh.position.y = 5;
      mesh.scale.set(0.01, 0.01, 0.01);
      furniture.add(mesh);

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
  }

  updateSimulation(simMins: number) {
    const loadFactor = getLoadFactor(simMins / 60);
    buildHeatmap(this.heatmapGroup, this.sceneData, loadFactor);
    this.updateTempReadout(loadFactor);
    this.heatmapGroup.visible = heatmapVisible;

    this.updateEquipmentGlow(loadFactor);
  }

  updateTempReadout(loadFactor: number) {
    if (!this.tempReadout) return;
    const { peakIntake } = computeAshraeMetrics(this.sceneData, loadFactor);
    this.tempReadout.textContent = `최고온도 ${peakIntake.toFixed(1)}°C`;
  }

  updateEquipmentGlow(loadFactor: number) {
    this.furnitureGroup.children.forEach((child) => {
      if (!(child instanceof THREE.Group)) return;
      const { category, heatOutput } = child.userData;
      if (!category) return;
      child.traverse((obj) => {
        if (!(obj instanceof THREE.Mesh)) return;
        const mat = obj.material as THREE.MeshBasicMaterial | THREE.MeshStandardMaterial;
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

  setHeatmap(visible: boolean) {
    this.heatmapGroup.visible = visible;
  }

  setAirflow(visible: boolean) {
    this.airflowGroup.visible = visible;
  }

  setZones(visible: boolean) {
    this.zonesGroup.visible = visible;
  }

  render() {
    this.controls.update();
    if (airflowVisible && this.airflowSystem.isReady()) {
      const loadFactor = getLoadFactor(simMinutes / 60);
      this.airflowSystem.update(this.sceneData, loadFactor);
    }
    this.renderer.render(this.scene, this.camera);
  }

  onResize() {
    const canvas = this.renderer.domElement;
    const parent = canvas.parentElement!;
    const w = parent.clientWidth;
    const h = parent.clientHeight;
    this.renderer.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }
}

// ===== Init =====
let leftViewer: SceneViewer;
let rightViewer: SceneViewer;

function init() {
  const leftCanvas = document.getElementById("canvas-left") as HTMLCanvasElement;
  const rightCanvas = document.getElementById("canvas-right") as HTMLCanvasElement;

  leftViewer = new SceneViewer(leftCanvas, allScenes[0], document.getElementById("temp-left"));
  rightViewer = new SceneViewer(rightCanvas, allScenes[1], document.getElementById("temp-right"));

  setupButtons();
  setupCameraSync();
  window.addEventListener("resize", onResize);
  animate();
}

function onResize() {
  leftViewer.onResize();
  rightViewer.onResize();
}

function animate(now: number = 0) {
  requestAnimationFrame(animate);

  if (simPlaying && now - lastSimTick > 50) {
    lastSimTick = now;
    simMinutes += simSpeed;
    if (simMinutes >= 1440) simMinutes = 0;
    updateSimSlider();
    updateSimulation();
  }

  leftViewer.render();
  rightViewer.render();
}

function updateSimSlider() {
  const slider = document.getElementById("sim-slider") as HTMLInputElement;
  slider.value = String(simMinutes);
  const h = Math.floor(simMinutes / 60);
  const m = Math.floor(simMinutes % 60);
  document.getElementById("sim-clock")!.textContent =
    `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}`;
}

function updateSimulation() {
  leftViewer.updateSimulation(simMinutes);
  rightViewer.updateSimulation(simMinutes);
}

function updateLegendVisibility() {
  document.getElementById("airflow-legend")!.style.display = zonesVisible ? "flex" : "none";
}

// ===== Camera Sync =====
let syncing = false;

function syncCameras(source: SceneViewer, target: SceneViewer) {
  if (syncing) return;
  syncing = true;
  target.camera.position.copy(source.camera.position);
  target.controls.target.copy(source.controls.target);
  target.controls.update();
  syncing = false;
}

function setupCameraSync() {
  leftViewer.controls.addEventListener("change", () => syncCameras(leftViewer, rightViewer));
  rightViewer.controls.addEventListener("change", () => syncCameras(rightViewer, leftViewer));
}

function setupButtons() {
  document.getElementById("btn-heatmap")?.addEventListener("click", () => {
    heatmapVisible = !heatmapVisible;
    leftViewer.setHeatmap(heatmapVisible);
    rightViewer.setHeatmap(heatmapVisible);
    document.getElementById("btn-heatmap")!.classList.toggle("active-toggle", heatmapVisible);
  });

  document.getElementById("btn-airflow")!.addEventListener("click", () => {
    airflowVisible = !airflowVisible;
    leftViewer.setAirflow(airflowVisible);
    rightViewer.setAirflow(airflowVisible);
    document.getElementById("btn-airflow")!.classList.toggle("active-toggle", airflowVisible);
    updateLegendVisibility();
  });

  document.getElementById("btn-zones")!.addEventListener("click", () => {
    zonesVisible = !zonesVisible;
    leftViewer.setZones(zonesVisible);
    rightViewer.setZones(zonesVisible);
    document.getElementById("btn-zones")!.classList.toggle("active-toggle", zonesVisible);
    updateLegendVisibility();
  });

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

  updateSimulation();
}

init();
