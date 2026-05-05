import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { AirflowSystem } from "./components/airflow";
import { createEquipmentMesh } from "./components/equipment";
import { buildHeatmap } from "./components/heatmap";
import { buildRoom } from "./components/room";
import { buildZones } from "./components/zones";
import type { Equipment, SceneGraph } from "./data/sceneGraphs";

interface GalleryItem {
  category: string;
  name: string;
  description: string;
  equipment: Equipment;
}

const TILE = 3.6;
const ROOM_HEIGHT = 3.0;

function eq(
  category: string,
  size: [number, number, number],
  color: string,
  heatOutput = 0,
): Equipment {
  return {
    id: `gallery_${category}`,
    category,
    label: category,
    position: [0, 0, 0],
    rotation: [0, 0, 0],
    size,
    color,
    heatOutput,
    relations: [],
  };
}

const items: GalleryItem[] = [
  {
    category: "server_rack",
    name: "Server Rack",
    description: "42U 캐비닛: 본체, 유닛 패널, LED, 후면 exhaust glow.",
    equipment: eq("server_rack", [0.6, 2.0, 1.0], "#37474F", 12),
  },
  {
    category: "ceiling_ac",
    name: "Ceiling AC (천장형)",
    description: "4-way 카세트형: flange, intake louver, 4방향 vent.",
    equipment: eq("ceiling_ac", [1.0, 0.25, 1.0], "#0097A7"),
  },
  {
    category: "network_switch",
    name: "Network Switch",
    description: "코어 스위치 + 24포트 LED + 상단 라벨 표시.",
    equipment: eq("network_switch", [0.6, 1.8, 0.8], "#1565C0", 3),
  },
  {
    category: "ups",
    name: "UPS",
    description: "전원 장치 + 배터리 라벨 + 상태 LED.",
    equipment: eq("ups", [0.8, 1.5, 0.6], "#E65100", 2),
  },
  {
    category: "cable_tray",
    name: "Cable Tray",
    description: "케이블 트레이 + 가드레일 + 케이블 묶음.",
    equipment: eq("cable_tray", [3.0, 0.1, 0.4], "#FDD835"),
  },
  {
    category: "fallback",
    name: "Fallback",
    description: "지정되지 않은 카테고리에 대한 기본 박스.",
    equipment: eq("unknown_category", [0.6, 1.2, 0.6], "#546E7A"),
  },
];

const canvas = document.getElementById("canvas") as HTMLCanvasElement;
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xf4f2ec);

scene.add(new THREE.AmbientLight(0xffffff, 0.55));
const dirLight = new THREE.DirectionalLight(0xfff7e8, 0.85);
dirLight.position.set(8, 14, 6);
dirLight.castShadow = true;
dirLight.shadow.mapSize.set(1024, 1024);
scene.add(dirLight);

const cols = 4;
const rows = Math.ceil(items.length / cols);

// === Floor grid + per-tile pads ===
const padMat = new THREE.MeshStandardMaterial({ color: 0xf4f2ec, roughness: 0.85 });
const labelGroup = new THREE.Group();
scene.add(labelGroup);

function makeLabelSprite(text: string): THREE.Sprite {
  const c = document.createElement("canvas");
  c.width = 512;
  c.height = 96;
  const ctx = c.getContext("2d")!;
  ctx.font = "500 30px -apple-system, system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillStyle = "rgba(253, 252, 249, 0.92)";
  const tw = ctx.measureText(text).width + 28;
  ctx.beginPath();
  ctx.roundRect((512 - tw) / 2, 18, tw, 60, 8);
  ctx.fill();
  ctx.strokeStyle = "rgba(232, 229, 221, 1)";
  ctx.lineWidth = 1;
  ctx.stroke();
  ctx.fillStyle = "#2C2C2A";
  ctx.fillText(text, 256, 48);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false });
  const sprite = new THREE.Sprite(mat);
  sprite.scale.set(2.6, 0.5, 1);
  return sprite;
}

const equipmentGroup = new THREE.Group();
scene.add(equipmentGroup);

items.forEach((item, idx) => {
  const col = idx % cols;
  const row = Math.floor(idx / cols);
  const x = (col - (cols - 1) / 2) * TILE;
  const z = (row - (rows - 1) / 2) * TILE;

  const pad = new THREE.Mesh(new THREE.BoxGeometry(TILE * 0.92, 0.04, TILE * 0.92), padMat);
  pad.position.set(x, 0.02, z);
  pad.receiveShadow = true;
  scene.add(pad);

  const outline = new THREE.LineSegments(
    new THREE.EdgesGeometry(new THREE.BoxGeometry(TILE * 0.92, 0.04, TILE * 0.92)),
    new THREE.LineBasicMaterial({ color: 0xe8e5dd }),
  );
  outline.position.copy(pad.position);
  scene.add(outline);

  const eqInstance: Equipment = { ...item.equipment, position: [x, 0, z] };
  const mesh = createEquipmentMesh(eqInstance, { roomHeight: ROOM_HEIGHT });
  equipmentGroup.add(mesh);

  const label = makeLabelSprite(item.name);
  label.position.set(x, 2.7, z + TILE * 0.42);
  labelGroup.add(label);
});

// === Camera + controls ===
const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 200);
camera.position.set(0, 9, 11);
camera.lookAt(0, 1, 0);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.target.set(0, 1, 0);
controls.maxPolarAngle = Math.PI / 2.05;

// === Demo SceneGraph for the physics layers (heatmap / airflow / zones) ===
const demoZ = (rows / 2) * TILE + 3;
const demoSceneData: SceneGraph = {
  id: "gallery_demo",
  name: "demo",
  description: "",
  room: {
    type: "demo_room",
    dimensions: [6, ROOM_HEIGHT, 4.5],
    openings: [],
  },
  furniture: [
    {
      id: "rack_demo",
      category: "server_rack",
      label: "demo rack",
      position: [2.0, 0, 2.25],
      rotation: [0, 0, 0],
      size: [0.6, 2.0, 1.0],
      color: "#37474F",
      heatOutput: 12,
      relations: [],
    },
    {
      id: "ac_demo",
      category: "ceiling_ac",
      label: "demo ac",
      position: [4.5, 0, 2.25],
      rotation: [0, 0, 0],
      size: [1.0, 0.25, 1.0],
      color: "#0097A7",
      heatOutput: 0,
      relations: [],
    },
  ],
  score: {
    total: 0.85,
    thermal: 0.85,
    cooling: 0.85,
    cable: 0.85,
    proximity: 0.85,
    constraint: 0.85,
  },
};

const demoOrigin = new THREE.Vector3(-3, 0, demoZ);
const demoGroup = new THREE.Group();
demoGroup.position.copy(demoOrigin);
scene.add(demoGroup);

const roomGroup = new THREE.Group();
const heatmapGroup = new THREE.Group();
const airflowGroup = new THREE.Group();
const zonesGroup = new THREE.Group();
const demoFurniture = new THREE.Group();
demoGroup.add(roomGroup);
demoGroup.add(heatmapGroup);
demoGroup.add(airflowGroup);
demoGroup.add(zonesGroup);
demoGroup.add(demoFurniture);

buildRoom(roomGroup, demoSceneData);
buildHeatmap(heatmapGroup, demoSceneData, 1.0);
buildZones(zonesGroup, demoSceneData);

const airflowSystem = new AirflowSystem(airflowGroup);
airflowSystem.init(demoSceneData);

for (const f of demoSceneData.furniture) {
  demoFurniture.add(createEquipmentMesh(f, { roomHeight: ROOM_HEIGHT }));
}

const demoLabel = makeLabelSprite("Heatmap · Airflow · Zones (demo)");
demoLabel.position.set(0, 3.4, demoZ + 4);
demoLabel.scale.set(4.0, 0.7, 1);
labelGroup.add(demoLabel);

let heatmapVisible = true;
let airflowVisible = true;
let zonesVisible = false;
heatmapGroup.visible = heatmapVisible;
airflowGroup.visible = airflowVisible;
zonesGroup.visible = zonesVisible;

function syncToggle(id: string, on: boolean) {
  document.getElementById(id)?.classList.toggle("active", on);
}
syncToggle("btn-heatmap", heatmapVisible);
syncToggle("btn-airflow", airflowVisible);
syncToggle("btn-zones", zonesVisible);

document.getElementById("btn-heatmap")?.addEventListener("click", () => {
  heatmapVisible = !heatmapVisible;
  heatmapGroup.visible = heatmapVisible;
  syncToggle("btn-heatmap", heatmapVisible);
});
document.getElementById("btn-airflow")?.addEventListener("click", () => {
  airflowVisible = !airflowVisible;
  airflowGroup.visible = airflowVisible;
  syncToggle("btn-airflow", airflowVisible);
});
document.getElementById("btn-zones")?.addEventListener("click", () => {
  zonesVisible = !zonesVisible;
  zonesGroup.visible = zonesVisible;
  syncToggle("btn-zones", zonesVisible);
});

// === Sidebar item list ===
const list = document.getElementById("item-list")!;
list.innerHTML = items
  .map(
    (it) => `
  <div class="item-card">
    <div class="name">${it.name}</div>
    <div class="cat">category: ${it.equipment.category}</div>
    <div class="desc">${it.description}</div>
  </div>`,
  )
  .join("");

// === Resize + render loop ===
function onResize() {
  const parent = canvas.parentElement!;
  const w = parent.clientWidth;
  const h = parent.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", onResize);
onResize();

function tick() {
  requestAnimationFrame(tick);
  controls.update();
  if (airflowVisible && airflowSystem.isReady()) {
    airflowSystem.update(demoSceneData, 1.0);
  }
  renderer.render(scene, camera);
}
tick();
