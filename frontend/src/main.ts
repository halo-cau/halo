import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import {
  allScenes,
  trainingHistory,
  getLoadFactor,
  coolingEnergyBase,
  peakTempBase,
  type SceneGraph,
  type Equipment,
  type Score,
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
  camera = new THREE.PerspectiveCamera(
    50,
    canvas.clientWidth / canvas.clientHeight,
    0.1,
    200
  );
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
  scene.add(roomGroup);
  scene.add(furnitureGroup);
  scene.add(heatmapGroup);

  // Grid
  const grid = new THREE.GridHelper(18, 36, 0x1a2744, 0x111a2a);
  grid.position.set(6, -0.01, 4.5);
  scene.add(grid);

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
  const tileMat = new THREE.LineBasicMaterial({ color: 0xb0b0b0, linewidth: 1 });
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
  const edgeMat = new THREE.LineBasicMaterial({ color: 0x00bcd4, linewidth: 2 });
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
        opening.wall === "north" ? 0.2 : opening.wall === "south" ? rd - 0.2 : opening.position[2]
      );
      if (opening.wall === "east") arrowOrigin.set(rw - 0.2, opening.position[1], opening.position[2]);
      if (opening.wall === "west") arrowOrigin.set(0.2, opening.position[1], opening.position[2]);

      const arrow = new THREE.ArrowHelper(arrowDir, arrowOrigin, 1.2, 0x00bcd4, 0.3, 0.15);
      roomGroup.add(arrow);
    }
  }
}

// ===== Heatmap =====
function buildHeatmap(sceneData: SceneGraph, loadFactor = 1.0) {
  heatmapGroup.clear();
  const [rw, , rd] = sceneData.room.dimensions;
  const resolution = 0.5;

  // 각 장비의 발열량 * 부하계수로 히트맵 생성
  const heatSources = sceneData.furniture
    .filter((f) => f.heatOutput > 0)
    .map((f) => ({ x: f.position[0], z: f.position[2], heat: f.heatOutput * loadFactor }));

  const coolingSources = sceneData.furniture.filter((f) => f.category === "cooling_unit");

  let maxHeatValue = 0;

  for (let x = resolution / 2; x < rw; x += resolution) {
    for (let z = resolution / 2; z < rd; z += resolution) {
      let totalHeat = 0;
      for (const src of heatSources) {
        const dist = Math.sqrt((x - src.x) ** 2 + (z - src.z) ** 2);
        totalHeat += (src.heat / (1 + dist * dist)) * 2;
      }

      // 냉각 장치 효과
      for (const cool of coolingSources) {
        const dist = Math.sqrt((x - cool.position[0]) ** 2 + (z - cool.position[2]) ** 2);
        totalHeat -= (3 * loadFactor) / (1 + dist * dist);
      }

      totalHeat = Math.max(0, Math.min(totalHeat, 10));
      if (totalHeat > maxHeatValue) maxHeatValue = totalHeat;

      // 색상: 파랑(시원) → 초록 → 노랑 → 빨강(뜨거움)
      const t = totalHeat / 10;
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
  return maxHeatValue;
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
      if (category === "server_rack" && mat.transparent && 'opacity' in mat && mat.side === THREE.DoubleSide) {
        const effectiveHeat = (heatOutput || 0) * loadFactor;
        mat.opacity = 0.05 + effectiveHeat * 0.02;
        if (effectiveHeat > 10) mat.color.setHex(0xff3300);
        else if (effectiveHeat > 5) mat.color.setHex(0xff8800);
        else mat.color.setHex(0xffcc00);
      }

      // 냉각 장치 glow 업데이트
      if (category === "cooling_unit" && mat.transparent && 'opacity' in mat && mat.side === THREE.DoubleSide) {
        mat.opacity = 0.03 + 0.12 * loadFactor;
      }
    });
  });
}

function updateMonitorPanel(sceneData: SceneGraph, loadFactor: number, maxHeat: number) {
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
  tempEl.className = "mon-value " + (
    currentMaxTemp >= 40 ? "temp-danger" :
    currentMaxTemp >= 32 ? "temp-warn" : "temp-ok"
  );

  // Cooling Energy
  const baseCooling = coolingEnergyBase[sceneData.id] || 30;
  const currentCooling = Math.round(baseCooling * loadFactor * 10) / 10;
  document.getElementById("mon-cooling")!.innerHTML =
    `${currentCooling.toFixed(1)}<span class="mon-unit">kW</span>`;

  // PUE (Power Usage Effectiveness)
  // PUE = (IT Power + Cooling Power) / IT Power
  const totalServerPower = sceneData.furniture
    .filter(f => f.heatOutput > 0)
    .reduce((sum, f) => sum + f.heatOutput, 0) * loadFactor;
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
        const ledColor = item.heatOutput > 10 ? 0xff1744 : item.heatOutput > 5 ? 0xffab00 : 0x00e676;
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
      new THREE.MeshStandardMaterial({ color: 0x212121, metalness: 0.7 })
    );
    topFrame.position.y = h;
    group.add(topFrame);

    // 발열 표시 (상단 glow)
    if (item.heatOutput > 0) {
      const glowGeo = new THREE.PlaneGeometry(w * 0.6, d * 0.6);
      const glowColor = item.heatOutput > 10 ? 0xff3300 : item.heatOutput > 5 ? 0xff8800 : 0xffcc00;
      const glowMat = new THREE.MeshBasicMaterial({
        color: glowColor,
        transparent: true,
        opacity: 0.15 + item.heatOutput * 0.01,
        side: THREE.DoubleSide,
      });
      const glow = new THREE.Mesh(glowGeo, glowMat);
      glow.rotation.x = -Math.PI / 2;
      glow.position.y = h + 0.05;
      group.add(glow);
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
      const grillMat = new THREE.MeshStandardMaterial({ color: 0x004d40, metalness: 0.5 });
      const grill = new THREE.Mesh(grillGeo, grillMat);
      grill.position.set(0, h * 0.3 + i * h * 0.08, d / 2 + 0.01);
      group.add(grill);
    }

    // 상단 팬
    const fanGeo = new THREE.CylinderGeometry(w * 0.3, w * 0.3, 0.05, 24);
    const fanMat = new THREE.MeshStandardMaterial({ color: 0x004d40, metalness: 0.6 });
    const fan = new THREE.Mesh(fanGeo, fanMat);
    fan.position.y = h + 0.025;
    group.add(fan);

    // 냉각 효과 표시 (파란 glow)
    const coolGlow = new THREE.Mesh(
      new THREE.PlaneGeometry(w * 1.5, d * 1.5),
      new THREE.MeshBasicMaterial({
        color: 0x00bcd4,
        transparent: true,
        opacity: 0.08,
        side: THREE.DoubleSide,
      })
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
        port.position.set(
          -w * 0.3 + col * w * 0.12,
          h * 0.4 + row * h * 0.12,
          d / 2 + 0.011
        );
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
    const railMat = new THREE.MeshStandardMaterial({ color: 0xf9a825, metalness: 0.5 });
    const rail1 = new THREE.Mesh(railGeo, railMat);
    rail1.position.set(0, 0.19, d / 2);
    group.add(rail1);

    const rail2 = new THREE.Mesh(railGeo, railMat);
    rail2.position.set(0, 0.19, -d / 2);
    group.add(rail2);

    // 케이블 묶음 시뮬레이션
    const cableGeo = new THREE.CylinderGeometry(0.04, 0.04, w * 0.9, 8);
    const cableMat = new THREE.MeshStandardMaterial({ color: 0x212121, roughness: 0.8 });
    const cable = new THREE.Mesh(cableGeo, cableMat);
    cable.rotation.z = Math.PI / 2;
    cable.position.y = 0.19;
    group.add(cable);

  } else {
    // Fallback
    const geo = new THREE.BoxGeometry(w, h, d);
    const mat = new THREE.MeshStandardMaterial({ color: item.color, roughness: 0.5 });
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
    THREE.MathUtils.degToRad(item.rotation[2])
  );

  group.userData = { id: item.id, label: item.label, category: item.category, heatOutput: item.heatOutput };
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
        const ease = 1 - Math.pow(1 - t, 3);
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
    "total", "thermal", "cooling", "cable", "proximity", "constraint",
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
      .map(
        (r) =>
          `<span class="rel rel-${r.type}">${r.type} &rarr; ${r.target}</span>`
      )
      .join("");
    const heatStr = f.heatOutput > 0
      ? `<div class="sg-heat">heat: ${f.heatOutput} kW</div>`
      : "";
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
    total: "Total", thermal: "Thermal", cooling: "Cooling",
    cable: "Cable", proximity: "Proximity", constraint: "Constraint",
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
function setupButtons() {
  document.getElementById("btn-prev")!.addEventListener("click", () => {
    if (animating || currentSceneIndex === 0) return;
    currentSceneIndex--;
    buildScene(allScenes[currentSceneIndex]);
    updateUI(allScenes[currentSceneIndex]);
  });

  document.getElementById("btn-next")!.addEventListener("click", () => {
    if (animating || currentSceneIndex === allScenes.length - 1) return;
    currentSceneIndex++;
    buildScene(allScenes[currentSceneIndex]);
    updateUI(allScenes[currentSceneIndex]);
  });

  document.querySelectorAll(".step").forEach((el, idx) => {
    el.addEventListener("click", () => {
      if (animating || idx === currentSceneIndex) return;
      currentSceneIndex = idx;
      buildScene(allScenes[currentSceneIndex]);
      updateUI(allScenes[currentSceneIndex]);
    });
  });

  document.getElementById("btn-relations")!.addEventListener("click", () => {
    const btn = document.getElementById("btn-relations")!;
    const showing = btn.dataset.showing === "true";
    furnitureGroup.children.forEach((child) => {
      if (child instanceof THREE.Line) child.visible = !showing;
    });
    btn.dataset.showing = showing ? "false" : "true";
    btn.textContent = showing ? "Show Relations" : "Hide Relations";
  });

  document.getElementById("btn-heatmap")!.addEventListener("click", () => {
    heatmapVisible = !heatmapVisible;
    heatmapGroup.visible = heatmapVisible;
    const btn = document.getElementById("btn-heatmap")!;
    btn.classList.toggle("active-toggle", heatmapVisible);
  });

  // Simulation controls
  document.getElementById("sim-play")!.addEventListener("click", () => {
    simPlaying = !simPlaying;
    const btn = document.getElementById("sim-play")!;
    btn.innerHTML = simPlaying ? "&#9646;&#9646;" : "&#9654;";
    // 시뮬레이션 시작 시 히트맵 자동 켜기
    if (simPlaying && !heatmapVisible) {
      heatmapVisible = true;
      heatmapGroup.visible = true;
      document.getElementById("btn-heatmap")!.classList.add("active-toggle");
    }
  });

  const simSlider = document.getElementById("sim-slider") as HTMLInputElement;
  simSlider.addEventListener("input", () => {
    simMinutes = Number(simSlider.value);
    updateSimSlider();
    updateSimulation();
  });

  document.getElementById("sim-speed")!.addEventListener("click", () => {
    const speeds = [1, 2, 4, 8];
    const idx = speeds.indexOf(simSpeed);
    simSpeed = speeds[(idx + 1) % speeds.length];
    document.getElementById("sim-speed")!.textContent = `${simSpeed}x`;
  });

  // 초기 모니터 업데이트
  updateSimulation();
}

// ===== Start =====
init();
