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

interface VisualizeResult {
  raw_glb: string;
  cleaned_glb: string;
  semantic_glb: string | null;
}
let result: VisualizeResult | null = null;

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

// ── Stage switching ───────────────────────────────────────
function setActiveStage(stage: string) {
  if (!result) return;

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

  // Update button state
  stageBtns.querySelectorAll(".stage-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.getAttribute("data-stage") === stage);
  });

  // Show legend only for semantic
  legendEl.style.display = stage === "semantic" ? "block" : "none";
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

    setActiveStage("raw");
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err);
    setStatus(`Error: ${msg}`, true);
  } finally {
    demoBtn.disabled = false;
  }
});
