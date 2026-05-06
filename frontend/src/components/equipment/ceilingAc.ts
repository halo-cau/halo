import * as THREE from "three";
import type { Equipment } from "../../data/sceneGraphs";

// Ceiling-mounted cassette air conditioner (4-way blow type).
// The data file places cooling units at y=0 with full-room height,
// but a real ceiling AC mounts flush against the ceiling. We render
// the unit at `ceilingY - depth` so it visually sits on the ceiling
// while physics keeps reading the item's [x, z] anchor.
export function createCeilingAc(item: Equipment, roomHeight: number): THREE.Group {
  const group = new THREE.Group();
  const [w, , d] = item.size;
  const cassetteH = 0.18;
  const grilleInset = 0.02;
  const ceilingY = Math.max(roomHeight - 0.02, cassetteH + 0.5);

  // === Outer housing (flange flush against ceiling) ===
  const flangeMat = new THREE.MeshStandardMaterial({
    color: 0xeceff1,
    roughness: 0.4,
    metalness: 0.2,
  });
  const flangeGeo = new THREE.BoxGeometry(w + 0.12, 0.04, d + 0.12);
  const flange = new THREE.Mesh(flangeGeo, flangeMat);
  flange.position.y = ceilingY - 0.02;
  group.add(flange);

  // === Cassette body (the rectangular box hanging just below the ceiling) ===
  const bodyMat = new THREE.MeshStandardMaterial({
    color: 0xf5f5f5,
    roughness: 0.5,
    metalness: 0.15,
  });
  const bodyGeo = new THREE.BoxGeometry(w, cassetteH, d);
  const body = new THREE.Mesh(bodyGeo, bodyMat);
  body.position.y = ceilingY - cassetteH / 2 - 0.04;
  group.add(body);

  // === Center intake grille (sucks return air up) ===
  const intakeMat = new THREE.MeshStandardMaterial({
    color: 0x37474f,
    roughness: 0.7,
    metalness: 0.3,
  });
  const intakeGeo = new THREE.BoxGeometry(w * 0.55, 0.015, d * 0.55);
  const intake = new THREE.Mesh(intakeGeo, intakeMat);
  intake.position.y = ceilingY - cassetteH - 0.045;
  group.add(intake);

  // Intake louver lines for visual texture
  const louverMat = new THREE.LineBasicMaterial({ color: 0x546e7a });
  const louverY = ceilingY - cassetteH - 0.04;
  for (let i = 1; i < 6; i++) {
    const t = i / 6;
    const lx = -w * 0.275 + t * w * 0.55;
    const lineGeo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(lx, louverY, -d * 0.275),
      new THREE.Vector3(lx, louverY, d * 0.275),
    ]);
    group.add(new THREE.Line(lineGeo, louverMat));
  }

  // === 4-way blow vents (around the perimeter, angled outward) ===
  const ventMat = new THREE.MeshStandardMaterial({
    color: 0xb0bec5,
    roughness: 0.4,
    metalness: 0.5,
  });
  const ventY = ceilingY - cassetteH - 0.02;
  const halfW = w / 2 - grilleInset;
  const halfD = d / 2 - grilleInset;

  const vents: { geo: THREE.BoxGeometry; pos: [number, number, number] }[] = [
    { geo: new THREE.BoxGeometry(w * 0.7, 0.025, 0.08), pos: [0, ventY, halfD - 0.04] },
    { geo: new THREE.BoxGeometry(w * 0.7, 0.025, 0.08), pos: [0, ventY, -halfD + 0.04] },
    { geo: new THREE.BoxGeometry(0.08, 0.025, d * 0.7), pos: [halfW - 0.04, ventY, 0] },
    { geo: new THREE.BoxGeometry(0.08, 0.025, d * 0.7), pos: [-halfW + 0.04, ventY, 0] },
  ];
  for (const v of vents) {
    const m = new THREE.Mesh(v.geo, ventMat);
    m.position.set(v.pos[0], v.pos[1], v.pos[2]);
    group.add(m);
  }

  // === Cool air glow underneath (visual cue of supply airflow) ===
  const glowGeo = new THREE.PlaneGeometry(w * 1.4, d * 1.4);
  const glowMat = new THREE.MeshBasicMaterial({
    color: 0x00bcd4,
    transparent: true,
    opacity: 0.18,
    side: THREE.DoubleSide,
    depthWrite: false,
  });
  const glow = new THREE.Mesh(glowGeo, glowMat);
  glow.rotation.x = -Math.PI / 2;
  glow.position.y = ceilingY - cassetteH - 0.05;
  group.add(glow);

  // Soft floor halo (where cold air lands) — keeps the "cooling presence"
  // visual hint even though the unit itself is now overhead.
  const floorHaloGeo = new THREE.PlaneGeometry(w * 2.4, d * 2.4);
  const floorHaloMat = new THREE.MeshBasicMaterial({
    color: 0x00bcd4,
    transparent: true,
    opacity: 0.08,
    side: THREE.DoubleSide,
    depthWrite: false,
  });
  const floorHalo = new THREE.Mesh(floorHaloGeo, floorHaloMat);
  floorHalo.rotation.x = -Math.PI / 2;
  floorHalo.position.y = 0.005;
  group.add(floorHalo);

  // Mounting brackets (4 thin posts disappearing into the ceiling)
  const bracketMat = new THREE.MeshStandardMaterial({
    color: 0x90a4ae,
    metalness: 0.6,
    roughness: 0.4,
  });
  const bracketGeo = new THREE.BoxGeometry(0.04, 0.06, 0.04);
  const bx = w / 2 - 0.06;
  const bz = d / 2 - 0.06;
  for (const [sx, sz] of [
    [bx, bz],
    [-bx, bz],
    [bx, -bz],
    [-bx, -bz],
  ]) {
    const br = new THREE.Mesh(bracketGeo, bracketMat);
    br.position.set(sx, ceilingY - 0.03, sz);
    group.add(br);
  }

  return group;
}
