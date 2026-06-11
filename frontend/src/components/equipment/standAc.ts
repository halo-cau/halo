import * as THREE from "three";
import type { Equipment } from "../../data/sceneGraphs";

// Floor-standing package AC (스탠드형 에어컨), modelled on the scanned Century unit:
// cream body, vertical louver bank up top, a small LCD control panel, and a tall
// horizontal intake/discharge grille over the lower two-thirds.
export function createStandAc(item: Equipment): THREE.Group {
  const group = new THREE.Group();
  const [w, h, d] = item.size;
  const front = d / 2;

  const bodyMat = new THREE.MeshStandardMaterial({
    color: 0xeae7df, // off-white / cream casing
    roughness: 0.55,
    metalness: 0.1,
  });
  const body = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), bodyMat);
  body.position.y = h / 2;
  group.add(body);

  // ── Top louver bank: angled black discharge slats across the top ~28% ──
  const louverMat = new THREE.MeshStandardMaterial({
    color: 0x14171a,
    roughness: 0.5,
    metalness: 0.4,
  });
  const louverTop = h * 0.96;
  const louverBot = h * 0.7;
  const nLouver = 5;
  for (let i = 0; i < nLouver; i++) {
    const ly = louverBot + ((louverTop - louverBot) * (i + 0.5)) / nLouver;
    const slat = new THREE.Mesh(
      new THREE.BoxGeometry(w * 0.86, ((louverTop - louverBot) / nLouver) * 0.6, 0.015),
      louverMat,
    );
    slat.position.set(0, ly, front + 0.004);
    slat.rotation.x = -0.35; // tilt the discharge louvers downward
    group.add(slat);
  }

  // ── Control panel: small dark module with a cyan LCD, upper-centre ──
  const panel = new THREE.Mesh(
    new THREE.BoxGeometry(w * 0.22, h * 0.07, 0.012),
    new THREE.MeshStandardMaterial({ color: 0x20242a, roughness: 0.4 }),
  );
  panel.position.set(0, h * 0.64, front + 0.006);
  group.add(panel);
  const lcd = new THREE.Mesh(
    new THREE.BoxGeometry(w * 0.12, h * 0.035, 0.008),
    new THREE.MeshStandardMaterial({ color: 0x4fd2e0, emissive: 0x2bb6c4, emissiveIntensity: 0.9 }),
  );
  lcd.position.set(0, h * 0.64, front + 0.012);
  group.add(lcd);

  // ── Lower grille: fine horizontal intake slats over the lower ~58% ──
  const grilleMat = new THREE.MeshStandardMaterial({
    color: 0xb9bdbf,
    roughness: 0.6,
    metalness: 0.5,
  });
  const grilleTop = h * 0.58;
  const grilleBot = h * 0.04;
  const nGrille = 22;
  for (let i = 0; i < nGrille; i++) {
    const gy = grilleBot + ((grilleTop - grilleBot) * (i + 0.5)) / nGrille;
    const slat = new THREE.Mesh(
      new THREE.BoxGeometry(w * 0.84, ((grilleTop - grilleBot) / nGrille) * 0.55, 0.01),
      grilleMat,
    );
    slat.position.set(0, gy, front + 0.003);
    group.add(slat);
  }

  // Base plinth so it reads as floor-standing.
  const plinth = new THREE.Mesh(
    new THREE.BoxGeometry(w * 1.02, h * 0.03, d * 1.02),
    new THREE.MeshStandardMaterial({ color: 0x2b2e31, roughness: 0.7 }),
  );
  plinth.position.y = h * 0.015;
  group.add(plinth);

  // Blue discharge arrow out of the upper-front louver vent — the discharge direction (forward + a slight
  // downward tilt, matching the louvers and the airflow streamlines). Rotates with the unit.
  const dischargeArrow = new THREE.ArrowHelper(
    new THREE.Vector3(0, -0.28, 1).normalize(),
    new THREE.Vector3(0, (louverTop + louverBot) / 2, front + 0.05),
    1.0,
    0x378add,
    0.28,
    0.2,
  );
  group.add(dischargeArrow);

  return group;
}
