import * as THREE from "three";
import type { Equipment } from "../../data/sceneGraphs";

// Wall/floor electrical distribution cabinet (분전반 / power panel), modelled on the scanned
// "L4-2 분전반" unit: gray-green steel body, a slightly recessed hinged door, a vertical handle,
// and a small white nameplate near the top.
export function createPowerCabinet(item: Equipment): THREE.Group {
  const group = new THREE.Group();
  const [w, h, d] = item.size;
  const front = d / 2;

  const steel = new THREE.MeshStandardMaterial({ color: 0x8a978f, roughness: 0.45, metalness: 0.55 });
  const body = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), steel);
  body.position.y = h / 2;
  group.add(body);

  // Recessed door panel (slightly inset + proud of the front face for a seam).
  const door = new THREE.Mesh(
    new THREE.BoxGeometry(w * 0.9, h * 0.92, 0.02),
    new THREE.MeshStandardMaterial({ color: 0x94a199, roughness: 0.4, metalness: 0.55 }),
  );
  door.position.set(0, h / 2, front + 0.005);
  group.add(door);

  // Vertical handle / latch on the right side of the door.
  const handle = new THREE.Mesh(
    new THREE.BoxGeometry(0.025, h * 0.18, 0.03),
    new THREE.MeshStandardMaterial({ color: 0x2c2f31, roughness: 0.4, metalness: 0.7 }),
  );
  handle.position.set(w * 0.34, h * 0.5, front + 0.02);
  group.add(handle);

  // White nameplate near the top (the "L4-2 분전반" label).
  const plate = new THREE.Mesh(
    new THREE.BoxGeometry(w * 0.3, h * 0.06, 0.006),
    new THREE.MeshStandardMaterial({ color: 0xf3f4f1, roughness: 0.6 }),
  );
  plate.position.set(0, h * 0.82, front + 0.018);
  group.add(plate);

  return group;
}
