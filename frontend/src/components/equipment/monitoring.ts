import * as THREE from "three";
import type { Equipment } from "../../data/sceneGraphs";

export function createMonitoring(item: Equipment): THREE.Group {
  const group = new THREE.Group();
  const [w, h, d] = item.size;

  const bodyMat = new THREE.MeshStandardMaterial({
    color: item.color,
    roughness: 0.3,
    metalness: 0.3,
  });

  const bodyGeo = new THREE.BoxGeometry(w, h * 0.7, d);
  const body = new THREE.Mesh(bodyGeo, bodyMat);
  body.position.y = h * 0.35;
  group.add(body);

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

  const antennaGeo = new THREE.CylinderGeometry(0.01, 0.01, h * 0.3, 6);
  const antenna = new THREE.Mesh(antennaGeo, new THREE.MeshStandardMaterial({ color: 0x424242 }));
  antenna.position.set(w * 0.3, h * 0.85 + h * 0.15, 0);
  group.add(antenna);

  return group;
}
