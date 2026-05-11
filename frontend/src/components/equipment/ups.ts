import * as THREE from "three";
import type { Equipment } from "../../data/sceneGraphs";

export function createUps(item: Equipment): THREE.Group {
  const group = new THREE.Group();
  const [w, h, d] = item.size;

  const bodyMat = new THREE.MeshStandardMaterial({
    color: item.color,
    roughness: 0.4,
    metalness: 0.3,
  });

  const bodyGeo = new THREE.BoxGeometry(w, h, d);
  const body = new THREE.Mesh(bodyGeo, bodyMat);
  body.position.y = h / 2;
  group.add(body);

  const ledGeo = new THREE.BoxGeometry(w * 0.3, 0.04, 0.01);
  const ledMat = new THREE.MeshStandardMaterial({
    color: 0x00e676,
    emissive: 0x00e676,
    emissiveIntensity: 1.0,
  });
  const led = new THREE.Mesh(ledGeo, ledMat);
  led.position.set(0, h * 0.8, d / 2 + 0.011);
  group.add(led);

  const battGeo = new THREE.BoxGeometry(w * 0.5, h * 0.3, 0.01);
  const battMat = new THREE.MeshStandardMaterial({
    color: 0xbf360c,
    roughness: 0.5,
  });
  const batt = new THREE.Mesh(battGeo, battMat);
  batt.position.set(0, h * 0.4, d / 2 + 0.011);
  group.add(batt);

  return group;
}
