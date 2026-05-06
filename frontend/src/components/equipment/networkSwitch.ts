import * as THREE from "three";
import type { Equipment } from "../../data/sceneGraphs";

export function createNetworkSwitch(item: Equipment): THREE.Group {
  const group = new THREE.Group();
  const [w, h, d] = item.size;

  const bodyMat = new THREE.MeshStandardMaterial({
    color: item.color,
    roughness: 0.3,
    metalness: 0.5,
  });

  const bodyGeo = new THREE.BoxGeometry(w, h, d);
  const body = new THREE.Mesh(bodyGeo, bodyMat);
  body.position.y = h / 2;
  group.add(body);

  for (let row = 0; row < 4; row++) {
    for (let col = 0; col < 6; col++) {
      const portGeo = new THREE.BoxGeometry(0.03, 0.03, 0.01);
      const portMat = new THREE.MeshStandardMaterial({
        color: 0x00e676,
        emissive: 0x00e676,
        emissiveIntensity: 0.8,
      });
      const port = new THREE.Mesh(portGeo, portMat);
      port.position.set(-w * 0.3 + col * w * 0.12, h * 0.4 + row * h * 0.12, d / 2 + 0.011);
      group.add(port);
    }
  }

  const labelGeo = new THREE.BoxGeometry(w * 0.6, 0.02, 0.01);
  const labelMat = new THREE.MeshStandardMaterial({
    color: 0x42a5f5,
    emissive: 0x42a5f5,
    emissiveIntensity: 0.5,
  });
  const label = new THREE.Mesh(labelGeo, labelMat);
  label.position.set(0, h * 0.85, d / 2 + 0.011);
  group.add(label);

  return group;
}
