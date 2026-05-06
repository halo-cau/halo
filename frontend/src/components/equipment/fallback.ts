import * as THREE from "three";
import type { Equipment } from "../../data/sceneGraphs";

export function createFallback(item: Equipment): THREE.Group {
  const group = new THREE.Group();
  const [w, h, d] = item.size;

  const geo = new THREE.BoxGeometry(w, h, d);
  const mat = new THREE.MeshStandardMaterial({
    color: item.color,
    roughness: 0.5,
  });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.position.y = h / 2;
  group.add(mesh);

  return group;
}
