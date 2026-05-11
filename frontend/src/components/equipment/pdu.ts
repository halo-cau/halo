import * as THREE from "three";
import type { Equipment } from "../../data/sceneGraphs";

export function createPdu(item: Equipment): THREE.Group {
  const group = new THREE.Group();
  const [w, h, d] = item.size;

  const bodyMat = new THREE.MeshStandardMaterial({
    color: item.color,
    roughness: 0.4,
    metalness: 0.4,
  });

  const bodyGeo = new THREE.BoxGeometry(w, h, d);
  const body = new THREE.Mesh(bodyGeo, bodyMat);
  body.position.y = h / 2;
  group.add(body);

  for (let i = 0; i < 6; i++) {
    const socketGeo = new THREE.BoxGeometry(0.04, 0.04, 0.01);
    const socketMat = new THREE.MeshStandardMaterial({ color: 0x212121 });
    const socket = new THREE.Mesh(socketGeo, socketMat);
    socket.position.set(0, 0.3 + i * 0.25, d / 2 + 0.011);
    group.add(socket);
  }

  return group;
}
