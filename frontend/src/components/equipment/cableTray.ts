import * as THREE from "three";
import type { Equipment } from "../../data/sceneGraphs";

export function createCableTray(item: Equipment): THREE.Group {
  const group = new THREE.Group();
  const [w, , d] = item.size;

  const trayMat = new THREE.MeshStandardMaterial({
    color: item.color,
    roughness: 0.5,
    metalness: 0.4,
    transparent: true,
    opacity: 0.7,
  });

  const trayGeo = new THREE.BoxGeometry(w, 0.05, d);
  const tray = new THREE.Mesh(trayGeo, trayMat);
  tray.position.y = 0.15;
  group.add(tray);

  const railGeo = new THREE.BoxGeometry(w, 0.08, 0.02);
  const railMat = new THREE.MeshStandardMaterial({
    color: 0xf9a825,
    metalness: 0.5,
  });
  const rail1 = new THREE.Mesh(railGeo, railMat);
  rail1.position.set(0, 0.19, d / 2);
  group.add(rail1);

  const rail2 = new THREE.Mesh(railGeo, railMat);
  rail2.position.set(0, 0.19, -d / 2);
  group.add(rail2);

  const cableGeo = new THREE.CylinderGeometry(0.04, 0.04, w * 0.9, 8);
  const cableMat = new THREE.MeshStandardMaterial({
    color: 0x212121,
    roughness: 0.8,
  });
  const cable = new THREE.Mesh(cableGeo, cableMat);
  cable.rotation.z = Math.PI / 2;
  cable.position.y = 0.19;
  group.add(cable);

  return group;
}
