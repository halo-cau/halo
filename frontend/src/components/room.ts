import * as THREE from "three";
import type { SceneGraph } from "../data/sceneGraphs";

export function buildRoom(group: THREE.Group, sceneData: SceneGraph): void {
  group.clear();
  const [rw, rh, rd] = sceneData.room.dimensions;

  // Floor — raised-floor look
  const floorGeo = new THREE.PlaneGeometry(rw, rd);
  const floorMat = new THREE.MeshStandardMaterial({
    color: 0x282828,
    roughness: 0.85,
  });
  const floor = new THREE.Mesh(floorGeo, floorMat);
  floor.rotation.x = -Math.PI / 2;
  floor.position.set(rw / 2, 0, rd / 2);
  floor.receiveShadow = true;
  group.add(floor);

  // Floor tile grid
  const tileSize = 0.6;
  const tileMat = new THREE.LineBasicMaterial({ color: 0x5a5a5a, linewidth: 1 });
  for (let x = 0; x <= rw; x += tileSize) {
    const geo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(x, 0.002, 0),
      new THREE.Vector3(x, 0.002, rd),
    ]);
    group.add(new THREE.Line(geo, tileMat));
  }
  for (let z = 0; z <= rd; z += tileSize) {
    const geo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(0, 0.002, z),
      new THREE.Vector3(rw, 0.002, z),
    ]);
    group.add(new THREE.Line(geo, tileMat));
  }

  // Walls
  const wallMat = new THREE.MeshStandardMaterial({
    color: 0x606060,
    transparent: true,
    opacity: 0.40,
    side: THREE.DoubleSide,
  });

  const northWall = new THREE.Mesh(new THREE.PlaneGeometry(rw, rh), wallMat.clone());
  northWall.position.set(rw / 2, rh / 2, 0);
  group.add(northWall);

  const southWall = new THREE.Mesh(new THREE.PlaneGeometry(rw, rh), wallMat.clone());
  southWall.position.set(rw / 2, rh / 2, rd);
  southWall.rotation.y = Math.PI;
  group.add(southWall);

  const westWall = new THREE.Mesh(new THREE.PlaneGeometry(rd, rh), wallMat.clone());
  westWall.rotation.y = Math.PI / 2;
  westWall.position.set(0, rh / 2, rd / 2);
  group.add(westWall);

  const eastWall = new THREE.Mesh(new THREE.PlaneGeometry(rd, rh), wallMat.clone());
  eastWall.rotation.y = -Math.PI / 2;
  eastWall.position.set(rw, rh / 2, rd / 2);
  group.add(eastWall);

  // Floor outline
  const edgeMat = new THREE.LineBasicMaterial({ color: 0x777777, linewidth: 1 });
  const floorOutline = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(0, 0.005, 0),
    new THREE.Vector3(rw, 0.005, 0),
    new THREE.Vector3(rw, 0.005, rd),
    new THREE.Vector3(0, 0.005, rd),
    new THREE.Vector3(0, 0.005, 0),
  ]);
  group.add(new THREE.Line(floorOutline, edgeMat));

  // Corner pillars
  const pillarGeo = new THREE.BoxGeometry(0.25, rh, 0.25);
  const pillarMat = new THREE.MeshStandardMaterial({ color: 0x404040 });
  const corners: [number, number, number][] = [
    [0, rh / 2, 0],
    [rw, rh / 2, 0],
    [rw, rh / 2, rd],
    [0, rh / 2, rd],
  ];
  for (const [cx, cy, cz] of corners) {
    const pillar = new THREE.Mesh(pillarGeo, pillarMat);
    pillar.position.set(cx, cy, cz);
    pillar.castShadow = true;
    group.add(pillar);
  }

  // Openings
  for (const opening of sceneData.room.openings) {
    const isDoor = opening.type === "door";
    const isVent = opening.type === "vent";
    const color = isDoor ? 0xe89e4f : isVent ? 0x378add : 0x1d9e75;
    const markerH = isDoor ? 2.4 : 0.8;

    const markerGeo = new THREE.BoxGeometry(opening.width, markerH, 0.08);
    const markerMat = new THREE.MeshStandardMaterial({
      color,
      transparent: true,
      opacity: isDoor ? 0.5 : 0.35,
    });
    const marker = new THREE.Mesh(markerGeo, markerMat);

    if (opening.wall === "north") {
      marker.position.set(opening.position[0], isDoor ? markerH / 2 : opening.position[1], 0);
    } else if (opening.wall === "south") {
      marker.position.set(opening.position[0], isDoor ? markerH / 2 : opening.position[1], rd);
    } else if (opening.wall === "east") {
      marker.rotation.y = Math.PI / 2;
      marker.position.set(rw, isDoor ? markerH / 2 : opening.position[1], opening.position[2]);
    } else {
      marker.rotation.y = Math.PI / 2;
      marker.position.set(0, isDoor ? markerH / 2 : opening.position[1], opening.position[2]);
    }
    group.add(marker);

    if (isVent) {
      const arrowDir = new THREE.Vector3(0, 0, 1);
      if (opening.wall === "south") arrowDir.set(0, 0, -1);
      else if (opening.wall === "east") arrowDir.set(-1, 0, 0);
      else if (opening.wall === "west") arrowDir.set(1, 0, 0);

      const arrowOrigin = new THREE.Vector3(
        opening.position[0],
        opening.position[1],
        opening.wall === "north" ? 0.2 : opening.wall === "south" ? rd - 0.2 : opening.position[2],
      );
      if (opening.wall === "east")
        arrowOrigin.set(rw - 0.2, opening.position[1], opening.position[2]);
      if (opening.wall === "west") arrowOrigin.set(0.2, opening.position[1], opening.position[2]);

      const arrow = new THREE.ArrowHelper(arrowDir, arrowOrigin, 1.2, 0x378add, 0.3, 0.15);
      group.add(arrow);
    }
  }
}
