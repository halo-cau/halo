import * as THREE from "three";
import type { SceneGraph } from "../data/sceneGraphs";

function makeTextSprite(text: string, color: string): THREE.Sprite {
  const canvas = document.createElement("canvas");
  canvas.width = 128;
  canvas.height = 48;
  const ctx = canvas.getContext("2d")!;
  ctx.font = "500 22px -apple-system, system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillStyle = color;
  ctx.globalAlpha = 0.25;
  const tw = ctx.measureText(text).width + 16;
  ctx.beginPath();
  ctx.roundRect((128 - tw) / 2, 6, tw, 36, 6);
  ctx.fill();
  ctx.globalAlpha = 0.9;
  ctx.fillStyle = color;
  ctx.fillText(text, 64, 24);
  const tex = new THREE.CanvasTexture(canvas);
  const mat = new THREE.SpriteMaterial({
    map: tex,
    transparent: true,
    depthWrite: false,
  });
  const sprite = new THREE.Sprite(mat);
  sprite.scale.set(1.2, 0.45, 1);
  return sprite;
}

export function buildZones(group: THREE.Group, sceneData: SceneGraph): void {
  group.clear();
  const quality = sceneData.score.total;
  const serverRacks = sceneData.furniture.filter((f) => f.heatOutput > 0);
  const coolingSources = sceneData.furniture.filter((f) => f.category === "cooling_unit");

  for (const rack of serverRacks) {
    const rotY = THREE.MathUtils.degToRad(rack.rotation[1]);
    const frontX = Math.sin(rotY);
    const frontZ = Math.cos(rotY);
    const [rackW, rackH] = rack.size;
    const zoneDepth = 1.2;
    const zoneW = rackW + 0.6;
    const zoneH = rackH + 0.3;

    // Cold zone (front)
    const coldGeo = new THREE.BoxGeometry(zoneW, zoneH, zoneDepth);
    const coldZone = new THREE.Mesh(
      coldGeo,
      new THREE.MeshBasicMaterial({
        color: 0x378add,
        transparent: true,
        opacity: 0.28,
        side: THREE.DoubleSide,
        depthWrite: false,
      }),
    );
    coldZone.position.set(
      rack.position[0] + frontX * (zoneDepth / 2 - 0.1),
      rackH / 2,
      rack.position[2] + frontZ * (zoneDepth / 2 - 0.1),
    );
    coldZone.rotation.y = rotY;
    group.add(coldZone);

    const coldEdge = new THREE.LineSegments(
      new THREE.EdgesGeometry(coldGeo),
      new THREE.LineBasicMaterial({ color: 0x378add, transparent: true, opacity: 0.35 }),
    );
    coldEdge.position.copy(coldZone.position);
    coldEdge.rotation.copy(coldZone.rotation);
    group.add(coldEdge);

    // Hot zone (back)
    const hotGeo = new THREE.BoxGeometry(zoneW, zoneH + 0.5, zoneDepth);
    const hotZone = new THREE.Mesh(
      hotGeo,
      new THREE.MeshBasicMaterial({
        color: 0xe89e4f,
        transparent: true,
        opacity: 0.28,
        side: THREE.DoubleSide,
        depthWrite: false,
      }),
    );
    hotZone.position.set(
      rack.position[0] - frontX * (zoneDepth / 2 - 0.1),
      (rackH + 0.5) / 2,
      rack.position[2] - frontZ * (zoneDepth / 2 - 0.1),
    );
    hotZone.rotation.y = rotY;
    group.add(hotZone);

    const hotEdge = new THREE.LineSegments(
      new THREE.EdgesGeometry(hotGeo),
      new THREE.LineBasicMaterial({ color: 0xe89e4f, transparent: true, opacity: 0.35 }),
    );
    hotEdge.position.copy(hotZone.position);
    hotEdge.rotation.copy(hotZone.rotation);
    group.add(hotEdge);

    // Arrows
    const coldArrow = new THREE.ArrowHelper(
      new THREE.Vector3(-frontX, 0, -frontZ),
      new THREE.Vector3(
        rack.position[0] + frontX * (zoneDepth + 0.1),
        rackH * 0.5,
        rack.position[2] + frontZ * (zoneDepth + 0.1),
      ),
      0.7,
      0x378add,
      0.15,
      0.1,
    );
    group.add(coldArrow);

    for (const yFrac of [0.25, 0.5, 0.75]) {
      const hotArrow = new THREE.ArrowHelper(
        new THREE.Vector3(-frontX, 0, -frontZ),
        new THREE.Vector3(
          rack.position[0] - frontX * (zoneDepth * 0.15),
          rackH * yFrac,
          rack.position[2] - frontZ * (zoneDepth * 0.15),
        ),
        0.7,
        0xe24b4a,
        0.15,
        0.1,
      );
      group.add(hotArrow);
    }
  }

  if (quality < 0.6) {
    for (let a = 0; a < serverRacks.length; a++) {
      const rackA = serverRacks[a];
      const rotA = THREE.MathUtils.degToRad(rackA.rotation[1]);
      const backAx = rackA.position[0] - Math.sin(rotA) * 1.5;
      const backAz = rackA.position[2] - Math.cos(rotA) * 1.5;

      for (let b = 0; b < serverRacks.length; b++) {
        if (a === b) continue;
        const rackB = serverRacks[b];
        const rotB = THREE.MathUtils.degToRad(rackB.rotation[1]);
        const frontBx = rackB.position[0] + Math.sin(rotB) * 1.0;
        const frontBz = rackB.position[2] + Math.cos(rotB) * 1.0;
        const dist = Math.sqrt((backAx - frontBx) ** 2 + (backAz - frontBz) ** 2);

        if (dist < 2.5) {
          const midX = (backAx + frontBx) / 2;
          const midZ = (backAz + frontBz) / 2;
          const overlapH = Math.max(rackA.size[1], rackB.size[1]) + 0.3;
          const spanX = Math.abs(backAx - frontBx) + 1.2;
          const spanZ = Math.abs(backAz - frontBz) + 1.2;
          const recircGeo = new THREE.BoxGeometry(spanX, overlapH, spanZ);
          const recircZone = new THREE.Mesh(
            recircGeo,
            new THREE.MeshBasicMaterial({
              color: 0xe24b4a,
              transparent: true,
              opacity: 0.14,
              side: THREE.DoubleSide,
              depthWrite: false,
            }),
          );
          recircZone.position.set(midX, overlapH / 2, midZ);
          group.add(recircZone);

          const recircEdge = new THREE.LineSegments(
            new THREE.EdgesGeometry(recircGeo),
            new THREE.LineBasicMaterial({ color: 0xe24b4a, transparent: true, opacity: 0.45 }),
          );
          recircEdge.position.copy(recircZone.position);
          group.add(recircEdge);
        }
      }
    }

    for (const cool of coolingSources) {
      let reachesAny = false;
      for (const rack of serverRacks) {
        const rotR = THREE.MathUtils.degToRad(rack.rotation[1]);
        const frontRx = rack.position[0] + Math.sin(rotR) * 0.5;
        const frontRz = rack.position[2] + Math.cos(rotR) * 0.5;
        const dist = Math.sqrt(
          (cool.position[0] - frontRx) ** 2 + (cool.position[2] - frontRz) ** 2,
        );
        if (dist < 4.0) {
          reachesAny = true;
          break;
        }
      }
      if (!reachesAny) {
        const bypassLabel = makeTextSprite("BYPASS", "#E89E4F");
        bypassLabel.position.set(cool.position[0], cool.size[1] + 0.5, cool.position[2]);
        group.add(bypassLabel);
      }
    }
  }
}
