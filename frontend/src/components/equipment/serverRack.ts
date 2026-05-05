import * as THREE from "three";
import type { Equipment } from "../../data/sceneGraphs";

export function createServerRack(item: Equipment): THREE.Group {
  const group = new THREE.Group();
  const [w, h, d] = item.size;

  const cabinetMat = new THREE.MeshStandardMaterial({
    color: item.color,
    roughness: 0.4,
    metalness: 0.6,
  });

  const bodyGeo = new THREE.BoxGeometry(w, h, d);
  const body = new THREE.Mesh(bodyGeo, cabinetMat);
  body.position.y = h / 2;
  group.add(body);

  const unitCount = 8;
  const unitHeight = (h - 0.2) / unitCount;
  for (let i = 0; i < unitCount; i++) {
    const unitGeo = new THREE.BoxGeometry(w * 0.85, unitHeight * 0.8, 0.02);
    const isActive = Math.random() > 0.2;
    const unitMat = new THREE.MeshStandardMaterial({
      color: isActive ? 0x263238 : 0x1a1a1a,
      roughness: 0.3,
      metalness: 0.5,
    });
    const unit = new THREE.Mesh(unitGeo, unitMat);
    unit.position.set(0, 0.1 + unitHeight * (i + 0.5), d / 2 + 0.011);
    group.add(unit);

    if (isActive) {
      const ledGeo = new THREE.BoxGeometry(0.02, 0.02, 0.01);
      const ledColor = item.heatOutput > 10 ? 0xff1744 : item.heatOutput > 5 ? 0xffab00 : 0x00e676;
      const ledMat = new THREE.MeshStandardMaterial({
        color: ledColor,
        emissive: ledColor,
        emissiveIntensity: 1.5,
      });
      const led = new THREE.Mesh(ledGeo, ledMat);
      led.position.set(w * 0.35, 0.1 + unitHeight * (i + 0.5), d / 2 + 0.016);
      group.add(led);
    }
  }

  const topFrame = new THREE.Mesh(
    new THREE.BoxGeometry(w + 0.02, 0.03, d + 0.02),
    new THREE.MeshStandardMaterial({ color: 0x212121, metalness: 0.7 }),
  );
  topFrame.position.y = h;
  group.add(topFrame);

  if (item.heatOutput > 0) {
    const glowColor = item.heatOutput > 10 ? 0xff3300 : item.heatOutput > 5 ? 0xff8800 : 0xffcc00;
    const glowOpacity = 0.15 + item.heatOutput * 0.01;

    const backGlowGeo = new THREE.PlaneGeometry(w * 0.85, h * 0.8);
    const backGlowMat = new THREE.MeshBasicMaterial({
      color: glowColor,
      transparent: true,
      opacity: glowOpacity * 1.5,
      side: THREE.DoubleSide,
    });
    const backGlow = new THREE.Mesh(backGlowGeo, backGlowMat);
    backGlow.position.set(0, h / 2, -d / 2 - 0.02);
    group.add(backGlow);

    const topGlowGeo = new THREE.PlaneGeometry(w * 0.5, d * 0.5);
    const topGlowMat = new THREE.MeshBasicMaterial({
      color: glowColor,
      transparent: true,
      opacity: glowOpacity * 0.7,
      side: THREE.DoubleSide,
    });
    const topGlow = new THREE.Mesh(topGlowGeo, topGlowMat);
    topGlow.rotation.x = -Math.PI / 2;
    topGlow.position.y = h + 0.05;
    group.add(topGlow);
  }

  return group;
}
