import * as THREE from "three";
import type { Equipment, SceneGraph } from "../data/sceneGraphs";

const STREAMLINE_COUNT = 500;
const TRAIL_LENGTH = 40;
const SEGMENT_COUNT = STREAMLINE_COUNT * (TRAIL_LENGTH - 1);
const VERTEX_COUNT = SEGMENT_COUNT * 2;

interface HeatSource {
  x: number;
  z: number;
  heat: number;
  h: number;
  frontX: number;
  frontZ: number;
  backX: number;
  backZ: number;
}

// CFD-standard Jet colormap (MATLAB Jet) for temperature visualization.
// Dark blue (cold) → blue → cyan → green → yellow → red → dark red (hot).
// Saturated and contains no white, which keeps trails legible on a light canvas.
const JET_STOPS: ReadonlyArray<readonly [number, [number, number, number]]> = [
  [0.0, [0.0, 0.0, 0.5]], // dark blue
  [0.125, [0.0, 0.0, 1.0]], // blue
  [0.375, [0.0, 1.0, 1.0]], // cyan
  [0.625, [1.0, 1.0, 0.0]], // yellow
  [0.875, [1.0, 0.0, 0.0]], // red
  [1.0, [0.5, 0.0, 0.0]], // dark red
];

function computeTemperatureColor(temp: number): [number, number, number] {
  const t = Math.max(0, Math.min(1, temp));
  for (let i = 1; i < JET_STOPS.length; i++) {
    const [pHi, cHi] = JET_STOPS[i];
    if (t <= pHi) {
      const [pLo, cLo] = JET_STOPS[i - 1];
      const span = pHi - pLo || 1;
      const k = (t - pLo) / span;
      return [
        cLo[0] + (cHi[0] - cLo[0]) * k,
        cLo[1] + (cHi[1] - cLo[1]) * k,
        cLo[2] + (cHi[2] - cLo[2]) * k,
      ];
    }
  }
  return JET_STOPS[JET_STOPS.length - 1][1] as [number, number, number];
}

export class AirflowSystem {
  readonly group: THREE.Group;

  private streamlineHeads!: Float32Array;
  private streamlineVelocities!: Float32Array;
  private streamlineAges!: Float32Array;
  private streamlineMaxAges!: Float32Array;
  private trailHistory!: Float32Array;
  private trailHistoryIdx!: Int32Array;
  private trailPositions!: Float32Array;
  private trailColors!: Float32Array;
  private trailGeometry!: THREE.BufferGeometry;
  private trailLines!: THREE.LineSegments;
  private cachedHeatSources: HeatSource[] = [];
  private initialised = false;

  constructor(group: THREE.Group) {
    this.group = group;
  }

  init(sceneData: SceneGraph): void {
    this.group.clear();

    this.streamlineHeads = new Float32Array(STREAMLINE_COUNT * 3);
    this.streamlineVelocities = new Float32Array(STREAMLINE_COUNT * 3);
    this.streamlineAges = new Float32Array(STREAMLINE_COUNT);
    this.streamlineMaxAges = new Float32Array(STREAMLINE_COUNT);
    this.trailHistory = new Float32Array(STREAMLINE_COUNT * TRAIL_LENGTH * 3);
    this.trailHistoryIdx = new Int32Array(STREAMLINE_COUNT);
    this.trailPositions = new Float32Array(VERTEX_COUNT * 3);
    this.trailColors = new Float32Array(VERTEX_COUNT * 3);

    const coolingSources = sceneData.furniture.filter((f) => f.category === "cooling_unit");
    const [rw, , rd] = sceneData.room.dimensions;

    this.cachedHeatSources = sceneData.furniture
      .filter((f) => f.heatOutput > 0)
      .map((f) => {
        const rotY = THREE.MathUtils.degToRad(f.rotation[1]);
        const frontX = Math.sin(rotY);
        const frontZ = Math.cos(rotY);
        return {
          x: f.position[0],
          z: f.position[2],
          heat: f.heatOutput,
          h: f.size[1],
          frontX,
          frontZ,
          backX: -frontX,
          backZ: -frontZ,
        };
      });

    for (let i = 0; i < STREAMLINE_COUNT; i++) {
      this.resetStreamline(i, coolingSources, rw, rd);
      this.streamlineAges[i] = Math.random() * this.streamlineMaxAges[i] * 0.5;
      const hx = this.streamlineHeads[i * 3];
      const hy = this.streamlineHeads[i * 3 + 1];
      const hz = this.streamlineHeads[i * 3 + 2];
      for (let t = 0; t < TRAIL_LENGTH; t++) {
        const ti = (i * TRAIL_LENGTH + t) * 3;
        this.trailHistory[ti] = hx;
        this.trailHistory[ti + 1] = hy;
        this.trailHistory[ti + 2] = hz;
      }
    }

    this.trailGeometry = new THREE.BufferGeometry();
    this.trailGeometry.setAttribute("position", new THREE.BufferAttribute(this.trailPositions, 3));
    this.trailGeometry.setAttribute("color", new THREE.BufferAttribute(this.trailColors, 3));

    // NormalBlending (not Additive): on the light canvas, additive blending
    // saturates to white. Normal blending preserves the spec hues (calm blue →
    // green → amber → red) directly.
    const trailMat = new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.95,
      blending: THREE.NormalBlending,
      depthWrite: false,
      linewidth: 1,
    });

    this.trailLines = new THREE.LineSegments(this.trailGeometry, trailMat);
    this.group.add(this.trailLines);
    this.initialised = true;
  }

  isReady(): boolean {
    return this.initialised;
  }

  private resetStreamline(i: number, coolingSources: Equipment[], rw: number, rd: number): void {
    const idx = i * 3;
    const roll = Math.random();

    if (roll < 0.45 && this.cachedHeatSources.length > 0) {
      const src = this.cachedHeatSources[Math.floor(Math.random() * this.cachedHeatSources.length)];
      const backOff = 0.3 + Math.random() * 0.7;
      this.streamlineHeads[idx] = src.x + src.backX * backOff + (Math.random() - 0.5) * 0.3;
      this.streamlineHeads[idx + 1] = 0.3 + Math.random() * src.h;
      this.streamlineHeads[idx + 2] = src.z + src.backZ * backOff + (Math.random() - 0.5) * 0.3;

      this.streamlineVelocities[idx] = src.backX * 0.012 + (Math.random() - 0.5) * 0.003;
      this.streamlineVelocities[idx + 1] = 0.01 + Math.random() * 0.015;
      this.streamlineVelocities[idx + 2] = src.backZ * 0.012 + (Math.random() - 0.5) * 0.003;
    } else if (roll < 0.8 && coolingSources.length > 0) {
      const cool = coolingSources[Math.floor(Math.random() * coolingSources.length)];
      this.streamlineHeads[idx] = cool.position[0] + (Math.random() - 0.5) * cool.size[0];
      this.streamlineHeads[idx + 1] = 0.05 + Math.random() * 0.3;
      this.streamlineHeads[idx + 2] = cool.position[2] + (Math.random() - 0.5) * cool.size[2];

      let bestDx = (Math.random() - 0.5) * 0.02;
      let bestDz = (Math.random() - 0.5) * 0.02;
      if (this.cachedHeatSources.length > 0) {
        const target =
          this.cachedHeatSources[Math.floor(Math.random() * this.cachedHeatSources.length)];
        const frontTargetX = target.x + target.frontX * 0.6;
        const frontTargetZ = target.z + target.frontZ * 0.6;
        const dx = frontTargetX - cool.position[0];
        const dz = frontTargetZ - cool.position[2];
        const dist = Math.sqrt(dx * dx + dz * dz) + 0.1;
        bestDx = (dx / dist) * 0.02 + (Math.random() - 0.5) * 0.003;
        bestDz = (dz / dist) * 0.02 + (Math.random() - 0.5) * 0.003;
      }
      this.streamlineVelocities[idx] = bestDx;
      this.streamlineVelocities[idx + 1] = -0.001 + Math.random() * 0.001;
      this.streamlineVelocities[idx + 2] = bestDz;
    } else {
      const roomH = 3.5;
      if (this.cachedHeatSources.length > 0) {
        const src =
          this.cachedHeatSources[Math.floor(Math.random() * this.cachedHeatSources.length)];
        this.streamlineHeads[idx] = src.x + (Math.random() - 0.5) * 2;
        this.streamlineHeads[idx + 1] = roomH - 0.3 - Math.random() * 0.5;
        this.streamlineHeads[idx + 2] = src.z + (Math.random() - 0.5) * 2;
      } else {
        this.streamlineHeads[idx] = Math.random() * rw;
        this.streamlineHeads[idx + 1] = roomH - 0.3 - Math.random() * 0.5;
        this.streamlineHeads[idx + 2] = Math.random() * rd;
      }

      let driftX = (Math.random() - 0.5) * 0.01;
      let driftZ = (Math.random() - 0.5) * 0.01;
      if (coolingSources.length > 0) {
        const cool = coolingSources[Math.floor(Math.random() * coolingSources.length)];
        const dx = cool.position[0] - this.streamlineHeads[idx];
        const dz = cool.position[2] - this.streamlineHeads[idx + 2];
        const dist = Math.sqrt(dx * dx + dz * dz) + 0.1;
        driftX = (dx / dist) * 0.015;
        driftZ = (dz / dist) * 0.015;
      }
      this.streamlineVelocities[idx] = driftX;
      this.streamlineVelocities[idx + 1] = -0.003 - Math.random() * 0.003;
      this.streamlineVelocities[idx + 2] = driftZ;
    }

    this.streamlineAges[i] = 0;
    this.streamlineMaxAges[i] = 200 + Math.random() * 300;
    this.trailHistoryIdx[i] = 0;

    const hx = this.streamlineHeads[idx];
    const hy = this.streamlineHeads[idx + 1];
    const hz = this.streamlineHeads[idx + 2];
    for (let t = 0; t < TRAIL_LENGTH; t++) {
      const ti = (i * TRAIL_LENGTH + t) * 3;
      this.trailHistory[ti] = hx;
      this.trailHistory[ti + 1] = hy;
      this.trailHistory[ti + 2] = hz;
    }
  }

  update(sceneData: SceneGraph, loadFactor: number): void {
    if (!this.initialised) return;

    const quality = sceneData.score.total;
    const chaos = 1 - quality;
    const turbulence = 0.0003 + chaos * 0.003;
    const damping = 0.96 - chaos * 0.04;
    const directionality = quality;

    this.cachedHeatSources = sceneData.furniture
      .filter((f) => f.heatOutput > 0)
      .map((f) => {
        const rotY = THREE.MathUtils.degToRad(f.rotation[1]);
        const frontX = Math.sin(rotY);
        const frontZ = Math.cos(rotY);
        return {
          x: f.position[0],
          z: f.position[2],
          heat: f.heatOutput * loadFactor,
          h: f.size[1],
          frontX,
          frontZ,
          backX: -frontX,
          backZ: -frontZ,
        };
      });

    const coolingSources = sceneData.furniture.filter((f) => f.category === "cooling_unit");
    const [rw, rh, rd] = sceneData.room.dimensions;

    for (let i = 0; i < STREAMLINE_COUNT; i++) {
      const idx = i * 3;
      this.streamlineAges[i]++;

      const px = this.streamlineHeads[idx];
      const py = this.streamlineHeads[idx + 1];
      const pz = this.streamlineHeads[idx + 2];

      if (
        this.streamlineAges[i] > this.streamlineMaxAges[i] ||
        py > rh + 0.3 ||
        py < -0.1 ||
        px < -0.5 ||
        px > rw + 0.5 ||
        pz < -0.5 ||
        pz > rd + 0.5
      ) {
        this.resetStreamline(i, coolingSources, rw, rd);
        continue;
      }

      let heatInfluence = 0;
      let forceX = 0;
      let forceY = 0;
      let forceZ = 0;

      for (const src of this.cachedHeatSources) {
        const dx = px - src.x;
        const dz = pz - src.z;
        const distSq = dx * dx + dz * dz;
        const dist = Math.sqrt(distSq);

        if (dist < 3.5) {
          const strength = src.heat / (1 + distSq);
          heatInfluence += strength;
          const dotFront = dx * src.frontX + dz * src.frontZ;

          if (dist < 2.0 && py <= src.h) {
            if (dotFront > 0) {
              const pull = strength * (0.003 + directionality * 0.004);
              forceX -= (dx / (dist + 0.1)) * pull;
              forceZ -= (dz / (dist + 0.1)) * pull;
              if (py < 0.5) forceY -= 0.001;
            } else {
              const push = strength * (0.004 + directionality * 0.005);
              forceX += src.backX * push;
              forceZ += src.backZ * push;
              forceY += strength * (0.008 + directionality * 0.008);
            }

            if (chaos > 0.5) {
              const leak = (chaos - 0.5) * 2 * strength * 0.004;
              forceX += (Math.random() - 0.5) * leak * 2;
              forceZ += (Math.random() - 0.5) * leak * 2;
              forceY -= leak * 0.5;
            }
          } else if (py > src.h && dist < 2.0) {
            forceY += strength * (0.006 + directionality * 0.006);
            const push = strength * 0.001;
            forceX += (dx / (dist + 0.1)) * push;
            forceZ += (dz / (dist + 0.1)) * push;
          } else if (py < 0.5 && dist < 3.5) {
            const toFrontX = src.x + src.frontX * 0.6 - px;
            const toFrontZ = src.z + src.frontZ * 0.6 - pz;
            const toFrontDist = Math.sqrt(toFrontX * toFrontX + toFrontZ * toFrontZ) + 0.1;
            const pull = strength * 0.002 * directionality;
            forceX += (toFrontX / toFrontDist) * pull;
            forceZ += (toFrontZ / toFrontDist) * pull;

            if (chaos > 0.4) {
              const bypass = (chaos - 0.4) * 0.005;
              forceX += (Math.random() - 0.5) * bypass;
              forceZ += (Math.random() - 0.5) * bypass;
            }
          }
        }
      }

      for (const cool of coolingSources) {
        const dx = px - cool.position[0];
        const dz = pz - cool.position[2];
        const dist = Math.sqrt(dx * dx + dz * dz);

        if (py < 1.0 && dist < 4.0) {
          const blowStrength = (0.008 * loadFactor) / (1 + dist * 0.3);
          forceX += (dx / (dist + 0.1)) * blowStrength;
          forceZ += (dz / (dist + 0.1)) * blowStrength;
          forceY -= 0.002;
        }

        if (py > 1.5 && dist < 6.0) {
          const pullStrength = ((0.003 + directionality * 0.003) * loadFactor) / (1 + dist * 0.3);
          forceX -= (dx / (dist + 0.1)) * pullStrength;
          forceZ -= (dz / (dist + 0.1)) * pullStrength;
          if (dist < 2.0) {
            forceY -= pullStrength * 1.5;
          }
        }
      }

      if (heatInfluence > 0.2) {
        forceY += heatInfluence * 0.005;
      }

      if (py > rh - 0.5) {
        const penetration = (py - (rh - 0.5)) / 0.5;
        forceY -= 0.015 + penetration * 0.02;
        for (const cool of coolingSources) {
          const dx = cool.position[0] - px;
          const dz = cool.position[2] - pz;
          const dist = Math.sqrt(dx * dx + dz * dz) + 0.1;
          forceX += (dx / dist) * 0.005;
          forceZ += (dz / dist) * 0.005;
        }
      }

      if (py < 0.1 && heatInfluence < 0.5) {
        forceY += 0.001;
      }

      if (px < 0.3) forceX += 0.004;
      if (px > rw - 0.3) forceX -= 0.004;
      if (pz < 0.3) forceZ += 0.004;
      if (pz > rd - 0.3) forceZ -= 0.004;

      this.streamlineVelocities[idx] = this.streamlineVelocities[idx] * damping + forceX;
      this.streamlineVelocities[idx + 1] = this.streamlineVelocities[idx + 1] * damping + forceY;
      this.streamlineVelocities[idx + 2] = this.streamlineVelocities[idx + 2] * damping + forceZ;

      const speed = Math.sqrt(
        this.streamlineVelocities[idx] ** 2 +
          this.streamlineVelocities[idx + 1] ** 2 +
          this.streamlineVelocities[idx + 2] ** 2,
      );
      const maxSpeed = 0.07 * (0.5 + loadFactor);
      if (speed > maxSpeed) {
        const s = maxSpeed / speed;
        this.streamlineVelocities[idx] *= s;
        this.streamlineVelocities[idx + 1] *= s;
        this.streamlineVelocities[idx + 2] *= s;
      }

      this.streamlineVelocities[idx] += (Math.random() - 0.5) * turbulence * 2;
      this.streamlineVelocities[idx + 1] += (Math.random() - 0.5) * turbulence;
      this.streamlineVelocities[idx + 2] += (Math.random() - 0.5) * turbulence * 2;

      this.streamlineHeads[idx] += this.streamlineVelocities[idx];
      this.streamlineHeads[idx + 1] += this.streamlineVelocities[idx + 1];
      this.streamlineHeads[idx + 2] += this.streamlineVelocities[idx + 2];

      if (this.streamlineHeads[idx + 1] < 0.02) {
        this.streamlineHeads[idx + 1] = 0.02;
        this.streamlineVelocities[idx + 1] = Math.abs(this.streamlineVelocities[idx + 1]) * 0.1;
      }

      if (this.streamlineHeads[idx + 1] > rh - 0.05) {
        this.streamlineHeads[idx + 1] = rh - 0.05;
        this.streamlineVelocities[idx + 1] = -Math.abs(this.streamlineVelocities[idx + 1]) * 0.3;
      }

      const cursor = this.trailHistoryIdx[i] % TRAIL_LENGTH;
      const hi = (i * TRAIL_LENGTH + cursor) * 3;
      this.trailHistory[hi] = this.streamlineHeads[idx];
      this.trailHistory[hi + 1] = this.streamlineHeads[idx + 1];
      this.trailHistory[hi + 2] = this.streamlineHeads[idx + 2];
      this.trailHistoryIdx[i]++;

      const segBase = i * (TRAIL_LENGTH - 1);
      const currentCursor = this.trailHistoryIdx[i];

      for (let s = 0; s < TRAIL_LENGTH - 1; s++) {
        const vIdx = (segBase + s) * 2;

        const older = (currentCursor + s) % TRAIL_LENGTH;
        const newer = (currentCursor + s + 1) % TRAIL_LENGTH;

        const oldOff = (i * TRAIL_LENGTH + older) * 3;
        const newOff = (i * TRAIL_LENGTH + newer) * 3;

        this.trailPositions[vIdx * 3] = this.trailHistory[oldOff];
        this.trailPositions[vIdx * 3 + 1] = this.trailHistory[oldOff + 1];
        this.trailPositions[vIdx * 3 + 2] = this.trailHistory[oldOff + 2];

        this.trailPositions[(vIdx + 1) * 3] = this.trailHistory[newOff];
        this.trailPositions[(vIdx + 1) * 3 + 1] = this.trailHistory[newOff + 1];
        this.trailPositions[(vIdx + 1) * 3 + 2] = this.trailHistory[newOff + 2];

        const segX = this.trailHistory[oldOff];
        const segY = this.trailHistory[oldOff + 1];
        const segZ = this.trailHistory[oldOff + 2];

        let localHeat = 0;
        for (const src of this.cachedHeatSources) {
          const sdx = segX - src.x;
          const sdz = segZ - src.z;
          const sd = sdx * sdx + sdz * sdz;
          localHeat += src.heat / (1 + sd);
        }
        const heightTemp = Math.min(segY / rh, 1) * 0.35;
        const segTemp = Math.min(localHeat / 8 + heightTemp, 1);
        const [cr, cg, cb] = computeTemperatureColor(segTemp);

        const ageFade = s / (TRAIL_LENGTH - 1);
        const tailBright = ageFade * ageFade * 0.9 + 0.1;
        const headBright = Math.min(tailBright * 1.3, 1);
        // Trail fades by darkening (× brightness): with NormalBlending on the
        // light canvas, dark hues stay legible — the older end shows as a thin
        // deep-coloured trace, the newer end at full Jet saturation.

        this.trailColors[vIdx * 3] = cr * tailBright;
        this.trailColors[vIdx * 3 + 1] = cg * tailBright;
        this.trailColors[vIdx * 3 + 2] = cb * tailBright;

        this.trailColors[(vIdx + 1) * 3] = cr * headBright;
        this.trailColors[(vIdx + 1) * 3 + 1] = cg * headBright;
        this.trailColors[(vIdx + 1) * 3 + 2] = cb * headBright;
      }
    }

    this.trailGeometry.attributes.position.needsUpdate = true;
    this.trailGeometry.attributes.color.needsUpdate = true;
  }
}
