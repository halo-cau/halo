import * as THREE from "three";
import type { Equipment, SceneGraph } from "../data/sceneGraphs";

const STREAMLINE_COUNT = 500;
const TRAIL_LENGTH = 40;
const SEGMENT_COUNT = STREAMLINE_COUNT * (TRAIL_LENGTH - 1);
const VERTEX_COUNT = SEGMENT_COUNT * 2;

// Per-source streamline speed caps, derived from REAL airflow so the AC power jet and the rack exhaust
// read at their true relative speeds. Measured ranges (provided by the team): a floor-standing AC in
// power mode discharges 6–8 m/s; a server-rack exhaust reads 1.5–4.5 m/s; room recirculation drifts
// ~1 m/s. Pick a representative speed inside each range and convert with one calibration constant — so
// these caps stay honestly tied to m/s and are trivial to retune to new measurements.
const MPS_TO_VIZ = 0.012; // viz units (per frame) per 1 m/s of real airflow
const SPEED_AC = 7.5 * MPS_TO_VIZ; // 0.090 — floor-standing AC, power mode (6–8 m/s)
const SPEED_RACK = 3.75 * MPS_TO_VIZ; // 0.045 — server-rack exhaust (1.5–4.5 m/s)
const SPEED_AMBIENT = 1.0 * MPS_TO_VIZ; // 0.012 — room recirculation / drift

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

// A cooling unit resolved into its discharge geometry. `n` is the front-face normal (the direction the
// unit discharges), `t` is the in-face width axis (along the vent). `ceiling` true = cassette that blows
// down in four directions; false = floor-standing unit that blows out of its upper-front louver vent.
interface CoolSource {
  x: number;
  z: number;
  y0: number; // base height (position.y)
  w: number;
  h: number;
  d: number;
  nX: number;
  nZ: number;
  tX: number;
  tZ: number;
  ceiling: boolean;
}

function isCoolingUnit(f: Equipment): boolean {
  return f.category === "cooling_unit" || f.category === "stand_ac" || f.category === "ceiling_ac";
}

function makeCoolSources(sceneData: SceneGraph): CoolSource[] {
  return sceneData.furniture.filter(isCoolingUnit).map((f) => {
    const rotY = THREE.MathUtils.degToRad(f.rotation[1]);
    return {
      x: f.position[0],
      z: f.position[2],
      y0: f.position[1],
      w: f.size[0],
      h: f.size[1],
      d: f.size[2],
      nX: Math.sin(rotY), // front normal (local +Z after the Y rotation)
      nZ: Math.cos(rotY),
      tX: Math.cos(rotY), // in-face width axis (local +X)
      tZ: -Math.sin(rotY),
      ceiling: f.category === "ceiling_ac",
    };
  });
}

interface Obstacle {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
  minZ: number;
  maxZ: number;
}

// Solid equipment boxes the air must flow AROUND, not through. Each furniture item becomes a world-space
// axis-aligned box; a Y-rotation just swaps the footprint extents (exact for the 0/90/180/270° facings
// used in these scenes).
function makeObstacles(sceneData: SceneGraph): Obstacle[] {
  return sceneData.furniture.map((f) => {
    const [w, h, d] = f.size;
    const rotY = THREE.MathUtils.degToRad(f.rotation[1]);
    const ax = Math.abs(Math.cos(rotY)) * (w / 2) + Math.abs(Math.sin(rotY)) * (d / 2);
    const az = Math.abs(Math.sin(rotY)) * (w / 2) + Math.abs(Math.cos(rotY)) * (d / 2);
    const [cx, cy, cz] = f.position;
    return { minX: cx - ax, maxX: cx + ax, minY: cy, maxY: cy + h, minZ: cz - az, maxZ: cz + az };
  });
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
  private streamlineMaxSpeedBase!: Float32Array;
  private trailHistory!: Float32Array;
  private trailHistoryIdx!: Int32Array;
  private trailPositions!: Float32Array;
  private trailColors!: Float32Array;
  private trailGeometry!: THREE.BufferGeometry;
  private trailLines!: THREE.LineSegments;
  private cachedHeatSources: HeatSource[] = [];
  private obstacles: Obstacle[] = [];
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
    this.streamlineMaxSpeedBase = new Float32Array(STREAMLINE_COUNT);
    this.trailHistory = new Float32Array(STREAMLINE_COUNT * TRAIL_LENGTH * 3);
    this.trailHistoryIdx = new Int32Array(STREAMLINE_COUNT);
    this.trailPositions = new Float32Array(VERTEX_COUNT * 3);
    this.trailColors = new Float32Array(VERTEX_COUNT * 3);

    const coolSources = makeCoolSources(sceneData);
    this.obstacles = makeObstacles(sceneData);
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
      this.resetStreamline(i, coolSources, rw, rd);
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

  // Discharge one streamline from a cooling unit's vent. Floor-standing unit: out of the upper-front
  // louver band (the top rectangular area), forward + slightly down. Ceiling cassette: down in 4 dirs.
  private emitFromVent(idx: number, cool: CoolSource): void {
    if (cool.ceiling) {
      const dirs = [
        [1, 0],
        [-1, 0],
        [0, 1],
        [0, -1],
      ] as const;
      const [dxs, dzs] = dirs[Math.floor(Math.random() * 4)];
      this.streamlineHeads[idx] = cool.x + dxs * cool.w * 0.35;
      this.streamlineHeads[idx + 1] = cool.y0 + cool.h * 0.5;
      this.streamlineHeads[idx + 2] = cool.z + dzs * cool.d * 0.35;
      this.streamlineVelocities[idx] = dxs * 0.012 + (Math.random() - 0.5) * 0.002;
      this.streamlineVelocities[idx + 1] = -0.006 - Math.random() * 0.003;
      this.streamlineVelocities[idx + 2] = dzs * 0.012 + (Math.random() - 0.5) * 0.002;
      return;
    }
    // upper-front louver band (0.70h..0.96h, width w*0.86) on the front face → forward + slightly down
    const u = (Math.random() - 0.5) * cool.w * 0.86; // lateral along the vent width
    const ventY = cool.h * (0.7 + Math.random() * 0.26);
    const faceOff = cool.d / 2 + 0.03; // emit just outside the front face
    this.streamlineHeads[idx] = cool.x + cool.tX * u + cool.nX * faceOff;
    this.streamlineHeads[idx + 1] = cool.y0 + ventY;
    this.streamlineHeads[idx + 2] = cool.z + cool.tZ * u + cool.nZ * faceOff;
    const vOut = 0.05; // strong discharge so the jet shoots across the aisle
    this.streamlineVelocities[idx] = cool.nX * vOut + (Math.random() - 0.5) * 0.003;
    this.streamlineVelocities[idx + 1] = -0.014 - Math.random() * 0.004; // discharge aimed a little low
    this.streamlineVelocities[idx + 2] = cool.nZ * vOut + (Math.random() - 0.5) * 0.003;
  }

  // Block a streamline from entering a solid box: push it to the nearest face and cancel the velocity
  // component pointing into the surface, so the air slides along / around the equipment instead of through.
  private resolveObstacle(idx: number): void {
    for (const o of this.obstacles) {
      const px = this.streamlineHeads[idx];
      const py = this.streamlineHeads[idx + 1];
      const pz = this.streamlineHeads[idx + 2];
      if (
        px <= o.minX ||
        px >= o.maxX ||
        py <= o.minY ||
        py >= o.maxY ||
        pz <= o.minZ ||
        pz >= o.maxZ
      ) {
        continue;
      }
      const dxMin = px - o.minX;
      const dxMax = o.maxX - px;
      const dyMin = py - o.minY;
      const dyMax = o.maxY - py;
      const dzMin = pz - o.minZ;
      const dzMax = o.maxZ - pz;
      const m = Math.min(dxMin, dxMax, dyMin, dyMax, dzMin, dzMax);
      const eps = 0.01;
      if (m === dxMin) {
        this.streamlineHeads[idx] = o.minX - eps;
        if (this.streamlineVelocities[idx] > 0) this.streamlineVelocities[idx] = 0;
      } else if (m === dxMax) {
        this.streamlineHeads[idx] = o.maxX + eps;
        if (this.streamlineVelocities[idx] < 0) this.streamlineVelocities[idx] = 0;
      } else if (m === dyMin) {
        this.streamlineHeads[idx + 1] = o.minY - eps;
        if (this.streamlineVelocities[idx + 1] > 0) this.streamlineVelocities[idx + 1] = 0;
      } else if (m === dyMax) {
        this.streamlineHeads[idx + 1] = o.maxY + eps;
        if (this.streamlineVelocities[idx + 1] < 0) this.streamlineVelocities[idx + 1] = 0;
      } else if (m === dzMin) {
        this.streamlineHeads[idx + 2] = o.minZ - eps;
        if (this.streamlineVelocities[idx + 2] > 0) this.streamlineVelocities[idx + 2] = 0;
      } else {
        this.streamlineHeads[idx + 2] = o.maxZ + eps;
        if (this.streamlineVelocities[idx + 2] < 0) this.streamlineVelocities[idx + 2] = 0;
      }
    }
  }

  private resetStreamline(i: number, coolSources: CoolSource[], rw: number, rd: number): void {
    const idx = i * 3;
    const roll = Math.random();

    if (roll < 0.45 && this.cachedHeatSources.length > 0) {
      const src = this.cachedHeatSources[Math.floor(Math.random() * this.cachedHeatSources.length)];
      const backOff = 0.5 + Math.random() * 0.6; // start just behind the rack's exhaust face, not inside it
      this.streamlineHeads[idx] = src.x + src.backX * backOff + (Math.random() - 0.5) * 0.3;
      this.streamlineHeads[idx + 1] = 0.3 + Math.random() * src.h;
      this.streamlineHeads[idx + 2] = src.z + src.backZ * backOff + (Math.random() - 0.5) * 0.3;

      this.streamlineVelocities[idx] = src.backX * 0.012 + (Math.random() - 0.5) * 0.003;
      this.streamlineVelocities[idx + 1] = 0.01 + Math.random() * 0.015;
      this.streamlineVelocities[idx + 2] = src.backZ * 0.012 + (Math.random() - 0.5) * 0.003;
    } else if (roll < 0.8 && coolSources.length > 0) {
      // Cold supply: discharge from the AC's vent (top rectangular area), not the floor or the sides.
      this.emitFromVent(idx, coolSources[Math.floor(Math.random() * coolSources.length)]);
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
      if (coolSources.length > 0) {
        const cool = coolSources[Math.floor(Math.random() * coolSources.length)];
        const dx = cool.x - this.streamlineHeads[idx];
        const dz = cool.z - this.streamlineHeads[idx + 2];
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
    // per-source speed cap scaled to real airflow (AC power jet > rack exhaust > room drift)
    this.streamlineMaxSpeedBase[i] =
      roll < 0.45 ? SPEED_RACK : roll < 0.8 ? SPEED_AC : SPEED_AMBIENT;
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

    const coolSources = makeCoolSources(sceneData);
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
        this.resetStreamline(i, coolSources, rw, rd);
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

      for (const cool of coolSources) {
        const dx = px - cool.x;
        const dz = pz - cool.z;
        const dist = Math.sqrt(dx * dx + dz * dz);

        if (cool.ceiling) {
          if (py < 1.0 && dist < 4.0) {
            const blow = (0.008 * loadFactor) / (1 + dist * 0.3); // cassette: spread radially outward
            forceX += (dx / (dist + 0.1)) * blow;
            forceZ += (dz / (dist + 0.1)) * blow;
            forceY -= 0.002;
          }
        } else {
          // Floor-standing unit: a STRONG directed jet, sustained down a forward corridor so the cold air
          // shoots almost to the far end of the aisle before it sinks and the racks draw it in.
          const fwd = dx * cool.nX + dz * cool.nZ; // distance in front of the vent
          const reach = (Math.abs(cool.nX) > Math.abs(cool.nZ) ? rw : rd) * 0.92;
          if (fwd > -0.2 && fwd < reach && py < cool.y0 + cool.h * 0.83 + 0.5) {
            const perpX = dx - fwd * cool.nX;
            const perpZ = dz - fwd * cool.nZ;
            const lateral = Math.exp(-(perpX * perpX + perpZ * perpZ) / 0.7); // tight jet core
            const jet = 0.06 * loadFactor * (1 - fwd / reach) * lateral;
            forceX += cool.nX * jet;
            forceZ += cool.nZ * jet;
            forceY -= jet * 0.35; // discharge tilts downward (louver angle) — aim the jet a little low
          }
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

      // Recirculation only (NOT a soft ceiling push — the hard ceiling clamp handles solidity): warm air
      // pooled under the ceiling is drawn back toward the AC return.
      if (py > rh - 0.5) {
        for (const cool of coolSources) {
          const dx = cool.x - px;
          const dz = cool.z - pz;
          const dist = Math.sqrt(dx * dx + dz * dz) + 0.1;
          forceX += (dx / dist) * 0.005;
          forceZ += (dz / dist) * 0.005;
        }
      }

      // No soft wall / floor pushes: solids (walls, floor, ceiling, equipment) are enforced purely by the
      // hard clamps + resolveObstacle below. The only forces left are real airflow (rack fans, AC jet,
      // buoyancy, recirculation) and turbulence.
      this.streamlineVelocities[idx] = this.streamlineVelocities[idx] * damping + forceX;
      this.streamlineVelocities[idx + 1] = this.streamlineVelocities[idx + 1] * damping + forceY;
      this.streamlineVelocities[idx + 2] = this.streamlineVelocities[idx + 2] * damping + forceZ;

      const speed = Math.sqrt(
        this.streamlineVelocities[idx] ** 2 +
          this.streamlineVelocities[idx + 1] ** 2 +
          this.streamlineVelocities[idx + 2] ** 2,
      );
      const maxSpeed = this.streamlineMaxSpeedBase[i] * (0.5 + loadFactor);
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

      // Hard walls — keep air inside the VISIBLE room shell. Previously the only wall limit was a soft
      // push plus a reset 0.5 m BEYOND the wall, so air leaked out and looked bounded by a larger
      // invisible shell. Clamp to the shell (room.dimensions) and cancel the outward velocity.
      if (this.streamlineHeads[idx] < 0.02) {
        this.streamlineHeads[idx] = 0.02;
        if (this.streamlineVelocities[idx] < 0) this.streamlineVelocities[idx] = 0;
      } else if (this.streamlineHeads[idx] > rw - 0.02) {
        this.streamlineHeads[idx] = rw - 0.02;
        if (this.streamlineVelocities[idx] > 0) this.streamlineVelocities[idx] = 0;
      }
      if (this.streamlineHeads[idx + 2] < 0.02) {
        this.streamlineHeads[idx + 2] = 0.02;
        if (this.streamlineVelocities[idx + 2] < 0) this.streamlineVelocities[idx + 2] = 0;
      } else if (this.streamlineHeads[idx + 2] > rd - 0.02) {
        this.streamlineHeads[idx + 2] = rd - 0.02;
        if (this.streamlineVelocities[idx + 2] > 0) this.streamlineVelocities[idx + 2] = 0;
      }

      this.resolveObstacle(idx); // air flows around the equipment, not through it

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
