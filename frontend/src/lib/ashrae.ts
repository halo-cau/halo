// ASHRAE TC 9.9 metrics — compute per-rack intake/exhaust + RCI/SHI/RHI
// for any SceneGraph using the same physics that drive the heatmap.
//
// References:
//   ASHRAE TC 9.9 — Thermal Guidelines for Data Processing Environments (5th ed.)
//   Herrlin, M.K. (2008). "Airflow and Cooling Performance of Data Centers:
//                          Two Performance Metrics." ASHRAE Transactions.

import type { SceneGraph } from "../data/sceneGraphs";

// === ASHRAE Thermal Guidelines (Class A1, recommended/allowable bounds) ===
export const T_MIN_ALLOW = 15; // °C — allowable lower bound
export const T_MIN_REC = 18; // °C — recommended lower bound
export const T_MAX_REC = 27; // °C — recommended upper bound
// Allowable upper bound. The HALO mid-presentation deck (page 8) lists
// 15–35 °C as the allowable range; this matches a generous interpretation
// of TC 9.9's allowable envelope (Class A2). Keep this in sync with the
// thermometer scale CSS in dashboard.html and the legend wording.
export const T_MAX_ALLOW = 35; // °C — allowable upper bound (per HALO deck)
export const AMBIENT_T = 22; // °C — assumed room/return air baseline
export const AC_SUPPLY_T = 14; // °C — CRAC / ceiling AC supply temperature

// === Air properties / flow constants (kept in sync with heatmap.ts) ===
const AIR_RHO = 1.2;
const AIR_CP = 1006;
const CFM_TO_M3S = 0.000472;
const DEFAULT_RACK_CFM = 800;
const AC_CFM = 2000;
const PLUME_SIGMA_RAD = 0.35;
const HEATMAP_HEIGHT = 1.0;

export type RackZone =
  | "below-allowable" // < 15 °C
  | "below-recommended" // 15–18
  | "recommended" // 18–27
  | "above-recommended" // 27–35
  | "above-allowable"; // > 35

export type RackStatus = "ok" | "warn" | "danger";

export interface RackMetrics {
  id: string;
  label: string;
  intakeTemp: number;
  exhaustTemp: number;
  deltaT: number;
  zone: RackZone;
  status: RackStatus;
  heatOutput: number; // kW × loadFactor
}

export interface SceneAshraeMetrics {
  sceneId: string;
  sceneName: string;
  perRack: RackMetrics[];
  rciHi: number; // 0-1 (1.0 = perfect)
  rciLo: number; // 0-1
  shi: number; // 0-1 (lower is better)
  rhi: number; // 0-1 (higher is better)
  meanIntake: number;
  meanExhaust: number;
  meanDeltaT: number;
  peakIntake: number;
  totalHeatKw: number;
  // Compliance summaries
  recommendedCount: number; // racks fully within recommended range
  allowableCount: number; // within allowable but outside recommended
  violationCount: number; // outside allowable (DANGER)
}

interface ResolvedHeatSource {
  x: number;
  z: number;
  backX: number;
  backZ: number;
  frontX: number;
  frontZ: number;
  rackW: number;
  deltaTExhaust: number;
  // Axis-aligned-bounding-box of the rack body, used for occlusion checks
  // so a neighbor's hot exhaust can't pass through this rack to reach
  // its cold-aisle intake (= proper hot/cold aisle containment).
  bboxX0: number;
  bboxX1: number;
  bboxZ0: number;
  bboxZ1: number;
  selfId: string;
}

interface ResolvedCooling {
  x: number;
  z: number;
  baseDt: number;
}

function buildSources(
  scene: SceneGraph,
  loadFactor: number,
): { heat: ResolvedHeatSource[]; cool: ResolvedCooling[] } {
  const heat = scene.furniture
    .filter((f) => f.heatOutput > 0 && f.category === "server_rack")
    .map((f): ResolvedHeatSource => {
      const rotY = (f.rotation[1] * Math.PI) / 180;
      const frontX = Math.sin(rotY);
      const frontZ = Math.cos(rotY);
      const rackW = f.size[0] || 0.6;
      const rackD = f.size[2] || 1.0;
      const halfDepth = rackD / 2;
      const Q = f.heatOutput * 1000 * loadFactor; // W
      const m_dot = AIR_RHO * DEFAULT_RACK_CFM * CFM_TO_M3S;
      const dt = Q / (m_dot * AIR_CP);
      // Use axis-aligned half-extents — racks here are mostly cardinal-axis
      // aligned, and a tiny rotation tolerance is fine for occlusion intent.
      const halfX = Math.max(rackW, rackD) / 2;
      const halfZ = Math.max(rackW, rackD) / 2;
      return {
        x: f.position[0] + -frontX * halfDepth, // back face = exhaust source
        z: f.position[2] + -frontZ * halfDepth,
        backX: -frontX,
        backZ: -frontZ,
        frontX,
        frontZ,
        rackW,
        deltaTExhaust: dt,
        bboxX0: f.position[0] - halfX,
        bboxX1: f.position[0] + halfX,
        bboxZ0: f.position[2] - halfZ,
        bboxZ1: f.position[2] + halfZ,
        selfId: f.id,
      };
    });

  const cool = scene.furniture
    .filter((f) => f.category === "cooling_unit" || f.category === "ceiling_ac")
    .map((f): ResolvedCooling => {
      const coolQ = AIR_RHO * AC_CFM * CFM_TO_M3S * AIR_CP * (AMBIENT_T - AC_SUPPLY_T);
      const m = AIR_RHO * AC_CFM * CFM_TO_M3S;
      return {
        x: f.position[0],
        z: f.position[2],
        baseDt: (coolQ * loadFactor) / (m * AIR_CP),
      };
    });

  return { heat, cool };
}

// 2-D segment vs axis-aligned bounding box. Returns true if the segment
// from (x0,z0)→(x1,z1) intersects the box. Slab method.
function segmentHitsAabb(
  x0: number,
  z0: number,
  x1: number,
  z1: number,
  bx0: number,
  bx1: number,
  bz0: number,
  bz1: number,
): boolean {
  let tMin = 0;
  let tMax = 1;
  const dx = x1 - x0;
  const dz = z1 - z0;
  for (const [p, dp, lo, hi] of [[x0, dx, bx0, bx1] as const, [z0, dz, bz0, bz1] as const]) {
    if (Math.abs(dp) < 1e-9) {
      if (p < lo || p > hi) return false;
    } else {
      let t1 = (lo - p) / dp;
      let t2 = (hi - p) / dp;
      if (t1 > t2) [t1, t2] = [t2, t1];
      tMin = Math.max(tMin, t1);
      tMax = Math.min(tMax, t2);
      if (tMin > tMax) return false;
    }
  }
  return true;
}

// Sample the same Gaussian-plume + AC-mixing model used by buildHeatmap
// at an arbitrary (x, z) point on the floor. Returns ΔT above ambient.
//
// Any rack body other than the source itself that lies on the line
// (src exhaust → sample point) heavily attenuates the contribution —
// this is what gives a contained hot/cold-aisle layout a low intake temp.
// In particular, when we sample a rack's *own* intake, that rack's body
// must remain in the occluder set so a neighbor's hot exhaust cannot
// magically tunnel through it.
function sampleDeltaT(
  x: number,
  z: number,
  sources: ResolvedHeatSource[],
  cools: ResolvedCooling[],
): number {
  let dt = 0;
  for (const src of sources) {
    const dx = x - src.x;
    const dz = z - src.z;
    const dist = Math.sqrt(dx * dx + dz * dz);
    if (dist < 0.01) {
      dt += src.deltaTExhaust;
      continue;
    }
    const dot = (dx * src.backX + dz * src.backZ) / dist;
    const theta = Math.acos(Math.min(1, Math.max(-1, dot)));
    const angularDecay = Math.exp(-(theta * theta) / (2 * PLUME_SIGMA_RAD * PLUME_SIGMA_RAD));
    const b0 = src.rackW / 2;
    const coreLen = b0 * 5;
    const axialDecay = dist < coreLen ? 1.0 : Math.sqrt(coreLen / dist);

    // Nudge the segment away from the source's own body so the exhaust
    // is reported as starting just outside its back face, not on it.
    const eps = 0.05;
    const sx = src.x + src.backX * eps;
    const sz = src.z + src.backZ * eps;

    let occluded = false;
    for (const other of sources) {
      if (other.selfId === src.selfId) continue;
      if (segmentHitsAabb(sx, sz, x, z, other.bboxX0, other.bboxX1, other.bboxZ0, other.bboxZ1)) {
        occluded = true;
        break;
      }
    }
    // Real rack bodies don't perfectly seal — some hot air leaks around.
    // 0.35 captures "the segment is blocked but plumes wrap around it"
    // without making bad layouts look magically good.
    const occlusionAtten = occluded ? 0.35 : 1.0;
    const recirc = 0.03;
    dt += src.deltaTExhaust * (angularDecay * axialDecay * occlusionAtten * (1 - recirc) + recirc);
  }
  for (const cool of cools) {
    const dx = x - cool.x;
    const dz = z - cool.z;
    const dist = Math.sqrt(dx * dx + dz * dz);
    const r0 = 0.3;
    const spatial = dist < r0 ? 1.0 : r0 / dist;
    const vertMix = Math.exp(-HEATMAP_HEIGHT / 1.5);
    const cooling = cool.baseDt * spatial * vertMix;
    dt -= Math.min(cooling, dt);
  }
  return Math.max(0, dt);
}

function classify(intake: number): { zone: RackZone; status: RackStatus } {
  if (intake < T_MIN_ALLOW) return { zone: "below-allowable", status: "danger" };
  if (intake < T_MIN_REC) return { zone: "below-recommended", status: "warn" };
  if (intake <= T_MAX_REC) return { zone: "recommended", status: "ok" };
  if (intake <= T_MAX_ALLOW) return { zone: "above-recommended", status: "warn" };
  return { zone: "above-allowable", status: "danger" };
}

// RCI-Hi / RCI-Lo per Herrlin (2008)
//   RCI-Hi = 1 − Σmax(0, T - T_max_rec) / Σ(T_max_allow − T_max_rec)
//   RCI-Lo = 1 − Σmax(0, T_min_rec − T) / Σ(T_min_rec − T_min_allow)
// Both are 1.0 when every intake sits inside the recommended range.
function rci(
  temps: number[],
  minRec: number,
  maxRec: number,
  minAllow: number,
  maxAllow: number,
): {
  hi: number;
  lo: number;
} {
  if (temps.length === 0) return { hi: 1, lo: 1 };
  let overSum = 0;
  let underSum = 0;
  for (const t of temps) {
    if (t > maxRec) overSum += t - maxRec;
    if (t < minRec) underSum += minRec - t;
  }
  const overBudget = (maxAllow - maxRec) * temps.length;
  const underBudget = (minRec - minAllow) * temps.length;
  return {
    hi: Math.max(0, 1 - overSum / overBudget),
    lo: Math.max(0, 1 - underSum / underBudget),
  };
}

export function computeAshraeMetrics(
  scene: SceneGraph,
  loadFactor: number = 1.0,
): SceneAshraeMetrics {
  const { heat, cool } = buildSources(scene, loadFactor);
  const racks = scene.furniture.filter((f) => f.heatOutput > 0 && f.category === "server_rack");

  const perRack: RackMetrics[] = racks.map((rack, i) => {
    // Sample intake just in front of the rack (cold-aisle side, ~0.5 m away)
    const src = heat[i];
    const sampleX = src ? rack.position[0] + src.frontX * 0.5 : rack.position[0];
    const sampleZ = src ? rack.position[2] + src.frontZ * 0.5 : rack.position[2];
    const intakeDt = sampleDeltaT(sampleX, sampleZ, heat, cool);
    const intakeTemp = AMBIENT_T + intakeDt;
    const exhaustTemp = intakeTemp + (src?.deltaTExhaust ?? 0);
    const deltaT = src?.deltaTExhaust ?? 0;
    const cls = classify(intakeTemp);
    return {
      id: rack.id,
      label: rack.label,
      intakeTemp,
      exhaustTemp,
      deltaT,
      zone: cls.zone,
      status: cls.status,
      heatOutput: rack.heatOutput * loadFactor,
    };
  });

  const intakes = perRack.map((r) => r.intakeTemp);
  const exhausts = perRack.map((r) => r.exhaustTemp);
  const { hi: rciHi, lo: rciLo } = rci(intakes, T_MIN_REC, T_MAX_REC, T_MIN_ALLOW, T_MAX_ALLOW);

  const meanIntake = avg(intakes);
  const meanExhaust = avg(exhausts);
  const meanDeltaT = meanExhaust - meanIntake;
  const peakIntake = intakes.length ? Math.max(...intakes) : AMBIENT_T;
  const totalHeatKw = perRack.reduce((s, r) => s + r.heatOutput, 0);

  // SHI = (T_intake_avg − T_supply) / (T_exhaust_avg − T_supply)
  // Lower is better — measures how much heat leaks back to intake before
  // the air reaches the rack.
  const supplyDelta = Math.max(0.5, meanExhaust - AC_SUPPLY_T);
  const shi = Math.min(1, Math.max(0, (meanIntake - AC_SUPPLY_T) / supplyDelta));
  // RHI = 1 − SHI in a 1-D mass balance. Higher = better return path.
  const rhi = 1 - shi;

  const recommendedCount = perRack.filter((r) => r.zone === "recommended").length;
  const allowableCount = perRack.filter(
    (r) => r.zone === "below-recommended" || r.zone === "above-recommended",
  ).length;
  const violationCount = perRack.filter(
    (r) => r.zone === "below-allowable" || r.zone === "above-allowable",
  ).length;

  return {
    sceneId: scene.id,
    sceneName: scene.name,
    perRack,
    rciHi,
    rciLo,
    shi,
    rhi,
    meanIntake,
    meanExhaust,
    meanDeltaT,
    peakIntake,
    totalHeatKw,
    recommendedCount,
    allowableCount,
    violationCount,
  };
}

function avg(xs: number[]): number {
  if (!xs.length) return AMBIENT_T;
  return xs.reduce((s, x) => s + x, 0) / xs.length;
}

// Plain-Korean labels for non-experts.
export const ZONE_LABEL: Record<RackZone, { ko: string; emoji: string; color: string }> = {
  "below-allowable": { ko: "위험: 너무 차가움", emoji: "🥶", color: "#1565c0" },
  "below-recommended": { ko: "주의: 권장보다 낮음", emoji: "❄️", color: "#42a5f5" },
  recommended: { ko: "정상 (권장 범위)", emoji: "✅", color: "#00e676" },
  "above-recommended": { ko: "주의: 권장보다 높음", emoji: "♨️", color: "#ffab00" },
  "above-allowable": { ko: "위험: 즉시 조치 필요", emoji: "🔥", color: "#ff1744" },
};
