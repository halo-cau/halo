import * as THREE from "three";
import type { SceneGraph } from "../data/sceneGraphs";

// References:
//   ASHRAE TC 9.9 — Thermal Guidelines for Data Processing Environments
//   Turbulent plane jet: Rajaratnam, "Turbulent Jets" (1976)
//   ΔT = Q / (ṁ · Cₚ)  where  ṁ = ρ · V̇
const AIR_RHO = 1.2; // kg/m³ at ~22 °C sea level
const AIR_CP = 1006; // J/(kg·K) — specific heat of dry air
const CFM_TO_M3S = 0.000472; // 1 CFM → m³/s
const DEFAULT_CFM = 800; // mid-range 42U rack airflow
const AC_CFM = 2000; // CRAC airflow
const AC_SUPPLY_T = 14; // °C — supply temperature
const AMBIENT_T = 22; // °C — ASHRAE recommended inlet
// Turbulent plane-jet half-angle spreading rate (Rajaratnam 1976):
const PLUME_SIGMA_RAD = 0.35; // ≈ 20° effective half-angle on floor
const HEATMAP_HEIGHT = 1.0; // m — rack mid-height sample plane

export function buildHeatmap(group: THREE.Group, sceneData: SceneGraph, loadFactor = 1.0): number {
  group.clear();
  const [rw, , rd] = sceneData.room.dimensions;
  const resolution = 0.5;

  const heatSources = sceneData.furniture
    .filter((f) => f.heatOutput > 0)
    .map((f) => {
      const rotY = THREE.MathUtils.degToRad(f.rotation[1]);
      const backX = -Math.sin(rotY);
      const backZ = -Math.cos(rotY);
      const rackW = f.size[0] || 0.6;
      const rackH = f.size[1] || 2.0;
      const rackD = f.size[2] || 1.0;
      const halfDepth = rackD / 2;

      const Q_watts = f.heatOutput * 1000 * loadFactor;
      const m_dot = AIR_RHO * DEFAULT_CFM * CFM_TO_M3S;
      const deltaT = Q_watts / (m_dot * AIR_CP);

      const vertFrac =
        HEATMAP_HEIGHT <= rackH ? 1.0 : Math.exp(-0.5 * ((HEATMAP_HEIGHT - rackH) / 0.5) ** 2);

      return {
        x: f.position[0] + backX * halfDepth,
        z: f.position[2] + backZ * halfDepth,
        backX,
        backZ,
        rackW,
        deltaT: deltaT * vertFrac,
      };
    });

  const coolingSources = sceneData.furniture
    .filter((f) => f.category === "cooling_unit")
    .map((cool) => {
      const coolQ = AIR_RHO * AC_CFM * CFM_TO_M3S * AIR_CP * (AMBIENT_T - AC_SUPPLY_T);
      return {
        x: cool.position[0],
        z: cool.position[2],
        coolingW: coolQ * loadFactor,
      };
    });

  let maxDeltaT = 0;
  const nx = Math.ceil(rw / resolution);
  const nz = Math.ceil(rd / resolution);
  const field = new Float32Array(nx * nz);

  for (let xi = 0; xi < nx; xi++) {
    const x = resolution / 2 + xi * resolution;
    for (let zi = 0; zi < nz; zi++) {
      const z = resolution / 2 + zi * resolution;
      let localDeltaT = 0;

      for (const src of heatSources) {
        const dx = x - src.x;
        const dz = z - src.z;
        const dist = Math.sqrt(dx * dx + dz * dz);

        if (dist < 0.01) {
          localDeltaT += src.deltaT;
          continue;
        }

        const dot = (dx * src.backX + dz * src.backZ) / dist;
        const theta = Math.acos(Math.min(1, Math.max(-1, dot)));
        const angularDecay = Math.exp(-(theta * theta) / (2 * PLUME_SIGMA_RAD * PLUME_SIGMA_RAD));

        const b0 = src.rackW / 2;
        const coreLen = b0 * 5;
        const axialDecay = dist < coreLen ? 1.0 : Math.sqrt(coreLen / dist);

        const recirc = 0.03;
        const contribution = src.deltaT * (angularDecay * axialDecay * (1 - recirc) + recirc);
        localDeltaT += contribution;
      }

      for (const cool of coolingSources) {
        const dx = x - cool.x;
        const dz = z - cool.z;
        const dist = Math.sqrt(dx * dx + dz * dz);
        const r0 = 0.3;
        const m_dot_cool = AIR_RHO * AC_CFM * CFM_TO_M3S;
        const baseCoolingDT = cool.coolingW / (m_dot_cool * AIR_CP);
        const spatialDecay = dist < r0 ? 1.0 : r0 / dist;
        const vertMix = Math.exp(-HEATMAP_HEIGHT / 1.5);
        const cooling = baseCoolingDT * spatialDecay * vertMix;
        localDeltaT -= Math.min(cooling, localDeltaT);
      }

      localDeltaT = Math.max(0, localDeltaT);
      field[xi * nz + zi] = localDeltaT;
      if (localDeltaT > maxDeltaT) maxDeltaT = localDeltaT;
    }
  }

  if (maxDeltaT < 0.1) maxDeltaT = 1;
  for (let xi = 0; xi < nx; xi++) {
    const x = resolution / 2 + xi * resolution;
    for (let zi = 0; zi < nz; zi++) {
      const z = resolution / 2 + zi * resolution;
      const dt = field[xi * nz + zi];
      const t = Math.min(dt / maxDeltaT, 1);

      // CFD-standard Jet colormap (MATLAB Jet): dark blue → blue → cyan →
      // green → yellow → red → dark red. Saturated, no white.
      const color = new THREE.Color();
      const stops = [
        [0.0, 0.0, 0.5],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
      ];
      const positions = [0.0, 0.125, 0.375, 0.625, 0.875, 1.0];
      let lo = 0;
      for (let i = 1; i < positions.length; i++) {
        if (t <= positions[i]) {
          lo = i - 1;
          break;
        }
        lo = i;
      }
      const hi = Math.min(lo + 1, positions.length - 1);
      const span = positions[hi] - positions[lo] || 1;
      const t01 = (t - positions[lo]) / span;
      color.setRGB(
        stops[lo][0] + (stops[hi][0] - stops[lo][0]) * t01,
        stops[lo][1] + (stops[hi][1] - stops[lo][1]) * t01,
        stops[lo][2] + (stops[hi][2] - stops[lo][2]) * t01,
      );

      const tileGeo = new THREE.PlaneGeometry(resolution * 0.95, resolution * 0.95);
      // Heatmap tiles use NormalBlending (default) so the saturated Jet hues
      // stay readable on the light canvas without saturating to white.
      const tileMat = new THREE.MeshBasicMaterial({
        color,
        transparent: true,
        opacity: 0.3 + 0.45 * loadFactor,
        side: THREE.DoubleSide,
      });
      const tile = new THREE.Mesh(tileGeo, tileMat);
      tile.rotation.x = -Math.PI / 2;
      tile.position.set(x, 0.01, z);
      group.add(tile);
    }
  }

  return maxDeltaT;
}
