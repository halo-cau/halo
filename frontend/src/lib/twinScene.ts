/**
 * Runtime twin → 3D SceneGraph converter for the 시나리오 page.
 *
 * Builds a scene live from the running backend instead of the static frontend/src/data/realScenes.ts
 * snapshot. It mirrors tools/gen_real_scenes.py so the runtime result matches the static one:
 *   - "scanned"   = racks/infra where the scan found them (GET .../artifact/placements.json + .../thermal)
 *   - "optimized" = the imitation layout policy's proposal (POST .../optimize)
 *
 * Coordinate map (twin → scene): center [tx,ty,tz] → [tx,0,ty]; ext [dx,dy,dz] → dimensions [dx,dz,dy];
 * facing PLUS_Y/MINUS_Y/PLUS_X/MINUS_X → Y-rotation 0/180/90/270.
 */
import type { Equipment, MidTemp, SceneGraph, Score } from "../data/sceneGraphs";

export type SceneVariant = "scanned" | "optimized";

const RACK_KW = [12, 10, 6, 8, 5, 7, 11, 6, 9, 5, 8, 6]; // per physical rack; same in both variants
const RACK_COLORS = ["#37474F", "#455A64"];
const FACE_ROT: Record<string, number> = { PLUS_Y: 0, MINUS_Y: 180, PLUS_X: 90, MINUS_X: 270 };

interface TwinInstance {
  name?: string;
  kind?: string;
  facing?: string;
  center: [number, number, number];
}
interface Placements {
  ext: [number, number, number];
  instances: TwinInstance[];
}
interface ThermalResp {
  n_racks?: number | null;
  compliant_racks?: number;
  temp_max_c?: number;
  racks?: { intake_temp?: number }[];
  mid_temp?: MidTemp | null;
}

// Hottest rack-intake temperature across the per-rack solver results — the worst inlet a server sees, and
// the metric that actually distinguishes layouts (the peak air temperature is the rack exhaust and is
// layout-independent). Returns undefined when no per-rack data is present.
function maxIntake(racks?: { intake_temp?: number }[]): number | undefined {
  const vals = (racks ?? [])
    .map((r) => r.intake_temp)
    .filter((v): v is number => typeof v === "number");
  return vals.length ? Math.max(...vals) : undefined;
}
interface OptimizeResp {
  ext: [number, number, number];
  instances: TwinInstance[];
  fixed: TwinInstance[];
  thermal: ThermalResp;
  mid_temp?: MidTemp | null;
}

const round3 = (x: number) => Math.round(x * 1000) / 1000;
const round2 = (x: number) => Math.round(x * 100) / 100;

function rackIdx(name: string | undefined): number {
  const last = String(name ?? "")
    .trim()
    .split(/\s+/)
    .pop();
  const n = Number.parseInt(last ?? "", 10);
  return Number.isFinite(n) && n > 0 ? n - 1 : 0;
}

function rackEquip(inst: TwinInstance): Equipment {
  const idx = rackIdx(inst.name);
  const [cx, cy] = inst.center;
  return {
    id: `rack_${String(idx + 1).padStart(2, "0")}`,
    category: "server_rack",
    label: `서버 랙 ${idx + 1}`,
    position: [round3(cx), 0, round3(cy)],
    rotation: [0, FACE_ROT[inst.facing ?? "PLUS_Y"] ?? 0, 0],
    size: [0.6, 1.95, 0.9],
    color: RACK_COLORS[idx % 2],
    heatOutput: RACK_KW[idx % RACK_KW.length],
    relations: [],
  };
}

// Map a non-rack scanned/proposed instance (AC / network rack / power cabinet) to scene Equipment.
function infraEquip(inst: TwinInstance): Equipment | null {
  const name = String(inst.name ?? "");
  const [cx, cy] = inst.center;
  const position: [number, number, number] = [round3(cx), 0, round3(cy)];
  if (name.startsWith("ac_unit")) {
    return {
      id: "cooling_01",
      category: "cooling_unit",
      label: "스탠드 에어컨 (스캔)",
      position,
      rotation: [0, 90, 0],
      size: [1.5, 2.0, 0.5],
      color: "#eae7df",
      heatOutput: 0,
      relations: [],
    };
  }
  if (name === "network rack") {
    return {
      id: "core_switch",
      category: "network_switch",
      label: "네트워크 랙 (스캔)",
      position,
      // Honour the instance facing (like the server racks) so the network rack's intake/exhaust line up
      // with the row it sits in, rather than always facing PLUS_Y and reading flipped next to a MINUS_Y row.
      rotation: [0, FACE_ROT[inst.facing ?? "PLUS_Y"] ?? 0, 0],
      size: [0.6, 2.0, 0.8],
      color: "#1565C0",
      heatOutput: 3,
      relations: [],
    };
  }
  if (name === "power cabinet") {
    return {
      id: "power_cabinet_01",
      category: "power_cabinet",
      label: "분전반 (스캔)",
      position,
      rotation: [0, 0, 0],
      size: [0.7, 2.0, 0.4],
      color: "#8a978f",
      heatOutput: 0,
      relations: [],
    };
  }
  return null; // fire hose etc. are movable, not fixed infra → not part of the layout scene
}

function notNull(e: Equipment | null): e is Equipment {
  return e !== null;
}

function scoreFromThermal(th: ThermalResp): Score {
  const n = Math.max(1, th.n_racks || th.racks?.length || 1); // /optimize sends n_racks=null + a racks[]
  const compliant = th.compliant_racks ?? n;
  const peak = th.temp_max_c ?? 30;
  const total = round2(
    Math.min(0.95, Math.max(0.1, (compliant / n) * (1 - Math.max(0, peak - 30) / 50))),
  );
  return {
    total,
    thermal: round2(total * 0.97),
    cooling: round2(total * 0.95),
    cable: round2(Math.min(0.95, total + 0.05)),
    proximity: round2(total * 0.98),
    constraint: round2(Math.min(0.98, total + 0.1)),
  };
}

function assemble(
  ext: [number, number, number],
  furniture: Equipment[],
  id: string,
  name: string,
  description: string,
  score: Score,
): SceneGraph {
  // twin ext = [X length, Y depth, Z up]; scene dimensions = [width, height, depth] = [ext0, ext2, ext1].
  const [w, d, h] = ext;
  return {
    id,
    name,
    description,
    room: {
      type: "server_room",
      dimensions: [round2(w), round2(h), round2(d)],
      openings: [
        { type: "door", wall: "south", position: [w / 2, 0, d], width: 1.2 },
        { type: "vent", wall: "west", position: [0, 2.0, d - 0.85], width: 1.4 },
      ],
    },
    furniture,
    score,
  };
}

async function getJson<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, init);
  if (!r.ok) throw new Error(`${url} → ${r.status}`);
  return (await r.json()) as T;
}

/** Build a 시나리오 SceneGraph for one job at runtime, from the live backend. */
export async function sceneFromTwin(jobId: string, variant: SceneVariant): Promise<SceneGraph> {
  if (variant === "optimized") {
    const [opt, pm] = await Promise.all([
      getJson<OptimizeResp>(`/api/v1/twin/${jobId}/optimize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: "{}",
      }),
      getJson<Placements>(`/api/v1/twin/${jobId}/artifact/placements.json`),
    ]);
    const furniture: Equipment[] = [
      ...opt.instances.filter((i) => i.kind === "rack").map(rackEquip),
      ...opt.fixed.map(infraEquip).filter(notNull),
      // the layout policy does not move the power cabinet → keep it at the scanned position
      ...pm.instances
        .filter((i) => i.name === "power cabinet")
        .map(infraEquip)
        .filter(notNull),
    ];
    const sg = assemble(
      opt.ext,
      furniture,
      "rl_ppo_v1",
      "RL 최적화 배치 (PPO)",
      "imitation 학습 정책이 스캔된 방에 제안한 정렬 배치.",
      scoreFromThermal(opt.thermal),
    );
    sg.midTemp = opt.mid_temp ?? undefined;
    sg.maxIntakeC = maxIntake(opt.thermal?.racks);
    return sg;
  }

  const [pm, th] = await Promise.all([
    getJson<Placements>(`/api/v1/twin/${jobId}/artifact/placements.json`),
    getJson<ThermalResp>(`/api/v1/twin/${jobId}/thermal`),
  ]);
  const furniture: Equipment[] = [
    ...pm.instances.filter((i) => i.kind === "rack").map(rackEquip),
    ...pm.instances.map(infraEquip).filter(notNull),
  ];
  const sg = assemble(
    pm.ext,
    furniture,
    "random_v1",
    "스캔 원본 배치",
    "실제 스캔으로 복원한 서버실 — 랙이 발견된 그대로의 배치.",
    scoreFromThermal(th),
  );
  sg.midTemp = th.mid_temp ?? undefined;
  sg.maxIntakeC = maxIntake(th.racks);
  return sg;
}
