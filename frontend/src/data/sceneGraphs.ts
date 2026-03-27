// HALO Scene Graph 데이터
// 서버실(Data Center) 공간 — 3단계 배치 결과: random → rule-based → RL 최적화

export interface Opening {
  type: "door" | "window" | "vent" | "entrance";
  wall: "north" | "south" | "east" | "west";
  position: [number, number, number];
  width: number;
}

export interface Relation {
  type: "cooling_serves" | "cable_connected" | "hot_aisle" | "cold_aisle" | "adjacent_to" | "grouped_with";
  target: string;
}

export interface Equipment {
  id: string;
  category: string;
  label: string;
  position: [number, number, number];
  rotation: [number, number, number];
  size: [number, number, number]; // width, height, depth
  color: string;
  heatOutput: number; // kW 발열량 (0 for non-heat sources)
  relations: Relation[];
}

export interface Score {
  total: number;
  thermal: number;     // 열 균등 분포
  cooling: number;     // 냉각 효율
  cable: number;       // 케이블 길이 최소화
  proximity: number;   // 메인 서버 근접성
  constraint: number;  // 제약 조건 충족
}

export interface SceneGraph {
  id: string;
  name: string;
  description: string;
  room: {
    type: string;
    dimensions: [number, number, number]; // width, height, depth
    openings: Opening[];
  };
  furniture: Equipment[];
  score: Score;
}

// 비정형 서버실: 12m x 3.5m x 9m (기존 건물 전환 공간)
const SERVER_ROOM = {
  type: "server_room",
  dimensions: [12, 3.5, 9] as [number, number, number],
  openings: [
    { type: "door" as const, wall: "south" as const, position: [6, 0, 9] as [number, number, number], width: 1.8 },
    { type: "vent" as const, wall: "north" as const, position: [3, 2.8, 0] as [number, number, number], width: 2.0 },
    { type: "vent" as const, wall: "north" as const, position: [9, 2.8, 0] as [number, number, number], width: 2.0 },
    { type: "vent" as const, wall: "east" as const, position: [12, 2.8, 4.5] as [number, number, number], width: 1.5 },
  ],
};

// ===== Stage 1: 무작위 배치 (Random) =====
export const randomPlacement: SceneGraph = {
  id: "random_v1",
  name: "무작위 배치",
  description:
    "서버 랙이 무작위로 흩어져 핫스팟 발생, 냉각 효율 극저. 케이블 경로 비효율적.",
  room: SERVER_ROOM,
  furniture: [
    {
      id: "rack_01",
      category: "server_rack",
      label: "서버 랙 A (GPU)",
      position: [2.5, 0, 2.0],
      rotation: [0, 35, 0],
      size: [0.6, 2.0, 1.0],
      color: "#37474F",
      heatOutput: 15,
      relations: [],
    },
    {
      id: "rack_02",
      category: "server_rack",
      label: "서버 랙 B (GPU)",
      position: [8.0, 0, 7.0],
      rotation: [0, -20, 0],
      size: [0.6, 2.0, 1.0],
      color: "#455A64",
      heatOutput: 12,
      relations: [],
    },
    {
      id: "rack_03",
      category: "server_rack",
      label: "서버 랙 C (Storage)",
      position: [5.5, 0, 5.5],
      rotation: [0, 70, 0],
      size: [0.6, 2.0, 1.0],
      color: "#37474F",
      heatOutput: 5,
      relations: [],
    },
    {
      id: "rack_04",
      category: "server_rack",
      label: "서버 랙 D (CPU)",
      position: [10.0, 0, 3.0],
      rotation: [0, 110, 0],
      size: [0.6, 2.0, 1.0],
      color: "#455A64",
      heatOutput: 10,
      relations: [],
    },
    {
      id: "rack_05",
      category: "server_rack",
      label: "서버 랙 E (CPU)",
      position: [3.5, 0, 7.5],
      rotation: [0, 15, 0],
      size: [0.6, 2.0, 1.0],
      color: "#37474F",
      heatOutput: 8,
      relations: [],
    },
    {
      id: "core_switch",
      category: "network_switch",
      label: "코어 스위치",
      position: [7.0, 0, 1.0],
      rotation: [0, 45, 0],
      size: [0.6, 1.8, 0.8],
      color: "#1565C0",
      heatOutput: 3,
      relations: [],
    },
    {
      id: "cooling_01",
      category: "cooling_unit",
      label: "냉각 장치 A (CRAC)",
      position: [1.0, 0, 5.0],
      rotation: [0, 0, 0],
      size: [1.0, 2.2, 0.8],
      color: "#0097A7",
      heatOutput: 0,
      relations: [],
    },
    {
      id: "cooling_02",
      category: "cooling_unit",
      label: "냉각 장치 B (CRAC)",
      position: [11.0, 0, 6.5],
      rotation: [0, -60, 0],
      size: [1.0, 2.2, 0.8],
      color: "#00838F",
      heatOutput: 0,
      relations: [],
    },
    {
      id: "ups_01",
      category: "ups",
      label: "UPS 전원 장치",
      position: [9.0, 0, 1.5],
      rotation: [0, 80, 0],
      size: [0.8, 1.5, 0.6],
      color: "#E65100",
      heatOutput: 2,
      relations: [],
    },
    {
      id: "pdu_01",
      category: "pdu",
      label: "PDU 분전반",
      position: [4.5, 0, 3.5],
      rotation: [0, 0, 0],
      size: [0.4, 1.8, 0.3],
      color: "#F57C00",
      heatOutput: 1,
      relations: [],
    },
    {
      id: "monitor_01",
      category: "monitoring",
      label: "환경 모니터링 장비",
      position: [6.0, 0, 8.0],
      rotation: [0, 180, 0],
      size: [0.5, 1.2, 0.4],
      color: "#7B1FA2",
      heatOutput: 0.5,
      relations: [],
    },
    {
      id: "cable_tray_01",
      category: "cable_tray",
      label: "케이블 트레이",
      position: [6.0, 0, 4.5],
      rotation: [0, 30, 0],
      size: [4.0, 0.1, 0.4],
      color: "#FDD835",
      heatOutput: 0,
      relations: [],
    },
  ],
  score: {
    total: 0.15,
    thermal: 0.08,
    cooling: 0.12,
    cable: 0.18,
    proximity: 0.20,
    constraint: 0.25,
  },
};

// ===== Stage 2: 규칙 기반 (Rule-Based) =====
export const ruleBasedPlacement: SceneGraph = {
  id: "rule_v1",
  name: "규칙 기반 배치",
  description:
    "Hot/Cold Aisle 규칙 적용. 서버 랙 정렬되었으나 비정형 공간 활용 미흡, 냉각 장치 배치 부최적.",
  room: SERVER_ROOM,
  furniture: [
    {
      id: "rack_01",
      category: "server_rack",
      label: "서버 랙 A (GPU)",
      position: [3.0, 0, 3.0],
      rotation: [0, 0, 0],
      size: [0.6, 2.0, 1.0],
      color: "#37474F",
      heatOutput: 15,
      relations: [{ type: "hot_aisle", target: "rack_02" }],
    },
    {
      id: "rack_02",
      category: "server_rack",
      label: "서버 랙 B (GPU)",
      position: [3.0, 0, 5.0],
      rotation: [0, 180, 0],
      size: [0.6, 2.0, 1.0],
      color: "#455A64",
      heatOutput: 12,
      relations: [
        { type: "hot_aisle", target: "rack_01" },
        { type: "cold_aisle", target: "rack_03" },
      ],
    },
    {
      id: "rack_03",
      category: "server_rack",
      label: "서버 랙 C (Storage)",
      position: [3.0, 0, 7.0],
      rotation: [0, 0, 0],
      size: [0.6, 2.0, 1.0],
      color: "#37474F",
      heatOutput: 5,
      relations: [{ type: "cold_aisle", target: "rack_02" }],
    },
    {
      id: "rack_04",
      category: "server_rack",
      label: "서버 랙 D (CPU)",
      position: [7.0, 0, 3.0],
      rotation: [0, 0, 0],
      size: [0.6, 2.0, 1.0],
      color: "#455A64",
      heatOutput: 10,
      relations: [{ type: "hot_aisle", target: "rack_05" }],
    },
    {
      id: "rack_05",
      category: "server_rack",
      label: "서버 랙 E (CPU)",
      position: [7.0, 0, 5.0],
      rotation: [0, 180, 0],
      size: [0.6, 2.0, 1.0],
      color: "#37474F",
      heatOutput: 8,
      relations: [{ type: "hot_aisle", target: "rack_04" }],
    },
    {
      id: "core_switch",
      category: "network_switch",
      label: "코어 스위치",
      position: [5.0, 0, 1.5],
      rotation: [0, 0, 0],
      size: [0.6, 1.8, 0.8],
      color: "#1565C0",
      heatOutput: 3,
      relations: [
        { type: "cable_connected", target: "rack_01" },
        { type: "cable_connected", target: "rack_04" },
      ],
    },
    {
      id: "cooling_01",
      category: "cooling_unit",
      label: "냉각 장치 A (CRAC)",
      position: [0.8, 0, 4.5],
      rotation: [0, 90, 0],
      size: [1.0, 2.2, 0.8],
      color: "#0097A7",
      heatOutput: 0,
      relations: [{ type: "cooling_serves", target: "rack_01" }],
    },
    {
      id: "cooling_02",
      category: "cooling_unit",
      label: "냉각 장치 B (CRAC)",
      position: [11.0, 0, 4.5],
      rotation: [0, -90, 0],
      size: [1.0, 2.2, 0.8],
      color: "#00838F",
      heatOutput: 0,
      relations: [{ type: "cooling_serves", target: "rack_04" }],
    },
    {
      id: "ups_01",
      category: "ups",
      label: "UPS 전원 장치",
      position: [10.0, 0, 1.5],
      rotation: [0, 0, 0],
      size: [0.8, 1.5, 0.6],
      color: "#E65100",
      heatOutput: 2,
      relations: [{ type: "adjacent_to", target: "pdu_01" }],
    },
    {
      id: "pdu_01",
      category: "pdu",
      label: "PDU 분전반",
      position: [10.0, 0, 3.0],
      rotation: [0, 0, 0],
      size: [0.4, 1.8, 0.3],
      color: "#F57C00",
      heatOutput: 1,
      relations: [{ type: "adjacent_to", target: "ups_01" }],
    },
    {
      id: "monitor_01",
      category: "monitoring",
      label: "환경 모니터링 장비",
      position: [6.0, 0, 8.0],
      rotation: [0, 0, 0],
      size: [0.5, 1.2, 0.4],
      color: "#7B1FA2",
      heatOutput: 0.5,
      relations: [],
    },
    {
      id: "cable_tray_01",
      category: "cable_tray",
      label: "케이블 트레이",
      position: [5.0, 0, 3.0],
      rotation: [0, 0, 0],
      size: [4.0, 0.1, 0.4],
      color: "#FDD835",
      heatOutput: 0,
      relations: [
        { type: "cable_connected", target: "core_switch" },
        { type: "cable_connected", target: "rack_04" },
      ],
    },
  ],
  score: {
    total: 0.48,
    thermal: 0.42,
    cooling: 0.50,
    cable: 0.55,
    proximity: 0.52,
    constraint: 0.65,
  },
};

// ===== Stage 3: RL 최적화 (PPO) =====
export const rlOptimizedPlacement: SceneGraph = {
  id: "rl_ppo_v1",
  name: "RL 최적화 배치 (PPO)",
  description:
    "강화학습으로 열 분포 최적화. Hot/Cold Aisle 최적 구성, 고발열 GPU 랙을 냉각 장치 인근에 배치, 케이블 경로 최소화.",
  room: SERVER_ROOM,
  furniture: [
    {
      id: "rack_01",
      category: "server_rack",
      label: "서버 랙 A (GPU)",
      position: [4.0, 0, 2.5],
      rotation: [0, 0, 0],
      size: [0.6, 2.0, 1.0],
      color: "#37474F",
      heatOutput: 15,
      relations: [
        { type: "hot_aisle", target: "rack_02" },
        { type: "cold_aisle", target: "rack_04" },
      ],
    },
    {
      id: "rack_02",
      category: "server_rack",
      label: "서버 랙 B (GPU)",
      position: [4.0, 0, 4.5],
      rotation: [0, 180, 0],
      size: [0.6, 2.0, 1.0],
      color: "#455A64",
      heatOutput: 12,
      relations: [
        { type: "hot_aisle", target: "rack_01" },
        { type: "cold_aisle", target: "rack_03" },
      ],
    },
    {
      id: "rack_03",
      category: "server_rack",
      label: "서버 랙 C (Storage)",
      position: [4.0, 0, 6.5],
      rotation: [0, 0, 0],
      size: [0.6, 2.0, 1.0],
      color: "#37474F",
      heatOutput: 5,
      relations: [
        { type: "cold_aisle", target: "rack_02" },
        { type: "cable_connected", target: "core_switch" },
      ],
    },
    {
      id: "rack_04",
      category: "server_rack",
      label: "서버 랙 D (CPU)",
      position: [8.0, 0, 2.5],
      rotation: [0, 0, 0],
      size: [0.6, 2.0, 1.0],
      color: "#455A64",
      heatOutput: 10,
      relations: [
        { type: "cold_aisle", target: "rack_01" },
        { type: "hot_aisle", target: "rack_05" },
      ],
    },
    {
      id: "rack_05",
      category: "server_rack",
      label: "서버 랙 E (CPU)",
      position: [8.0, 0, 4.5],
      rotation: [0, 180, 0],
      size: [0.6, 2.0, 1.0],
      color: "#37474F",
      heatOutput: 8,
      relations: [
        { type: "hot_aisle", target: "rack_04" },
        { type: "cold_aisle", target: "rack_03" },
        { type: "cable_connected", target: "core_switch" },
      ],
    },
    {
      id: "core_switch",
      category: "network_switch",
      label: "코어 스위치",
      position: [6.0, 0, 1.5],
      rotation: [0, 0, 0],
      size: [0.6, 1.8, 0.8],
      color: "#1565C0",
      heatOutput: 3,
      relations: [
        { type: "cable_connected", target: "rack_01" },
        { type: "cable_connected", target: "rack_02" },
        { type: "cable_connected", target: "rack_04" },
        { type: "cable_connected", target: "rack_05" },
      ],
    },
    {
      id: "cooling_01",
      category: "cooling_unit",
      label: "냉각 장치 A (CRAC)",
      position: [1.0, 0, 3.5],
      rotation: [0, 90, 0],
      size: [1.0, 2.2, 0.8],
      color: "#0097A7",
      heatOutput: 0,
      relations: [
        { type: "cooling_serves", target: "rack_01" },
        { type: "cooling_serves", target: "rack_02" },
      ],
    },
    {
      id: "cooling_02",
      category: "cooling_unit",
      label: "냉각 장치 B (CRAC)",
      position: [11.0, 0, 3.5],
      rotation: [0, -90, 0],
      size: [1.0, 2.2, 0.8],
      color: "#00838F",
      heatOutput: 0,
      relations: [
        { type: "cooling_serves", target: "rack_04" },
        { type: "cooling_serves", target: "rack_05" },
      ],
    },
    {
      id: "cooling_03",
      category: "cooling_unit",
      label: "냉각 장치 C (보조)",
      position: [6.0, 0, 7.5],
      rotation: [0, 0, 0],
      size: [0.8, 2.0, 0.6],
      color: "#00ACC1",
      heatOutput: 0,
      relations: [
        { type: "cooling_serves", target: "rack_03" },
      ],
    },
    {
      id: "ups_01",
      category: "ups",
      label: "UPS 전원 장치",
      position: [10.5, 0, 1.0],
      rotation: [0, -90, 0],
      size: [0.8, 1.5, 0.6],
      color: "#E65100",
      heatOutput: 2,
      relations: [
        { type: "adjacent_to", target: "pdu_01" },
        { type: "grouped_with", target: "pdu_01" },
      ],
    },
    {
      id: "pdu_01",
      category: "pdu",
      label: "PDU 분전반",
      position: [10.5, 0, 2.5],
      rotation: [0, -90, 0],
      size: [0.4, 1.8, 0.3],
      color: "#F57C00",
      heatOutput: 1,
      relations: [
        { type: "adjacent_to", target: "ups_01" },
        { type: "grouped_with", target: "ups_01" },
      ],
    },
    {
      id: "monitor_01",
      category: "monitoring",
      label: "환경 모니터링 장비",
      position: [1.5, 0, 8.0],
      rotation: [0, 0, 0],
      size: [0.5, 1.2, 0.4],
      color: "#7B1FA2",
      heatOutput: 0.5,
      relations: [],
    },
    {
      id: "cable_tray_01",
      category: "cable_tray",
      label: "케이블 트레이 (메인)",
      position: [6.0, 0, 2.5],
      rotation: [0, 0, 0],
      size: [4.5, 0.1, 0.4],
      color: "#FDD835",
      heatOutput: 0,
      relations: [
        { type: "cable_connected", target: "core_switch" },
      ],
    },
    {
      id: "cable_tray_02",
      category: "cable_tray",
      label: "케이블 트레이 (보조)",
      position: [6.0, 0, 4.5],
      rotation: [0, 0, 0],
      size: [4.5, 0.1, 0.4],
      color: "#F9A825",
      heatOutput: 0,
      relations: [
        { type: "cable_connected", target: "cable_tray_01" },
      ],
    },
  ],
  score: {
    total: 0.91,
    thermal: 0.94,
    cooling: 0.92,
    cable: 0.88,
    proximity: 0.90,
    constraint: 0.95,
  },
};

// 학습 과정 에피소드별 점수 (차트용)
export const trainingHistory = {
  episodes: [0, 100, 200, 400, 600, 800, 1000, 1500, 2000, 3000, 5000, 8000, 10000],
  scores: {
    total:      [0.15, 0.22, 0.30, 0.40, 0.50, 0.60, 0.68, 0.76, 0.82, 0.87, 0.90, 0.91, 0.91],
    thermal:    [0.08, 0.15, 0.24, 0.35, 0.48, 0.60, 0.70, 0.80, 0.87, 0.92, 0.94, 0.94, 0.94],
    cooling:    [0.12, 0.20, 0.30, 0.42, 0.54, 0.64, 0.72, 0.82, 0.88, 0.91, 0.92, 0.92, 0.92],
    cable:      [0.18, 0.24, 0.32, 0.42, 0.50, 0.58, 0.65, 0.73, 0.80, 0.85, 0.88, 0.88, 0.88],
    proximity:  [0.20, 0.26, 0.33, 0.42, 0.52, 0.60, 0.68, 0.76, 0.82, 0.87, 0.90, 0.90, 0.90],
    constraint: [0.25, 0.34, 0.44, 0.55, 0.65, 0.74, 0.82, 0.88, 0.92, 0.95, 0.95, 0.95, 0.95],
  },
};

export const allScenes: SceneGraph[] = [
  randomPlacement,
  ruleBasedPlacement,
  rlOptimizedPlacement,
];

// ===== 시간대별 서버 부하 프로파일 (24시간) =====
// 각 시간(0~23)에 대한 부하 계수 (0.0 ~ 1.0)
export const loadProfile: number[] = [
  // 0h    1h    2h    3h    4h    5h
  0.15, 0.12, 0.10, 0.10, 0.12, 0.18,
  // 6h    7h    8h    9h   10h   11h
  0.30, 0.50, 0.70, 0.82, 0.88, 0.92,
  // 12h  13h   14h   15h   16h   17h
  0.95, 1.00, 0.98, 0.93, 0.88, 0.82,
  // 18h  19h   20h   21h   22h   23h
  0.72, 0.60, 0.45, 0.35, 0.25, 0.18,
];

// 시간대별 보간된 부하 계수 (분 단위 부드러운 전환)
export function getLoadFactor(hour: number): number {
  const h0 = Math.floor(hour) % 24;
  const h1 = (h0 + 1) % 24;
  const t = hour - Math.floor(hour);
  return loadProfile[h0] * (1 - t) + loadProfile[h1] * t;
}

// 배치별 peak time 냉각 에너지 (kW) — 배치 품질에 따라 다름
// random: 비효율적 → 냉각 에너지 높음, RL: 효율적 → 냉각 에너지 낮음
export const coolingEnergyBase: Record<string, number> = {
  random_v1: 42,   // kW (비효율)
  rule_v1: 30,     // kW (중간)
  rl_ppo_v1: 18,   // kW (효율적, ~40% 절감)
};

// 배치별 peak time 최고 온도 (°C)
export const peakTempBase: Record<string, number> = {
  random_v1: 48,   // 핫스팟 심각
  rule_v1: 38,     // 개선되었지만 미흡
  rl_ppo_v1: 29,   // 균등 분포
};
