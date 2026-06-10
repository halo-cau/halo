// Cooling dashboard — single-layout view inspired by the team's mock.
// Status bar → KPI cards → heatmap floor plan + 24h time series → savings.
// The "랙별 상세" table from the mock is intentionally omitted.

import * as echarts from "echarts";
import "./lib/twinIndicator";
import {
  allScenes,
  coolingEnergyBase,
  type Equipment,
  getLoadFactor,
  loadProfile,
  peakTempBase,
  type SceneGraph,
} from "./data/sceneGraphs";
import {
  AMBIENT_T,
  computeAshraeMetrics,
  type RackMetrics,
  type SceneAshraeMetrics,
  T_MAX_REC,
} from "./lib/ashrae";
import { loadSavedJob } from "./lib/twinJob";

// Design tokens (kept in lockstep with the CSS custom properties in dashboard.html)
const C = {
  bg: "#FAFAF7",
  bgCard: "#FDFCF9",
  bgSunken: "#F4F2EC",
  border: "#E8E5DD",
  text1: "#2C2C2A",
  text2: "#5F5E5A",
  text3: "#888780",
  rackGray: "#B4B2A9",
  // Sequential single-hue green ramp for "OK" range, then amber/red for warn/danger.
  // Replaces the old rainbow palette.
  greenDeep: "#1D9E75",
  greenMid: "#5DBE9A",
  greenSoft: "#9DD8B8",
  blueCool: "#378ADD",
  amber: "#E89E4F",
  red: "#E24B4A",
  successText: "#0F6E56",
  warnText: "#A86B1A",
  dangerText: "#A32D2D",
  infoText: "#0C447C",
};

const PEAK_HOUR = 13;
const PEAK_LOAD = getLoadFactor(PEAK_HOUR);

// Dashboard compares baseline (random) vs optimized (RL) — rule-based
// is shown on the scenarios page only.
const dashboardScenes = allScenes.filter((s) => s.id !== "rule_v1");

let currentSceneIdx = dashboardScenes.findIndex((s) => s.id === "rl_ppo_v1");
if (currentSceneIdx < 0) currentSceneIdx = 0;

// ---------- Helpers ----------
function intakeColor(t: number): string {
  // Match ASHRAE bands: <15 / 15-18 / 18-23 / 23-27 / 27-35 / >35.
  // Single-hue green for the recommended band, calm blue for "too cold",
  // amber for warn, muted red for danger. No rainbow.
  if (t < 15) return C.blueCool;
  if (t < 18) return "#7CB7E8"; // lighter calm blue
  if (t <= 23) return C.greenDeep;
  if (t <= 27) return C.greenMid;
  if (t <= 35) return C.amber;
  return C.red;
}

function pueFor(
  scene: SceneGraph,
  m?: SceneAshraeMetrics,
): { pue: number; itKw: number; coolKw: number } {
  const itKw =
    scene.furniture.filter((f) => f.heatOutput > 0).reduce((s, f) => s + f.heatOutput, 0) *
    PEAK_LOAD;
  let coolKw = coolingEnergyBase[scene.id] ?? 0;
  if (scene.id === "detected" && m) {
    // Real PUE from the thermal solve (the static table has no scanned room): cooling electrical
    // power = IT heat / effective chiller COP, plus a fixed fan/pump overhead. The COP scales with the
    // Rack Cooling Index (how well intakes sit inside ASHRAE) -- a poorly cooled room makes the chiller
    // work harder, raising PUE.
    const cop = 4.0 * Math.max(0.4, m.rciHi);
    coolKw = itKw / cop + 0.08 * itKw;
  }
  const pue = itKw > 0 ? (itKw + coolKw) / itKw : 1.0;
  return { pue, itKw, coolKw };
}

function escapeHtml(s: string): string {
  return s.replace(
    /[&<>"']/g,
    (c) =>
      ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;",
      })[c]!,
  );
}

// ---------- Scene picker ----------
function renderScenePicker() {
  const host = document.getElementById("scene-picker")!;
  host.innerHTML = "";
  dashboardScenes.forEach((s, i) => {
    const btn = document.createElement("button");
    btn.className = `scene-btn${i === currentSceneIdx ? " active" : ""}`;
    btn.innerHTML = `${escapeHtml(s.name)}<span class="scene-meta">${escapeHtml(s.id)} · ${s.furniture.length} 항목</span>`;
    btn.addEventListener("click", () => {
      currentSceneIdx = i;
      render();
    });
    host.appendChild(btn);
  });
}

// ---------- Status bar ----------
function renderStatusBar(scene: SceneGraph, m: SceneAshraeMetrics) {
  const host = document.getElementById("status-bar")!;
  const violations = m.violationCount;
  const warns = m.allowableCount;
  let dot = C.greenDeep;
  let title = "냉각 시스템 정상 운영";
  let sub = "모든 랙 ASHRAE 권장 범위 내 · 마지막 업데이트 14초 전";
  if (violations > 0) {
    dot = C.red;
    title = "냉각 시스템 주의 — 즉시 조치 필요";
    sub = `${violations}개 랙이 허용 범위(15–35°C)를 벗어났습니다 · 마지막 업데이트 14초 전`;
  } else if (warns > 0) {
    dot = C.amber;
    title = "냉각 시스템 모니터링";
    sub = `${warns}개 랙이 권장 범위 밖 (허용 범위 내) · 마지막 업데이트 14초 전`;
  }
  const totalIt = scene.furniture
    .filter((f) => f.heatOutput > 0)
    .reduce((s, f) => s + f.heatOutput, 0);
  host.innerHTML = `
    <div class="status-left">
      <span class="status-dot" style="background:${dot}; box-shadow: 0 0 0 4px ${dot}2A;"></span>
      <div>
        <p class="title">${title}</p>
        <p class="sub">${sub}</p>
      </div>
    </div>
    <div class="status-meta">
      ${escapeHtml(scene.name)} · ${m.perRack.length} 랙 · ${(totalIt * PEAK_LOAD).toFixed(1)} kW 부하
    </div>`;
}

// ---------- KPI cards ----------
function renderKpis(scene: SceneGraph, m: SceneAshraeMetrics) {
  const host = document.getElementById("kpi-grid")!;

  const peak = m.peakIntake;
  const peakPct = Math.max(0, Math.min(1, (peak - 18) / (35 - 18))) * 100;
  const peakColor = peak <= T_MAX_REC ? C.greenDeep : peak <= 35 ? C.amber : C.red;
  const peakSubCls = peak <= T_MAX_REC ? "good" : peak <= 35 ? "warn" : "bad";
  const peakSub =
    peak <= T_MAX_REC ? "권장 27°C 이하" : peak <= 35 ? "권장 초과 (허용 범위)" : "허용 범위 초과";

  const { pue, coolKw } = pueFor(scene, m);
  const random = allScenes.find((s) => s.id === "random_v1");
  const baseCool = random ? (coolingEnergyBase[random.id] ?? coolKw) : coolKw;
  const savedKw = baseCool - coolKw;
  const savedPct = baseCool > 0 ? (savedKw / baseCool) * 100 : 0;

  // Pips: green = recommended, amber = allowable, red = violation
  const total = m.perRack.length;
  const pips: string[] = [];
  for (let i = 0; i < total; i++) {
    let color = C.greenDeep;
    if (i < m.violationCount) color = C.red;
    else if (i < m.violationCount + m.allowableCount) color = C.amber;
    pips.push(`<div style="background:${color};"></div>`);
  }

  const monthCostMan = (coolKw * 24 * 30 * 130) / 10000; // ₩130/kWh ballpark
  const violationCls = m.violationCount === 0 ? "good" : "bad";

  host.innerHTML = `
    <div class="kpi-card">
      <div class="lbl">최고 흡입 온도</div>
      <div class="val" style="color:${peakColor};">${peak.toFixed(1)}<span class="unit">°C</span></div>
      <div class="sub ${peakSubCls}">${peakSub}</div>
      <div class="kpi-bar"><div style="width:${peakPct}%; background:${peakColor};"></div></div>
    </div>

    <div class="kpi-card">
      <div class="lbl">냉각 효율 (PUE)</div>
      <div class="val">${pue.toFixed(2)}</div>
      <div class="sub ${pue <= 1.4 ? "good" : pue <= 1.6 ? "warn" : "bad"}">
        ${pue <= 1.4 ? "목표 1.4 이하 달성" : pue <= 1.6 ? "업계 평균 근접" : "비효율 — 개선 필요"}
      </div>
      <div class="sub" style="margin-top:2px;">목표: 1.4 이하 · 업계 평균 1.58</div>
    </div>

    <div class="kpi-card">
      <div class="lbl">냉방 전력</div>
      <div class="val">${coolKw.toFixed(1)} <span class="unit">kW</span></div>
      <div class="sub ${savedKw > 0 ? "good" : ""}">
        ${savedKw > 0 ? `−${savedPct.toFixed(0)}% vs 무작위` : "기준선"}
      </div>
      <div class="sub" style="margin-top:2px;">월 ₩${monthCostMan.toFixed(0)}만 추정</div>
    </div>

    <div class="kpi-card">
      <div class="lbl">과열 위험 랙</div>
      <div class="val" style="color:${m.violationCount === 0 ? C.greenDeep : C.red};">
        ${m.violationCount} <span class="unit">/ ${total}</span>
      </div>
      <div class="sub ${violationCls}">
        ${m.violationCount === 0 ? "허용 범위 내 안정 운영" : "즉시 조치 필요"}
      </div>
      <div class="kpi-pips">${pips.join("")}</div>
    </div>`;
}

// ---------- Heatmap floor plan ----------
// Top-down view of the room; each rack drawn as a colored cell whose hue
// reflects intake temperature. Cooling units shown as rack-gray blocks.
// rack id -> facing for the detected scene, so the floor plan can mark intake/exhaust faces. Empty for
// the static demo scenes (which have no scanned facing), so they draw no face bars.
const detRackFacing = new Map<string, string>();

function renderFloorPlan(scene: SceneGraph, m: SceneAshraeMetrics) {
  const svg = document.getElementById("floor-svg")!;
  const W = 480;
  const H = 280;
  const padX = 16;
  const padY = 16;
  const drawW = W - padX * 2 - 80; // leave room for legend on right
  const drawH = H - padY * 2;

  const [roomW, , roomD] = scene.room.dimensions;
  const sx = (x: number) => padX + (x / roomW) * drawW;
  const sz = (z: number) => padY + (z / roomD) * drawH;

  const intakeById: Record<string, number> = {};
  m.perRack.forEach((r) => {
    intakeById[r.id] = r.intakeTemp;
  });

  const parts: string[] = [];
  // Room outline — hairline
  parts.push(
    `<rect x="${padX}" y="${padY}" width="${drawW}" height="${drawH}" fill="${C.bg}" stroke="${C.border}" stroke-width="0.5"/>`,
  );

  // Two passes: rotated bodies first, then upright labels on top so labels
  // never appear upside-down when the rack faces 180°.
  const labelParts: string[] = [];

  // Only AC units and server racks are drawn; UPS/PDU/switches are hidden
  // to keep the heatmap readable.
  scene.furniture.forEach((f) => {
    const isAc = f.category === "cooling_unit" || f.category === "ceiling_ac";
    const isNet = f.category === "network_rack";
    const isRack = f.category === "server_rack" && f.heatOutput > 0;
    if (!isAc && !isRack && !isNet) return;

    const x = sx(f.position[0]);
    const z = sz(f.position[2]);
    const w = (f.size[0] / roomW) * drawW;
    const d = (f.size[2] / roomD) * drawH;
    const rotation = f.rotation[1];

    if (isAc) {
      parts.push(
        `<g transform="translate(${x},${z}) rotate(${rotation})">
          <rect x="${-w / 2}" y="${-d / 2}" width="${w}" height="${d}" fill="${C.rackGray}" stroke="${C.text2}" stroke-width="0.5" rx="1"/>
        </g>`,
      );
      // Pill-backed AC label so it reads cleanly over any rack/AC color.
      const acPillW = 18;
      const acPillH = 12;
      labelParts.push(
        `<rect x="${x - acPillW / 2}" y="${z - acPillH / 2}" width="${acPillW}" height="${acPillH}" rx="3" fill="${C.bgCard}" opacity="0.85"/>
         <text x="${x}" y="${z + 3}" text-anchor="middle" font-size="8" fill="${C.text1}" font-weight="500">AC</text>`,
      );
      return;
    }

    if (isNet) {
      parts.push(
        `<g transform="translate(${x},${z})">
          <rect x="${-w / 2}" y="${-d / 2}" width="${w}" height="${d}" fill="#7C6FD6" stroke="${C.bgCard}" stroke-width="0.5" rx="1.5"/>
        </g>`,
      );
      const npW = 22;
      const npH = 12;
      labelParts.push(
        `<rect x="${x - npW / 2}" y="${z - npH / 2}" width="${npW}" height="${npH}" rx="3" fill="${C.bgCard}" opacity="0.85"/>
         <text x="${x}" y="${z + 3}" text-anchor="middle" font-size="8" fill="${C.text1}" font-weight="500">NET</text>`,
      );
      return;
    }

    const t = intakeById[f.id] ?? AMBIENT_T;
    const color = intakeColor(t);
    // Blue intake / red exhaust face bars so the rack's direction reads at a glance in the heatmap.
    const facing = detRackFacing.get(f.id);
    let faceBars = "";
    if (facing) {
      const bt = Math.min(3, w / 3, d / 3);
      const bar = (c: string, bx: number, by: number, bw: number, bh: number) =>
        `<rect x="${bx}" y="${by}" width="${bw}" height="${bh}" fill="${c}"/>`;
      const edge = (e: string, c: string) =>
        e === "top"
          ? bar(c, -w / 2, -d / 2, w, bt)
          : e === "bottom"
            ? bar(c, -w / 2, d / 2 - bt, w, bt)
            : e === "left"
              ? bar(c, -w / 2, -d / 2, bt, d)
              : bar(c, w / 2 - bt, -d / 2, bt, d);
      // svg z is flipped (twin +Y is up); intake is the face the rack looks toward.
      const sides: Record<string, [string, string]> = {
        PLUS_Y: ["top", "bottom"],
        MINUS_Y: ["bottom", "top"],
        PLUS_X: ["right", "left"],
        MINUS_X: ["left", "right"],
      };
      const [inEdge, exEdge] = sides[facing] ?? ["top", "bottom"];
      faceBars = edge(inEdge, "#2E73E8") + edge(exEdge, "#E0402F");
    }
    parts.push(
      `<g transform="translate(${x},${z}) rotate(${rotation})">
        <rect x="${-w / 2}" y="${-d / 2}" width="${w}" height="${d}" fill="${color}" stroke="${C.bgCard}" stroke-width="0.5" rx="1.5"/>
        ${faceBars}
      </g>`,
    );
    // Pill-backed temperature label: a translucent card-color background
    // keeps a single unified text color readable across every band.
    const pillW = 26;
    const pillH = 13;
    labelParts.push(
      `<rect x="${x - pillW / 2}" y="${z - pillH / 2}" width="${pillW}" height="${pillH}" rx="3" fill="${C.bgCard}" opacity="0.85"/>
       <text x="${x}" y="${z + 3.5}" text-anchor="middle" font-size="9" font-weight="500" fill="${C.text1}">${t.toFixed(1)}</text>`,
    );
  });

  // Append upright labels last so they sit above the rotated bodies.
  parts.push(...labelParts);

  // Color scale legend (right side)
  const legX = padX + drawW + 14;
  const legY = padY + 10;
  const stops = [
    { c: C.greenDeep, t: "≤23°C" },
    { c: C.greenMid, t: "23–27°C" },
    { c: C.amber, t: "27–35°C" },
    { c: C.red, t: "≥35°C" },
  ];
  stops.forEach((s, i) => {
    const ly = legY + i * 22;
    parts.push(
      `<rect x="${legX}" y="${ly}" width="14" height="14" fill="${s.c}" rx="2"/>
       <text x="${legX + 18}" y="${ly + 11}" font-size="9" fill="${C.text2}">${s.t}</text>`,
    );
  });

  svg.innerHTML = parts.join("");
}

// ---------- 24h time series (ECharts) ----------
// Server intake = ambient + (peakTemp - ambient) * loadFactor(t)
// Outside     = sinusoidal day curve peaking around 14h
let tsChart: echarts.ECharts | null = null;

function renderTimeSeries(scene: SceneGraph, peakOverride?: number) {
  const host = document.getElementById("ts-chart") as HTMLElement;
  if (!host) return;

  if (!tsChart) {
    tsChart = echarts.init(host, undefined, { renderer: "svg" });
    window.addEventListener("resize", () => tsChart?.resize());
  }

  // For the scanned room, anchor the daily profile to its REAL peak intake (a projection over the load
  // cycle, not measured history); static demo scenes use their tabulated peak.
  const peakT = peakOverride ?? peakTempBase[scene.id] ?? 32;

  // Sample every 30 min
  const samples = 49;
  const hours: string[] = [];
  const serverData: number[] = [];
  const outsideData: number[] = [];
  for (let i = 0; i < samples; i++) {
    const h = (i / (samples - 1)) * 24;
    const lf = getLoadFactor(h);
    const serverT = AMBIENT_T + (peakT - AMBIENT_T) * lf;
    const outT = 24.5 - 8.5 * Math.cos(((h - 14) / 24) * 2 * Math.PI);
    hours.push(h.toFixed(1));
    serverData.push(Number(serverT.toFixed(2)));
    outsideData.push(Number(outT.toFixed(2)));
  }

  const peakLf = getLoadFactor(PEAK_HOUR);
  const peakServerT = AMBIENT_T + (peakT - AMBIENT_T) * peakLf;

  // Dynamic y-axis bounds. peakT varies wildly across scenes (29 / 38 / 48 °C),
  // so a fixed 19-30 range clips Random and Rule lines off the top of the
  // plot. Snap min/max to round numbers and make sure the 27 °C ASHRAE line
  // stays visible.
  const dataMin = Math.min(...serverData);
  const dataMax = Math.max(...serverData);
  const yServerMin = Math.floor(Math.min(20, dataMin - 1));
  const yServerMax = Math.ceil(Math.max(30, dataMax + 2));
  const yServerInterval = Math.max(2, Math.ceil((yServerMax - yServerMin) / 4));

  const fontFamily =
    '-apple-system, BlinkMacSystemFont, "Segoe UI", "Pretendard", system-ui, sans-serif';

  tsChart.setOption(
    {
      animation: false,
      backgroundColor: "transparent",
      textStyle: { fontFamily, color: C.text2 },
      grid: { left: 38, right: 36, top: 28, bottom: 32, containLabel: false },
      tooltip: {
        trigger: "axis",
        backgroundColor: C.bgCard,
        borderColor: C.border,
        borderWidth: 0.5,
        padding: [6, 10],
        textStyle: { color: C.text1, fontSize: 11, fontFamily },
        axisPointer: {
          lineStyle: { color: C.border, width: 0.5, type: "dashed" },
        },
        valueFormatter: (v: number | string) =>
          typeof v === "number" ? `${v.toFixed(1)}°C` : String(v),
      },
      legend: {
        data: ["서버 흡입", "외기", "부하"],
        right: 8,
        top: 4,
        textStyle: { color: C.text2, fontSize: 10, fontFamily },
        itemWidth: 14,
        itemHeight: 2,
        itemGap: 12,
        icon: "rect",
      },
      xAxis: {
        type: "category",
        data: hours,
        axisLine: { lineStyle: { color: C.border, width: 0.5 } },
        axisTick: { show: false },
        axisLabel: {
          color: C.text2,
          fontSize: 9,
          fontFamily,
          // Match exact integer hours only — using Math.round here would
          // print "12시" twice (once at 11.5 and again at 12.0).
          interval: (_idx: number, val: string) => {
            const h = parseFloat(val);
            return [0, 6, 12, 18, 24].includes(h);
          },
          formatter: (val: string) => `${parseFloat(val).toFixed(0)}시`,
        },
      },
      yAxis: [
        {
          type: "value",
          name: "흡입 °C",
          nameTextStyle: { color: C.successText, fontSize: 9, fontFamily, padding: [0, 0, 4, 0] },
          min: yServerMin,
          max: yServerMax,
          interval: yServerInterval,
          axisLine: { show: false },
          axisTick: { show: false },
          splitLine: {
            lineStyle: { color: C.border, width: 0.5, type: "dashed" },
          },
          axisLabel: { color: C.successText, fontSize: 8, fontFamily },
        },
        {
          type: "value",
          name: "외기 °C",
          nameTextStyle: { color: C.warnText, fontSize: 9, fontFamily, padding: [0, 0, 4, 0] },
          min: 14,
          max: 35,
          interval: 6,
          axisLine: { show: false },
          axisTick: { show: false },
          splitLine: { show: false },
          axisLabel: { color: C.warnText, fontSize: 8, fontFamily },
        },
        {
          type: "value",
          show: false,
          min: 0,
          max: 1,
        },
      ],
      series: [
        {
          name: "부하",
          type: "bar",
          xAxisIndex: 0,
          yAxisIndex: 2,
          data: loadProfile.map((lf) => lf),
          barCategoryGap: "55%",
          itemStyle: { color: C.blueCool, opacity: 0.15 },
          tooltip: {
            valueFormatter: (v: number | string) =>
              typeof v === "number" ? `${(v * 100).toFixed(0)}%` : String(v),
          },
          z: 1,
        },
        {
          name: "외기",
          type: "line",
          yAxisIndex: 1,
          data: outsideData,
          showSymbol: false,
          smooth: 0.3,
          lineStyle: { color: C.amber, width: 1.2, opacity: 0.85 },
          z: 2,
        },
        {
          name: "서버 흡입",
          type: "line",
          yAxisIndex: 0,
          data: serverData,
          showSymbol: false,
          smooth: 0.3,
          lineStyle: { color: C.greenDeep, width: 1.6 },
          markLine: {
            silent: true,
            symbol: "none",
            label: {
              formatter: `권장 상한 ${T_MAX_REC}°C`,
              color: C.dangerText,
              fontSize: 7,
              fontFamily,
              position: "insideEndTop",
            },
            lineStyle: {
              color: C.red,
              type: "dashed",
              width: 0.6,
              opacity: 0.7,
            },
            data: [{ yAxis: T_MAX_REC }],
          },
          markPoint: {
            symbolSize: 6,
            symbol: "circle",
            data: [
              {
                coord: [String(PEAK_HOUR.toFixed(1)), Number(peakServerT.toFixed(2))],
                value: `피크 ${peakServerT.toFixed(1)}°C`,
                label: {
                  formatter: `피크 ${peakServerT.toFixed(1)}°C`,
                  color: C.successText,
                  fontSize: 8,
                  fontFamily,
                  position: "top",
                  distance: 6,
                },
                itemStyle: {
                  color: C.greenDeep,
                  borderColor: C.bgCard,
                  borderWidth: 1,
                },
              },
            ],
          },
          z: 3,
        },
      ],
    },
    true,
  );
  // The chart container uses `flex: 1` and is stretched by the grid; the
  // size at init() time can predate layout, so re-measure after setOption.
  tsChart.resize();
}

// ---------- Compliance summary (replaces rack-detail table) ----------
function renderComplianceSummary(scene: SceneGraph, m: SceneAshraeMetrics) {
  const host = document.getElementById("comp-card")!;
  const total = m.perRack.length;
  const rciAvg = (m.rciHi + m.rciLo) / 2;
  const meanDt = m.meanDeltaT;

  const rec = m.recommendedCount;
  const allow = m.allowableCount;
  const danger = m.violationCount;

  const okText = C.successText;
  const warnText = C.warnText;
  const dangerText = C.dangerText;

  host.innerHTML = `
    <h3>ASHRAE 준수 요약 — ${escapeHtml(scene.name)}</h3>
    <div class="comp-row">
      <div class="l">권장 범위 (18–27°C) 안 랙
        <small>모든 랙이 안에 있을 때 만점</small>
      </div>
      <div class="v" style="color:${rec === total ? okText : warnText};">${rec} / ${total}</div>
    </div>
    <div class="comp-row">
      <div class="l">허용 범위 내 (권장 범위 밖)
        <small>15–35°C 안이지만 18–27°C 밖</small>
      </div>
      <div class="v" style="color:${allow === 0 ? C.text2 : warnText};">${allow}</div>
    </div>
    <div class="comp-row">
      <div class="l">허용 범위 초과
        <small>즉시 조치 필요 (15°C 미만 또는 35°C 초과)</small>
      </div>
      <div class="v" style="color:${danger === 0 ? okText : dangerText};">${danger}</div>
    </div>
    <div class="comp-row">
      <div class="l">RCI 평균
        <small>권장 범위 준수율 (100% = 만점)</small>
      </div>
      <div class="v" style="color:${rciAvg >= 0.95 ? okText : rciAvg >= 0.8 ? warnText : dangerText};">${(rciAvg * 100).toFixed(1)}%</div>
    </div>
    <div class="comp-row">
      <div class="l">SHI / RHI
        <small>SHI 0 / RHI 1 이 이상적 (격리 효율)</small>
      </div>
      <div class="v" style="color:${C.text1};">${m.shi.toFixed(2)} / ${m.rhi.toFixed(2)}</div>
    </div>
    <div class="comp-row">
      <div class="l">평균 ΔT (배기−흡기)
        <small>10–15°C가 정상</small>
      </div>
      <div class="v" style="color:${C.text1};">${meanDt.toFixed(1)}°C</div>
    </div>`;
}

// ---------- Savings card ----------
function renderSavings(scene: SceneGraph, m: SceneAshraeMetrics) {
  const host = document.getElementById("savings-card") as HTMLElement;
  // The scanned room has no RL-optimized counterpart yet (the savings comparison needs that run), so
  // show its REAL current cooling cost and ASHRAE state instead of a fabricated "vs random" figure.
  if (scene.id === "detected") {
    const { coolKw } = pueFor(scene, m);
    const annualMan = (coolKw * 24 * 365 * 130) / 10000;
    host.classList.remove("baseline");
    host.innerHTML = `
      <div class="savings-lbl">현재 냉방 비용 (연간 추정)</div>
      <div class="savings-val" style="color:${C.text1};">₩${annualMan.toFixed(0)}만</div>
      <div class="savings-sub">스캔된 방 · ${m.perRack.length}개 랙 · RL 최적화 후 절감 비교 가능</div>
      <hr/>
      <div class="savings-grid">
        <div class="item"><div class="lbl">평균 흡입온도</div>
          <div class="v">${m.meanIntake.toFixed(1)}°C</div></div>
        <div class="item"><div class="lbl">냉방 전력</div>
          <div class="v">${coolKw.toFixed(1)} kW</div></div>
        <div class="item"><div class="lbl">위반 랙</div>
          <div class="v" style="color:${m.violationCount === 0 ? C.successText : C.dangerText};">${m.violationCount}회</div></div>
        <div class="item"><div class="lbl">RCI</div>
          <div class="v">${(m.rciHi * 100).toFixed(0)}%</div></div>
      </div>`;
    return;
  }
  const random = allScenes.find((s) => s.id === "random_v1");
  const baseCool = random ? (coolingEnergyBase[random.id] ?? 0) : 0;
  const cool = coolingEnergyBase[scene.id] ?? 0;
  const savedKw = baseCool - cool;
  const savedPct = baseCool > 0 ? (savedKw / baseCool) * 100 : 0;
  const annualMan = (savedKw * 24 * 365 * 130) / 10000;

  const basePeak = random ? (peakTempBase[random.id] ?? 0) : 0;
  const peakDelta = basePeak - (peakTempBase[scene.id] ?? 0);

  const co2Tons = (savedKw * 24 * 365 * 0.42) / 1000;

  const isBaseline = scene.id === "random_v1";
  host.classList.toggle("baseline", isBaseline);

  const headline = isBaseline ? "기준선 (개선 전 대조군)" : "개선 후 누적 절감 (연간 기준)";
  const value = isBaseline ? `${cool.toFixed(0)} kW` : `₩${annualMan.toFixed(0)}만`;

  host.innerHTML = `
    <div class="savings-lbl">${headline}</div>
    <div class="savings-val">${value}</div>
    <div class="savings-sub">
      ${
        isBaseline
          ? "이 배치는 비교 기준선입니다 — 다른 시나리오와 비교해보세요."
          : `무작위 배치 대비 · ${m.perRack.length}개 랙 기준`
      }
    </div>
    <hr/>
    <div class="savings-grid">
      <div class="item">
        <div class="lbl">평균 흡입온도 변화</div>
        <div class="v" style="color:${peakDelta > 0 ? C.successText : C.text2};">${peakDelta > 0 ? "−" : ""}${Math.abs(peakDelta).toFixed(1)}°C</div>
      </div>
      <div class="item">
        <div class="lbl">전력 사용량 변화</div>
        <div class="v" style="color:${savedPct > 0 ? C.successText : C.text2};">${savedPct > 0 ? "−" : ""}${Math.abs(savedPct).toFixed(0)}%</div>
      </div>
      <div class="item">
        <div class="lbl">위반 랙</div>
        <div class="v" style="color:${m.violationCount === 0 ? C.successText : C.dangerText};">${m.violationCount}회</div>
      </div>
      <div class="item">
        <div class="lbl">CO₂ 절감 (연간)</div>
        <div class="v" style="color:${co2Tons > 0 ? C.successText : C.text2};">${co2Tons > 0 ? co2Tons.toFixed(1) : "0.0"}t</div>
      </div>
    </div>`;
}

// ---------- Detected-room thermal: render the SCANNED twin in the dashboard ----------
// The static scenes above are the demo fallback. This builds a SceneGraph + REAL ASHRAE metrics from the
// scanned twin -- its layout from placements.json, its thermal/ASHRAE from the 3-D engine
// (GET /api/v1/twin/{id}/thermal) -- and renders it through the same panels. The RL-optimised comparison
// is deferred (no model yet), so the scanned scene stands in for "current"; "optimised" is the same scene.
const T_MIN_ALLOW = 15;
const T_MIN_REC = 18;
const T_MAX_ALLOW = 35;

function zoneFor(t: number): { zone: RackMetrics["zone"]; status: RackMetrics["status"] } {
  if (t < T_MIN_ALLOW) return { zone: "below-allowable", status: "danger" };
  if (t < T_MIN_REC) return { zone: "below-recommended", status: "warn" };
  if (t <= T_MAX_REC) return { zone: "recommended", status: "ok" };
  if (t <= T_MAX_ALLOW) return { zone: "above-recommended", status: "warn" };
  return { zone: "above-allowable", status: "danger" };
}

interface ThermalResp {
  racks: { rack_index: number; intake_temp: number; exhaust_temp: number; delta_t: number }[];
  room: {
    rci_hi: number;
    rci_lo: number;
    shi: number;
    rhi: number;
    mean_intake: number;
    mean_exhaust: number;
  };
}
interface Manifest {
  ext: [number, number, number];
  instances: {
    name?: string;
    kind?: string;
    vox_id?: number;
    facing?: string;
    center: [number, number, number];
    dims: [number, number, number];
  }[];
}

// Twin frame: X length, Y depth, Z up. Dashboard floor plan: X width, Z depth, Y up.
async function buildDetectedScene(
  jobId: string,
): Promise<{ scene: SceneGraph; metrics: SceneAshraeMetrics }> {
  const [pm, th] = await Promise.all([
    fetch(`/api/v1/twin/${jobId}/artifact/placements.json`).then((r) => {
      if (!r.ok) throw new Error(`placements ${r.status}`);
      return r.json() as Promise<Manifest>;
    }),
    fetch(`/api/v1/twin/${jobId}/thermal`).then((r) => {
      if (!r.ok) throw new Error(`thermal ${r.status}`);
      return r.json() as Promise<ThermalResp>;
    }),
  ]);
  const rackInst = pm.instances.filter(
    (p) => p.kind === "rack" || (p.name ?? "").startsWith("server rack"),
  );
  // SVG y grows downward, so map twin Y -> (depth - Y) for a true top-down BEV (twin Y=0, the front
  // wall, at the BOTTOM) instead of a mirrored from-below view.
  const depthY = pm.ext[1];
  detRackFacing.clear();
  const furniture: Equipment[] = rackInst.map((p, i) => {
    detRackFacing.set(`rack_${i}`, p.facing ?? "PLUS_Y");
    return {
      id: `rack_${i}`,
      category: "server_rack",
      label: p.name ?? `Rack ${i + 1}`,
      position: [p.center[0], 0, depthY - p.center[1]] as [number, number, number],
      // Detected racks are axis-aligned; `size` already carries the true X/Y footprint, so the floor
      // plan must NOT re-rotate it by facing (that would draw the rack sideways).
      rotation: [0, 0, 0] as [number, number, number],
      size: [p.dims[0], p.dims[2], p.dims[1]] as [number, number, number],
      color: "#37474F",
      heatOutput: 5,
      relations: [],
    };
  });
  for (const [i, p] of pm.instances
    .filter((p) => (p.name ?? "").startsWith("ac_unit") || p.vox_id === 3)
    .entries()) {
    furniture.push({
      id: `ac_${i}`,
      category: "cooling_unit",
      label: "AC",
      position: [p.center[0], 0, depthY - p.center[1]],
      rotation: [0, 0, 0],
      size: [p.dims[0], p.dims[2], p.dims[1]],
      color: C.rackGray,
      heatOutput: 0,
      relations: [],
    });
  }
  // The network rack (ups) is not a thermal rack (no intake temp), so draw it as a neutral box so the
  // floor plan matches the voxel room's contiguous 7-rack front row instead of showing only 6.
  const net = pm.instances.find((p) => p.name === "network rack" || p.vox_id === 8);
  if (net) {
    furniture.push({
      id: "netrack",
      category: "network_rack",
      label: "NET",
      position: [net.center[0], 0, depthY - net.center[1]],
      rotation: [0, 0, 0],
      size: [net.dims[0], net.dims[2], net.dims[1]],
      color: "#7C6FD6",
      heatOutput: 0,
      relations: [],
    });
  }
  const scene: SceneGraph = {
    id: "detected",
    name: "스캔된 방 (현재 배치)",
    description: "Scanned room — current placement",
    room: { type: "scanned", dimensions: [pm.ext[0], pm.ext[2], pm.ext[1]], openings: [] },
    furniture,
    score: { total: 0, thermal: 0, cooling: 0, cable: 0, proximity: 0, constraint: 0 },
  };
  const perRack: RackMetrics[] = th.racks.map((r) => ({
    id: `rack_${r.rack_index}`,
    label: `Rack ${r.rack_index + 1}`,
    intakeTemp: r.intake_temp,
    exhaustTemp: r.exhaust_temp,
    deltaT: r.delta_t,
    ...zoneFor(r.intake_temp),
    heatOutput: 5,
  }));
  const intakes = perRack.map((r) => r.intakeTemp);
  const metrics: SceneAshraeMetrics = {
    sceneId: "detected",
    sceneName: scene.name,
    perRack,
    rciHi: th.room.rci_hi / 100,
    rciLo: th.room.rci_lo / 100,
    shi: th.room.shi,
    rhi: th.room.rhi,
    meanIntake: th.room.mean_intake,
    meanExhaust: th.room.mean_exhaust,
    meanDeltaT: perRack.length ? perRack.reduce((s, r) => s + r.deltaT, 0) / perRack.length : 0,
    peakIntake: intakes.length ? Math.max(...intakes) : AMBIENT_T,
    totalHeatKw: perRack.reduce((s, r) => s + r.heatOutput, 0),
    recommendedCount: perRack.filter((r) => r.zone === "recommended").length,
    allowableCount: perRack.filter((r) => r.status === "warn").length,
    violationCount: perRack.filter((r) => r.status === "danger").length,
  };
  return { scene, metrics };
}

async function showDetectedRoom(): Promise<void> {
  const out = document.getElementById("detected-thermal");
  const saved = loadSavedJob();
  if (!saved) {
    if (out) out.textContent = "스캔된 방이 없습니다 — 먼저 스캔을 완료하세요. (no finished scan)";
    return;
  }
  if (out) out.textContent = "스캔된 방의 열·ASHRAE 해석 중… (computing)";
  try {
    const { scene, metrics } = await buildDetectedScene(saved.jobId);
    // Every panel is now driven by the scanned room's real /thermal metrics (ASHRAE + heat). The time
    // series is a daily projection anchored to the measured peak; savings shows the real current cost.
    renderStatusBar(scene, metrics);
    renderKpis(scene, metrics);
    renderFloorPlan(scene, metrics);
    renderComplianceSummary(scene, metrics);
    renderTimeSeries(scene, metrics.peakIntake); // daily profile anchored to the room's real peak
    renderSavings(scene, metrics); // real current cooling cost / state (no fabricated comparison)
    if (out) {
      out.textContent =
        `스캔된 방 적용됨 — 권장 준수 ${metrics.recommendedCount}/${metrics.perRack.length} 랙 · ` +
        `RCI ${(metrics.rciHi * 100).toFixed(0)}% · 최고 흡기 ${metrics.peakIntake.toFixed(1)}°C`;
    }
  } catch (e) {
    if (out) out.textContent = `오류: ${e instanceof Error ? e.message : String(e)}`;
  }
}
document
  .getElementById("compute-thermal-btn")
  ?.addEventListener("click", () => void showDetectedRoom());

// ---------- Top-level render ----------
function render() {
  const scene = dashboardScenes[currentSceneIdx];
  const m = computeAshraeMetrics(scene, PEAK_LOAD);
  detRackFacing.clear(); // static demo scenes have no scanned facing -> no face bars
  renderScenePicker();
  renderStatusBar(scene, m);
  renderKpis(scene, m);
  renderFloorPlan(scene, m);
  renderTimeSeries(scene);
  renderComplianceSummary(scene, m);
  renderSavings(scene, m);
}

render();
