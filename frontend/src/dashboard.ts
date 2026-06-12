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
import { latestScan, listScans, loadSavedJob } from "./lib/twinJob";

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
  if ((scene.id === "detected" || scene.id === "optimized") && m) {
    // Cooling electrical power from the ACTUAL thermal solve. The cooling LOAD equals the IT heat (energy
    // balance), and the chiller's effective coefficient of performance (COP) degrades with recirculation --
    // the Supply Heat Index (SHI), the share of exhaust heat that leaks back into the intakes. Better
    // containment lowers SHI, which raises the COP and lowers both cooling power and PUE. SHI is used rather
    // than the ASHRAE compliance fraction (RCI): RCI saturates at 100 for any compliant layout, so it
    // cannot distinguish two compliant layouts -- which is why cooling power and PUE did not move with
    // optimization even though the optimized layout has markedly lower recirculation.
    const cop = 6.0 * Math.max(0.3, 1 - (m.shi ?? 0));
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
  // show its real current cooling cost and ASHRAE state.
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
// The static scenes above are the fallback. This builds a SceneGraph + REAL ASHRAE metrics from the
// scanned twin -- its layout from placements.json, its thermal/ASHRAE from the 3-D engine
// (GET /api/v1/twin/{id}/thermal) -- and renders it through the same panels. The optimised layout is
// built separately via the optimize action.
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
  compliant_racks: number;
  temp_max_c: number;
  racks: { rack_index: number; intake_temp: number; exhaust_temp: number; delta_t: number }[];
  room: {
    rci_hi: number;
    rci_lo: number;
    shi: number;
    rhi: number;
    rti: number;
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

// The dashboard shows a SAVED completed scan (from the registry), not the in-progress scan job, so a brand
// new scan can run on the scan page while a previously scanned room is viewed and edited here. The picker
// selects which saved room is shown; it defaults to the most recent, falling back to the legacy single job.
let currentScanJob: string | null = latestScan()?.jobId ?? loadSavedJob()?.jobId ?? null;

interface ServerRoom {
  id: string;
  n_racks: number;
  precomputed: boolean;
}

// Populate the room picker from BOTH the browser's completed-scan registry AND the rooms that actually
// exist on the server (GET /twin/runs). The browser registry is empty on a fresh browser or a teammate's
// machine, so without the server fallback the picker is blank and every panel reports "no finished scan" --
// which is exactly the "I don't see results" failure. Merging guarantees a selectable room, then the
// detected room is rendered immediately so the dashboard opens on the real scan.
async function populateScanPicker(): Promise<void> {
  const sel = document.getElementById("scan-select") as HTMLSelectElement | null;
  const editBtn = document.getElementById("open-editor-btn") as HTMLButtonElement | null;
  if (!sel) return;

  const local = listScans().map((s) => ({
    id: s.jobId,
    label: `${escapeHtml(s.name)}${s.kind === "images" ? " (멀티뷰)" : ""}`,
  }));
  let server: { id: string; label: string }[] = [];
  let precomputedOnly = false;
  try {
    const r = await fetch("/api/v1/twin/runs").then((x) => (x.ok ? x.json() : { rooms: [] }));
    const rooms = (r.rooms ?? []) as ServerRoom[];
    const pre = rooms.filter((rm) => rm.precomputed);
    precomputedOnly = pre.length > 0; // when a precomputed sample exists, show only it
    server = (precomputedOnly ? pre : rooms).map((rm) => ({
      id: rm.id,
      label: `${rm.id.slice(0, 8)} · ${rm.n_racks}랙${rm.precomputed ? " (기본)" : ""}`,
    }));
  } catch {
    /* offline or no backend: fall back to the local registry only */
  }

  // When a precomputed sample exists it is the only selectable room;
  // otherwise merge the browser's local scans with the server runs.
  const seen = new Set<string>();
  const merged: { id: string; label: string }[] = [];
  for (const o of precomputedOnly ? server : [...local, ...server]) {
    if (!seen.has(o.id)) {
      seen.add(o.id);
      merged.push(o);
    }
  }
  if (merged.length === 0) {
    const j = loadSavedJob();
    if (j) merged.push({ id: j.jobId, label: "스캔된 방" });
  }
  if (merged.length === 0) {
    sel.style.display = "none";
    if (editBtn) editBtn.disabled = true;
    return;
  }
  sel.style.display = "";
  if (editBtn) editBtn.disabled = false;
  sel.innerHTML = merged.map((o) => `<option value="${o.id}">${o.label}</option>`).join("");
  if (!currentScanJob || !merged.some((o) => o.id === currentScanJob)) {
    currentScanJob = merged[0].id;
  }
  sel.value = currentScanJob;
  void showDetectedRoom(); // render the selected room immediately so results are visible on load
}

// ---------- 시나리오 비교 선택기: pick two (room, variant) views and jump to the side-by-side page ----------
// Each view is a room rendered as either 원본 (as-scanned, GET placements) or RL 최적화 (POST /optimize).
// The button hands the pair to index.html via ?lj/lv/rj/rv, which builds both panels with lib/twinScene.
async function populateCompareChooser(): Promise<void> {
  const left = document.getElementById("cmp-left") as HTMLSelectElement | null;
  const right = document.getElementById("cmp-right") as HTMLSelectElement | null;
  const btn = document.getElementById("compare-scenario-btn") as HTMLButtonElement | null;
  if (!left || !right || !btn) return;

  let rooms: ServerRoom[] = [];
  try {
    const r = await fetch("/api/v1/twin/runs").then((x) => (x.ok ? x.json() : { rooms: [] }));
    rooms = (r.rooms ?? []) as ServerRoom[];
  } catch {
    /* no backend */
  }
  const pre = rooms.filter((rm) => rm.precomputed);
  const pool = pre.length > 0 ? pre : rooms; // restrict to the precomputed run when present
  if (pool.length === 0) {
    btn.disabled = true;
    return;
  }
  const opts = pool.flatMap((rm) => [
    { value: `${rm.id}~scanned`, label: `${rm.id.slice(0, 8)} · 원본` },
    { value: `${rm.id}~optimized`, label: `${rm.id.slice(0, 8)} · RL 최적화` },
  ]);
  const html = opts
    .map((o) => `<option value="${o.value}">${escapeHtml(o.label)}</option>`)
    .join("");
  left.innerHTML = html;
  right.innerHTML = html;
  left.value = opts[0].value; // default: as-scanned ...
  right.value = opts[1]?.value ?? opts[0].value; // ... vs its RL-optimised proposal
  btn.disabled = false;

  btn.addEventListener("click", () => {
    const [lj, lv] = left.value.split("~");
    const [rj, rv] = right.value.split("~");
    const p = new URLSearchParams({ lj, lv, rj, rv });
    window.location.href = `./index.html?${p.toString()}`;
  });
}

document.getElementById("scan-select")?.addEventListener("change", (e) => {
  currentScanJob = (e.target as HTMLSelectElement).value;
  void showDetectedRoom();
});

// Open the SELECTED room in the editor in a NEW TAB, so editing one room and scanning a new one happen at
// the same time (the editor reads that job's placements.json and posts restamps to that job alone).
document.getElementById("open-editor-btn")?.addEventListener("click", () => {
  if (!currentScanJob) return;
  window.open(`./editor.html?job=${encodeURIComponent(currentScanJob)}`, "_blank");
});

async function showDetectedRoom(): Promise<void> {
  const out = document.getElementById("detected-thermal");
  if (!currentScanJob) {
    if (out) out.textContent = "스캔된 방이 없습니다 — 먼저 스캔을 완료하세요. (no finished scan)";
    return;
  }
  if (out) out.textContent = "스캔된 방의 열·ASHRAE 해석 중… (computing)";
  try {
    const { scene, metrics } = await buildDetectedScene(currentScanJob);
    // Every panel is now driven by the scanned room's real /thermal metrics (ASHRAE + heat). The time
    // series is a daily projection anchored to the measured peak; savings shows the real current cost.
    renderStatusBar(scene, metrics);
    renderKpis(scene, metrics);
    renderFloorPlan(scene, metrics);
    renderComplianceSummary(scene, metrics);
    // Drive the time series from the REAL per-hour steady-state solve (POST /temporal), not the synthetic
    // daily projection. That projection collapses to a flat line at ambient when the room's peak intake is
    // ~22 C (a well-cooled room), which is misleading on first load; the solver curve varies with the
    // outside-air cycle and shows the true result. Async (~seconds); leaves the panels rendered meanwhile.
    void runTemporalAnalysis();
    renderSavings(scene, metrics); // real current cooling cost / state
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

// ---------- RL layout optimization: run the trained policy on the scanned room ----------
// Posts the scanned job to POST /api/v1/twin/{id}/optimize, which feeds the twin to the MaskablePPO policy
// (engine/rl via twin_bridge) and returns the layout it proposes plus that layout's own 3-D thermal score.
// We render the proposed racks in the same BEV panels and a HONEST current-vs-RL comparison card.
interface OptimizeResp {
  n_racks: number;
  rack_num: number;
  total_energy: number;
  ext: [number, number, number];
  rack_type: string;
  instances: {
    name: string;
    facing: string;
    center: [number, number, number];
    dims: [number, number, number];
    intake_temp?: number;
    exhaust_temp?: number;
  }[];
  fixed: {
    name?: string;
    vox_id?: number;
    center: [number, number, number];
    dims: [number, number, number];
  }[];
  thermal: {
    compliant_racks: number;
    temp_max_c: number;
    racks: { rack_index: number; intake_temp: number; exhaust_temp: number; delta_t: number }[];
    room: {
      rci_hi: number;
      rci_lo: number;
      shi: number;
      rhi: number;
      rti: number;
      mean_intake: number;
      mean_exhaust: number;
    };
  } | null;
}

function buildOptimizedScene(r: OptimizeResp): { scene: SceneGraph; metrics: SceneAshraeMetrics } {
  const depthY = r.ext[1];
  detRackFacing.clear();
  const furniture: Equipment[] = r.instances.map((p, i) => {
    detRackFacing.set(`rack_${i}`, p.facing ?? "PLUS_Y");
    return {
      id: `rack_${i}`,
      category: "server_rack",
      label: p.name ?? `Rack ${i + 1}`,
      position: [p.center[0], 0, depthY - p.center[1]] as [number, number, number],
      rotation: [0, 0, 0] as [number, number, number],
      size: [p.dims[0], p.dims[2], p.dims[1]] as [number, number, number],
      color: "#37474F",
      heatOutput: 5,
      relations: [],
    };
  });
  for (const [i, p] of r.fixed
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
  const net = r.fixed.find((p) => p.name === "network rack" || p.vox_id === 8);
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
    id: "optimized",
    name: "RL 최적화 배치 (제안)",
    description: "RL-proposed layout",
    room: { type: "scanned", dimensions: [r.ext[0], r.ext[2], r.ext[1]], openings: [] },
    furniture,
    score: { total: 0, thermal: 0, cooling: 0, cable: 0, proximity: 0, constraint: 0 },
  };
  const th = r.thermal;
  const perRack: RackMetrics[] = (th?.racks ?? []).map((rk) => ({
    id: `rack_${rk.rack_index}`,
    label: `Rack ${rk.rack_index + 1}`,
    intakeTemp: rk.intake_temp,
    exhaustTemp: rk.exhaust_temp,
    deltaT: rk.delta_t,
    ...zoneFor(rk.intake_temp),
    heatOutput: 5,
  }));
  const intakes = perRack.map((r) => r.intakeTemp);
  const metrics: SceneAshraeMetrics = {
    sceneId: "optimized",
    sceneName: scene.name,
    perRack,
    rciHi: (th?.room.rci_hi ?? 100) / 100,
    rciLo: (th?.room.rci_lo ?? 100) / 100,
    shi: th?.room.shi ?? 0,
    rhi: th?.room.rhi ?? 0,
    meanIntake: th?.room.mean_intake ?? AMBIENT_T,
    meanExhaust: th?.room.mean_exhaust ?? AMBIENT_T,
    meanDeltaT: perRack.length ? perRack.reduce((s, r) => s + r.deltaT, 0) / perRack.length : 0,
    peakIntake: intakes.length ? Math.max(...intakes) : AMBIENT_T,
    totalHeatKw: perRack.reduce((s, r) => s + r.heatOutput, 0),
    recommendedCount: perRack.filter((r) => r.zone === "recommended").length,
    allowableCount: perRack.filter((r) => r.status === "warn").length,
    violationCount: perRack.filter((r) => r.status === "danger").length,
  };
  return { scene, metrics };
}

// Current-vs-RL comparison written into the savings card.
function renderOptimizeComparison(cur: ThermalResp, r: OptimizeResp): void {
  const host = document.getElementById("savings-card") as HTMLElement;
  host.classList.remove("baseline");
  const th = r.thermal;
  if (!th) {
    host.innerHTML = `<div class="savings-lbl">RL 제안 배치</div>
      <div class="savings-val" style="color:${C.text1};">${r.n_racks}개 랙</div>
      <div class="savings-sub">열 해석 없음 — 배치만 표시</div>`;
    return;
  }
  const curComp = cur.compliant_racks;
  const rlComp = th.compliant_racks;
  const curMaxIn = Math.max(...cur.racks.map((rk) => rk.intake_temp), Number.NEGATIVE_INFINITY);
  const rlMaxIn = Math.max(
    ...(th.racks ?? []).map((rk) => rk.intake_temp),
    Number.NEGATIVE_INFINITY,
  );
  // Compliance/RCI are usually saturated on a tidy room; SHI (cold-air isolation) and RTI (airflow
  // balance) are the metrics that actually move, so "improved" counts those too — not just rack count.
  const improved =
    rlComp > curComp ||
    th.room.shi < cur.room.shi - 0.01 ||
    Math.abs(th.room.rti - 100) < Math.abs(cur.room.rti - 100) - 1 ||
    (rlComp === curComp && th.temp_max_c < cur.temp_max_c - 0.05);
  const headline = improved ? "RL 최적화 — 개선됨" : "RL 최적화 — 현재 배치가 이미 우수";
  // one "before → after" row with a coloured delta; better(cur, rl) decides the colour.
  const mrow = (
    lbl: string,
    sub: string,
    c: number,
    rl: number,
    unit: string,
    dp: number,
    better: (a: number, b: number) => boolean,
  ): string => {
    const dlt = rl - c;
    const col =
      Math.abs(dlt) < 0.5 * 10 ** -dp ? C.text2 : better(c, rl) ? C.successText : C.dangerText;
    return `<div class="item"><div class="lbl">${lbl}<small style="color:${C.text3};"> ${sub}</small></div>
      <div class="v" style="font-size:13px;">${c.toFixed(dp)}${unit}
      <span style="color:${C.text3};">→</span>
      <span style="color:${col};">${rl.toFixed(dp)}${unit}</span>
      <span style="color:${col};font-size:11px;"> (${dlt >= 0 ? "+" : ""}${dlt.toFixed(dp)})</span></div></div>`;
  };
  host.innerHTML = `
    <div class="savings-lbl">${headline}</div>
    <div class="savings-val" style="color:${improved ? C.successText : C.text1};font-size:18px;">
      준수 ${curComp}/${r.n_racks} → ${rlComp}/${r.n_racks}</div>
    <div class="savings-sub">현재 스캔 배치 대비 · RL 정책 추론 결과 (변화량 표시)</div>
    <hr/>
    <div class="savings-grid">
      ${mrow("RCI", "高 ↑", cur.room.rci_hi, th.room.rci_hi, "%", 0, (a, b) => b >= a)}
      ${mrow("SHI", "격리 ↓", cur.room.shi, th.room.shi, "", 2, (a, b) => b <= a)}
      ${mrow("RTI", "100% 이상적", cur.room.rti, th.room.rti, "%", 0, (a, b) => Math.abs(b - 100) <= Math.abs(a - 100))}
      ${mrow("최고 흡기", "↓", curMaxIn, rlMaxIn, "°C", 1, (a, b) => b <= a)}
      ${mrow("평균 흡기", "↓", cur.room.mean_intake, th.room.mean_intake, "°C", 1, (a, b) => b <= a)}
    </div>`;
}

async function showOptimizedRoom(): Promise<void> {
  const out = document.getElementById("detected-thermal");
  if (!currentScanJob) {
    if (out) out.textContent = "스캔된 방이 없습니다 — 먼저 스캔을 완료하세요. (no finished scan)";
    return;
  }
  const jobId = currentScanJob;
  if (out) out.textContent = "RL 정책으로 배치 최적화 추론 중… (running RL inference)";
  try {
    // Run the policy AND fetch the current scanned layout's thermal in parallel so we can show before/after.
    const [r, cur] = await Promise.all([
      fetch(`/api/v1/twin/${jobId}/optimize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: "{}",
      }).then((x) => {
        if (!x.ok) throw new Error(`optimize ${x.status}`);
        return x.json() as Promise<OptimizeResp>;
      }),
      fetch(`/api/v1/twin/${jobId}/thermal`)
        .then((x) => (x.ok ? (x.json() as Promise<ThermalResp>) : null))
        .catch(() => null),
    ]);
    const { scene, metrics } = buildOptimizedScene(r);
    renderStatusBar(scene, metrics);
    renderKpis(scene, metrics);
    renderFloorPlan(scene, metrics);
    renderComplianceSummary(scene, metrics);
    renderTimeSeries(scene, metrics.peakIntake);
    if (cur) renderOptimizeComparison(cur, r);
    else renderSavings(scene, metrics);
    if (out) {
      const c = r.thermal
        ? `준수 ${r.thermal.compliant_racks}/${r.n_racks} · RCI ${r.thermal.room.rci_hi.toFixed(0)}% · 최고흡기 ${metrics.peakIntake.toFixed(1)}°C`
        : `${r.n_racks}개 랙 배치`;
      out.textContent = `RL 최적화 추론 완료 — ${r.n_racks}개 랙 재배치 · ${c}`;
    }
  } catch (e) {
    if (out) out.textContent = `오류: ${e instanceof Error ? e.message : String(e)}`;
  }
}
document
  .getElementById("rl-optimize-btn")
  ?.addEventListener("click", () => void showOptimizedRoom());

// ---------- Temporal (per-time-t) ASHRAE analysis ----------
// Real steady-state physics, not a projection: POST /api/v1/twin/{id}/temporal runs the 3-D solver once per
// discrete hour with the room default set to the OUTSIDE temperature at t, the AC fixed at 18 °C, and rack
// power following the load at t; it returns the converged per-t metrics, which we plot over the day.
interface TemporalSample {
  t: number;
  outside_c: number;
  load: number;
  rack_kw: number;
  mean_intake: number;
  max_intake: number;
  mean_exhaust: number;
  max_temp: number;
  compliant_racks: number;
  allowable_racks: number;
  rci_hi: number;
  mean_delta_t: number;
}
interface TemporalResp {
  n_racks: number;
  ac_supply_c: number;
  base_rack_kw: number;
  n_samples: number;
  series: TemporalSample[];
}

function renderTemporalSeries(resp: TemporalResp): void {
  const host = document.getElementById("ts-chart") as HTMLElement;
  if (!host || resp.series.length === 0) return;
  if (!tsChart) {
    tsChart = echarts.init(host, undefined, { renderer: "svg" });
    window.addEventListener("resize", () => tsChart?.resize());
  }
  const s = resp.series;
  const n = resp.n_racks;
  const hours = s.map((x) => x.t.toFixed(1));
  const fontFamily =
    '-apple-system, BlinkMacSystemFont, "Segoe UI", "Pretendard", system-ui, sans-serif';
  const tMax = Math.ceil(Math.max(...s.map((x) => x.max_temp)) + 2);
  const tMin = Math.floor(Math.min(...s.map((x) => Math.min(x.max_intake, x.outside_c))) - 2);
  // Per-t compliance colour: green = all racks within recommended, amber = within allowable, red = violated.
  const statusColor = (x: TemporalSample) =>
    x.compliant_racks === n ? C.greenDeep : x.allowable_racks === n ? C.amber : C.red;
  const compScatter = s.map((x, i) => ({
    value: [hours[i], x.max_intake] as [string, number],
    itemStyle: { color: statusColor(x) },
  }));

  tsChart.setOption(
    {
      animation: false,
      backgroundColor: "transparent",
      textStyle: { fontFamily, color: C.text2 },
      grid: { left: 38, right: 40, top: 28, bottom: 32, containLabel: false },
      tooltip: {
        trigger: "axis",
        backgroundColor: C.bgCard,
        borderColor: C.border,
        borderWidth: 0.5,
        padding: [6, 10],
        textStyle: { color: C.text1, fontSize: 11, fontFamily },
        formatter: (params: unknown) => {
          const arr = Array.isArray(params) ? params : [params];
          const i = (arr[0] as { dataIndex: number }).dataIndex;
          const x = s[i];
          const tag =
            x.compliant_racks === n ? "권장 준수" : x.allowable_racks === n ? "허용 범위" : "위반";
          return (
            `${x.t.toFixed(1)}시<br/>외기 ${x.outside_c.toFixed(1)}°C · 부하 ${(x.load * 100).toFixed(0)}% (${x.rack_kw.toFixed(1)}kW)<br/>` +
            `흡입 최고 ${x.max_intake.toFixed(1)}°C · 최고온도 ${x.max_temp.toFixed(1)}°C<br/>` +
            `준수 ${x.compliant_racks}/${n} · 허용 ${x.allowable_racks}/${n} · RCI ${x.rci_hi.toFixed(0)}% · ${tag}`
          );
        },
      },
      legend: {
        data: ["부하", "외기", "흡입 최고", "최고 온도"],
        right: 8,
        top: 4,
        textStyle: { color: C.text2, fontSize: 10, fontFamily },
        itemWidth: 14,
        itemHeight: 2,
        itemGap: 10,
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
          formatter: (val: string) => `${parseFloat(val).toFixed(0)}시`,
        },
      },
      yAxis: [
        {
          type: "value",
          name: "°C",
          nameTextStyle: { color: C.text2, fontSize: 9, fontFamily, padding: [0, 0, 4, 0] },
          min: tMin,
          max: tMax,
          axisLine: { show: false },
          axisTick: { show: false },
          splitLine: { lineStyle: { color: C.border, width: 0.5, type: "dashed" } },
          axisLabel: { color: C.text2, fontSize: 8, fontFamily },
        },
        {
          type: "value",
          name: "외기 °C",
          nameTextStyle: { color: C.warnText, fontSize: 9, fontFamily, padding: [0, 0, 4, 0] },
          min: tMin,
          max: tMax,
          axisLine: { show: false },
          axisTick: { show: false },
          splitLine: { show: false },
          axisLabel: { color: C.warnText, fontSize: 8, fontFamily },
        },
        { type: "value", show: false, min: 0, max: 1 },
      ],
      series: [
        {
          name: "부하",
          type: "bar",
          yAxisIndex: 2,
          data: s.map((x) => x.load),
          barCategoryGap: "55%",
          itemStyle: { color: C.blueCool, opacity: 0.15 },
          z: 1,
        },
        {
          name: "외기",
          type: "line",
          yAxisIndex: 1,
          data: s.map((x) => x.outside_c),
          showSymbol: false,
          smooth: 0.2,
          lineStyle: { color: C.amber, width: 1.2, opacity: 0.85 },
          z: 2,
        },
        {
          name: "최고 온도",
          type: "line",
          yAxisIndex: 0,
          data: s.map((x) => x.max_temp),
          showSymbol: false,
          smooth: 0.2,
          lineStyle: { color: C.red, width: 1, type: "dashed", opacity: 0.65 },
          z: 2,
        },
        {
          name: "흡입 최고",
          type: "line",
          yAxisIndex: 0,
          data: s.map((x) => x.max_intake),
          showSymbol: false,
          smooth: 0.2,
          lineStyle: { color: C.greenDeep, width: 1.6 },
          markLine: {
            silent: true,
            symbol: "none",
            lineStyle: { color: C.red, type: "dashed", width: 0.6, opacity: 0.6 },
            data: [
              {
                yAxis: T_MAX_REC,
                label: {
                  formatter: `권장 상한 ${T_MAX_REC}°C`,
                  color: C.dangerText,
                  fontSize: 7,
                  fontFamily,
                  position: "insideEndTop",
                },
              },
              {
                yAxis: 18,
                label: {
                  formatter: "권장 하한 18°C",
                  color: C.infoText,
                  fontSize: 7,
                  fontFamily,
                  position: "insideEndBottom",
                },
              },
            ],
          },
          z: 3,
        },
        {
          name: "준수상태",
          type: "scatter",
          yAxisIndex: 0,
          data: compScatter,
          symbolSize: 7,
          tooltip: { show: false },
          z: 4,
        },
      ],
    },
    true,
  );
  tsChart.resize();
}

async function runTemporalAnalysis(): Promise<void> {
  const out = document.getElementById("detected-thermal");
  if (!currentScanJob) {
    if (out) out.textContent = "스캔된 방이 없습니다 — 먼저 스캔을 완료하세요. (no finished scan)";
    return;
  }
  if (out) out.textContent = "시간대별 정상상태 열 해석 중… 각 시각마다 3D 솔버 수렴 (수 초 소요)";
  try {
    const resp = await fetch(`/api/v1/twin/${currentScanJob}/temporal`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    }).then((x) => {
      if (!x.ok) throw new Error(`temporal ${x.status}`);
      return x.json() as Promise<TemporalResp>;
    });
    renderTemporalSeries(resp);
    const s = resp.series;
    const okHours = s
      .filter((x) => x.compliant_racks === resp.n_racks)
      .map((x) => `${x.t.toFixed(0)}시`);
    const worst = s.reduce((a, b) => (b.rci_hi < a.rci_hi ? b : a), s[0]);
    if (out) {
      out.textContent =
        `시간대별 분석 완료 — AC ${resp.ac_supply_c}°C 고정 · ${resp.n_samples}개 시각 정상상태 해석 · ` +
        `권장 준수 시간대 ${okHours.length ? okHours.join(", ") : "없음"} · ` +
        `최악 ${worst.t.toFixed(0)}시 (외기 ${worst.outside_c.toFixed(0)}°C, RCI ${worst.rci_hi.toFixed(0)}%)`;
    }
  } catch (e) {
    if (out) out.textContent = `오류: ${e instanceof Error ? e.message : String(e)}`;
  }
}
document
  .getElementById("temporal-btn")
  ?.addEventListener("click", () => void runTemporalAnalysis());

// ---------- Containment comparison: hot/cold aisle vs non-contained baseline ----------
// POST /api/v1/twin/{id}/compare scores the room twice -- the contained hot/cold-aisle layout and the
// same racks all facing one way (recirculation) -- so the delta is the cooling benefit of containment.
interface CompareMetrics {
  compliant: number;
  rci_hi: number;
  rci_lo: number;
  shi: number;
  rhi: number;
  mean_intake: number;
  max_intake: number;
  mean_exhaust: number;
  max_temp: number;
  mean_delta_t: number;
}
interface CompareResp {
  n_racks: number;
  designed: CompareMetrics;
  baseline: CompareMetrics;
}

function renderContainmentComparison(r: CompareResp): void {
  const host = document.getElementById("savings-card") as HTMLElement;
  host.classList.remove("baseline");
  const n = r.n_racks;
  const d = r.designed;
  const b = r.baseline;
  // Each row: label, designed value, baseline value, and whether the design is better (green) or not.
  const row = (lbl: string, dv: string, bv: string, better: boolean | null) => {
    const col = better === null ? C.text2 : better ? C.successText : C.dangerText;
    return `<div class="item"><div class="lbl">${lbl}</div>
      <div class="v" style="font-size:13px;"><span style="color:${col};font-weight:600;">${dv}</span>
      <span style="color:${C.text3};">vs</span> ${bv}</div></div>`;
  };
  host.innerHTML = `
    <div class="savings-lbl">냉각 격리 효과 (설계 vs 비격리 기준선)</div>
    <div class="savings-val" style="color:${C.successText};font-size:18px;">
      준수 ${d.compliant}/${n} <span style="color:${C.text3};font-size:13px;">vs</span> ${b.compliant}/${n}</div>
    <div class="savings-sub">동일 방·동일 AC·동일 랙 — 통로 격리만 차이 (핫/콜드 아일 vs 재순환)</div>
    <hr/>
    <div class="savings-grid">
      ${row("RCI (高)", `${d.rci_hi.toFixed(0)}%`, `${b.rci_hi.toFixed(0)}%`, d.rci_hi >= b.rci_hi)}
      ${row("평균 흡기", `${d.mean_intake.toFixed(1)}°C`, `${b.mean_intake.toFixed(1)}°C`, d.mean_intake <= b.mean_intake)}
      ${row("최고 흡기", `${d.max_intake.toFixed(1)}°C`, `${b.max_intake.toFixed(1)}°C`, d.max_intake <= b.max_intake)}
      ${row("SHI (低우수)", `${d.shi.toFixed(2)}`, `${b.shi.toFixed(2)}`, d.shi <= b.shi)}
    </div>`;
}

async function runContainmentComparison(): Promise<void> {
  const out = document.getElementById("detected-thermal");
  if (!currentScanJob) {
    if (out) out.textContent = "스캔된 방이 없습니다 — 먼저 스캔을 완료하세요. (no finished scan)";
    return;
  }
  if (out) out.textContent = "격리 효과 비교 계산 중… 설계/기준선 두 배치 정상상태 해석";
  try {
    const r = await fetch(`/api/v1/twin/${currentScanJob}/compare`, { method: "POST" }).then(
      (x) => {
        if (!x.ok) throw new Error(`compare ${x.status}`);
        return x.json() as Promise<CompareResp>;
      },
    );
    renderContainmentComparison(r);
    if (out) {
      out.textContent =
        `격리 효과 — 설계 준수 ${r.designed.compliant}/${r.n_racks} (RCI ${r.designed.rci_hi.toFixed(0)}%) · ` +
        `비격리 기준선 ${r.baseline.compliant}/${r.n_racks} (RCI ${r.baseline.rci_hi.toFixed(0)}%)`;
    }
  } catch (e) {
    if (out) out.textContent = `오류: ${e instanceof Error ? e.message : String(e)}`;
  }
}
document
  .getElementById("compare-btn")
  ?.addEventListener("click", () => void runContainmentComparison());

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
void populateScanPicker();
void populateCompareChooser();
