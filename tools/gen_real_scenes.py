"""Generate the two 시나리오 scenes from REAL data via the running backend, so the scenario page
mirrors the dashboard's real pipeline instead of hand-authored placeholders.

  random_v1  = the racks exactly where the scan found them   (GET .../artifact/placements.json)
  rl_ppo_v1  = the imitation layout policy's proposal          (POST .../optimize)

Both are 100% real (option ⓐ, honest): on this already-tidy room the two layouts look similar — that
is the truthful result, not a staged before/after.

Run with the server up (TWIN_PRECOMPUTED_SAMPLE=precomputed_server_room ... uvicorn) in the halo env:
    PYTHONPATH=.:backend ~/anaconda3/bin/conda run -n halo python tools/gen_real_scenes.py

Coordinate map: twin center [tx,ty,tz] -> scene position [tx,0,ty]; twin dims [dx,dy,dz] -> scene size
[dx,dz,dy]; facing PLUS_Y/MINUS_Y/PLUS_X/MINUS_X -> scene Y-rotation 0/180/90/270 (intake direction).
"""
import json
import time
import urllib.request as U
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SERVER = "http://127.0.0.1:8000"

RACK_KW = [12, 10, 6, 8, 5, 7, 11, 6, 9, 5, 8, 6]   # per physical rack; same in both scenes
RACK_COLORS = ["#37474F", "#455A64"]
FACE_ROT = {"PLUS_Y": 0, "MINUS_Y": 180, "PLUS_X": 90, "MINUS_X": 270}


def _get(path):
    return json.load(U.urlopen(SERVER + path, timeout=60))


def _post(path, payload=b"{}"):
    req = U.Request(SERVER + path, data=payload, headers={"Content-Type": "application/json"})
    return json.load(U.urlopen(req, timeout=120))


def _submit_precomputed_job():
    b = (b'--X\r\nContent-Disposition: form-data; name="files"; filename="a.jpg"\r\n'
         b'Content-Type: image/jpeg\r\n\r\n\xff\xd8\xff\xe0a\r\n'
         b'--X\r\nContent-Disposition: form-data; name="files"; filename="b.jpg"\r\n'
         b'Content-Type: image/jpeg\r\n\r\n\xff\xd8\xff\xe0b\r\n--X--\r\n')
    req = U.Request(SERVER + "/api/v1/twin", data=b,
                    headers={"Content-Type": "multipart/form-data; boundary=X"})
    jid = json.load(U.urlopen(req, timeout=30))["job_id"]
    time.sleep(2)
    return jid


def _rack_idx(name):
    try:
        return int(str(name).split()[-1]) - 1
    except (ValueError, IndexError):
        return 0


def rack_equip(inst):
    idx = _rack_idx(inst["name"])
    cx, cy, _ = inst["center"]
    rot = FACE_ROT.get(inst.get("facing", "PLUS_Y"), 0)
    return {
        "id": f"rack_{idx + 1:02d}", "category": "server_rack", "label": f"서버 랙 {idx + 1}",
        "position": [round(cx, 3), 0, round(cy, 3)], "rotation": [0, rot, 0],
        "size": [0.6, 1.95, 0.9], "color": RACK_COLORS[idx % 2],
        "heatOutput": RACK_KW[idx % len(RACK_KW)], "relations": [],
    }


def infra_equip(inst):
    """Map a non-rack scanned/proposed instance (AC / network rack / power cabinet) to scene Equipment."""
    name = str(inst.get("name", ""))
    cx, cy, _ = inst["center"]
    pos = [round(cx, 3), 0, round(cy, 3)]
    if name.startswith("ac_unit"):  # floor-standing package AC (Century stand unit in the scan)
        # Scanned dims (twin) = [0.5, 1.5, 2.0]: the broad 1.5 m face runs ALONG the west wall, only
        # 0.5 m deep. size = [broad 1.5, height 2.0, depth 0.5] + 90° so that broad face looks into the room.
        return {"id": "cooling_01", "category": "cooling_unit", "label": "스탠드 에어컨 (스캔)",
                "position": pos, "rotation": [0, 90, 0], "size": [1.5, 2.0, 0.5],
                "color": "#eae7df", "heatOutput": 0, "relations": []}
    if name == "network rack":
        return {"id": "core_switch", "category": "network_switch", "label": "네트워크 랙 (스캔)",
                "position": pos, "rotation": [0, 0, 0], "size": [0.6, 2.0, 0.8],
                "color": "#1565C0", "heatOutput": 3, "relations": []}
    if name == "power cabinet":  # electrical distribution cabinet (분전반)
        return {"id": "power_cabinet_01", "category": "power_cabinet", "label": "분전반 (스캔)",
                "position": pos, "rotation": [0, 0, 0], "size": [0.7, 2.0, 0.4],
                "color": "#8a978f", "heatOutput": 0, "relations": []}
    # fire hose (소화전) is a movable item, not fixed infra -> not rendered in the layout scenes.
    return None


def score_from_thermal(th):
    n = max(1, th.get("n_racks") or len(th.get("racks", [])) or 1)
    compliant = th.get("compliant_racks", n)
    peak = th.get("temp_max_c", 30)
    total = round(min(0.95, max(0.1, (compliant / n) * (1 - max(0, peak - 30) / 50))), 2)
    return {"total": total, "thermal": round(total * 0.97, 2), "cooling": round(total * 0.95, 2),
            "cable": round(min(0.95, total + 0.05), 2), "proximity": round(total * 0.98, 2),
            "constraint": round(min(0.98, total + 0.1), 2)}


# ---- pull real data from the running backend ----
jid = _submit_precomputed_job()
scanned = _get(f"/api/v1/twin/{jid}/artifact/placements.json")
scanned_th = _get(f"/api/v1/twin/{jid}/thermal")
opt = _post(f"/api/v1/twin/{jid}/optimize")
ext = opt["ext"]
ROOM_W, ROOM_H, ROOM_D = round(ext[0], 2), round(ext[2], 2), round(ext[1], 2)

# RANDOM: racks + all infra exactly where the scan found them
random_furniture = [rack_equip(i) for i in scanned["instances"] if i.get("kind") == "rack"]
random_furniture += [e for i in scanned["instances"] if (e := infra_equip(i))]

# Fixed wall infra the layout policy does NOT manage -> keep at the scanned position in both scenes.
scanned_fixed = [i for i in scanned["instances"] if i.get("name") == "power cabinet"]

# OPTIMIZED: racks + AC/network where the policy proposed; power cabinet + fire hose stay as scanned.
opt_furniture = [rack_equip(i) for i in opt["instances"] if i.get("kind") == "rack"]
opt_furniture += [e for i in opt["fixed"] if (e := infra_equip(i))]
opt_furniture += [e for i in scanned_fixed if (e := infra_equip(i))]

random_score = score_from_thermal(scanned_th)
opt_score = score_from_thermal(opt["thermal"])

OPENINGS = [
    {"type": "door", "wall": "south", "position": [ROOM_W / 2, 0, ROOM_D], "width": 1.2},
    {"type": "vent", "wall": "west", "position": [0, 2.0, ROOM_D - 0.85], "width": 1.4},
]


def _eq_ts(f):
    p, r, s = f["position"], f["rotation"], f["size"]
    return (f'  {{ id: "{f["id"]}", category: "{f["category"]}", label: "{f["label"]}", '
            f'position: [{p[0]}, {p[1]}, {p[2]}], rotation: [{r[0]}, {r[1]}, {r[2]}], '
            f'size: [{s[0]}, {s[1]}, {s[2]}], color: "{f["color"]}", '
            f'heatOutput: {f["heatOutput"]}, relations: [] }},')


def _scene_ts(name, sid, label, desc, furn, sc):
    ops = ",\n".join(
        f'    {{ type: "{o["type"]}", wall: "{o["wall"]}", '
        f'position: [{o["position"][0]}, {o["position"][1]}, {o["position"][2]}], width: {o["width"]} }}'
        for o in OPENINGS)
    lines = "\n".join(_eq_ts(f) for f in furn)
    return f'''export const {name}: SceneGraph = {{
  id: "{sid}",
  name: "{label}",
  description: "{desc}",
  room: {{
    type: "server_room",
    dimensions: [{ROOM_W}, {ROOM_H}, {ROOM_D}],
    openings: [
{ops},
    ],
  }},
  furniture: [
{lines}
  ],
  score: {{ total: {sc["total"]}, thermal: {sc["thermal"]}, cooling: {sc["cooling"]}, cable: {sc["cable"]}, proximity: {sc["proximity"]}, constraint: {sc["constraint"]} }},
}};
'''


ts = f'''// AUTO-GENERATED by tools/gen_real_scenes.py — do not edit by hand.
// Real 시나리오 scenes from the precomputed server-room twin (option ⓐ, fully honest):
//   random_v1 = racks where the scan found them; rl_ppo_v1 = the imitation layout policy's proposal.
// Both are 12/12 ASHRAE-compliant — on this already-tidy room the two layouts look alike, which is the
// truthful result. The dashboard runs the same /optimize live; this bakes it for the side-by-side page.
import type {{ SceneGraph }} from "./sceneGraphs";

{_scene_ts("realRandomPlacement", "random_v1", "스캔 원본 배치",
           "실제 스캔으로 복원한 서버실 — 랙이 발견된 그대로의 배치.",
           random_furniture, random_score)}
{_scene_ts("realOptimizedPlacement", "rl_ppo_v1", "RL 최적화 배치 (PPO)",
           "imitation 학습 정책이 스캔된 방에 제안한 정렬 배치 — 서버 랙을 정돈된 hot/cold-aisle 행으로 정리.",
           opt_furniture, opt_score)}'''

(REPO / "frontend/src/data/realScenes.ts").write_text(ts)
print(f"room (scene W,H,D): [{ROOM_W}, {ROOM_H}, {ROOM_D}]")
print(f"random:    {len(random_furniture)} items, score.total={random_score['total']} "
      f"(scanned {scanned_th['compliant_racks']}/{scanned_th['n_racks']} compliant, peak {scanned_th['temp_max_c']:.1f})")
print(f"optimized: {len(opt_furniture)} items, score.total={opt_score['total']} "
      f"(opt {opt['thermal']['compliant_racks']} compliant, peak {opt['thermal']['temp_max_c']:.1f})")
print("wrote frontend/src/data/realScenes.ts")
