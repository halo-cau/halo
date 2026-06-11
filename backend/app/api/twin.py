"""Async digital-twin endpoint: ANY input (multi-view images OR one geometry scan) -> voxel twin.

The multi-view front is multi-minute and GPU-bound, so this uses a JOB model (submit -> poll -> fetch):
``POST /api/v1/twin`` enqueues a job and returns a ``job_id``; a single background worker serializes the
GPU pipeline (one at a time), running ``scripts/recon/pipeline_web.py`` in the ``halo`` env as a
subprocess; ``GET /api/v1/twin/{id}`` reports status; ``GET /api/v1/twin/{id}/artifact/{name}`` serves a
stage artifact (recon / labeled / voxel / placements). Decoupling the request from the long compute keeps
the API responsive and the single GPU safe. See engine/vision/twin.py for the shared pipeline.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import uuid
from pathlib import Path

from fastapi import APIRouter, Body, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

router = APIRouter()

REPO = Path(__file__).resolve().parents[3]
RUNS = Path(os.environ.get("TWIN_RUNS", REPO / "tools" / "recon_web" / "runs"))
RUNS.mkdir(parents=True, exist_ok=True)
PIPELINE = REPO / "scripts" / "recon" / "pipeline_web.py"
RESTAMP = REPO / "scripts" / "recon" / "restamp_room.py"
# The GPU pipeline (pi3 + SAM3) lives in the `halo` conda env; the API process need not. Override with
# the HALO_PY env var if the interpreter path differs.
HALO_PY = os.environ.get("HALO_PY", "/home/ppco915/ENTER/envs/halo/bin/python")
# The multi-view reconstruction (Pi3) + segmentation (SAM3) over ~1M points takes several minutes on the
# GPU. When TWIN_PRECOMPUTED_SAMPLE names a finished run, submitted jobs are served from its cached
# artifacts and flagged ``precomputed`` instead of re-running that pipeline; unset (the default) every job
# runs the full pipeline.
PRECOMPUTED_SAMPLE = os.environ.get("TWIN_PRECOMPUTED_SAMPLE", "").strip()

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".bmp", ".tif", ".tiff", ".webp"}
GEOMETRY_SUFFIXES = {".obj", ".ply", ".las", ".laz"}

_jobs: "queue.Queue[tuple[str, dict]]" = queue.Queue()


def _env() -> dict:
    return dict(os.environ, HF_HUB_OFFLINE="1",
                PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", PYOPENGL_PLATFORM="egl")


def _serve_precomputed(sample: Path, dst: Path) -> None:
    """Serve a finished run's cached artifacts for a job and flag it ``precomputed``.

    The job returns immediately instead of re-running the multi-minute Pi3 + SAM3 GPU reconstruction (see
    TWIN_PRECOMPUTED_SAMPLE). The recon / labeled / point_labels / voxel artifacts of the sample run are
    copied into the new job
    dir unchanged; the real pipeline is the default whenever no sample is configured.

    Ordering matters: copy every artifact EXCEPT status.json first, then write the ``done`` status LAST.
    The sample's own status.json already says ``state: done``; copying it in the loop (in arbitrary order,
    while ~70 MB of PLY files are still being copied) let a fast poller see ``done`` before voxel.ply
    existed, which surfaced as a 404 on the artifact fetch the instant the scan page reached the done state.
    """
    import shutil
    for f in sample.iterdir():
        if f.is_file() and f.name != "status.json":
            shutil.copy2(f, dst / f.name)
    src_status = sample / "status.json"
    st = json.loads(src_status.read_text()) if src_status.exists() else {}
    st.update({"state": "done", "step": "done", "pct": 100, "precomputed": True,
               "message": "reconstruction complete (Pi3 + SAM3)"})
    (dst / "status.json").write_text(json.dumps(st))


def _worker() -> None:
    """Serialize jobs: run one pipeline subprocess (in halo) at a time so the single GPU never thrashes."""
    while True:
        job_id, p = _jobs.get()
        d = RUNS / job_id
        if PRECOMPUTED_SAMPLE and (RUNS / PRECOMPUTED_SAMPLE).is_dir():
            try:                                               # serve cached artifacts for this run
                _serve_precomputed(RUNS / PRECOMPUTED_SAMPLE, d)
            except Exception as e:  # noqa: BLE001 — never let one bad job kill the daemon worker
                (d / "status.json").write_text(json.dumps(
                    {"state": "error", "step": "precomputed", "pct": 0,
                     "message": "precomputed serve failed", "error": str(e), "outputs": {}}))
            finally:
                _jobs.task_done()
            continue
        if p["kind"] == "images":
            # Multi-view front: Pi3 reconstruction + SAM3 per-instance segmentation, then the shared
            # voxelizer recovers the metric room from the CV alone (per-axis scale from the rack cuboid +
            # the detected walls). No room dimensions are passed.
            cmd = [HALO_PY, str(PIPELINE), "--images", str(d / "input"), "--out", str(d),
                   "--model", p.get("model", "pi3"),
                   "--pi3-frames", "32", "--pi3-pixel-limit", "245000", "--rack-instancing", "sam3"]
        else:
            cmd = [HALO_PY, str(PIPELINE), "--scan", str(p["scan"]), "--out", str(d)]
        try:
            with (d / "driver.log").open("w") as f:
                subprocess.run(cmd, cwd=str(REPO), env=_env(), stdout=f, stderr=subprocess.STDOUT)
        finally:
            sj = d / "status.json"
            st = json.loads(sj.read_text()) if sj.exists() else {}
            if st.get("state") not in ("done", "error"):           # crashed without a terminal state
                tail = ("\n".join((d / "driver.log").read_text().splitlines()[-15:])
                        if (d / "driver.log").exists() else "")
                sj.write_text(json.dumps({"state": "error", "step": st.get("step", "?"),
                                          "pct": st.get("pct", 0), "message": "pipeline crashed",
                                          "error": tail or "no output", "outputs": {}}))
            _jobs.task_done()


threading.Thread(target=_worker, daemon=True).start()


@router.post("/twin")
async def create_twin(files: list[UploadFile] = File(...)) -> dict:
    """Accept multi-view images (>= 2) OR one geometry scan; enqueue a twin job; return its id + kind."""
    sfx = [Path(f.filename or "").suffix.lower() for f in files]
    imgs = [f for f, s in zip(files, sfx) if s in IMAGE_SUFFIXES]
    geo = [f for f, s in zip(files, sfx) if s in GEOMETRY_SUFFIXES]
    if len(imgs) >= 2 and not geo:
        kind = "images"
    elif len(geo) == 1 and not imgs:
        kind = "geometry"
    else:
        raise HTTPException(400, "provide >= 2 images (multi-view) OR exactly one .obj/.ply/.las/.laz")

    job_id = uuid.uuid4().hex[:12]
    d = RUNS / job_id
    (d / "input").mkdir(parents=True, exist_ok=True)
    p: dict = {"kind": kind}
    if kind == "images":
        for f in imgs:
            (d / "input" / Path(f.filename).name).write_bytes(await f.read())
    else:
        scan = d / ("scan" + Path(geo[0].filename).suffix.lower())
        scan.write_bytes(await geo[0].read())
        p["scan"] = str(scan)
    (d / "status.json").write_text(json.dumps(
        {"state": "queued", "step": "queued", "pct": 0, "message": "queued", "outputs": {}, "kind": kind}))
    _jobs.put((job_id, p))
    return {"job_id": job_id, "kind": kind}


@router.get("/twin/precomputed")
def twin_precomputed() -> JSONResponse:
    """Report the cached twin sample available to serve in place of a live pipeline run.

    Returns the configured sample's id + a summary, or ``enabled: false`` in the normal case where none
    is set and every submitted job runs the full Pi3 + SAM3 pipeline. (Declared before /twin/{job_id} so
    the literal path is not captured as a job id.)"""
    samples = []
    if PRECOMPUTED_SAMPLE and (RUNS / PRECOMPUTED_SAMPLE / "status.json").exists():
        st = json.loads((RUNS / PRECOMPUTED_SAMPLE / "status.json").read_text())
        samples.append({"id": PRECOMPUTED_SAMPLE, "outputs": list(st.get("outputs", {}).keys()),
                        "label_counts": st.get("label_counts", {})})
    return JSONResponse({"enabled": bool(samples), "samples": samples})


@router.get("/twin/runs")
def twin_runs() -> JSONResponse:
    """List analysable rooms on the server (run dirs with voxel_grid.npy + placements.json), newest first.

    The dashboard's room picker is driven by browser localStorage, which is empty on a fresh browser or a
    teammate's machine; this lets it fall back to (or merge with) what actually exists on disk, so the
    thermal / compare / temporal / RL panels always have a room to act on. (Declared before /twin/{job_id}
    so the literal path is not captured as a job id.)"""
    rooms = []
    for d in RUNS.iterdir():
        if not d.is_dir() or not (d / "voxel_grid.npy").exists() or not (d / "placements.json").exists():
            continue
        try:
            m = json.loads((d / "placements.json").read_text())
        except Exception:  # noqa: BLE001 -- skip a run whose manifest is unreadable
            continue
        insts = m.get("instances", [])
        n_racks = sum(1 for p in insts if p.get("kind") == "rack")
        if n_racks == 0:
            continue
        rooms.append({
            "id": d.name,
            "n_racks": n_racks,
            "ext": [round(float(e), 1) for e in m.get("ext", [])],
            "precomputed": d.name == PRECOMPUTED_SAMPLE,
            "mtime": (d / "placements.json").stat().st_mtime,
        })
    rooms.sort(key=lambda r: r["mtime"], reverse=True)
    return JSONResponse({"rooms": rooms[:30]})


@router.get("/twin/{job_id}")
def twin_status(job_id: str) -> JSONResponse:
    """Poll a job: returns its status.json (state / step / pct / message / outputs)."""
    sj = RUNS / job_id / "status.json"
    if not sj.exists():
        raise HTTPException(404, "no such job")
    return JSONResponse(json.loads(sj.read_text()))


@router.get("/twin/{job_id}/artifact/{name}")
def twin_artifact(job_id: str, name: str) -> FileResponse:
    """Serve a stage artifact (recon.ply / labeled.ply / voxel.ply / placements.json / ...)."""
    if "/" in name or ".." in name:
        raise HTTPException(400, "bad artifact name")
    p = RUNS / job_id / name
    if not p.exists():
        raise HTTPException(404, "no such artifact")
    return FileResponse(str(p), headers={"Cache-Control": "no-cache, must-revalidate"})


@router.post("/twin/{job_id}/restamp")
def twin_restamp(job_id: str, payload: dict = Body(default={})) -> JSONResponse:
    """Re-voxelize a job from its (edited) placements.json. The per-instance editor sends the moved /
    removed instance list; we persist it, then rebuild voxel_grid.npy + voxel.ply via restamp_room.py
    in halo (subprocess), keeping the API process decoupled from the heavy CV deps. Shares
    apply_placements with the voxelizer, so an unedited manifest reproduces the grid bit-for-bit."""
    d = RUNS / job_id
    pj = d / "placements.json"
    if not pj.exists():
        raise HTTPException(404, "no placements.json for this job")
    if "instances" in payload:
        manifest = json.loads(pj.read_text())
        manifest["instances"] = payload["instances"]          # editor sends the full edited list
        pj.write_text(json.dumps(manifest, indent=2))
    r = subprocess.run([HALO_PY, str(RESTAMP), "--run", str(d)], cwd=str(REPO),
                       env=_env(), capture_output=True, text=True)
    if r.returncode != 0:
        raise HTTPException(500, f"restamp failed: {r.stderr[-800:]}")
    n = len(json.loads(pj.read_text()).get("instances", []))
    return JSONResponse({"ok": True, "n_instances": n, "log": r.stdout.strip()})


def _ac_supply_dir(center, dims, ext) -> tuple[float, float, float]:
    """Discharge direction for a floor-standing AC: horizontal, along its shallow footprint axis, pointing
    AWAY from the nearest wall (into the room). The unit blows from its front vent into the room, not
    straight down (the CoolingUnit default (0,0,-1)), so the cold jet reaches the racks instead of pooling
    below the unit. (Grid frame: X, Y horizontal, Z up.)"""
    dx, dy = float(dims[0]), float(dims[1])
    cx, cy = float(center[0]), float(center[1])
    ex = float(ext[0]) if ext and len(ext) > 0 else 0.0
    ey = float(ext[1]) if ext and len(ext) > 1 else 0.0
    if dx <= dy:  # shallow along X -> front/back are the X faces -> discharge ±X
        return (1.0, 0.0, 0.0) if (ex <= 0 or cx < ex / 2) else (-1.0, 0.0, 0.0)
    return (0.0, 1.0, 0.0) if (ey <= 0 or cy < ey / 2) else (0.0, -1.0, 0.0)


def _mid_height_grid(temp, origin, z_m: float = 1.0, res_m: float = 0.5):
    """Down-sample the solved 3-D field to a 2-D mid-height (z ~ 1 m) temperature grid for the floor heatmap.

    Returns absolute °C tiles on a ``res_m`` lattice in world XY so the viewer can show the SAME field the
    ASHRAE metrics come from (cold aisle, hot aisles, the AC jet) rather than a separate frontend plume
    approximation. A thin band (+/- 0.2 m) around the plane is averaged for stability. ``values[i][j]`` is the
    temperature at world (x = ox + i*res, y = oy + j*res, z = z_m)."""
    from engine.core.config import VOXEL_SIZE
    vx = float(VOXEL_SIZE)
    k = int(round((z_m - float(origin[2])) / vx))
    k = min(max(k, 0), temp.shape[2] - 1)
    band = max(1, int(round(0.2 / vx)))
    slab = temp[:, :, max(0, k - band):min(temp.shape[2], k + band + 1)].mean(axis=2)
    step = max(1, int(round(res_m / vx)))
    nx, ny = slab.shape[0] // step, slab.shape[1] // step
    if nx == 0 or ny == 0:
        return None
    block = slab[:nx * step, :ny * step].reshape(nx, step, ny, step).mean(axis=(1, 3))
    return {
        "z_m": z_m, "res_m": res_m, "nx": int(nx), "ny": int(ny),
        "ox": round(float(origin[0]), 3), "oy": round(float(origin[1]), 3),
        "values": [[round(float(v), 2) for v in row] for row in block],
        "tmin": round(float(block.min()), 2), "tmax": round(float(block.max()), 2),
    }


@router.get("/twin/{job_id}/thermal")
def twin_thermal(job_id: str) -> JSONResponse:
    """Run the 3-D thermal engine on the job's DETECTED voxel room and return its ASHRAE TC 9.9 score.

    Loads ``voxel_grid.npy`` + ``placements.json`` (the scanned twin), rebuilds the rack / cooling objects
    from the manifest, and runs ``compute_thermal_field`` + ``compute_metrics`` -- the same engine as the
    ``/visualize`` route, but on the twin's own grid rather than a re-voxelized mesh. The thermal engine is
    numpy-only (no open3d / torch), so it is imported lazily here without breaking the API process's CV
    boundary. This is the 'compute thermals on the detected room' step of the dashboard workflow."""
    import numpy as np

    from engine.core.data_types import Coordinate, CoolingUnit, RackFacing, RackPlacement
    from engine.thermal.metrics import compute_metrics
    from engine.thermal.solver import compute_thermal_field

    d = RUNS / job_id
    gp, pp = d / "voxel_grid.npy", d / "placements.json"
    if not gp.exists() or not pp.exists():
        raise HTTPException(404, "no voxel_grid.npy / placements.json for this job (run the twin first)")
    grid = np.load(gp)
    m = json.loads(pp.read_text())
    origin = np.array(m.get("origin", [0.0, 0.0, 0.0]), float)
    racks = [RackPlacement(position=Coordinate(*p["pos"]), facing=RackFacing[p["facing"]],
                           rack_type=p.get("rack_type", "42U"))
             for p in m.get("instances", []) if p.get("kind") == "rack"]
    if not racks:
        raise HTTPException(400, "no racks in this twin to analyse")
    ac_ext = m.get("ext", [])
    cooling = [
        CoolingUnit(
            position=Coordinate(*p["center"]),
            capacity_kw=12.0,
            # Floor-standing unit: discharge horizontally into the room (front vent), not straight down.
            supply_direction=_ac_supply_dir(p["center"], p.get("dims", [1.0, 1.0, 1.0]), ac_ext),
        )
        for p in m.get("instances", [])
        if str(p.get("name", "")).startswith("ac_unit") or int(p.get("vox_id", -1)) == 3
    ]
    temp = compute_thermal_field(grid, racks, origin, cooling_units=cooling)
    mr = compute_metrics(grid, temp, racks, origin, cooling)
    return JSONResponse({
        "n_racks": len(racks), "n_cooling": len(cooling),
        "temp_mean_c": round(float(np.mean(temp)), 2), "temp_max_c": round(float(np.max(temp)), 2),
        "compliant_racks": int(sum(r.inlet_compliant for r in mr.racks)),
        "racks": [{"rack_index": r.rack_index, "intake_temp": round(r.intake_temp, 2),
                   "exhaust_temp": round(r.exhaust_temp, 2), "delta_t": round(r.delta_t, 2),
                   "inlet_compliant": bool(r.inlet_compliant),
                   "inlet_within_allowable": bool(r.inlet_within_allowable)} for r in mr.racks],
        "room": {"rci_hi": round(mr.room.rci_hi, 1), "rci_lo": round(mr.room.rci_lo, 1),
                 "shi": round(mr.room.shi, 3), "rhi": round(mr.room.rhi, 3),
                 "rti": round(mr.room.rti, 1),
                 "mean_intake": round(mr.room.mean_intake, 2), "mean_exhaust": round(mr.room.mean_exhaust, 2),
                 "vertical_profile": [round(float(x), 2) for x in mr.room.vertical_profile]},
        "mid_temp": _mid_height_grid(temp, origin),
    })


# Equipment voxel ids that are MOVABLE (racks + their stamped intake/exhaust faces, network rack, AC and
# its grille, power) -- stripped to recover the empty room shell. Structural ids (wall 1, fire-hose cabinet
# 11) stay as fixed obstacles the RL must route around.
_MOVABLE_VOX_IDS = (3, 5, 6, 7, 8, 9, 12)
# RL action ``dir`` -> rack INTAKE facing (the opposite of the exhaust direction the policy encodes:
# dir 0 exhausts +X so the intake faces -X, and so on). Matches engine/rl twin decode conventions.
_DIR_FACE = ("MINUS_X", "PLUS_X", "MINUS_Y", "PLUS_Y")
_FACE_SIGN = {"PLUS_X": (1, 0), "MINUS_X": (-1, 0), "PLUS_Y": (0, 1), "MINUS_Y": (0, -1)}


def _rack_center_dims(pos, facing, rack_type):
    """World center + axis-aligned footprint for a decoded RL rack. ``pos`` is the intake-face
    front-bottom-center (the cell the policy chose); the body extends ``depth/2`` opposite the facing.
    When the rack faces along X the footprint is rotated (depth along X, width along Y)."""
    from engine.core.config import RACK_DIMENSIONS

    w, d, h = RACK_DIMENSIONS.get(rack_type, RACK_DIMENSIONS["42U"])
    sx, sy = _FACE_SIGN[facing]
    fx, fy = (d, w) if facing in ("PLUS_X", "MINUS_X") else (w, d)
    cx, cy = pos[0] - sx * d / 2.0, pos[1] - sy * d / 2.0
    return [round(cx, 3), round(cy, 3), round(h / 2.0, 3)], [round(fx, 3), round(fy, 3), round(h, 3)]


@router.post("/twin/{job_id}/optimize")
def twin_optimize(job_id: str, payload: dict = Body(default={})) -> JSONResponse:
    """Run the trained reinforcement-learning (RL) policy on the job's scanned room and return the layout
    it proposes. This is the twin -> RL leg: ``twin_bridge.twin_to_rl_input`` maps the scanned voxel room
    (walls + fixed infrastructure, movable equipment stripped) to the policy's ``(obstacle, cooling_pos,
    rack_num, ceiling_m)`` observation, ``rl_service`` runs MaskablePPO to place ``rack_num`` racks, and the
    chosen cells are decoded back to world-frame rack placements (the same ``pos`` / ``facing`` / ``center``
    / ``dims`` schema as ``placements.json``) so the dashboard can render the proposed layout in the BEV.

    The RL stack (stable-baselines3 / sb3-contrib + the model files) is imported lazily: if it is absent the
    route returns 503, exactly like the optional ``/inference`` router, and the CV ``/twin`` flow is
    unaffected. ``rack_num`` may be overridden in the body; it defaults to the twin's stamped rack count."""
    import numpy as np

    from engine.rl.twin_bridge import twin_to_rl_input

    d = RUNS / job_id
    gp, pp = d / "voxel_grid.npy", d / "placements.json"
    if not gp.exists() or not pp.exists():
        raise HTTPException(404, "no voxel_grid.npy / placements.json for this job (run the twin first)")

    # Empty room: prefer the dedicated empty grid if the pipeline wrote one, else recover it by stripping
    # movable equipment voxels from the full grid (racks/AC/network rack), leaving the structural shell.
    egp = d / "voxel_empty_grid.npy"
    if egp.exists():
        empty = np.load(egp)
    else:
        empty = np.load(gp).copy()
        empty[np.isin(empty, _MOVABLE_VOX_IDS)] = 0
    manifest = json.loads(pp.read_text())

    rack_num = payload.get("rack_num")
    rack_num = int(rack_num) if rack_num is not None else None
    rl_in = twin_to_rl_input(empty, manifest, rack_num=rack_num)

    # Use the imitation-trained layout policy (it proposes aligned hot/cold-aisle rows) when a compatible
    # one is loaded; otherwise the generic per-rack policy in ``rl_service``.
    from app.core import layout_service

    roll = None
    if layout_service.available_for(d):
        try:
            roll = layout_service.rollout(d)
        except Exception as exc:  # noqa: BLE001 -- surface layout-policy failures with the reason
            raise HTTPException(500, f"layout policy failed: {exc}") from exc
        result = {"data": roll["racks"], "total_energy": 0.0, "max_temp": []}
    else:
        # Lazy RL import -- keeps the CV API process import-safe when the RL deps / model are absent.
        try:
            from app.core.rl_service import rl_service
        except Exception as exc:  # noqa: BLE001 -- RL stack optional; report cleanly instead of 500-ing
            raise HTTPException(503, f"RL serving stack unavailable: {exc}") from exc
        try:
            result = rl_service.optimize(
                obstacle=rl_in["obstacle"].tolist(),
                cooling_pos=rl_in["cooling_pos"].tolist(),
                rack_num=rl_in["rack_num"],
                ceiling_m=rl_in["ceiling_m"],
            )
        except Exception as exc:  # noqa: BLE001 -- surface inference failures as 500 with the reason
            raise HTTPException(500, f"RL inference failed: {exc}") from exc

    # Match the scanned racks' own type (so the proposed footprint matches), not the manifest's coarser
    # top-level default.
    scanned_racks = [p for p in manifest.get("instances", []) if p.get("kind") == "rack"]
    rack_type = (scanned_racks[0].get("rack_type") if scanned_racks
                 else manifest.get("rack_type")) or "42U_real"
    cell_m = float(rl_in.get("cell_m", 0.2))
    instances = []
    for i, r in enumerate(result.get("data", [])):
        facing = _DIR_FACE[int(r["dir"]) % 4]
        pos = [round(int(r["x"]) * cell_m, 3), round(int(r["y"]) * cell_m, 3), 0.0]
        center, dims = _rack_center_dims(pos, facing, rack_type)
        instances.append({
            "name": f"server rack {i + 1}", "kind": "rack", "pos": pos, "facing": facing,
            "rack_type": rack_type, "vox_id": 5, "movable": True, "center": center, "dims": dims,
        })

    # Fixed (non-rack) instances for the BEV. When the layout policy ran, use the AC / network rack at the
    # positions the POLICY proposed -- NOT the room's current ones -- so the optimized layout shows the
    # equipment relocated. Otherwise carry the room's manifest instances through.
    if roll is not None:
        fixed = roll["fixed"]
    else:
        fixed = [p for p in manifest.get("instances", [])
                 if not (p.get("kind") == "rack" or str(p.get("name", "")).startswith("server rack"))]

    # Score the PROPOSED layout on the same 3-D thermal engine as ``/thermal`` so the dashboard can show a
    # true before/after. The solver takes the rack list as heat sources + the grid only for airflow
    # geometry, so no voxel re-stamping (and no open3d) is needed: clear the movable server-rack voxels
    # (5/6/7) from the full grid, keep the fixed shell + infrastructure, and place the proposed racks.
    thermal = None
    mid_temp = None
    try:
        import numpy as _np

        from engine.core.data_types import Coordinate, CoolingUnit, RackFacing, RackPlacement
        from engine.thermal.metrics import compute_metrics
        from engine.thermal.solver import compute_thermal_field
        from engine.vision.voxelizer import stamp_rack_on_grid

        # Clear the movable server-rack voxels (5/6/7) from the full grid, then STAMP the proposed racks
        # back in: the solver builds its heat-source field from RACK_EXHAUST voxels, so the proposed racks
        # must be present as voxels (not just as a placement list) for the field to be non-ambient.
        tgrid = _np.load(gp).copy()
        tgrid[_np.isin(tgrid, (5, 6, 7))] = 0
        origin = _np.array(manifest.get("origin", [0.0, 0.0, 0.0]), float)
        proposed = [RackPlacement(position=Coordinate(*p["pos"]), facing=RackFacing[p["facing"]],
                                  rack_type=p["rack_type"]) for p in instances]
        for rk in proposed:
            stamp_rack_on_grid(tgrid, rk, origin)
        opt_ext = manifest.get("ext", [])
        # Score with the PROPOSED cooling unit: when a layout was proposed the AC may
        # have been relocated, so the thermal field must reflect the moved unit -- sourcing it from the
        # original manifest would score the optimized racks against the OLD AC position and hide the gain.
        ac_source = fixed if roll is not None else manifest.get("instances", [])
        cooling = [
            CoolingUnit(
                position=Coordinate(*p["center"]),
                capacity_kw=12.0,
                supply_direction=_ac_supply_dir(p["center"], p.get("dims", [1.0, 1.0, 1.0]), opt_ext),
            )
            for p in ac_source
            if str(p.get("name", "")).startswith("ac_unit") or int(p.get("vox_id", -1)) == 3
        ]
        tfield = compute_thermal_field(tgrid, proposed, origin, cooling_units=cooling)
        mr = compute_metrics(tgrid, tfield, proposed, origin, cooling)
        for inst, rk in zip(instances, mr.racks):
            inst["intake_temp"] = round(rk.intake_temp, 2)
            inst["exhaust_temp"] = round(rk.exhaust_temp, 2)
        thermal = {
            "compliant_racks": int(sum(rk.inlet_compliant for rk in mr.racks)),
            "temp_max_c": round(float(_np.max(tfield)), 2),
            "racks": [{"rack_index": rk.rack_index, "intake_temp": round(rk.intake_temp, 2),
                       "exhaust_temp": round(rk.exhaust_temp, 2), "delta_t": round(rk.delta_t, 2),
                       "inlet_compliant": bool(rk.inlet_compliant)} for rk in mr.racks],
            "room": {"rci_hi": round(mr.room.rci_hi, 1), "rci_lo": round(mr.room.rci_lo, 1),
                     "shi": round(mr.room.shi, 3), "rhi": round(mr.room.rhi, 3),
                     "rti": round(mr.room.rti, 1),
                     "mean_intake": round(mr.room.mean_intake, 2),
                     "mean_exhaust": round(mr.room.mean_exhaust, 2)},
        }
        mid_temp = _mid_height_grid(tfield, origin)
    except Exception:  # noqa: BLE001 -- thermal scoring is best-effort; placements still return without it
        thermal = None

    return JSONResponse({
        "n_racks": len(instances),
        "rack_num": int(rl_in["rack_num"]),
        "total_energy": round(float(result.get("total_energy", 0.0)), 2),
        "ext": manifest.get("ext"),
        "origin": manifest.get("origin", [0.0, 0.0, 0.0]),
        "voxel_size": manifest.get("voxel_size", 0.1),
        "rack_type": rack_type,
        "instances": instances,
        "fixed": fixed,
        "cooling_pos": rl_in["cooling_pos"].tolist(),
        "thermal": thermal,
        "mid_temp": mid_temp,
    })


# Default daily profile when the caller does not supply its own time samples: a diurnal outside-air
# sinusoid (coldest before dawn, hottest mid-afternoon) and a gentle daytime IT-load curve.
_DEFAULT_TEMPORAL_N = 12          # discrete samples across 24 h (every 2 h) — a coarse-t analysis
_OUTSIDE_MIN_C = 16.0
_OUTSIDE_MAX_C = 32.0
_PEAK_HOUR = 14.0                 # hottest hour of the day
_LOAD_BASE = 0.55                 # minimum IT load fraction
_LOAD_SWING = 0.45                # extra load at the daytime peak


def _default_temporal_samples(payload: dict) -> list[dict]:
    """Build the discrete (hour, outside_c, load) samples for the temporal sweep. The caller may override
    every knob; absent that, a diurnal outdoor sinusoid drives the room default temperature at each t."""
    import math

    n = int(payload.get("n", _DEFAULT_TEMPORAL_N))
    omin = float(payload.get("outside_min_c", _OUTSIDE_MIN_C))
    omax = float(payload.get("outside_max_c", _OUTSIDE_MAX_C))
    peak = float(payload.get("peak_hour", _PEAK_HOUR))
    out = []
    for i in range(max(1, n)):
        h = i * 24.0 / max(1, n)
        # 1 at the peak hour, 0 twelve hours away — drives both the outdoor temp and the IT load.
        warm = 0.5 * (1.0 + math.cos(2.0 * math.pi * (h - peak) / 24.0))
        out.append({
            "t": round(h, 2),
            "ambient_c": round(omin + (omax - omin) * warm, 2),
            "load": round(_LOAD_BASE + _LOAD_SWING * warm, 3),
        })
    return out


@router.post("/twin/{job_id}/temporal")
def twin_temporal(job_id: str, payload: dict = Body(default={})) -> JSONResponse:
    """Temporal ASHRAE analysis: a per-time-t steady-state thermal sweep, not a fabricated projection.

    For each discrete time t the method is exactly: (1) set the room default temperature to the OUTSIDE air
    temperature at t; (2) run every machine at once -- the AC fixed at ``ac_supply_c`` (default 18 C) and the
    racks following their power usage (``base_rack_kw`` scaled by the load at t); (3) solve the 3-D field to
    a stationary point (``compute_thermal_field`` iterates to convergence); (4) record that t's ASHRAE / heat
    metrics. t is discrete by design -- the room is assumed to re-settle at each sample; continuous t is not
    modelled. Samples may be supplied in the body (``samples=[{t, ambient_c, load}]``); otherwise a default
    diurnal outdoor profile is used. Each sample is one full solve (~1 s), so this runs for a few seconds."""
    import numpy as np

    from engine.core.config import DEFAULT_RACK_POWER_KW
    from engine.core.data_types import Coordinate, CoolingUnit, RackFacing, RackPlacement
    from engine.thermal.metrics import compute_metrics
    from engine.thermal.solver import compute_thermal_field

    d = RUNS / job_id
    gp, pp = d / "voxel_grid.npy", d / "placements.json"
    if not gp.exists() or not pp.exists():
        raise HTTPException(404, "no voxel_grid.npy / placements.json for this job (run the twin first)")
    grid = np.load(gp)
    m = json.loads(pp.read_text())
    origin = np.array(m.get("origin", [0.0, 0.0, 0.0]), float)

    ac_supply_c = float(payload.get("ac_supply_c", 18.0))
    base_rack_kw = float(payload.get("base_rack_kw", DEFAULT_RACK_POWER_KW))
    samples = payload.get("samples") or _default_temporal_samples(payload)

    # Build the rack / AC templates once; only power (per load) and the room default temp change per sample.
    base_racks = [(p["pos"], p["facing"], p.get("rack_type", "42U_real"))
                  for p in m.get("instances", []) if p.get("kind") == "rack"]
    if not base_racks:
        raise HTTPException(400, "no racks in this twin to analyse")
    ac_inst = [p for p in m.get("instances", [])
               if str(p.get("name", "")).startswith("ac_unit") or int(p.get("vox_id", -1)) == 3]

    series = []
    for s in samples:
        amb = float(s["ambient_c"])
        load = float(s.get("load", 1.0))
        racks = [RackPlacement(position=Coordinate(*pos), facing=RackFacing[f], rack_type=rt,
                               power_kw=base_rack_kw * load) for pos, f, rt in base_racks]
        cooling = [CoolingUnit(position=Coordinate(*p["center"]), capacity_kw=12.0,
                               supply_temp_c=ac_supply_c) for p in ac_inst]
        temp = compute_thermal_field(grid, racks, origin, cooling_units=cooling, ambient_c=amb)
        mr = compute_metrics(grid, temp, racks, origin, cooling)
        intakes = [r.intake_temp for r in mr.racks]
        series.append({
            "t": s.get("t"),
            "outside_c": round(amb, 2),
            "load": round(load, 3),
            "rack_kw": round(base_rack_kw * load, 2),
            "mean_intake": round(mr.room.mean_intake, 2),
            "max_intake": round(max(intakes, default=amb), 2),
            "mean_exhaust": round(mr.room.mean_exhaust, 2),
            "max_temp": round(float(np.max(temp)), 2),
            "compliant_racks": int(sum(r.inlet_compliant for r in mr.racks)),
            "allowable_racks": int(sum(r.inlet_within_allowable for r in mr.racks)),
            "rci_hi": round(mr.room.rci_hi, 1),
            "rti": round(mr.room.rti, 1),
            "mean_delta_t": round(float(np.mean([r.delta_t for r in mr.racks])) if mr.racks else 0.0, 2),
        })

    return JSONResponse({
        "n_racks": len(base_racks),
        "ac_supply_c": ac_supply_c,
        "base_rack_kw": base_rack_kw,
        "n_samples": len(series),
        "series": series,
    })


@router.post("/twin/{job_id}/compare")
def twin_compare(job_id: str) -> JSONResponse:
    """Side-by-side ASHRAE comparison that isolates the value of hot/cold-aisle containment.

    DESIGNED is the room's actual layout (rows facing each other across a shared cold aisle). BASELINE is
    the SAME racks, the SAME room, and the SAME AC, but with every rack turned to face one way, so exhaust
    recirculates into the next row's intakes. Only the aisle structure differs, so the delta is the cooling
    benefit of containment. Both are scored on the authoritative 3-D engine (racks stamped into the cleared
    grid as heat sources, exactly like ``/optimize`` and ``/thermal``)."""
    from collections import Counter

    import numpy as np

    from engine.core.data_types import Coordinate, CoolingUnit, RackFacing, RackPlacement
    from engine.thermal.metrics import compute_metrics
    from engine.thermal.solver import compute_thermal_field
    from engine.vision.voxelizer import stamp_rack_on_grid

    d = RUNS / job_id
    gp, pp = d / "voxel_grid.npy", d / "placements.json"
    if not gp.exists() or not pp.exists():
        raise HTTPException(404, "no voxel_grid.npy / placements.json for this job (run the twin first)")
    grid0 = np.load(gp)
    m = json.loads(pp.read_text())
    origin = np.array(m.get("origin", [0.0, 0.0, 0.0]), float)
    rack_insts = [p for p in m.get("instances", []) if p.get("kind") == "rack"]
    if not rack_insts:
        raise HTTPException(400, "no racks in this twin to compare")
    ac_ext = m.get("ext", [])
    cooling = [
        CoolingUnit(
            position=Coordinate(*p["center"]),
            capacity_kw=12.0,
            # Floor-standing unit: discharge horizontally into the room (front vent), not straight down.
            supply_direction=_ac_supply_dir(p["center"], p.get("dims", [1.0, 1.0, 1.0]), ac_ext),
        )
        for p in m.get("instances", [])
        if str(p.get("name", "")).startswith("ac_unit") or int(p.get("vox_id", -1)) == 3
    ]

    def score(facings: list[str]) -> dict:
        racks = [RackPlacement(position=Coordinate(*p["pos"]), facing=RackFacing[f],
                               rack_type=p.get("rack_type", "42U_real"))
                 for p, f in zip(rack_insts, facings)]
        grid = grid0.copy()
        grid[np.isin(grid, (5, 6, 7))] = 0
        for rk in racks:
            stamp_rack_on_grid(grid, rk, origin)
        temp = compute_thermal_field(grid, racks, origin, cooling_units=cooling)
        mr = compute_metrics(grid, temp, racks, origin, cooling)
        intk = [r.intake_temp for r in mr.racks]
        return {
            "compliant": int(sum(r.inlet_compliant for r in mr.racks)),
            "rci_hi": round(mr.room.rci_hi, 1),
            "rci_lo": round(mr.room.rci_lo, 1),
            "shi": round(mr.room.shi, 3),
            "rhi": round(mr.room.rhi, 3),
            "rti": round(mr.room.rti, 1),
            "mean_intake": round(mr.room.mean_intake, 2),
            "max_intake": round(max(intk), 2) if intk else None,
            "mean_exhaust": round(mr.room.mean_exhaust, 2),
            "max_temp": round(float(np.max(temp)), 2),
            "mean_delta_t": round(float(np.mean([r.delta_t for r in mr.racks])), 2),
        }

    designed_facings = [p["facing"] for p in rack_insts]
    majority = Counter(designed_facings).most_common(1)[0][0]
    baseline_facings = [majority] * len(rack_insts)
    return JSONResponse({
        "n_racks": len(rack_insts),
        "designed": score(designed_facings),
        "baseline": score(baseline_facings),
    })
