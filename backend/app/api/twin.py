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
# A precomputed twin run can stand in for the live pipeline during a walkthrough: the multi-view
# reconstruction (Pi3) + segmentation (SAM3) over ~1M points takes several minutes on the GPU, too long
# to run in front of an audience. When TWIN_PRECOMPUTED_SAMPLE names such a run, submitted jobs are served
# from it and flagged ``precomputed``; unset (the default) every job runs the real pipeline.
PRECOMPUTED_SAMPLE = os.environ.get("TWIN_PRECOMPUTED_SAMPLE", "").strip()

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".bmp", ".tif", ".tiff", ".webp"}
GEOMETRY_SUFFIXES = {".obj", ".ply", ".las", ".laz"}

_jobs: "queue.Queue[tuple[str, dict]]" = queue.Queue()


def _env() -> dict:
    return dict(os.environ, HF_HUB_OFFLINE="1",
                PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", PYOPENGL_PLATFORM="egl")


def _serve_precomputed(sample: Path, dst: Path) -> None:
    """Serve a precomputed twin run's artifacts for a job and flag it ``precomputed``.

    Lets a live walkthrough skip the multi-minute Pi3 + SAM3 GPU pipeline (see TWIN_PRECOMPUTED_SAMPLE).
    The recon / labeled / point_labels / voxel artifacts of the sample run are copied into the new job
    dir unchanged; the real pipeline is the default whenever no sample is configured."""
    import shutil
    for f in sample.iterdir():
        if f.is_file():
            shutil.copy2(f, dst / f.name)
    sj = dst / "status.json"
    st = json.loads(sj.read_text()) if sj.exists() else {}
    st.update({"state": "done", "step": "done", "pct": 100, "precomputed": True,
               "message": "precomputed sample (Pi3 + SAM3 pipeline pre-run)"})
    sj.write_text(json.dumps(st))


def _worker() -> None:
    """Serialize jobs: run one pipeline subprocess (in halo) at a time so the single GPU never thrashes."""
    while True:
        job_id, p = _jobs.get()
        d = RUNS / job_id
        if PRECOMPUTED_SAMPLE and (RUNS / PRECOMPUTED_SAMPLE).is_dir():
            try:                                               # demo mode: skip the multi-minute pipeline
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
    """Report the precomputed twin sample available to stand in for the live pipeline (demo mode).

    Returns the configured sample's id + a summary, or ``enabled: false`` in the normal case where none
    is set and every submitted job runs the real Pi3 + SAM3 pipeline. (Declared before /twin/{job_id} so
    the literal path is not captured as a job id.)"""
    samples = []
    if PRECOMPUTED_SAMPLE and (RUNS / PRECOMPUTED_SAMPLE / "status.json").exists():
        st = json.loads((RUNS / PRECOMPUTED_SAMPLE / "status.json").read_text())
        samples.append({"id": PRECOMPUTED_SAMPLE, "outputs": list(st.get("outputs", {}).keys()),
                        "label_counts": st.get("label_counts", {})})
    return JSONResponse({"enabled": bool(samples), "samples": samples})


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
    cooling = [CoolingUnit(position=Coordinate(*p["center"]), capacity_kw=12.0)
               for p in m.get("instances", [])
               if str(p.get("name", "")).startswith("ac_unit") or int(p.get("vox_id", -1)) == 3]
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
                 "mean_intake": round(mr.room.mean_intake, 2), "mean_exhaust": round(mr.room.mean_exhaust, 2),
                 "vertical_profile": [round(float(x), 2) for x in mr.room.vertical_profile]},
    })
