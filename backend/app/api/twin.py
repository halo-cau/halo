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

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".bmp", ".tif", ".tiff", ".webp"}
GEOMETRY_SUFFIXES = {".obj", ".ply", ".las", ".laz"}

_jobs: "queue.Queue[tuple[str, dict]]" = queue.Queue()


def _env() -> dict:
    return dict(os.environ, HF_HUB_OFFLINE="1",
                PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", PYOPENGL_PLATFORM="egl")


def _worker() -> None:
    """Serialize jobs: run one pipeline subprocess (in halo) at a time so the single GPU never thrashes."""
    while True:
        job_id, p = _jobs.get()
        d = RUNS / job_id
        if p["kind"] == "images":
            # The proven June-6 chest32 recipe: all 32 frames @245k px reproduce the reconstruction
            # byte-for-byte (run_pi3 is deterministic), and SAM3 instancing matches that run. The
            # voxelizer then anchors metric scale on the rack-ROW width (robust to the instance split)
            # and recovers the occluded depth from rack priors, so the drop-in lands on the target room.
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
