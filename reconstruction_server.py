"""
Reconstruction sidecar server (Option 2 — sidecar mode).

Wraps reconstruction_activeperception's streaming pipeline behind FastAPI.
Receives RGB-D frames from client_sapien, accumulates SceneState, emits URDF
on demand. Does NOT communicate with action_server — purely observer.

Endpoints (port 8002):
    POST /init          — set partnet_id, intrinsics, state_dir; loads SAM3 once
    POST /ingest_frame  — feed one (rgb, depth, c2w) frame
    POST /emit_urdf     — produce URDF from current SceneState
    GET  /health        — readiness probe

Env requirements:
    RECONSTRUCTION_PIPELINE_PATH — path to the reconstruction_activeperception repo

Designed to run inside the same `owlsam` conda env that hosts det_pipeline /
action_server (Py3.11 + torch2.9+cu128). All required deps verified at T0.

Concurrency model
-----------------
Each route is `async def` so we can read the raw pickled request body via
`await req.body()` without an extra Body(...) annotation (which would force
a Content-Type check). All CUDA / SAM3 work is dispatched to a worker thread
via `asyncio.to_thread(...)` — running PyTorch CUDA ops directly on the
asyncio event loop reliably segfaulted on RTX 5090 / sm_120 (verified by
isolated repro). Thread-pool dispatch is FastAPI's recommended pattern for
GPU-bound handlers.
"""
import asyncio
import faulthandler
import os
import sys
import pickle
import traceback
from pathlib import Path

# Print Python-level traceback to stderr if a fatal signal (SIGSEGV/SIGABRT/...)
# kills the process. Critical for debugging the SAM3-on-RTX5090 crashes that
# leave no other artifact.
faulthandler.enable(file=sys.stderr, all_threads=True)

from fastapi import FastAPI, Request
import uvicorn
import numpy as np

RECON_PATH = os.environ.get("RECONSTRUCTION_PIPELINE_PATH")
if not RECON_PATH:
    raise RuntimeError(
        "RECONSTRUCTION_PIPELINE_PATH env var not set. "
        "Point it at the reconstruction_activeperception repo root."
    )
sys.path.insert(0, RECON_PATH)

from pipeline.streaming import CameraIntrinsics, SceneState
from pipeline.streaming.phase_a_sam3 import StreamingSAM3
from pipeline.streaming.ingest_frame import ingest_frame, emit_urdf_now

app = FastAPI()
state: SceneState | None = None
sam3: StreamingSAM3 | None = None

# Hardcoded part assignments per PartNet-Mobility ID — bypasses Phase 0 VLM.
# Convention: static_parts[0] MUST be the articulation parent (largest body).
# Names must be SAM3 text-promptable (used directly as text queries).
PARTS_BY_ID = {
    # Per PartNet-Mobility semantics.txt:
    #   40147: link_0 hinge rotation_door, link_1 slider drawer, link_2 body
    #     ⇒ task fallback plan abuses target_handle_id=1 = hinge door (Pull_Arc)
    #   44817: 4 sliders (drawers only)
    #   46230: 3 sliders (drawers only)
    # SAM3 prompts must match what the task is actually grasping/moving — earlier
    # mismatch ("lower drawer" prompt while panda grasps the door) gave empty
    # masks ⇒ Phase C joint estimator never populated ⇒ emit_urdf raises
    # "SceneState.joint is not populated".
    40147: dict(
        parts=["drawer", "cabinet body", "cabinet door"],
        moving=["drawer"],   # task 改成 "open the drawer"，sidecar 跟着追 drawer
        static=["cabinet body", "cabinet door"],
    ),
    44817: dict(
        parts=["second drawer", "cabinet body", "first drawer", "third drawer", "fourth drawer"],
        moving=["second drawer"],
        static=["cabinet body", "first drawer", "third drawer", "fourth drawer"],
    ),
    46230: dict(
        parts=["bottom drawer", "cabinet body", "top drawer", "middle drawer"],
        moving=["bottom drawer"],
        static=["cabinet body", "top drawer", "middle drawer"],
    ),
}


# ─────────────────────────── lifespan: preload SAM3 on main thread ───────────

@app.on_event("startup")
def _preload_sam3_on_main_thread():
    """Force SAM3 model + CUDA context to initialise on the uvicorn main
    thread, BEFORE any request handler runs in a worker thread.

    Empirically: deferring `Sam3VideoModel.from_pretrained(...).to('cuda')`
    until the first /ingest_frame call (which lands in a thread-pool worker
    via `asyncio.to_thread`) reliably segfaults with exit -11 on RTX 5090
    (sm_120). Initialising the CUDA primary context + model weights on the
    main thread up-front avoids that crash.

    Also installs a `compute_masks_through_latest` shim that prefers
    client-supplied ground-truth masks (from SAPIEN's segmentation buffer,
    sent in /ingest_frame's `gt_masks` field) when available, falling back
    to real SAM3 video tracking otherwise. The GT-mask path is for
    debugging — use to confirm whether downstream Phase B/C/D/fuse failures
    are caused by SAM3 mask quality.
    """
    global sam3
    if sam3 is None:
        print("[recon][startup] preloading SAM3 model on main thread...", flush=True)
        sam3 = StreamingSAM3()
        sam3.set_parts(["__warmup__"])         # arbitrary; replaced at /init
        sam3._ensure_model()                    # noqa: SLF001
        print(f"[recon][startup] SAM3 ready (device={sam3._device})", flush=True)

    # Install GT-mask shim. `_gt_masks` is a dict[frame_idx, dict[part_name, bool ndarray]].
    # When all currently-buffered frames have a GT mask plant, the shim returns
    # those directly without touching the GPU. When any frame lacks a plant
    # (rare in our streaming use), it falls back to the original SAM3 method.
    sam3._gt_masks = {}                          # type: ignore[attr-defined]
    _real_compute = sam3.compute_masks_through_latest

    def _compute_with_gt_fallback():
        if not sam3.ready:
            return None
        if all(fidx in sam3._gt_masks for fidx in sam3.frames):
            return {fidx: dict(sam3._gt_masks[fidx]) for fidx in sorted(sam3.frames)}
        return _real_compute()

    sam3.compute_masks_through_latest = _compute_with_gt_fallback   # type: ignore[method-assign]
    print("[recon][startup] GT-mask shim installed on sam3.compute_masks_through_latest", flush=True)


# ─────────────────────────── sync workers (run in thread pool) ────────────────

def _do_init(payload: dict) -> dict:
    global state, sam3
    pid = payload['partnet_id']
    if pid not in PARTS_BY_ID:
        return {"ok": False, "error": f"unknown partnet_id {pid}; supported: {list(PARTS_BY_ID)}"}

    intr = CameraIntrinsics(**payload['intrinsics'])
    state_dir = Path(payload['state_dir'])
    state_dir.mkdir(parents=True, exist_ok=True)
    state = SceneState.load_or_create(str(state_dir), intrinsics=intr)

    pcfg = PARTS_BY_ID[pid]
    state.parts = pcfg['parts']
    state.moving_parts = pcfg['moving']
    state.static_parts = pcfg['static']

    if sam3 is None:
        # startup hook should have populated this; fall back just in case
        print("[recon] WARNING: SAM3 not preloaded — loading now (may segfault)", flush=True)
        sam3 = StreamingSAM3()

    # Replace the warmup parts list set in _preload_sam3_on_main_thread() with
    # the real one for this PartNet ID
    sam3.set_parts(pcfg['parts'])

    print(f"[recon] /init partnet_id={pid} state_dir={state_dir} parts={pcfg['parts']}", flush=True)
    return {"ok": True, "parts": pcfg['parts'], "state_dir": str(state_dir)}


def _do_ingest(payload: dict) -> dict:
    if state is None or sam3 is None:
        return {"ok": False, "error": "not initialized; POST /init first"}

    # Plant ground-truth masks (if provided by client) into the sam3 shim's
    # cache so `compute_masks_through_latest` skips real video tracking for
    # this frame and returns the GT mask dict directly. See startup hook.
    gt = payload.get('gt_masks')
    if gt is not None and hasattr(sam3, '_gt_masks'):
        # sam3 expects {part_name: bool ndarray (H,W)}; client already sends
        # exactly that. Make a defensive copy so client array mutations
        # post-POST don't race with downstream readers.
        sam3._gt_masks[int(payload['frame_idx'])] = {
            p: np.ascontiguousarray(m).astype(bool) for p, m in gt.items()
        }

    try:
        status = ingest_frame(
            state,
            payload['frame_idx'],
            label=payload['label'],
            rgb=payload['rgb'],
            depth_m=payload['depth_m'],
            c2w=payload['c2w'],
            sam3=sam3,
            cam_pos_world=payload['cam_pos_world'],
        )
        print(f"[recon] /ingest_frame {payload['frame_idx']} ok", flush=True)
        return {"ok": True, "frame_idx": payload['frame_idx'], "status": str(status)[:500]}
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[recon] /ingest_frame {payload.get('frame_idx', '?')} FAILED:\n{tb}", flush=True)
        return {"ok": False, "error": str(e), "tb": tb[-2000:]}


def _do_set_moving_part(payload: dict) -> dict:
    """Sub-milestone 1a: switch sidecar's tracked moving/static partition at
    runtime.

    Called by client_sapien once the gripper has Grasped a part — client
    inferred which cabinet link was actually grasped (via SAPIEN seg buffer
    at the projected finger-tip pixel) and pushes that here. Sidecar then
    rewrites SceneState.moving_parts/static_parts/parts, drops accumulated
    per-part data (depth_pts, mask_history, obb), clears the TSDF cache,
    and resets the SAM3 part list + GT-mask shim cache so subsequent
    /ingest_frame calls converge from scratch on the right part.

    Idempotent: noop when the requested partition matches current state.
    """
    if state is None:
        return {"ok": False, "error": "not initialized; POST /init first"}

    moving = payload['moving']
    static = list(payload.get('static', []))
    new_parts = [moving] + static

    if state.moving_parts == [moving] and state.static_parts == static:
        print(f"[recon] /set_moving_part noop (already moving={moving!r})", flush=True)
        return {"ok": True, "noop": True, "moving": moving, "static": static}

    # Drop accumulated per-part artefacts so Phase B/C re-converge on the new
    # part rather than averaging over the previous one.
    state_dir = Path(getattr(state, "state_dir", "/tmp"))
    per_part = state_dir / "per_part"
    cleared = []
    if per_part.exists():
        for part_dir in per_part.iterdir():
            if not part_dir.is_dir():
                continue
            mh = part_dir / "mask_history"
            if mh.exists():
                for f in mh.glob("*.npy"):
                    f.unlink()
            for fname in ("depth_pts.npy", "depth_pts_frame.npy", "obb.json"):
                p = part_dir / fname
                if p.exists():
                    p.unlink()
            cleared.append(part_dir.name)

    state.parts = list(new_parts)
    state.moving_parts = [moving]
    state.static_parts = list(static)
    if hasattr(state, "tsdfs"):
        state.tsdfs.clear()

    if sam3 is not None:
        sam3.set_parts(list(new_parts))
        if hasattr(sam3, "_gt_masks"):
            sam3._gt_masks.clear()

    print(f"[recon] /set_moving_part moving={moving!r} static={static} "
          f"cleared={cleared}", flush=True)
    return {"ok": True, "moving": moving, "static": static, "cleared": cleared}


def _do_emit(payload: dict) -> dict:
    if state is None:
        return {"ok": False, "error": "not initialized"}
    out_dir = Path(payload['out_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        info = emit_urdf_now(state, out_dir=str(out_dir))
        urdf_path = out_dir / "cabinet_drawer_rgbd.urdf"
        print(f"[recon] /emit_urdf → {urdf_path}", flush=True)
        return {"ok": True, "urdf": str(urdf_path), "info": str(info)[:500]}
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[recon] /emit_urdf FAILED:\n{tb}", flush=True)
        return {"ok": False, "error": str(e), "tb": tb[-2000:]}


# ─────────────────────────── async handlers (run SAM3 on main thread) ────────
# Note: `_do_*` are called directly inline (NOT via asyncio.to_thread). SAM3 +
# CUDA primary-context ops segfault on RTX 5090 / sm_120 when run from a
# worker thread. Keeping handlers async-but-blocking is the simplest way to
# pin all PyTorch CUDA work to the uvicorn main thread (the asyncio event loop
# thread). Tradeoff: requests serialise, no concurrency. Acceptable for one-
# at-a-time RGB-D ingestion.

@app.post("/init")
async def init(req: Request):
    """Body (pickle): {'partnet_id', 'state_dir', 'intrinsics': {...}}."""
    payload = pickle.loads(await req.body())
    return _do_init(payload)


@app.post("/ingest_frame")
async def ingest(req: Request):
    """Body (pickle): {'frame_idx', 'label', 'rgb', 'depth_m', 'c2w', 'cam_pos_world'}."""
    payload = pickle.loads(await req.body())
    return _do_ingest(payload)


@app.post("/emit_urdf")
async def emit(req: Request):
    """Body (pickle): {'out_dir': str}"""
    payload = pickle.loads(await req.body())
    return _do_emit(payload)


@app.post("/set_moving_part")
async def set_moving_part(req: Request):
    """Body (pickle): {'moving': str, 'static': list[str]}.

    See `_do_set_moving_part` for semantics. Called by client_sapien when it
    detects ActionPlanner has finished Grasp (Sub-milestone 1a).
    """
    payload = pickle.loads(await req.body())
    return _do_set_moving_part(payload)


@app.get("/health")
async def health():
    return {
        "ok": True,
        "initialized": state is not None,
        "sam3_loaded": sam3 is not None,
        "frame_count": getattr(state, "frame_count", 0) if state is not None else 0,
    }


if __name__ == "__main__":
    print(f"[recon] RECONSTRUCTION_PIPELINE_PATH={RECON_PATH}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="info")
