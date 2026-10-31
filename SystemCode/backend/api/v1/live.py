"""One-frame inference for simulated live playback.

Each call runs the released still-image model once and echoes the caller's
frame index and timestamp. This route does not queue frames; the client keeps
at most one request in flight and drops frames that arrive while it is busy.

A live session stores those frames on an analysis task so the recorded-video
report, frame list, and CSV export can read them. The session is not placed
on the background worker queue.
"""

import base64
import io
import math
from datetime import datetime, timezone

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from ...core.config import WARNINGS
from ...endolens_model import MODEL_VERSION, POSTPROCESS_VERSION
from ..dependencies import get_record, get_runtime, identity, image_bytes, now

router = APIRouter()

MAX_LIVE_FRAME_BYTES = 8 * 1024 * 1024


class LiveSessionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    video_id: str = Field(pattern=r"^[a-f0-9]{32}$")


def _elapsed(task: dict) -> float:
    started = datetime.fromisoformat(task["created_at"])
    return max(0.0, (datetime.now(timezone.utc) - started).total_seconds())


def _live_task(runtime, task_id: str) -> dict:
    if len(task_id) != 32 or any(char not in "0123456789abcdef" for char in task_id):
        raise HTTPException(422, "task_id must be 32 hex characters")
    task = get_record(runtime, "analysis", task_id)
    if task.get("configuration", {}).get("sampling_mode") != "live":
        raise HTTPException(409, "Task is not a live session")
    return task


def _sync_live_task(runtime, task: dict, *, finished: bool) -> dict:
    rows = runtime.store.list("frame", task["task_id"])
    good = [row for row in rows if row["status"] == "completed"]
    failed = len(rows) - len(good)
    updates = {
        "processed_frames": len(good),
        "failed_frames": failed,
        "candidate_frames": sum(row["is_candidate"] is True for row in good),
        "elapsed_seconds": _elapsed(task),
        "warnings": [*WARNINGS, "HAS_FAILED_FRAMES"] if failed else list(WARNINGS),
    }
    if finished:
        updates.update(status="completed", phase="done", finished_at=now(), error=None)
    return runtime.store.update(task["task_id"], **updates)


def _store_live_frame(runtime, task_id: str, content: bytes, rgb: np.ndarray, post, result, frame_index: int, timestamp_seconds: float):
    frame_id = f"{task_id}-f{frame_index:08d}"
    with runtime.store.lock:
        task = _live_task(runtime, task_id)
        if task["status"] != "running":
            raise HTTPException(409, "Live session is not accepting frames")
        try:
            runtime.store.get("frame", frame_id)
            return
        except KeyError:
            pass
        source = runtime.inference.source_image(rgb)
        runtime.store.write(f"frames/{frame_id}.jpg", content)
        runtime.store.write(f"thumbs/{frame_id}.jpg", image_bytes(runtime.inference.thumbnail(source), "JPEG"))
        runtime.store.write(f"masks/{frame_id}.png", image_bytes(runtime.inference.mask_image(post)))
        runtime.store.put(
            "frame",
            frame_id,
            {
                "frame_id": frame_id,
                "task_id": task_id,
                "frame_index": frame_index,
                "timestamp_seconds": timestamp_seconds,
                "width": int(rgb.shape[1]),
                "height": int(rgb.shape[0]),
                "model_version": MODEL_VERSION,
                "status": "completed",
                "is_candidate": bool(post.is_candidate),
                "max_region_area_ratio": post.max_region_area_ratio,
                "source_frame_url": f"/api/v1/assets/frames/{frame_id}.jpg",
                "thumbnail_url": f"/api/v1/assets/thumbs/{frame_id}.jpg",
                "mask_url": f"/api/v1/assets/masks/{frame_id}.png",
                "contours": post.contours,
                "regions": post.regions,
                "component_count": post.component_count,
                "display_geometry_truncated": post.display_geometry_truncated,
                "review_status": "unreviewed",
                "revision_id": None,
                "foreground_score": post.foreground_score,
                "score_type": "mean_foreground_score_of_largest_retained_component",
                "raw_component_count": post.raw_component_count,
                "removed_component_count": post.removed_component_count,
                "postprocessing_version": POSTPROCESS_VERSION,
                "segmented_area_ratio": post.segmented_area_ratio,
                "inference_time_ms": result.inference_time_ms,
                "device": result.device,
            },
            task_id,
            frame_index,
        )
        _sync_live_task(runtime, task, finished=False)


@router.post("/live/sessions", status_code=201)
def open_live_session(body: LiveSessionInput, request: Request):
    runtime = get_runtime(request)
    get_record(runtime, "video", body.video_id)
    description = runtime.adapter.describe()
    task_id = identity()
    value = {
        "task_id": task_id,
        "video_id": body.video_id,
        "status": "running",
        "configuration": {
            "video_id": body.video_id,
            "sampling_mode": "live",
            "interval_seconds": None,
            "model_version": MODEL_VERSION,
            "segmentation_threshold": 0.5,
            "min_region_area_ratio": 0.0,
            "min_region_mean_score": 0.8,
            "candidate_rule_status": "REGION_MEAN_0_8_FILTER",
            "source_checkpoint_sha256": description["source_checkpoint_sha256"],
            "model_weights_sha256": description.get("weights_sha256", "TEST_ONLY"),
            "model_device": description["device"],
            "preprocessing": description.get("input", {}),
            "postprocessing_version": POSTPROCESS_VERSION,
        },
        "created_at": now(),
        "started_at": now(),
        "target_frames": None,
        "processed_frames": 0,
        "failed_frames": 0,
        "candidate_frames": 0,
        "elapsed_seconds": 0.0,
        "cancel_requested": False,
        "phase": "live",
        "warnings": WARNINGS,
    }
    runtime.store.put("analysis", task_id, value)
    return {"task_id": task_id, "video_id": body.video_id, "status": "running"}


@router.post("/live/sessions/{task_id}/finish")
def finish_live_session(task_id: str, request: Request):
    runtime = get_runtime(request)
    with runtime.store.lock:
        task = _live_task(runtime, task_id)
        if task["status"] != "running":
            return {"task_id": task_id, "status": task["status"]}
        task = _sync_live_task(runtime, task, finished=True)
    return {"task_id": task_id, "status": task["status"]}


@router.post("/live/frames")
def infer_live_frame(
    request: Request,
    file: UploadFile = File(...),
    frame_index: int = Form(...),
    timestamp_seconds: float = Form(...),
    task_id: str | None = Form(None),
):
    if frame_index < 0 or frame_index > 10_000_000:
        raise HTTPException(422, "frame_index out of range")
    if (
        not math.isfinite(timestamp_seconds)
        or timestamp_seconds < 0
        or timestamp_seconds > 24 * 3600
    ):
        raise HTTPException(422, "timestamp_seconds out of range")
    try:
        content = file.file.read(MAX_LIVE_FRAME_BYTES + 1)
    finally:
        file.file.close()
    if len(content) > MAX_LIVE_FRAME_BYTES:
        raise HTTPException(413, "Frame exceeds 8MB")
    if not content:
        raise HTTPException(422, "Empty frame")
    try:
        with Image.open(io.BytesIO(content)) as image:
            rgb = np.ascontiguousarray(image.convert("RGB"))
    except Exception as error:
        raise HTTPException(422, "Unreadable image") from error
    runtime = get_runtime(request)
    try:
        result, post = runtime.inference.predict(rgb, MODEL_VERSION)
        if task_id:
            _store_live_frame(runtime, task_id, content, rgb, post, result, frame_index, timestamp_seconds)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    mask = base64.b64encode(image_bytes(runtime.inference.mask_image(post))).decode("ascii")
    return {
        "frame_index": frame_index,
        "timestamp_seconds": timestamp_seconds,
        "width": int(rgb.shape[1]),
        "height": int(rgb.shape[0]),
        "is_candidate": bool(post.is_candidate),
        "foreground_score": post.foreground_score,
        "max_region_area_ratio": post.max_region_area_ratio,
        "inference_time_ms": result.inference_time_ms,
        "device": result.device,
        "model_version": MODEL_VERSION,
        "mask_png_base64": mask,
        "timing_scope": "single frame: preprocessing, rotate4 forwards, resize, and region filtering",
    }
