"""One-frame inference for simulated live playback.

Each call runs the released still-image model once and echoes the caller's
frame index and timestamp. This route does not queue frames; the client keeps
at most one request in flight and drops frames that arrive while it is busy.
"""

import base64
import io
import math

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from PIL import Image

from ...endolens_model import MODEL_VERSION
from ..dependencies import get_runtime, image_bytes

router = APIRouter()

MAX_LIVE_FRAME_BYTES = 8 * 1024 * 1024


@router.post("/live/frames")
def infer_live_frame(
    request: Request,
    file: UploadFile = File(...),
    frame_index: int = Form(...),
    timestamp_seconds: float = Form(...),
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
