import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from ...core.config import MAX_UPLOAD
from ...services.video_service import VideoService
from ..dependencies import get_record, get_runtime, identity

router = APIRouter()
UPLOAD_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}


def _discard(store, path):
    if path is not None and path.exists():
        size = path.stat().st_size
        path.unlink()
        store.used_bytes -= size


def _stored_mp4(runtime, source, destination, size):
    """Probe a valid H264 MP4 as-is. Convert every other accepted upload first."""
    store = runtime.store
    if source.suffix.lower() == ".mp4":
        try:
            return VideoService.probe(source)
        except Exception:
            temporary = destination.with_name(destination.stem + ".converting.mp4")
            try:
                if store.used_bytes + size > store.max_bytes:
                    raise ValueError("Storage quota reached; preserve results and choose a new storage directory")
                VideoService.convert(source, temporary)
                metadata = VideoService.probe(temporary)
            except Exception:
                if temporary.exists():
                    temporary.unlink()
                raise
            _discard(store, source)
            temporary.replace(destination)
            store.used_bytes += destination.stat().st_size
            return metadata
    if store.used_bytes + size > store.max_bytes:
        raise ValueError("Storage quota reached; preserve results and choose a new storage directory")
    VideoService.convert(source, destination)
    try:
        metadata = VideoService.probe(destination)
    except Exception:
        if destination.exists():
            destination.unlink()
        raise
    _discard(store, source)
    store.used_bytes += destination.stat().st_size
    return metadata


@router.post("/videos", status_code=201)
async def upload(request: Request, file: UploadFile = File(...)):
    runtime = get_runtime(request)
    raw = None
    try:
        name = Path((file.filename or "upload").replace("\\", "/")).name
        suffix = Path(name).suffix.lower()
        if suffix not in UPLOAD_SUFFIXES:
            raise HTTPException(415, "Only .mp4, .mov, .m4v, .avi, or .mkv accepted")
        content = await file.read(MAX_UPLOAD + 1)
        if not content or len(content) > MAX_UPLOAD:
            raise HTTPException(413, f"Empty file or upload exceeds{MAX_UPLOAD // 1024**2}MiB")
        video_id = identity()
        raw_rel = f"videos/{video_id}.mp4" if suffix == ".mp4" else f"videos/{video_id}{suffix}"
        raw = runtime.store.write(raw_rel, content)
        final = runtime.store.path(f"videos/{video_id}.mp4")
        try:
            metadata = _stored_mp4(runtime, raw, final, len(content))
        except Exception as error:
            _discard(runtime.store, raw)
            if final.exists() and final != raw:
                final.unlink()
            raw = None
            raise HTTPException(422, f"Video rejected:{error}") from error
        value = {
            "video_id": video_id,
            "filename": name[:120],
            "video_url": f"/api/v1/videos/{video_id}/stream",
            "stored_path": f"videos/{video_id}.mp4",
            **metadata,
        }
        runtime.store.put("video", video_id, value)
        return {key: value[key] for key in value if key != "stored_path"}
    finally:
        await file.close()


@router.get("/videos/{video_id}/stream")
def stream(video_id: str, request: Request):
    runtime = get_runtime(request)
    value = get_record(runtime, "video", video_id)
    byte_range = request.headers.get("range", "")
    if byte_range and not re.fullmatch(r"bytes=(\d{1,18}-\d{0,18}|-\d{1,18})", byte_range):
        raise HTTPException(416, "Only a single byte range is supported")
    return FileResponse(runtime.store.path(value["stored_path"]), media_type="video/mp4")
