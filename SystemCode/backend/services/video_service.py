from pathlib import Path

from ..video import convert_to_h264_mp4, decoded, probe, schedule


class VideoService:
    """Owns video probing, PTS decoding, and sampling decisions."""

    probe = staticmethod(probe)
    decoded = staticmethod(decoded)
    schedule = staticmethod(schedule)
    convert = staticmethod(convert_to_h264_mp4)

    @staticmethod
    def filename(value: str | None) -> str:
        return Path((value or "upload.mp4").replace("\\", "/")).name[:120]
