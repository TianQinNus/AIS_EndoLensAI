import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# File size is not the processing-time limit. Duration (15 min) and the
# 600-frame sample cap bound inference. 512MiB fits the current local clips.
MAX_UPLOAD = 512 * 1024**2
WARNINGS = [
    "RESEARCH_DEMO_NOT_CLINICAL",
    "POSTPROCESSING_SELECTED_ON_REUSED_C6_EXPLORATORY",
    "CANDIDATE_COUNT_IS_FRAMES_NOT_DISTINCT_POLYPS",
    "FOREGROUND_SCORE_NOT_CALIBRATED_CLINICAL_PROBABILITY",
]


def area_gate() -> float:
    value = float(os.getenv("ENDO_MIN_REGION_AREA_RATIO", "0"))
    if not math.isfinite(value) or value != 0:
        raise ValueError(
            "Remove nonzero ENDO_MIN_REGION_AREA_RATIO: this release uses "
            "confidence filtering only, with no area gate"
        )
    return value


def storage_dir() -> str:
    return os.getenv("ENDO_STORAGE_DIR", str(ROOT / "runtime_data"))


def device(requested: str | None = None) -> str:
    return requested or os.getenv("ENDO_DEVICE", "auto")


def cors_origins() -> list[str]:
    origins = [
        origin.strip()
        for origin in os.getenv(
            "ENDO_CORS_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000",
        ).split(",")
        if origin.strip()
    ]
    if "*" in origins:
        raise ValueError(
            "Wildcard CORS is disabled; set explicit frontend origins in "
            "ENDO_CORS_ORIGINS"
        )
    return origins


def cors_methods() -> list[str]:
    return ["GET", "POST", "PUT", "OPTIONS"]


def cors_headers() -> list[str]:
    return ["Accept", "Content-Type", "Range"]
