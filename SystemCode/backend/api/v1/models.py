from fastapi import APIRouter, Request

from ...core.config import WARNINGS
from ...endolens_model import POSTPROCESS_VERSION
from ..dependencies import get_runtime

router = APIRouter()


@router.get("/models")
def models(request: Request):
    runtime = get_runtime(request)
    return {
        "items": [runtime.adapter.describe()],
        "warnings": WARNINGS,
        "postprocessing": {
            "segmentation_threshold": 0.5,
            "min_region_area_ratio": 0.0,
            "min_region_mean_score": 0.8,
            "mask_filtering": True,
            "postprocessing_version": POSTPROCESS_VERSION,
        },
    }
