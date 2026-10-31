"""Verify that the maintained application uses the same final E13 SDK release."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT.parents[1] / "backend"


def verify():
    files = ["__init__.py", "adapter.py", "network.py", "postprocess.py", "constants.py"]
    for name in files:
        expected = (ROOT / "endolens_model" / name).read_text(encoding="utf-8")
        actual = (BACKEND / "endolens_model" / name).read_text(encoding="utf-8")
        if expected != actual:
            raise ValueError("Backend/SDK source differs: " + name)
    model = json.loads((ROOT / "model.json").read_text(encoding="utf-8"))
    if json.loads((BACKEND / "model.json").read_text(encoding="utf-8")) != model:
        raise ValueError("Backend/SDK model configuration differs")
    fixture = "examples/synthetic_probability.npy"
    if hashlib.sha256((ROOT / fixture).read_bytes()).digest() != hashlib.sha256((BACKEND / fixture).read_bytes()).digest():
        raise ValueError("Synthetic regression fixtures differ")
    print("BACKEND_SDK_RELEASE_MATCH", model["model_version"], "source_files", len(files))


if __name__ == "__main__":
    verify()
