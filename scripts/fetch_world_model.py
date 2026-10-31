#!/usr/bin/env python3
"""Download a ready-made object recognizer for World mode (MobileNet v1, 1000 ImageNet classes).

    python scripts/fetch_world_model.py
    pip install ai-edge-litert       # the runtime that runs .tflite models

Writes models/world.tflite (~4 MB, int8) and models/world_labels.txt. It recognizes everyday
animals and objects well enough to drive World mode; it does not know Pokémon (train that one
with scripts/train_classifier.py).
"""

from __future__ import annotations

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
URL = "https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_1.0_224_quant_and_labels.zip"


def main() -> None:
    models = ROOT / "models"
    models.mkdir(exist_ok=True)
    print(f"Downloading {URL}")
    with urllib.request.urlopen(URL, timeout=60) as r:
        data = r.read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        names = z.namelist()
        model = next((n for n in names if n.endswith(".tflite")), None)
        labels = next((n for n in names if n.endswith(".txt")), None)
        if not model or not labels:
            sys.exit(f"Unexpected archive contents: {names}")
        (models / "world.tflite").write_bytes(z.read(model))
        (models / "world_labels.txt").write_bytes(z.read(labels))
    n = len((models / "world_labels.txt").read_text().splitlines())
    print(f"Wrote models/world.tflite and models/world_labels.txt ({n} labels)")
    try:
        import ai_edge_litert  # noqa: F401
    except ImportError:
        print("Next: pip install ai-edge-litert")


if __name__ == "__main__":
    main()
