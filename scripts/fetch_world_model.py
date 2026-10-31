#!/usr/bin/env python3
"""Download a ready-made object recognizer for World mode (MobileNet v1, 1000 ImageNet classes).

    python scripts/fetch_world_model.py              # float model, ~17 MB, runs anywhere
    python scripts/fetch_world_model.py --quantized  # int8 model, ~4 MB, for the Pi

Writes models/world.tflite and models/world_labels.txt. It recognizes everyday
animals and objects well enough to drive World mode; it does not know Pokémon (train that one
with scripts/train_classifier.py).
"""

from __future__ import annotations

import argparse
import io
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Same MobileNet v1 (1001 classes, background first). The float model runs everywhere, including
# OpenCV's DNN reader; the int8 one is ~4x smaller and faster but needs a TFLite runtime.
QUANT_ZIP = "https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_1.0_224_quant_and_labels.zip"
FLOAT_TGZ = "https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz"


def fetch(url: str) -> bytes:
    print(f"Downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--quantized", action="store_true",
                   help="int8 model: smaller and faster on a Pi, but needs ai-edge-litert (no OpenCV fallback)")
    args = p.parse_args()

    models = ROOT / "models"
    models.mkdir(exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(fetch(QUANT_ZIP))) as z:
        names = z.namelist()
        quant = next((n for n in names if n.endswith(".tflite")), None)
        labels = next((n for n in names if n.endswith(".txt")), None)
        if not quant or not labels:
            sys.exit(f"Unexpected archive contents: {names}")
        (models / "world_labels.txt").write_bytes(z.read(labels))
        if args.quantized:
            (models / "world.tflite").write_bytes(z.read(quant))

    if not args.quantized:
        with tarfile.open(fileobj=io.BytesIO(fetch(FLOAT_TGZ)), mode="r:gz") as t:
            member = next((m for m in t.getmembers() if m.name.endswith(".tflite")), None)
            if member is None:
                sys.exit(f"No .tflite in {FLOAT_TGZ}: {t.getnames()}")
            (models / "world.tflite").write_bytes(t.extractfile(member).read())

    size = (models / "world.tflite").stat().st_size / 1e6
    n = len((models / "world_labels.txt").read_text().splitlines())
    kind = "int8" if args.quantized else "float"
    print(f"Wrote models/world.tflite ({kind}, {size:.1f} MB) and models/world_labels.txt ({n} labels)")


if __name__ == "__main__":
    main()
