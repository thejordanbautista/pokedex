#!/usr/bin/env python3
"""End-to-end check of World mode: run the downloaded model on a photo and look it up in the dex.

    python scripts/check_world_model.py photo.jpg [--expect Dog]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from pokedex import classifier, config  # noqa: E402
from pokedex.dex import Dex  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("image", type=Path)
    p.add_argument("--expect", help="dex entry name the photo should match")
    args = p.parse_args()

    cfg = config.load()
    model = classifier.create(cfg.world_model, cfg.world_labels, [])
    if not model.available:
        sys.exit("No world model loaded: run scripts/fetch_world_model.py and pip install ai-edge-litert")
    dex = Dex.load(cfg.data_dir / "world.json")
    dex.expand_ranges(model.labels)

    frame = np.asarray(Image.open(args.image).convert("RGB"))
    model.predict(frame)  # warm-up
    t = time.perf_counter()
    preds = model.predict(frame, top_k=10)
    ms = (time.perf_counter() - t) * 1000
    entry, conf, top = dex.best(preds)
    print("Top guesses:", ", ".join(f"{l} {c:.2f}" for l, c in preds[:5]))
    print(f"Dex match: {entry.name if entry else 'none'} ({conf:.2f})  threshold {cfg.threshold}  in {ms:.0f} ms")
    if args.expect and (not entry or entry.name != args.expect or conf < cfg.threshold):
        sys.exit(f"FAIL: expected {args.expect}")
    print("OK")


if __name__ == "__main__":
    main()
