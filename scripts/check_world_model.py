#!/usr/bin/env python3
"""End-to-end check of World mode: run the downloaded model on a photo and look it up in the dex.

    python scripts/check_world_model.py photo.jpg [--expect Dog]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import traceback
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from pokedex import classifier, config  # noqa: E402
from pokedex.dex import Dex  # noqa: E402


IN_CI = os.environ.get("GITHUB_ACTIONS") == "true"


def report(level: str, msg: str) -> None:
    """Print, and in GitHub Actions also raise an annotation (readable via the API)."""
    print(msg)
    if IN_CI:
        print(f"::{level}::" + msg.replace("\n", "%0A"))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        run()
    except SystemExit:
        raise
    except Exception:
        report("error", "Crashed:\n" + traceback.format_exc()[-1500:])
        sys.exit(1)


def run() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("image", type=Path)
    p.add_argument("--expect", help="dex entry name the photo should match")
    args = p.parse_args()

    cfg = config.load()
    model = classifier.create(cfg.world_model, cfg.world_labels, [])
    if not model.available:
        report("error", "No world model loaded (see the warning above for why)")
        sys.exit(1)
    report("notice", f"Model input {model.w}x{model.h} {model.inp['dtype'].__name__} q={model.inp.get('quantization')}, "
           f"output q={model.out.get('quantization')}, {len(model.labels)} labels, first={model.labels[:2]}")
    dex = Dex.load(cfg.data_dir / "world.json")
    dex.expand_ranges(model.labels)

    frame = np.asarray(Image.open(args.image).convert("RGB"))
    model.predict(frame)  # warm-up
    t = time.perf_counter()
    preds = model.predict(frame, top_k=10)
    ms = (time.perf_counter() - t) * 1000
    entry, conf, top = dex.best(preds)
    summary = (f"Top guesses: {', '.join(f'{l} {c:.2f}' for l, c in preds[:5])} | "
               f"Dex match: {entry.name if entry else 'none'} ({conf:.2f}), threshold {cfg.threshold}, {ms:.0f} ms")
    if args.expect and (not entry or entry.name != args.expect or conf < cfg.threshold):
        report("error", f"FAIL: expected {args.expect}. {summary}")
        sys.exit(1)
    report("notice", f"OK. {summary}")


if __name__ == "__main__":
    main()
