#!/usr/bin/env python3
"""Assemble the phone app into _site/ for GitHub Pages.

    python scripts/build_web.py [--labels models/world_labels.txt]

Copies web/, the shared dex data (data/world.json), the ImageNet label list (from --labels or
downloaded) and generates PNG app icons. Serve locally with:  python -m http.server -d _site
"""

from __future__ import annotations

import argparse
import io
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LABELS_ZIP = "https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_1.0_224_quant_and_labels.zip"


def icon_png(size: int, path: Path) -> None:
    """Same design as web/icon.svg, drawn with Pillow (no SVG renderer needed)."""
    from PIL import Image, ImageDraw

    k = size / 512
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, size - 1, size - 1), radius=int(112 * k), fill="#C8102E")

    def circle(cx, cy, r, **kw):
        d.ellipse((int((cx - r) * k), int((cy - r) * k), int((cx + r) * k), int((cy + r) * k)), **kw)

    circle(256, 244, 150, fill="#F7F7F2")
    circle(256, 244, 118, fill="#0E1A2B")
    circle(256, 244, 88, outline="#5EE6FF", width=max(2, int(14 * k)))
    circle(226, 212, 22, fill="#E8FBFF")
    img.save(path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--labels", type=Path, help="ImageNet labels file (default: download)")
    p.add_argument("--out", type=Path, default=ROOT / "_site")
    args = p.parse_args()

    out = args.out
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(ROOT / "web", out, ignore=shutil.ignore_patterns("test", "package.json", "node_modules"))
    shutil.copy(ROOT / "data" / "world.json", out / "world.json")

    labels = args.labels or (ROOT / "models" / "world_labels.txt")
    if labels.exists():
        shutil.copy(labels, out / "labels.txt")
    else:
        try:
            with urllib.request.urlopen(LABELS_ZIP, timeout=60) as r, zipfile.ZipFile(io.BytesIO(r.read())) as z:
                name = next(n for n in z.namelist() if n.endswith(".txt"))
                (out / "labels.txt").write_bytes(z.read(name))
        except Exception as e:  # the app falls back to the model's own class names
            print(f"warning: no labels.txt ({e}); dog/cat ranges won't apply", file=sys.stderr)

    icon_png(180, out / "icon-180.png")
    icon_png(512, out / "icon-512.png")
    (out / ".nojekyll").write_text("")
    print(f"Built {out.relative_to(ROOT)}: {sorted(p.name for p in out.iterdir())}")


if __name__ == "__main__":
    main()
