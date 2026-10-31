#!/usr/bin/env python3
"""Render every screen to PNG without a display, for checking layout.

    python scripts/screenshots.py            # writes screenshots/<size>_<screen>.png
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pygame  # noqa: E402

from pokedex import config  # noqa: E402
from pokedex.controls import Action  # noqa: E402
from pokedex.main import App  # noqa: E402


def main() -> None:
    out = config.ROOT / "screenshots"
    out.mkdir(exist_ok=True)
    data = Path(tempfile.mkdtemp())
    (data / "world.json").write_text((config.ROOT / "data/world.json").read_text())

    for w, h in [(800, 480), (480, 320)]:
        cfg = config.load()
        cfg.width, cfg.height = w, h
        cfg.camera_source, cfg.tts, cfg.data_dir = "folder", "none", data
        app = App(cfg, max_frames=1)

        def shot(name: str) -> None:
            app.draw(time.monotonic())
            path = out / f"{w}x{h}_{name}.png"
            pygame.image.save(app.surface, str(path))
            print(path.relative_to(config.ROOT))

        shot("1_live")
        app.handle(Action.SCAN)
        shot("2_scanning")
        app.job.done.wait(5)
        app.finish_scan()
        shot("3_entry")
        app.handle(Action.RIGHT)
        shot("4_entry_page2")
        app.handle(Action.B)
        app.models["world"].predict = lambda f, top_k=3: [("toaster", 0.31)]
        app.handle(Action.SCAN)
        app.job.done.wait(5)
        app.finish_scan()
        shot("5_unknown")
        app.handle(Action.B)
        app.handle(Action.RIGHT)
        shot("6_list")
        app.cam.stop()
        pygame.quit()


if __name__ == "__main__":
    main()
