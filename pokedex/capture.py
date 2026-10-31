"""Capture training photos with the Pokédex's own camera.

    python -m pokedex.capture Pikachu                 # preview window; Space = burst of 10
    python -m pokedex.capture Pikachu --auto 60       # take 60 photos, one every 0.5 s
    python -m pokedex.capture background --auto 100   # random scenes for the "nothing here" class

Photos land in photos/<label>/ ready for scripts/train_classifier.py. Move the object, angle,
distance and lighting between shots: variety matters more than count.
"""

from __future__ import annotations

import argparse
import logging
import re
import time
from pathlib import Path

import numpy as np

from pokedex import camera, config

log = logging.getLogger("pokedex.capture")


def safe_label(label: str) -> str:
    """Folder-safe label that still matches dex lookups (case and spaces are kept)."""
    cleaned = re.sub(r"[^\w\- .'é♀♂]", "", label.strip())
    if not cleaned:
        raise ValueError(f"Label {label!r} has no usable characters")
    return cleaned


def next_index(folder: Path) -> int:
    nums = [int(m.group(1)) for p in folder.glob("*.jpg") if (m := re.search(r"_(\d+)\.jpg$", p.name))]
    return max(nums, default=0) + 1


def save_frame(frame: np.ndarray, folder: Path, label: str, index: int, size: int = 448) -> Path:
    """Center-crop to a square, downsize, save as JPEG. Training resizes again, so 448 px is plenty."""
    from PIL import Image

    h, w = frame.shape[:2]
    s = min(h, w)
    crop = frame[(h - s) // 2 : (h - s) // 2 + s, (w - s) // 2 : (w - s) // 2 + s]
    img = Image.fromarray(crop)
    if s > size:
        img = img.resize((size, size), Image.BILINEAR)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{label.replace(' ', '_')}_{index:04d}.jpg"
    img.save(path, quality=90)
    return path


def capture_auto(cam: camera.Camera, folder: Path, label: str, count: int, interval: float) -> list[Path]:
    saved, i = [], next_index(folder)
    for _ in range(count):
        frame = cam.capture()
        if frame is not None:
            saved.append(save_frame(frame, folder, label, i))
            i += 1
            log.info("%s (%d/%d)", saved[-1].name, len(saved), count)
        cam.next_sample()
        time.sleep(interval)
    return saved


def capture_interactive(cam: camera.Camera, folder: Path, label: str, burst: int, interval: float, size) -> int:
    import pygame

    pygame.init()
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption(f"Capture: {label}")
    font = pygame.font.Font(None, 32)
    clock = pygame.time.Clock()
    i, pending, last = next_index(folder), 0, 0.0
    total = i - 1
    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key in (pygame.K_q, pygame.K_ESCAPE)):
                running = False
            elif e.type == pygame.KEYDOWN and e.key in (pygame.K_SPACE, pygame.K_RETURN):
                pending += burst
        frame = cam.frame()
        now = time.monotonic()
        if pending and frame is not None and now - last >= interval:
            save_frame(cam.capture(), folder, label, i)
            i, pending, last, total = i + 1, pending - 1, now, total + 1
        if frame is not None:
            surf = pygame.surfarray.make_surface(np.ascontiguousarray(frame.swapaxes(0, 1)))
            screen.blit(pygame.transform.smoothscale(surf, size), (0, 0))
            # Show the square the model will actually see
            side = min(size)
            pygame.draw.rect(screen, (60, 200, 255), ((size[0] - side) // 2, (size[1] - side) // 2, side, side), 2)
        if pending:
            pygame.draw.circle(screen, (220, 10, 45), (size[0] - 30, 30), 14)
        msg = f"{label}: {total} photos  ·  Space = burst of {burst}  ·  Q = done"
        screen.blit(font.render(msg, True, (255, 255, 255), (0, 0, 0)), (10, size[1] - 34))
        pygame.display.flip()
        clock.tick(30)
    pygame.quit()
    return total


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Capture training photos for one class")
    p.add_argument("label", help="class name, matching the dex entry (e.g. Pikachu, background)")
    p.add_argument("--out", type=Path, default=config.ROOT / "photos")
    p.add_argument("--camera", default=None, choices=["auto", "picamera2", "opencv", "folder"])
    p.add_argument("--auto", type=int, metavar="N", help="take N photos without a window")
    p.add_argument("--burst", type=int, default=10, help="photos per Space press")
    p.add_argument("--interval", type=float, default=0.5, help="seconds between photos")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = config.load()
    label = safe_label(args.label)
    folder = args.out / label
    cam = camera.create(args.camera or cfg.camera_source, cfg.camera_index, cfg.sample_dir)
    try:
        if args.auto:
            n = len(capture_auto(cam, folder, label, args.auto, args.interval))
            print(f"Saved {n} photos to {folder}")
        else:
            total = capture_interactive(cam, folder, label, args.burst, args.interval, (cfg.width, cfg.height))
            print(f"{total} photos in {folder}")
    finally:
        cam.stop()


if __name__ == "__main__":
    main()
