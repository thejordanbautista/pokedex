#!/usr/bin/env python3
"""Turn a photo into a short .y4m video for Chrome's fake camera (browser tests).

    python scripts/make_fake_video.py photo.jpg out.y4m
"""

import sys

import numpy as np
from PIL import Image


def main() -> None:
    src, dst = sys.argv[1], sys.argv[2]
    img = Image.open(src).convert("RGB")
    w, h = 640, 480
    s = min(img.width / w, img.height / h)
    cw, ch = int(w * s), int(h * s)
    img = img.crop(((img.width - cw) // 2, (img.height - ch) // 2, (img.width + cw) // 2, (img.height + ch) // 2)).resize((w, h))
    rgb = np.asarray(img).astype(np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 0.257 * r + 0.504 * g + 0.098 * b + 16
    u = -0.148 * r - 0.291 * g + 0.439 * b + 128
    v = 0.439 * r - 0.368 * g - 0.071 * b + 128
    sub = lambda c: c.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))  # noqa: E731  4:2:0
    frame = b"".join(np.clip(c, 0, 255).astype(np.uint8).tobytes() for c in (y, sub(u), sub(v)))
    with open(dst, "wb") as f:
        f.write(f"YUV4MPEG2 W{w} H{h} F30:1 Ip A1:1 C420jpeg\n".encode())
        for _ in range(30):
            f.write(b"FRAME\n" + frame)


if __name__ == "__main__":
    main()
