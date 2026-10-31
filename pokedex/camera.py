"""Camera sources. Every camera returns frames as RGB uint8 numpy arrays (H, W, 3)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class Camera:
    name = "base"

    def start(self) -> None: ...

    def frame(self) -> np.ndarray | None:
        """Latest preview frame, or None if not ready."""
        raise NotImplementedError

    def capture(self) -> np.ndarray | None:
        """Still used for recognition. Defaults to the preview frame."""
        return self.frame()

    def next_sample(self) -> None:
        """Advance to the next test image (folder camera only)."""

    def stop(self) -> None: ...


class Picamera2Camera(Camera):
    """Raspberry Pi camera via Picamera2 (Camera Module 3, AI Camera, etc.)."""

    name = "picamera2"

    def __init__(self, size=(640, 480)):
        from picamera2 import Picamera2  # only available on the Pi

        self.cam = Picamera2()
        config = self.cam.create_preview_configuration(main={"size": size, "format": "RGB888"})
        self.cam.configure(config)

    def start(self) -> None:
        self.cam.start()
        try:  # continuous autofocus on Camera Module 3; ignored by fixed-focus cameras
            from libcamera import controls

            self.cam.set_controls({"AfMode": controls.AfModeEnum.Continuous})
        except Exception:
            pass

    def frame(self) -> np.ndarray | None:
        # Picamera2's "RGB888" is BGR byte order in memory; flip to RGB.
        return self.cam.capture_array()[:, :, ::-1].copy()

    def stop(self) -> None:
        self.cam.stop()


class OpenCVCamera(Camera):
    """USB webcam or laptop camera."""

    name = "opencv"

    def __init__(self, index: int = 0):
        import cv2

        self.cv2 = cv2
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"No webcam at index {index}")

    def frame(self) -> np.ndarray | None:
        ok, bgr = self.cap.read()
        return self.cv2.cvtColor(bgr, self.cv2.COLOR_BGR2RGB) if ok else None

    def stop(self) -> None:
        self.cap.release()


class FolderCamera(Camera):
    """Cycles through images in a folder. For developing without camera hardware."""

    name = "folder"

    def __init__(self, folder: Path):
        from PIL import Image

        self.paths = sorted(p for p in Path(folder).glob("*") if p.suffix.lower() in IMAGE_EXTS)
        if not self.paths:
            raise RuntimeError(f"No images in {folder}")
        self._Image = Image
        self.i = 0
        self._cache: np.ndarray | None = None

    @property
    def current_path(self) -> Path:
        return self.paths[self.i]

    def frame(self) -> np.ndarray:
        if self._cache is None:
            img = self._Image.open(self.current_path).convert("RGB")
            self._cache = np.asarray(img, dtype=np.uint8)
        return self._cache

    def next_sample(self) -> None:
        self.i = (self.i + 1) % len(self.paths)
        self._cache = None


def create(source: str, index: int = 0, sample_dir: Path | None = None) -> Camera:
    """Build a camera. 'auto' tries the Pi camera, then a webcam, then the sample folder."""
    order = ["picamera2", "opencv", "folder"] if source == "auto" else [source]
    errors = []
    for kind in order:
        try:
            if kind == "picamera2":
                cam: Camera = Picamera2Camera()
            elif kind == "opencv":
                cam = OpenCVCamera(index)
            elif kind == "folder":
                cam = FolderCamera(sample_dir or Path("samples"))
            else:
                raise ValueError(f"Unknown camera source {kind!r}")
            cam.start()
            log.info("Camera: %s", cam.name)
            return cam
        except Exception as e:  # try the next source
            errors.append(f"{kind}: {e}")
    raise RuntimeError("No camera available. " + "; ".join(errors))
