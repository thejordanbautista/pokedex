"""Camera sources. Every camera returns frames as RGB uint8 numpy arrays (H, W, 3)."""

from __future__ import annotations

import logging
import sys
import threading
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class Camera:
    name = "base"
    problems: list[str] = []  # why better cameras were skipped (set by create)

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


MAC_PERMISSION_HINT = (
    "On macOS: System Settings > Privacy & Security > Camera, allow your terminal app, "
    "then quit and reopen the terminal (the permission only applies to new launches)."
)


class OpenCVCamera(Camera):
    """USB webcam or laptop camera, read on a background thread so the UI never waits on it."""

    name = "opencv"

    def __init__(self, index: int = 0, warmup_seconds: float = 4.0, size=(640, 480)):
        import cv2

        self.cv2 = cv2
        self.index = index
        backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(index, backend)
        if not self.cap.isOpened():
            hint = f" {MAC_PERMISSION_HINT}" if sys.platform == "darwin" else ""
            raise RuntimeError(f"webcam {index} would not open.{hint}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])

        # Laptop cameras return empty or black frames for the first second or so.
        deadline = time.monotonic() + warmup_seconds
        first = None
        while time.monotonic() < deadline:
            ok, bgr = self.cap.read()
            if ok and bgr is not None and bgr.size and bgr.max() > 8:
                first = bgr
                break
            time.sleep(0.05)
        if first is None:
            self.cap.release()
            hint = f" {MAC_PERMISSION_HINT}" if sys.platform == "darwin" else ""
            raise RuntimeError(f"webcam {index} opened but sent no picture in {warmup_seconds:.0f} s.{hint}")

        self._latest = cv2.cvtColor(first, cv2.COLOR_BGR2RGB)
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        misses = 0
        while self._running:
            ok, bgr = self.cap.read()
            if ok and bgr is not None:
                rgb = self.cv2.cvtColor(bgr, self.cv2.COLOR_BGR2RGB)
                with self._lock:
                    self._latest = rgb
                misses = 0
            else:
                misses += 1
                if misses == 50:
                    log.warning("Webcam %d stopped sending frames", self.index)
                time.sleep(0.02)

    def frame(self) -> np.ndarray | None:
        with self._lock:
            return self._latest  # last good frame: a dropped frame never blanks the screen

    def stop(self) -> None:
        self._running = False
        self._thread.join(timeout=1)
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


def _plan(source: str, index: int) -> list[tuple[str, int]]:
    """Which cameras to try, in order. 'auto' tries the Pi camera only on Linux, then the
    configured webcam, then the other common webcam slots (on a Mac, slot 0 can be an iPhone
    via Continuity Camera), then the sample folder."""
    if source != "auto":
        return [(source, index)]
    plan: list[tuple[str, int]] = []
    if sys.platform.startswith("linux"):
        plan.append(("picamera2", 0))
    plan += [("opencv", i) for i in dict.fromkeys([index, 0, 1, 2])]
    plan.append(("folder", 0))
    return plan


def create(source: str, index: int = 0, sample_dir: Path | None = None) -> Camera:
    """Build a camera. The result's `problems` lists why earlier choices were skipped, so the
    UI can say "Using sample images: webcam 0 sent no picture" instead of failing silently."""
    problems: list[str] = []
    for kind, i in _plan(source, index):
        try:
            if kind == "picamera2":
                cam: Camera = Picamera2Camera()
            elif kind == "opencv":
                cam = OpenCVCamera(i)
            elif kind == "folder":
                cam = FolderCamera(sample_dir or Path("samples"))
            else:
                raise ValueError(f"Unknown camera source {kind!r}")
            cam.start()
            cam.problems = problems
            label = f"{cam.name} {i}" if kind == "opencv" else cam.name
            if problems:
                log.warning("Camera: %s (skipped: %s)", label, "; ".join(problems))
            else:
                log.info("Camera: %s", label)
            return cam
        except Exception as e:  # try the next source
            msg = str(e) if kind == "opencv" else f"{kind}: {e}"
            if kind == "picamera2" and isinstance(e, ImportError):
                msg = "no Pi camera software"
            if any(MAC_PERMISSION_HINT in p for p in problems):
                msg = msg.replace(" " + MAC_PERMISSION_HINT, "")  # say it once
            problems.append(msg)
    raise RuntimeError("No camera available. " + "; ".join(problems))


def diagnose(max_index: int = 4, out: Path | None = None) -> list[str]:
    """Try every webcam slot, report what works, and save one frame from each."""
    from PIL import Image

    lines = [f"Python {sys.version.split()[0]} on {sys.platform}"]
    try:
        import cv2

        lines.append(f"OpenCV {cv2.__version__}")
    except ImportError:
        return lines + ["OpenCV is not installed: pip install opencv-python-headless"]
    found = False
    for i in range(max_index):
        try:
            cam = OpenCVCamera(i, warmup_seconds=3)
        except Exception as e:
            lines.append(f"webcam {i}: {e}")
            continue
        f = cam.frame()
        cam.stop()
        found = True
        line = f"webcam {i}: OK, {f.shape[1]}x{f.shape[0]}, brightness {f.mean():.0f}/255"
        if out:
            out.mkdir(parents=True, exist_ok=True)
            path = out / f"webcam_{i}.jpg"
            Image.fromarray(f).save(path)
            line += f", saved {path}"
        lines.append(line)
    if found:
        lines.append("Pick the slot showing your laptop camera: python -m pokedex --camera-index N")
    return lines


if __name__ == "__main__":
    # python -m pokedex.camera  -> test every webcam slot and save a snapshot from each
    logging.basicConfig(level=logging.WARNING)
    for line in diagnose(out=Path("camera-test")):
        print(line)
