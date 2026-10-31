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

    def __init__(self, index: int = 0, warmup_seconds: float = 4.0, settle_seconds: float = 2.5, size=(640, 480)):
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

        # 1) Wait for any picture at all: laptop cameras send empty frames at first.
        deadline = time.monotonic() + warmup_seconds
        first = None
        while time.monotonic() < deadline:
            ok, bgr = self.cap.read()
            if ok and bgr is not None and bgr.size:
                first = bgr
                break
            time.sleep(0.05)
        if first is None:
            self.cap.release()
            hint = f" {MAC_PERMISSION_HINT}" if sys.platform == "darwin" else ""
            raise RuntimeError(f"webcam {index} opened but sent no picture in {warmup_seconds:.0f} s.{hint}")

        # 2) Let auto-exposure settle: Mac cameras start near-black and brighten over ~1-2 s.
        #    Stop once brightness has stopped rising for a few frames.
        last, steady, end = float(first.mean()), 0, time.monotonic() + settle_seconds
        while time.monotonic() < end and steady < 6:
            ok, bgr = self.cap.read()
            if ok and bgr is not None and bgr.size:
                first, m = bgr, float(bgr.mean())
                steady = steady + 1 if abs(m - last) < 1.5 and m > 3 else 0
                last = m
            time.sleep(0.03)
        self.brightness = float(first.mean())

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


DARK = 10  # mean brightness (0-255) below this = lens covered, phone in a pocket, or lights off

CONTINUITY_HINT = (
    "If this is your iPhone (Continuity Camera), press C to try the next camera, or turn it off on "
    "the phone: Settings > General > AirPlay & Continuity > Continuity Camera."
)


def _plan(source: str, index: int) -> list[tuple[str, int]]:
    """Which cameras to try, in order. 'auto' tries the Pi camera only on Linux, then the
    preferred webcam, then the other common webcam slots, then the sample folder."""
    if source != "auto":
        return [(source, index)]
    plan: list[tuple[str, int]] = []
    if sys.platform.startswith("linux"):
        plan.append(("picamera2", 0))
    plan += [("opencv", i) for i in dict.fromkeys([index, 0, 1, 2])]
    plan.append(("folder", 0))
    return plan


def _open(kind: str, i: int, sample_dir: Path | None) -> Camera:
    if kind == "picamera2":
        return Picamera2Camera()
    if kind == "opencv":
        return OpenCVCamera(i)
    if kind == "folder":
        return FolderCamera(sample_dir or Path("samples"))
    raise ValueError(f"Unknown camera source {kind!r}")


def create(source: str, index: int = 0, sample_dir: Path | None = None) -> Camera:
    """Build a camera. The result's `problems` lists why earlier choices were skipped, so the
    UI can say "Using sample images: webcam 0 sent no picture" instead of failing silently.

    In auto mode a webcam that only sends darkness (an iPhone in a pocket via Continuity
    Camera, a closed lid) is held back: the next slot is tried, and the dark one is used only
    if no other real camera works."""
    problems: list[str] = []
    dark: Camera | None = None
    for kind, i in _plan(source, index):
        if kind == "folder" and dark is not None:
            cam = dark  # a dark real camera beats sample pictures: the user can add light
            dark = None
        else:
            try:
                cam = _open(kind, i, sample_dir)
                cam.start()
            except Exception as e:  # try the next source
                msg = str(e) if kind == "opencv" else f"{kind}: {e}"
                if kind == "picamera2" and isinstance(e, ImportError):
                    msg = "no Pi camera software"
                if any(MAC_PERMISSION_HINT in p for p in problems):
                    msg = msg.replace(" " + MAC_PERMISSION_HINT, "")  # say it once
                problems.append(msg)
                continue
            if source == "auto" and kind == "opencv" and getattr(cam, "brightness", 255) < DARK:
                problems.append(f"webcam {i} is almost black (brightness {cam.brightness:.0f}/255)")
                if dark is None:
                    dark = cam
                else:
                    cam.stop()
                continue
        if dark is not None:
            dark.stop()
        cam.problems = problems
        label = f"{cam.name} {cam.index}" if cam.name == "opencv" else cam.name
        if problems:
            log.warning("Camera: %s (skipped: %s)", label, "; ".join(problems))
        else:
            log.info("Camera: %s", label)
        return cam
    if dark is not None:
        dark.problems = problems
        return dark
    raise RuntimeError("No camera available. " + "; ".join(problems))


def mac_camera_names() -> list[str]:
    """Camera names macOS knows about (not necessarily in OpenCV's slot order)."""
    if sys.platform != "darwin":
        return []
    import json
    import subprocess

    try:
        out = subprocess.run(["system_profiler", "SPCameraDataType", "-json"],
                             capture_output=True, text=True, timeout=15).stdout
        return [c.get("_name", "?") for c in json.loads(out).get("SPCameraDataType", [])]
    except Exception:
        return []


def diagnose(max_index: int = 4, out: Path | None = None) -> list[str]:
    """Try every webcam slot, report what works, save one frame from each and a report.txt."""
    from PIL import Image

    lines = [f"Python {sys.version.split()[0]} on {sys.platform}"]
    try:
        import cv2

        lines.append(f"OpenCV {cv2.__version__}")
    except ImportError:
        return lines + ["OpenCV is not installed: pip install opencv-python-headless"]
    names = mac_camera_names()
    if names:
        lines.append("Cameras macOS sees: " + ", ".join(names))
    good = []
    for i in range(max_index):
        try:
            cam = OpenCVCamera(i, warmup_seconds=3)
        except Exception as e:
            lines.append(f"webcam {i}: {e}")
            continue
        f = cam.frame()
        cam.stop()
        b = float(f.mean())
        line = f"webcam {i}: OK, {f.shape[1]}x{f.shape[0]}, brightness {b:.0f}/255"
        if b < DARK:
            line += f"  <- almost black. {CONTINUITY_HINT}"
        else:
            good.append(i)
        if out:
            out.mkdir(parents=True, exist_ok=True)
            path = out / f"webcam_{i}.jpg"
            Image.fromarray(f).save(path)
            line += f"  [saved {path}]"
        lines.append(line)
    if good:
        lines.append(f"Best slot: {good[0]}. Run: python -m pokedex --camera-index {good[0]}")
    else:
        lines.append("No camera gave a usable picture. Check the lens, the lighting and the camera permission.")
    if out:
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.txt").write_text("\n".join(lines) + "\n")
    return lines


if __name__ == "__main__":
    # python -m pokedex.camera  -> test every webcam slot and save a snapshot from each
    logging.basicConfig(level=logging.WARNING)
    for line in diagnose(out=Path("camera-test")):
        print(line)
