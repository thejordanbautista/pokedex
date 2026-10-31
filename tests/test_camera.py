"""Webcam handling, with a fake cv2 so the tests run without camera hardware."""

import sys
import types

import numpy as np
import pytest

from pokedex import camera, config


def fake_cv2(slots):
    """slots: {index: list of frames to return, or None = won't open}. The last frame repeats."""

    class Cap:
        def __init__(self, index, backend=None):
            self.frames = slots.get(index)
            self.n = 0

        def isOpened(self):
            return self.frames is not None

        def set(self, *a):
            return True

        def read(self):
            if not self.frames:
                return False, None
            f = self.frames[min(self.n, len(self.frames) - 1)]
            self.n += 1
            return (f is not None), f

        def release(self):
            pass

    return types.SimpleNamespace(
        VideoCapture=Cap, CAP_AVFOUNDATION=1200, CAP_ANY=0,
        CAP_PROP_FRAME_WIDTH=3, CAP_PROP_FRAME_HEIGHT=4, COLOR_BGR2RGB=4,
        cvtColor=lambda img, code: img[:, :, ::-1].copy(), __version__="fake",
    )


BLACK = np.zeros((480, 640, 3), np.uint8)
GOOD = np.full((480, 640, 3), (10, 120, 200), np.uint8)  # BGR


@pytest.fixture
def cams(monkeypatch):
    def install(slots):
        monkeypatch.setitem(sys.modules, "cv2", fake_cv2(slots))
    return install


def test_warmup_skips_black_and_empty_frames(cams):
    cams({0: [None, BLACK, BLACK, GOOD]})
    cam = camera.OpenCVCamera(0, warmup_seconds=1)
    f = cam.frame()
    assert f.shape == (480, 640, 3) and tuple(f[0, 0]) == (200, 120, 10)  # converted to RGB
    cam.stop()


def test_camera_that_never_sends_a_picture_is_rejected(cams):
    cams({0: [None]})  # opens, but every read fails
    with pytest.raises(RuntimeError, match="sent no picture"):
        camera.OpenCVCamera(0, warmup_seconds=0.2)


def test_waits_for_auto_exposure_to_settle(cams):
    # Mac cameras ramp from black to normal brightness: the settled frame is the one kept.
    ramp = [np.full((480, 640, 3), v, np.uint8) for v in (1, 3, 12, 40, 80, 110, 120)] + [GOOD]
    cams({0: ramp})
    cam = camera.OpenCVCamera(0, warmup_seconds=1, settle_seconds=2)
    assert cam.brightness > 100
    cam.stop()


def test_dark_iphone_is_skipped_for_working_camera(cams, monkeypatch):
    # Your Mac: slot 0 refused, slot 1 = iPhone in a pocket (black), slot 2 = real camera.
    monkeypatch.setattr(camera.sys, "platform", "darwin")
    cams({0: None, 1: [BLACK], 2: [GOOD]})
    cam = camera.create("auto", 0, config.ROOT / "samples")
    assert cam.name == "opencv" and cam.index == 2
    assert any("webcam 1 is almost black" in p for p in cam.problems)
    cam.stop()


def test_dark_camera_still_beats_sample_pictures(cams, monkeypatch):
    monkeypatch.setattr(camera.sys, "platform", "darwin")
    cams({0: None, 1: [BLACK]})  # the only real camera is dark: use it, the user can add light
    cam = camera.create("auto", 0, config.ROOT / "samples")
    assert cam.name == "opencv" and cam.index == 1
    cam.stop()


def test_dropped_frames_keep_last_good_picture(cams):
    cams({0: [GOOD, None]})  # one good frame, then failures forever
    cam = camera.OpenCVCamera(0, warmup_seconds=1)
    import time

    time.sleep(0.1)
    assert cam.frame() is not None
    cam.stop()


def test_auto_tries_next_slot_when_first_is_dead(cams, monkeypatch):
    monkeypatch.setattr(camera.sys, "platform", "darwin")
    cams({0: None, 1: [GOOD]})  # e.g. slot 0 refused, slot 1 is the laptop camera
    cam = camera.create("auto", 0, config.ROOT / "samples")
    assert cam.name == "opencv" and cam.index == 1
    assert any("webcam 0 would not open" in p for p in cam.problems)
    assert "System Settings" in cam.problems[0]  # macOS permission hint
    cam.stop()


def test_auto_falls_back_to_samples_and_says_why(cams, monkeypatch):
    monkeypatch.setattr(camera.sys, "platform", "darwin")
    cams({})  # no webcams at all (or permission denied)
    cam = camera.create("auto", 0, config.ROOT / "samples")
    assert cam.name == "folder"
    assert cam.problems and "would not open" in cam.problems[0]
    assert not any("picamera2" in p or "Pi camera" in p for p in cam.problems)  # not tried on a Mac


def test_diagnose_reports_each_slot(cams, tmp_path):
    cams({0: None, 1: [BLACK], 2: [GOOD]})
    lines = camera.diagnose(max_index=3, out=tmp_path)
    text = "\n".join(lines)
    assert "webcam 0: webcam 0 would not open" in text
    assert "webcam 1: OK" in text and "almost black" in text and "Continuity Camera" in text
    assert "webcam 2: OK, 640x480" in text
    assert "Best slot: 2" in text
    assert (tmp_path / "webcam_2.jpg").exists()
    assert (tmp_path / "report.txt").read_text() == text + "\n"
