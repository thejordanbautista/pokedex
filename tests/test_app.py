import pytest

pygame = pytest.importorskip("pygame")

from pokedex import config, controls  # noqa: E402
from pokedex.dex import Entry  # noqa: E402
from pokedex.main import App  # noqa: E402


@pytest.mark.parametrize("size", [(800, 480), (480, 320)])
def test_app_scans_and_shows_entry(tmp_path, size):
    cfg = config.load()
    cfg.width, cfg.height = size
    cfg.camera_source = "folder"
    cfg.tts = "none"
    cfg.mode = "world"
    cfg.data_dir = tmp_path
    (tmp_path / "world.json").write_text((config.ROOT / "data/world.json").read_text())

    app = App(cfg, max_frames=45)  # 1.5 s at 30 fps: long enough for one scan
    controls.post(controls.Action.SCAN)
    app.run()

    assert app.state == "result"
    entry, conf, new = app.result
    assert isinstance(entry, Entry) and new
    assert app.seen.has("world", entry.name)


def test_navigation_without_crash(tmp_path):
    cfg = config.load()
    cfg.camera_source, cfg.tts, cfg.mode, cfg.data_dir = "folder", "none", "world", tmp_path
    (tmp_path / "world.json").write_text((config.ROOT / "data/world.json").read_text())
    app = App(cfg, max_frames=1)
    A = controls.Action
    for a in [A.RIGHT, A.DOWN, A.DOWN, A.A, A.B, A.MODE, A.MODE, A.UP, A.SCAN]:
        assert app.handle(a)
        app.draw(0.0)
    assert app.handle(A.QUIT) is False
    pygame.quit()


def test_camera_choice_is_remembered(tmp_path, monkeypatch):
    import sys

    sys.path.insert(0, str(config.ROOT / "tests"))
    import test_camera as tc
    from pokedex import camera

    monkeypatch.setitem(sys.modules, "cv2", tc.fake_cv2({0: [tc.GOOD], 1: [tc.GOOD]}))
    cfg = config.load()
    cfg.tts, cfg.data_dir = "none", tmp_path
    app = App(cfg, max_frames=1)
    assert app.cam.index == 0
    app.handle(controls.Action.CAMERA)  # user presses C: moves to slot 1, saved
    assert app.cam.index == 1
    app.cam.stop()
    pygame.quit()

    again = App(cfg, max_frames=1)  # next launch starts on slot 1
    assert again.cam.index == 1
    again.cam.stop()
    pygame.quit()
