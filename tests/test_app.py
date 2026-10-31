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
