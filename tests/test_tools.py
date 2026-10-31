import pytest
from PIL import Image

from pokedex import camera, capture, config, evaluate
from pokedex.classifier import Classifier


def test_safe_label_keeps_dex_names():
    assert capture.safe_label("Mr. Mime") == "Mr. Mime"
    assert capture.safe_label("Nidoran♀") == "Nidoran♀"
    assert capture.safe_label("../evil") == "..evil"
    with pytest.raises(ValueError):
        capture.safe_label("///")


def test_auto_capture_numbers_and_crops(tmp_path):
    cam = camera.create("folder", sample_dir=config.ROOT / "samples")
    saved = capture.capture_auto(cam, tmp_path / "Pikachu", "Pikachu", count=3, interval=0)
    assert [p.name for p in saved] == ["Pikachu_0001.jpg", "Pikachu_0002.jpg", "Pikachu_0003.jpg"]
    assert Image.open(saved[0]).size == (448, 448)  # samples are 640x480 -> 480 square -> 448
    more = capture.capture_auto(cam, tmp_path / "Pikachu", "Pikachu", count=1, interval=0)
    assert more[0].name == "Pikachu_0004.jpg"  # continues numbering, never overwrites


class ColorModel(Classifier):
    """Predicts 'Red' for reddish images, 'Blue' otherwise, with a fixed delay."""

    def predict(self, frame, top_k=3):
        r, _, b = frame.reshape(-1, 3).mean(0)
        return [("Red", 0.9)] if r > b else [("Blue", 0.55)]


def _photos(root, name, color, n):
    d = root / name
    d.mkdir(parents=True)
    for i in range(n):
        Image.new("RGB", (64, 64), color).save(d / f"{i}.jpg")


def test_evaluate_reports_accuracy_and_confusions(tmp_path):
    _photos(tmp_path, "Red", (220, 20, 20), 4)
    _photos(tmp_path, "Blue", (20, 20, 220), 3)
    _photos(tmp_path, "Purple", (200, 20, 210), 2)  # model will wrongly call these Blue
    items = evaluate.labelled_photos(tmp_path)
    r = evaluate.evaluate(ColorModel(), items)
    assert (r.total, r.correct) == (9, 7)
    assert r.confusions[("Purple", "Blue")] == 2
    assert len(r.times_ms) == 9
    report = r.text(threshold=0.6)
    assert "77.8%" in report and "Purple → Blue ×2" in report
    assert r.suggest_threshold() == 0.55
