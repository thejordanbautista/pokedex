import json

import numpy as np

from pokedex import camera, classifier, config
from pokedex.dex import Dex, SeenLog


def test_world_dex_loads_and_maps_imagenet_labels():
    dex = Dex.load(config.ROOT / "data/world.json")
    assert len(dex) >= 5
    assert dex.lookup("golden retriever").name == "Dog"
    assert dex.lookup("Tiger_Cat").name == "Cat"
    assert dex.lookup("toaster") is None


def test_load_labels_formats(tmp_path):
    f = tmp_path / "labels.txt"
    f.write_text("0 tabby\n1: tiger cat\nbanana\n\n")
    assert classifier.load_labels(f) == ["tabby", "tiger cat", "banana"]


def test_to_probabilities_softmax_and_dequant():
    p = classifier.to_probabilities(np.array([2.0, 1.0, 0.0]))
    assert np.isclose(p.sum(), 1) and p.argmax() == 0
    # uint8 output already holding probabilities: 0..255 * (1/256)
    q = classifier.to_probabilities(np.array([230, 26, 0], dtype=np.uint8), scale=1 / 256, zero_point=0)
    assert np.isclose(q.sum(), 1, atol=0.05) and q.argmax() == 0


def test_center_crop_resize_shape():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    assert classifier.center_crop_resize(img, (224, 224)).shape == (224, 224, 3)


def test_mock_classifier_cycles():
    m = classifier.create(config.ROOT / "models/missing.tflite", config.ROOT / "models/missing.txt", ["A", "B"])
    assert isinstance(m, classifier.MockClassifier)
    assert [m.predict(None)[0][0] for _ in range(3)] == ["A", "B", "A"]


def test_seen_log_persists(tmp_path):
    log = SeenLog(tmp_path / "seen.json")
    assert log.mark("world", "Dog") is True
    assert log.mark("world", "Dog") is False
    again = SeenLog(tmp_path / "seen.json")
    assert again.has("world", "Dog") and again.count("world") == 1
    assert json.loads((tmp_path / "seen.json").read_text())["world"]["Dog"] > 0


def test_folder_camera_reads_samples():
    cam = camera.create("folder", sample_dir=config.ROOT / "samples")
    f = cam.frame()
    assert f.ndim == 3 and f.shape[2] == 3 and f.dtype == np.uint8
    cam.next_sample()
    assert cam.frame() is not None


def test_imagenet_ranges_map_every_dog_breed():
    dex = Dex.load(config.ROOT / "data/world.json")
    # 1001-label file like MobileNet's: "background" first, then ImageNet classes 0-999
    labels = ["background"] + [f"class{i}" for i in range(1000)]
    labels[1 + 151], labels[1 + 268], labels[1 + 200] = "Chihuahua", "Mexican hairless", "Tibetan terrier"
    labels[1 + 285] = "Egyptian cat"
    assert dex.expand_ranges(labels) > 100
    assert dex.lookup("Mexican hairless").name == "Dog"
    assert dex.lookup("Tibetan terrier").name == "Dog"
    assert dex.lookup("class150") is None and dex.lookup("class269") is None  # range edges
    assert dex.lookup("class286") is None  # wild cats aren't "Cat"


def test_best_pools_confidence_across_breeds():
    dex = Dex.load(config.ROOT / "data/world.json")
    preds = [("golden retriever", 0.30), ("tennis ball", 0.28), ("labrador retriever", 0.25), ("pug", 0.10)]
    entry, conf, top = dex.best(preds)
    assert entry.name == "Dog" and abs(conf - 0.65) < 1e-9 and top == "golden retriever"
    entry, conf, top = dex.best([("toaster", 0.9)])
    assert entry is None and conf == 0.9 and top == "toaster"
