"""Image classifiers. A classifier takes an RGB frame and returns ranked (label, confidence)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

Prediction = list[tuple[str, float]]


def load_labels(path: Path) -> list[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    labels = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Accept "0 label", "0: label" or plain "label"
        head, _, rest = line.partition(" ")
        if head.rstrip(":").isdigit() and rest:
            line = rest.strip()
        labels.append(line)
    return labels


def center_crop_resize(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Square center crop, then resize to (width, height)."""
    from PIL import Image

    h, w = frame.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    crop = frame[y0 : y0 + s, x0 : x0 + s]
    return np.asarray(Image.fromarray(crop).resize(size, Image.BILINEAR))


def to_probabilities(raw: np.ndarray, scale: float = 0.0, zero_point: int = 0) -> np.ndarray:
    """Dequantize model output if needed, and apply softmax if it isn't already probabilities."""
    out = raw.astype(np.float32).reshape(-1)
    if scale:
        out = (out - zero_point) * scale
    if out.min() < 0 or out.max() > 1.0001 or not np.isclose(out.sum(), 1.0, atol=0.05):
        e = np.exp(out - out.max())
        out = e / e.sum()
    return out


class Classifier:
    available = False

    def predict(self, frame: np.ndarray, top_k: int = 3) -> Prediction:
        raise NotImplementedError


class TFLiteClassifier(Classifier):
    available = True

    def __init__(self, model_path: Path, labels_path: Path, threads: int = 4):
        Interpreter = _interpreter_class()
        self.interp = Interpreter(model_path=str(model_path), num_threads=threads)
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]
        _, self.h, self.w, _ = self.inp["shape"]
        self.labels = load_labels(labels_path)
        n_out = int(np.prod(self.out["shape"]))
        if n_out != len(self.labels):
            # Some ImageNet models have an extra "background" class at index 0
            if n_out == len(self.labels) + 1:
                self.labels = ["background", *self.labels]
            else:
                log.warning("Model has %d outputs but %d labels", n_out, len(self.labels))
        log.info("Loaded %s (%dx%d, %s)", model_path.name, self.w, self.h, self.inp["dtype"].__name__)

    def _prepare(self, frame: np.ndarray) -> np.ndarray:
        img = center_crop_resize(frame, (self.w, self.h))
        dtype = self.inp["dtype"]
        if dtype == np.float32:
            # Our training script bakes [-1, 1] rescaling into the model; most float models expect it too.
            x = img.astype(np.float32) / 127.5 - 1.0
        else:
            scale, zero = self.inp.get("quantization", (0.0, 0))
            if scale and dtype == np.int8:
                x = np.clip(np.round(img / 255.0 / scale + zero), -128, 127).astype(np.int8)
            else:
                x = img.astype(dtype)
        return x[np.newaxis, ...]

    def predict(self, frame: np.ndarray, top_k: int = 3) -> Prediction:
        self.interp.set_tensor(self.inp["index"], self._prepare(frame))
        self.interp.invoke()
        raw = self.interp.get_tensor(self.out["index"])
        scale, zero = self.out.get("quantization", (0.0, 0))
        probs = to_probabilities(raw, scale, zero)
        idx = np.argsort(probs)[::-1][:top_k]
        return [(self.labels[i] if i < len(self.labels) else str(i), float(probs[i])) for i in idx]


class MockClassifier(Classifier):
    """Stands in until a model exists: returns dex entries in turn so the UI can be developed."""

    def __init__(self, labels: list[str]):
        self.labels = labels or ["unknown"]
        self.i = -1

    def predict(self, frame: np.ndarray, top_k: int = 3) -> Prediction:
        self.i = (self.i + 1) % len(self.labels)
        return [(self.labels[self.i], 0.9)]


def _interpreter_class():
    for mod, attr in (
        ("ai_edge_litert.interpreter", "Interpreter"),
        ("tflite_runtime.interpreter", "Interpreter"),
        ("tensorflow.lite", "Interpreter"),
    ):
        try:
            module = __import__(mod, fromlist=[attr])
            return getattr(module, attr)
        except ImportError:
            continue
    raise ImportError("Install ai-edge-litert (pip install ai-edge-litert) to run .tflite models")


def create(model_path: Path, labels_path: Path, fallback_labels: list[str]) -> Classifier:
    if Path(model_path).exists() and Path(labels_path).exists():
        try:
            return TFLiteClassifier(Path(model_path), Path(labels_path))
        except Exception as e:
            log.warning("Could not load %s (%s); using mock classifier", model_path, e)
    else:
        log.info("No model at %s; using mock classifier", model_path)
    return MockClassifier(fallback_labels)
