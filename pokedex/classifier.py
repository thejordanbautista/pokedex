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
    """Dequantize model output if needed, and apply softmax only if it isn't already probabilities.

    Quantized softmax outputs only roughly sum to 1 (each of ~1000 values is rounded to a
    multiple of 1/256), so "already probabilities" means: all in [0, 1] with a sum near 1.
    Those are renormalized, never softmaxed again: a second softmax flattens everything to ~1/N."""
    out = raw.astype(np.float32).reshape(-1)
    if scale:
        out = (out - zero_point) * scale
    total = float(out.sum())
    if out.min() >= 0 and out.max() <= 1.0001 and 0.5 <= total <= 1.5:
        return out / total
    e = np.exp(out - out.max())
    return e / e.sum()


class Classifier:
    available = False

    def predict(self, frame: np.ndarray, top_k: int = 3) -> Prediction:
        raise NotImplementedError


class TFLiteClassifier(Classifier):
    available = True
    backend = "litert"

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


class OpenCVTFLiteClassifier(Classifier):
    """Fallback that runs a .tflite model with OpenCV's DNN module (OpenCV 4.8+), for machines
    where no TFLite runtime installs cleanly (e.g. ai-edge-litert on macOS + Python 3.9)."""

    available = True
    backend = "opencv-dnn"

    def __init__(self, model_path: Path, labels_path: Path, size: int = 224):
        import cv2

        if not hasattr(cv2.dnn, "readNetFromTFLite"):
            raise ImportError(f"OpenCV {cv2.__version__} can't read .tflite (needs 4.8+)")
        self.cv2 = cv2
        self.net = cv2.dnn.readNetFromTFLite(str(model_path))
        self.w = self.h = size
        self.labels = load_labels(labels_path)
        # Same field names as TFLiteClassifier so diagnostics can print them
        self.inp = {"dtype": np.uint8, "quantization": "handled by OpenCV"}
        self.out = {"quantization": "handled by OpenCV"}
        log.info("Loaded %s with OpenCV DNN %s", model_path.name, cv2.__version__)

    def predict(self, frame: np.ndarray, top_k: int = 3) -> Prediction:
        img = center_crop_resize(frame, (self.w, self.h))
        blob = self.cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(self.w, self.h), swapRB=False)
        self.net.setInput(blob)
        raw = np.asarray(self.net.forward())
        probs = to_probabilities(raw)
        if len(probs) == len(self.labels) + 1:
            self.labels = ["background", *self.labels]
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
    tried = []
    for mod, attr in (
        ("ai_edge_litert.interpreter", "Interpreter"),
        ("tflite_runtime.interpreter", "Interpreter"),
        ("tensorflow.lite", "Interpreter"),
    ):
        try:
            module = __import__(mod, fromlist=[attr])
            return getattr(module, attr)
        except Exception as e:  # ImportError, or a broken install raising something else
            tried.append(f"{mod.split('.')[0]}: {type(e).__name__}: {e}")
    raise ImportError("No TFLite runtime could be loaded (pip install ai-edge-litert). Tried: " + " | ".join(tried))


LOAD_ERRORS: dict[str, str] = {}  # model path -> why it failed to load, for diagnostics


def create(model_path: Path, labels_path: Path, fallback_labels: list[str]) -> Classifier:
    if Path(model_path).exists() and Path(labels_path).exists():
        errors = []
        for cls in (TFLiteClassifier, OpenCVTFLiteClassifier):
            try:
                return cls(Path(model_path), Path(labels_path))
            except Exception as e:
                errors.append(f"{cls.backend}: {type(e).__name__}: {e}")
        LOAD_ERRORS[str(model_path)] = " || ".join(errors)
        log.warning("Could not load %s (%s); using mock classifier", model_path, LOAD_ERRORS[str(model_path)])
    else:
        log.info("No model at %s; using mock classifier", model_path)
    return MockClassifier(fallback_labels)
