"""Measure a model against labelled photos: accuracy, scan time, and a suggested threshold.

    python -m pokedex.evaluate photos/ --mode pokemon
    python -m pokedex.evaluate photos/ --model models/pokemon.tflite --labels models/pokemon_labels.txt

Use photos the model was NOT trained on (capture a fresh set, or hold some back).
Phase 3 exit criteria: accuracy >= 85% and median scan time under 1 s on the Pi.
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pokedex import classifier, config
from pokedex.camera import IMAGE_EXTS
from pokedex.dex import normalize

BACKGROUND = {"background", "nothing", "none"}


@dataclass
class Report:
    total: int = 0
    correct: int = 0
    times_ms: list[float] = field(default_factory=list)
    per_class: dict[str, list[int]] = field(default_factory=lambda: defaultdict(lambda: [0, 0]))  # [correct, total]
    confusions: Counter = field(default_factory=Counter)
    right_conf: list[float] = field(default_factory=list)  # confidence on correct answers
    wrong_conf: list[float] = field(default_factory=list)  # confidence on wrong answers

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def suggest_threshold(self) -> float | None:
        """Lowest threshold that rejects 90% of wrong answers, if that keeps most right ones."""
        if not self.wrong_conf:
            return None
        t = float(np.percentile(self.wrong_conf, 90))
        return round(min(max(t, 0.3), 0.95), 2)

    def text(self, threshold: float) -> str:
        lines = [f"Photos: {self.total}   Top-1 accuracy: {self.accuracy:.1%}"]
        if self.times_ms:
            t = sorted(self.times_ms)
            lines.append(
                f"Scan time: median {statistics.median(t):.0f} ms, "
                f"p95 {t[min(len(t) - 1, int(len(t) * 0.95))]:.0f} ms"
            )
        kept = sum(c >= threshold for c in self.right_conf)
        leaked = sum(c >= threshold for c in self.wrong_conf)
        lines.append(
            f"At threshold {threshold:.2f}: {kept}/{len(self.right_conf)} right answers shown, "
            f"{leaked}/{len(self.wrong_conf)} wrong answers shown (rest say Unknown)"
        )
        if (s := self.suggest_threshold()) is not None:
            lines.append(f"Suggested threshold: {s:.2f}")
        weakest = sorted(self.per_class.items(), key=lambda kv: kv[1][0] / kv[1][1])[:5]
        lines.append("Weakest classes: " + ", ".join(f"{k} {c}/{n}" for k, (c, n) in weakest))
        if self.confusions:
            lines.append("Top confusions: " + ", ".join(f"{a} → {b} ×{n}" for (a, b), n in self.confusions.most_common(5)))
        return "\n".join(lines)


def labelled_photos(root: Path) -> list[tuple[str, Path]]:
    items = []
    for folder in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        items += [(folder.name, f) for f in sorted(folder.iterdir()) if f.suffix.lower() in IMAGE_EXTS]
    return items


def evaluate(model: classifier.Classifier, items: list[tuple[str, Path]], aliases=None) -> Report:
    """aliases(label) -> canonical name, so ImageNet labels like 'tabby' count as 'Cat'."""
    from PIL import Image

    canon = aliases or normalize
    r = Report()
    for truth, path in items:
        frame = np.asarray(Image.open(path).convert("RGB"))
        start = time.perf_counter()
        preds = model.predict(frame, top_k=1)
        r.times_ms.append((time.perf_counter() - start) * 1000)
        guess, conf = preds[0] if preds else ("", 0.0)
        ok = canon(guess) == canon(truth) or (normalize(truth) in BACKGROUND and canon(guess) in BACKGROUND)
        r.total += 1
        r.per_class[truth][1] += 1
        if ok:
            r.correct += 1
            r.per_class[truth][0] += 1
            r.right_conf.append(conf)
        else:
            r.confusions[(truth, guess)] += 1
            r.wrong_conf.append(conf)
    return r


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Evaluate a recognition model on labelled photos")
    p.add_argument("photos", type=Path, help="folder with one sub-folder per class")
    p.add_argument("--mode", choices=["pokemon", "world"], default="pokemon")
    p.add_argument("--model", type=Path)
    p.add_argument("--labels", type=Path)
    args = p.parse_args(argv)

    cfg = config.load()
    model_path = args.model or getattr(cfg, f"{args.mode}_model")
    labels_path = args.labels or getattr(cfg, f"{args.mode}_labels")
    model = classifier.create(model_path, labels_path, [])
    if isinstance(model, classifier.MockClassifier):
        raise SystemExit(f"No usable model at {model_path}")

    from pokedex.dex import Dex

    dex = Dex.load(cfg.data_dir / f"{args.mode}.json")

    def aliases(label: str) -> str:
        e = dex.lookup(label)
        return normalize(e.name) if e else normalize(label)

    items = labelled_photos(args.photos)
    if not items:
        raise SystemExit(f"No photos found under {args.photos}/<class>/")
    model.predict(np.zeros((480, 640, 3), np.uint8))  # warm-up run, not timed
    print(evaluate(model, items, aliases).text(cfg.threshold))


if __name__ == "__main__":
    main()
