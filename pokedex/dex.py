"""Dex entries and the seen/caught log."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class Entry:
    number: int
    name: str
    category: str = ""
    types: list[str] = field(default_factory=list)
    height_m: float | None = None
    weight_kg: float | None = None
    description: str = ""
    evolutions: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)  # model labels that map to this entry
    sprite: str | None = None  # path relative to the assets dir
    cry: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "Entry":
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in known})


def normalize(label: str) -> str:
    return label.strip().lower().replace("_", " ").replace("-", " ")


class Dex:
    """One mode's set of entries, indexed by every label that maps to them."""

    def __init__(self, entries: list[Entry]):
        self.entries = sorted(entries, key=lambda e: e.number)
        self._by_label: dict[str, Entry] = {}
        for e in self.entries:
            for label in [e.name, *e.labels]:
                self._by_label[normalize(label)] = e

    @classmethod
    def load(cls, path: Path) -> "Dex":
        if not Path(path).exists():
            log.info("No dex file at %s", path)
            return cls([])
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([Entry.from_dict(d) for d in raw.get("entries", raw)])

    def lookup(self, label: str) -> Entry | None:
        return self._by_label.get(normalize(label))

    def __len__(self) -> int:
        return len(self.entries)

    def labels(self) -> list[str]:
        return [e.name for e in self.entries]


class SeenLog:
    """Remembers what has been scanned, per mode, across reboots."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.data: dict[str, dict[str, float]] = {}
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text())
            except json.JSONDecodeError:
                log.warning("Ignoring corrupt %s", self.path)

    def mark(self, mode: str, name: str) -> bool:
        """Record a sighting. Returns True if it's new."""
        seen = self.data.setdefault(mode, {})
        new = name not in seen
        seen.setdefault(name, time.time())
        self.save()
        return new

    def count(self, mode: str) -> int:
        return len(self.data.get(mode, {}))

    def has(self, mode: str, name: str) -> bool:
        return name in self.data.get(mode, {})

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2))
        tmp.replace(self.path)  # atomic, so a power cut can't leave half a file
