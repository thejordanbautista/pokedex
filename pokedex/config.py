"""Settings loaded from config.toml, with command-line overrides."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # Python 3.9-3.10 (e.g. macOS's built-in python3)
    import tomli as tomllib

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Pins:
    up: int = 5
    down: int = 6
    left: int = 13
    right: int = 26
    a: int = 16
    b: int = 17
    scan: int = 27
    leds: int = 22


@dataclass
class Config:
    width: int = 800
    height: int = 480
    fullscreen: bool = False

    camera_source: str = "auto"
    camera_index: int = 0
    sample_dir: Path = ROOT / "samples"

    mode: str = "pokemon"
    threshold: float = 0.6
    pokemon_model: Path = ROOT / "models/pokemon.tflite"
    pokemon_labels: Path = ROOT / "models/pokemon_labels.txt"
    world_model: Path = ROOT / "models/world.tflite"
    world_labels: Path = ROOT / "models/world_labels.txt"

    tts: str = "auto"
    piper_voice: str = ""
    volume: float = 0.8

    gpio: bool = False
    pins: Pins = field(default_factory=Pins)
    shutdown_hold_seconds: float = 3.0

    data_dir: Path = ROOT / "data"
    assets_dir: Path = ROOT / "assets"


def _path(value: str) -> Path:
    p = Path(value).expanduser()
    return p if p.is_absolute() else ROOT / p


def load(path: Path | None = None) -> Config:
    """Read config.toml (if present) into a Config."""
    cfg = Config()
    path = path or ROOT / "config.toml"
    if not path.exists():
        return cfg
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    d = raw.get("display", {})
    cfg.width = d.get("width", cfg.width)
    cfg.height = d.get("height", cfg.height)
    cfg.fullscreen = d.get("fullscreen", cfg.fullscreen)

    c = raw.get("camera", {})
    cfg.camera_source = c.get("source", cfg.camera_source)
    cfg.camera_index = c.get("index", cfg.camera_index)
    if "sample_dir" in c:
        cfg.sample_dir = _path(c["sample_dir"])

    r = raw.get("recognition", {})
    cfg.mode = r.get("mode", cfg.mode)
    cfg.threshold = r.get("threshold", cfg.threshold)
    for key in ("pokemon_model", "pokemon_labels", "world_model", "world_labels"):
        if key in r:
            setattr(cfg, key, _path(r[key]))

    a = raw.get("audio", {})
    cfg.tts = a.get("tts", cfg.tts)
    cfg.piper_voice = a.get("piper_voice", cfg.piper_voice)
    cfg.volume = a.get("volume", cfg.volume)

    g = raw.get("gpio", {})
    cfg.gpio = g.get("enabled", cfg.gpio)
    cfg.shutdown_hold_seconds = g.get("shutdown_hold_seconds", cfg.shutdown_hold_seconds)
    for name in Pins.__dataclass_fields__:
        if name in g:
            setattr(cfg.pins, name, g[name])

    return cfg
