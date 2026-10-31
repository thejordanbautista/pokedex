"""The app: a small state machine over LIVE -> SCANNING -> RESULT / UNKNOWN, plus a dex LIST."""

from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from dataclasses import dataclass, field

import pygame

from pokedex import audio, camera, classifier, config, controls, imaging
from pokedex.controls import Action
from pokedex.dex import Dex, Entry, SeenLog
from pokedex.ui import UI

log = logging.getLogger("pokedex")

MIN_SCAN_SECONDS = 1.0  # keep the scan animation on screen at least this long


@dataclass
class ScanJob:
    frame: object
    started: float
    result: list | None = None
    error: str | None = None
    done: threading.Event = field(default_factory=threading.Event)


class App:
    def __init__(self, cfg: config.Config, max_frames: int | None = None):
        self.cfg = cfg
        self.max_frames = max_frames  # for headless smoke tests
        flags = pygame.FULLSCREEN if cfg.fullscreen else 0
        pygame.init()
        pygame.display.set_caption("Pokédex")
        self.surface = pygame.display.set_mode((cfg.width, cfg.height), flags)
        if cfg.fullscreen:
            pygame.mouse.set_visible(False)
        self.ui = UI(self.surface, cfg.assets_dir)

        self.camera_file = cfg.data_dir / "camera.json"
        index = cfg.camera_index
        if cfg.camera_source == "auto" and not cfg.camera_index_set:
            index = self._remembered_camera(index)
        self.cam = camera.create(cfg.camera_source, index, cfg.sample_dir)
        self.audio = audio.Audio(cfg.tts, cfg.piper_voice, cfg.volume)
        self.hw = controls.NoLights()
        if cfg.gpio:
            try:
                self.hw = controls.GPIOButtons(cfg.pins, cfg.shutdown_hold_seconds)
            except Exception as e:
                log.warning("GPIO unavailable (%s); keyboard only", e)

        self.dexes = {
            "pokemon": Dex.load(cfg.data_dir / "pokemon.json"),
            "world": Dex.load(cfg.data_dir / "world.json"),
        }
        self.models = {
            "pokemon": classifier.create(cfg.pokemon_model, cfg.pokemon_labels, self.dexes["pokemon"].labels()),
            "world": classifier.create(cfg.world_model, cfg.world_labels, self.dexes["world"].labels()),
        }
        for mode, model in self.models.items():
            added = self.dexes[mode].expand_ranges(getattr(model, "labels", []) if model.available else [])
            if added:
                log.info("%s dex: %d extra model labels mapped by range", mode, added)
        self.seen = SeenLog(cfg.data_dir / "seen.json")

        self.mode = cfg.mode if cfg.mode in self.dexes else "pokemon"
        if not len(self.dexes[self.mode]) and len(self.dexes["world"]):
            log.info("No %s data yet; starting in World mode", self.mode)
            self.mode = "world"
        self.state = "live"
        self.job: ScanJob | None = None
        self.result: tuple[Entry, float | None, bool] | None = None
        self.unknown: tuple[str | None, float | None] = (None, None)
        self.page = 0
        self.cursor = 0
        self.still = None  # frame frozen at scan time
        self.toast, self.toast_until = "", 0.0

    def _remembered_camera(self, default: int) -> int:
        try:
            return int(json.loads(self.camera_file.read_text())["index"])
        except Exception:
            return default

    def _remember_camera(self) -> None:
        if self.cam.name != "opencv":
            return
        try:
            self.camera_file.parent.mkdir(parents=True, exist_ok=True)
            self.camera_file.write_text(json.dumps({"index": self.cam.index}))
        except OSError as e:
            log.warning("Could not save camera choice: %s", e)

    @property
    def dex(self) -> Dex:
        return self.dexes[self.mode]

    # ---------- scanning ----------
    def start_scan(self) -> None:
        frame = self.cam.capture()
        if frame is None:
            return
        frame = imaging.boost_low_light(frame)
        self.still = frame
        self.hw.lights(True)
        self.job = ScanJob(frame=frame, started=time.monotonic())
        threading.Thread(target=self._run_model, args=(self.job,), daemon=True).start()
        self.state = "scanning"

    def _run_model(self, job: ScanJob) -> None:
        try:
            job.result = self.models[self.mode].predict(job.frame, top_k=10)
        except Exception as e:
            log.exception("Recognition failed")
            job.error = str(e)
        job.done.set()

    def finish_scan(self) -> None:
        job, self.job = self.job, None
        self.hw.lights(False)
        preds = job.result or []
        entry, conf, top = self.dex.best(preds)
        log.info("Scan: %s", ", ".join(f"{l} {c:.2f}" for l, c in preds[:5]) or "no result")
        if entry and conf >= self.cfg.threshold:
            self.show_entry(entry, conf)
        else:
            self.unknown = (top, preds[0][1] if preds else None)
            self.state = "unknown"
            self.audio.say("No data. Please try again.")

    def show_entry(self, entry: Entry, conf: float | None) -> None:
        new = self.seen.mark(self.mode, entry.name)
        self.result = (entry, conf, new)
        self.page = 0
        self.state = "result"
        if entry.cry:
            self.audio.play(self.cfg.assets_dir / entry.cry)
        intro = f"{entry.name}. {entry.category}." if entry.category else f"{entry.name}."
        self.audio.say(f"{intro} {entry.description}", delay=1.2 if entry.cry else 0.0)

    # ---------- input ----------
    def handle(self, action: Action) -> bool:
        """Returns False to quit."""
        log.info("Button: %s (screen: %s)", action.value, self.state)
        if action == Action.QUIT:
            return False
        s = self.state
        if action == Action.CAMERA and s == "live":
            self.switch_camera()
        elif action == Action.MODE and s in ("live", "list"):
            self.mode = "world" if self.mode == "pokemon" else "pokemon"
            self.cursor = 0
        elif action == Action.SCAN and s in ("live", "unknown", "result"):
            self.audio.stop()
            self.start_scan()
        elif action == Action.B and s in ("result", "unknown", "list"):
            self.audio.stop()
            self.state = "live"
        elif s == "live" and action in (Action.LEFT, Action.RIGHT, Action.A):
            if len(self.dex):
                self.state = "list"
            else:
                self.cam.next_sample()
        elif s == "live" and action in (Action.UP, Action.DOWN):
            self.cam.next_sample()  # folder camera: flip through test images
        elif s == "result" and action in (Action.LEFT, Action.RIGHT):
            self.page = 1 - self.page
        elif s == "list" and len(self.dex):
            if action == Action.UP:
                self.cursor = (self.cursor - 1) % len(self.dex)
            elif action == Action.DOWN:
                self.cursor = (self.cursor + 1) % len(self.dex)
            elif action == Action.A:
                e = self.dex.entries[self.cursor]
                if self.seen.has(self.mode, e.name):
                    self.still = None
                    self.result = (e, None, False)
                    self.page = 0
                    self.state = "result"
        return True

    def switch_camera(self) -> None:
        """Try the next webcam slot (laptop camera, iPhone, USB), wrapping back to the samples."""
        current = getattr(self.cam, "index", -1)
        self.cam.stop()
        for i in [*range(current + 1, 4), "folder"]:
            try:
                if i == "folder":
                    self.cam = camera.create("folder", sample_dir=self.cfg.sample_dir)
                else:
                    self.cam = camera.create("opencv", i)
                break
            except Exception as e:
                log.info("Camera %s unavailable: %s", i, e)
        self._remember_camera()
        self.toast = f"Camera: {self.camera_label()} (press C again for the next one)"
        self.toast_until = time.monotonic() + 3.0
        log.info("Switched to %s", self.camera_label())

    def camera_label(self) -> str:
        if self.cam.name == "opencv":
            f = self.cam.frame()
            shape = f"{f.shape[1]}x{f.shape[0]}" if f is not None else "?"
            return f"webcam {self.cam.index} ({shape})"
        if self.cam.name == "folder":
            return "sample images"
        return "Pi camera"

    def camera_note(self) -> str:
        if time.monotonic() < self.toast_until:
            return self.toast
        if self.cam.name == "folder":
            why = self.cam.problems[0] if self.cam.problems else "chosen in settings"
            return f"Using sample images (UP/DOWN to flip). Webcam: {why}"
        return ""

    def light_note(self, frame) -> str:
        if frame is None or self.cam.name == "folder":
            return ""
        b = imaging.brightness(frame)
        if b < camera.DARK:
            return (f"{self.camera_label().capitalize()} is almost black. If it's your iPhone, "
                    "press C for the next camera. Otherwise check the lens and add light.")
        if b < imaging.DIM:
            return "Low light: brightened automatically. Add light for better scans."
        return ""

    # ---------- loop ----------
    def draw(self, t: float) -> None:
        self.ui.begin_frame()
        self.ui.chrome(self.mode, self.seen.count(self.mode), len(self.dex), t, lights=self.state == "scanning")
        if self.state == "live":
            raw = self.cam.frame()
            notes = [self.camera_note() or self.light_note(raw)]
            if not len(self.dex):
                notes.append("No dex data yet: run scripts/fetch_pokeapi.py")
            frame = imaging.boost_low_light(raw) if raw is not None else None
            self.ui.live(frame, "  ".join(n for n in notes if n))
        elif self.state == "scanning":
            self.ui.scanning(self.still, t)
        elif self.state == "result":
            e, conf, new = self.result
            self.ui.entry(e, self.still, conf, new, self.page)
        elif self.state == "unknown":
            self.ui.unknown(self.still, *self.unknown)
        elif self.state == "list":
            self.ui.listing(self.dex.entries, self.cursor, lambda n: self.seen.has(self.mode, n))
        pygame.display.flip()

    def run(self) -> None:
        clock = pygame.time.Clock()
        frames, running = 0, True
        try:
            while running:
                for event in pygame.event.get():
                    action = controls.from_event(event)
                    if action is None and event.type == pygame.MOUSEBUTTONDOWN and getattr(event, "button", 1) == 1:
                        name = self.ui.hit_test(event.pos)
                        action = Action(name) if name else None
                    if action and not self.handle(action):
                        running = False
                if self.job and self.job.done.is_set() and time.monotonic() - self.job.started >= MIN_SCAN_SECONDS:
                    self.finish_scan()
                self.draw(time.monotonic())
                clock.tick(30)
                frames += 1
                if self.max_frames and frames >= self.max_frames:
                    break
        finally:
            self.audio.stop()
            self.cam.stop()
            pygame.quit()


def setup_logging(cfg: config.Config, verbose: bool = False) -> None:
    """Log to the terminal and to pokedex.log (last run only), so problems can be read back later."""
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    try:
        handlers.append(logging.FileHandler(config.ROOT / "pokedex.log", mode="w", encoding="utf-8"))
    except OSError:
        pass
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format=fmt, handlers=handlers, force=True)
    import platform
    log.info("Python %s, %s %s", platform.python_version(), platform.system(), platform.machine())


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Real-life Pokédex")
    p.add_argument("--config", type=config.Path, help="path to config.toml")
    p.add_argument("--camera", choices=["auto", "picamera2", "opencv", "folder"])
    p.add_argument("--camera-index", type=int, help="webcam slot (try 1 if 0 is your iPhone)")
    p.add_argument("--mode", choices=["pokemon", "world"])
    p.add_argument("--fullscreen", action="store_true")
    p.add_argument("--gpio", action="store_true", help="read physical buttons")
    p.add_argument("--size", help="WIDTHxHEIGHT, e.g. 480x320")
    p.add_argument("--no-tts", action="store_true")
    p.add_argument("--frames", type=int, help="exit after N frames (testing)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    cfg = config.load(args.config)
    setup_logging(cfg, args.verbose)
    if args.camera:
        cfg.camera_source = args.camera
    if args.camera_index is not None:
        cfg.camera_index = args.camera_index
        cfg.camera_index_set = True
    if args.mode:
        cfg.mode = args.mode
    if args.fullscreen:
        cfg.fullscreen = True
    if args.gpio:
        cfg.gpio = True
    if args.size:
        cfg.width, cfg.height = (int(v) for v in args.size.lower().split("x"))
    if args.no_tts:
        cfg.tts = "none"

    import sys

    log.info("Starting Pokédex: %s", " ".join(sys.argv[1:] if argv is None else argv) or "default options")
    names = camera.mac_camera_names()
    if names:
        log.info("Cameras macOS sees (not in slot order): %s", ", ".join(names))
    try:
        App(cfg, max_frames=args.frames).run()
    except Exception:
        log.exception("Crashed")
        raise
