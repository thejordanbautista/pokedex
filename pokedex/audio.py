"""Text-to-speech and sound effects. Speech runs in a background thread so the UI never blocks."""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

log = logging.getLogger(__name__)


class Audio:
    def __init__(self, tts: str = "auto", piper_voice: str = "", volume: float = 0.8):
        self.volume = volume
        self.piper_voice = piper_voice
        self.engine = self._pick_engine(tts)
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self.mixer = False
        try:
            import pygame

            pygame.mixer.init()
            self.mixer = True
        except Exception as e:
            log.info("No audio output (%s)", e)
        log.info("TTS engine: %s", self.engine)

    def _pick_engine(self, tts: str) -> str:
        if tts == "none":
            return "none"
        if tts in ("auto", "piper") and shutil.which("piper") and self.piper_voice:
            return "piper"
        if tts in ("auto", "espeak"):
            for exe in ("espeak-ng", "espeak"):
                if shutil.which(exe):
                    return exe
        return "none"

    def play(self, path: str | Path | None) -> None:
        if not (self.mixer and path and Path(path).exists()):
            return
        import pygame

        try:
            snd = pygame.mixer.Sound(str(path))
            snd.set_volume(self.volume)
            snd.play()
        except pygame.error as e:
            log.warning("Can't play %s: %s", path, e)

    def say(self, text: str, delay: float = 0.0) -> None:
        """Speak text, cutting off anything already being spoken."""
        if self.engine == "none" or not text:
            return
        self.stop()
        threading.Thread(target=self._speak, args=(text, delay), daemon=True).start()

    def _speak(self, text: str, delay: float) -> None:
        if delay:
            threading.Event().wait(delay)
        with self._lock:
            if self.engine == "piper":
                wav = Path(tempfile.gettempdir()) / "pokedex_tts.wav"
                subprocess.run(
                    ["piper", "--model", self.piper_voice, "--output_file", str(wav)],
                    input=text.encode(), capture_output=True,
                )
                cmd = ["aplay", "-q", str(wav)]
            else:
                # Slightly fast and low-pitched: a decent robotic Pokédex voice
                cmd = [self.engine, "-s", "165", "-p", "35", "-a", str(int(self.volume * 200)), text]
            self._proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self._proc.wait()

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
