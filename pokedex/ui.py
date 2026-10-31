"""Drawing. Everything scales from the screen size, so 480x320 and 800x480 both work."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pygame

from pokedex.dex import Entry

RED = (220, 10, 45)
DARK_RED = (150, 0, 25)
SCREEN_BG = (24, 28, 32)
PANEL = (236, 240, 232)
INK = (24, 24, 24)
MUTED = (110, 110, 110)
CYAN = (60, 200, 255)
GREEN = (60, 200, 110)
WHITE = (255, 255, 255)

TYPE_COLORS = {
    "normal": (168, 167, 122), "fire": (238, 129, 48), "water": (99, 144, 240),
    "electric": (247, 208, 44), "grass": (122, 199, 76), "ice": (150, 217, 214),
    "fighting": (194, 46, 40), "poison": (163, 62, 161), "ground": (226, 191, 101),
    "flying": (169, 143, 243), "psychic": (249, 85, 135), "bug": (166, 185, 26),
    "rock": (182, 161, 54), "ghost": (115, 87, 151), "dragon": (111, 53, 252),
    "dark": (112, 87, 70), "steel": (183, 183, 206), "fairy": (214, 133, 173),
}


def wrap(text: str, font: pygame.font.Font, width: int) -> list[str]:
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if font.size(trial)[0] <= width:
            line = trial
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


class UI:
    def __init__(self, surface: pygame.Surface, assets_dir: Path):
        self.s = surface
        self.w, self.h = surface.get_size()
        self.assets = assets_dir
        u = self.h / 480  # scale unit
        self.u = u
        self.f_big = pygame.font.Font(None, int(44 * u))
        self.f_mid = pygame.font.Font(None, int(30 * u))
        self.f_small = pygame.font.Font(None, int(24 * u))
        pad = int(16 * u)
        self.pad = pad
        # Left: the "viewfinder" screen. Right: the info panel.
        vw = int(self.w * 0.52)
        self.view = pygame.Rect(pad, int(56 * u), vw - pad, self.h - int(56 * u) - pad)
        self.panel = pygame.Rect(vw + pad, int(56 * u), self.w - vw - 2 * pad, self.h - int(56 * u) - pad)
        self._sprites: dict[str, pygame.Surface | None] = {}
        self.hits: list[tuple[pygame.Rect, str]] = []  # clickable areas -> action name, rebuilt each frame

    # ---------- chrome ----------
    def chrome(self, mode: str, seen: int, total: int, t: float, lights: bool = False) -> None:
        s, u = self.s, self.u
        s.fill(RED)
        pygame.draw.rect(s, DARK_RED, (0, int(48 * u), self.w, int(3 * u)))
        # The big blue lens light, pulsing when idle
        r = int(18 * u)
        glow = 0.6 + 0.4 * (1 if lights else (0.5 + 0.5 * math.sin(t * 2)))
        pygame.draw.circle(s, WHITE, (int(28 * u), int(24 * u)), r + int(3 * u))
        pygame.draw.circle(s, tuple(int(c * glow) for c in CYAN), (int(28 * u), int(24 * u)), r)
        for i, c in enumerate([(255, 70, 70), (250, 220, 60), (70, 220, 90)]):
            pygame.draw.circle(s, c, (int((62 + 18 * i) * u), int(14 * u)), int(5 * u))
        label = "POKÉMON" if mode == "pokemon" else "WORLD"
        txt = self.f_small.render(f"{label}  ·  SEEN {seen}/{total}" if total else label, True, WHITE)
        s.blit(txt, (self.w - txt.get_width() - self.pad, int(16 * u)))

    def _screen(self) -> None:
        pygame.draw.rect(self.s, PANEL, self.view.inflate(int(12 * self.u), int(12 * self.u)), border_radius=int(10 * self.u))
        pygame.draw.rect(self.s, SCREEN_BG, self.view, border_radius=int(6 * self.u))

    def _panel(self) -> None:
        pygame.draw.rect(self.s, PANEL, self.panel, border_radius=int(10 * self.u))

    def _blit_frame(self, frame: np.ndarray | None) -> None:
        self._screen()
        if frame is None:
            return
        surf = pygame.surfarray.make_surface(np.ascontiguousarray(frame.swapaxes(0, 1)))
        fw, fh = surf.get_size()
        k = max(self.view.w / fw, self.view.h / fh)  # cover the viewfinder, crop overflow
        surf = pygame.transform.smoothscale(surf, (int(fw * k), int(fh * k)))
        area = pygame.Rect(0, 0, self.view.w, self.view.h)
        area.center = surf.get_rect().center
        self.s.blit(surf, self.view.topleft, area)

    def _text(self, text: str, font, color, x: int, y: int) -> int:
        img = font.render(text, True, color)
        self.s.blit(img, (x, y))
        return y + img.get_height()

    def begin_frame(self) -> None:
        self.hits = []

    def hit_test(self, pos) -> str | None:
        """Action name under a click or tap, or None."""
        for rect, action in reversed(self.hits):  # last drawn = on top
            if rect.collidepoint(pos):
                return action
        return None

    def buttons(self, items: list[tuple[str, str]]) -> None:
        """A row of clickable buttons along the bottom of the info panel.
        items: (label, action name). The first one is the main action and is drawn in red."""
        u, pad = self.u, self.pad
        h = int(44 * u)
        gap = int(8 * u)
        y = self.panel.bottom - pad - h
        x = self.panel.x + pad
        width = self.panel.w - 2 * pad
        weights = [2 if i == 0 else 1 for i in range(len(items))]
        unit = (width - gap * (len(items) - 1)) / sum(weights)
        for i, ((label, action), wt) in enumerate(zip(items, weights)):
            w = int(unit * wt)
            rect = pygame.Rect(x, y, w, h)
            main = i == 0
            pygame.draw.rect(self.s, RED if main else WHITE, rect, border_radius=int(10 * u))
            pygame.draw.rect(self.s, DARK_RED if main else MUTED, rect, max(1, int(2 * u)), border_radius=int(10 * u))
            img = self.f_small.render(label, True, WHITE if main else INK)
            self.s.blit(img, img.get_rect(center=rect.center))
            self.hits.append((rect, action))
            x += w + gap

    # ---------- screens ----------
    def live(self, frame, note: str = "") -> None:
        self._blit_frame(frame)
        # Viewfinder brackets
        c, L, w = self.view.inflate(-int(60 * self.u), -int(60 * self.u)), int(28 * self.u), max(2, int(3 * self.u))
        for (x, y, dx, dy) in [(c.left, c.top, 1, 1), (c.right, c.top, -1, 1), (c.left, c.bottom, 1, -1), (c.right, c.bottom, -1, -1)]:
            pygame.draw.line(self.s, WHITE, (x, y), (x + dx * L, y), w)
            pygame.draw.line(self.s, WHITE, (x, y), (x, y + dy * L), w)
        self._panel()
        x, y = self.panel.x + self.pad, self.panel.y + self.pad
        y = self._text("Ready to scan", self.f_big, INK, x, y) + int(10 * self.u)
        for line in wrap("Point the lens at a Pokémon or a creature and press SCAN.", self.f_small, self.panel.w - 2 * self.pad):
            y = self._text(line, self.f_small, MUTED, x, y)
        y += int(10 * self.u)
        for line in wrap(note, self.f_small, self.panel.w - 2 * self.pad)[:6]:
            y = self._text(line, self.f_small, DARK_RED, x, y)
        self.hits.append((self.view.copy(), "scan"))  # click the camera view to scan
        self.buttons([("SCAN", "scan"), ("DEX", "right"), ("MODE", "mode"), ("CAM", "camera")])

    def scanning(self, frame, t: float) -> None:
        self._blit_frame(frame)
        v = self.view
        y = v.y + int((0.5 + 0.5 * math.sin(t * 6)) * v.h)
        pygame.draw.line(self.s, CYAN, (v.x, y), (v.right, y), max(2, int(3 * self.u)))
        self._panel()
        dots = "." * (1 + int(t * 4) % 3)
        self._text(f"Analyzing{dots}", self.f_big, INK, self.panel.x + self.pad, self.panel.y + self.pad)

    def entry(self, e: Entry, frame, confidence: float | None, new: bool, page: int) -> None:
        self._screen()
        sprite = self._sprite(e.sprite)
        if sprite is not None:
            k = min(self.view.w / sprite.get_width(), self.view.h / sprite.get_height()) * 0.8
            img = pygame.transform.scale(sprite, (int(sprite.get_width() * k), int(sprite.get_height() * k)))
            self.s.blit(img, img.get_rect(center=self.view.center))
        elif frame is not None:
            self._blit_frame(frame)

        self._panel()
        p, u = self.pad, self.u
        x, y, width = self.panel.x + p, self.panel.y + p, self.panel.w - 2 * p
        y = self._text(f"No. {e.number:03d}", self.f_small, MUTED, x, y)
        y = self._text(e.name.upper(), self.f_big, INK, x, y)
        if e.category:
            y = self._text(e.category, self.f_small, MUTED, x, y)
        y += int(8 * u)

        if page == 0:
            tx = x
            for t in e.types:
                lbl = self.f_small.render(t.upper(), True, WHITE)
                chip = lbl.get_rect(topleft=(tx, y)).inflate(int(14 * u), int(8 * u))
                chip.topleft = (tx, y)
                pygame.draw.rect(self.s, TYPE_COLORS.get(t.lower(), MUTED), chip, border_radius=int(8 * u))
                self.s.blit(lbl, lbl.get_rect(center=chip.center))
                tx = chip.right + int(6 * u)
            if e.types:
                y += int(34 * u)
            stats = []
            if e.height_m is not None:
                stats.append(f"HT {e.height_m:g} m")
            if e.weight_kg is not None:
                stats.append(f"WT {e.weight_kg:g} kg")
            if stats:
                y = self._text("   ".join(stats), self.f_small, INK, x, y) + int(6 * u)
            for line in wrap(e.description, self.f_small, width)[:6]:
                y = self._text(line, self.f_small, INK, x, y)
        else:
            y = self._text("Evolutions", self.f_mid, INK, x, y) + int(4 * u)
            chain = e.evolutions or [e.name]
            for name in chain:
                mark = "▶ " if name.lower() == e.name.lower() else "   "
                y = self._text(mark + name, self.f_small, INK, x, y)

        foot = []
        if confidence is not None:
            foot.append(f"{confidence:.0%} match")
        if new:
            foot.append("NEW!")
        if foot:
            self._text("  ·  ".join(foot), self.f_small, GREEN if new else MUTED, x, self.panel.bottom - int(92 * u))
        self.buttons([("SCAN", "scan"), ("PAGE", "right"), ("BACK", "b")])

    def unknown(self, frame, best: str | None, confidence: float | None) -> None:
        self._blit_frame(frame)
        self._panel()
        x, y = self.panel.x + self.pad, self.panel.y + self.pad
        y = self._text("Unknown", self.f_big, INK, x, y) + int(8 * self.u)
        for line in wrap("No match in the database. Get closer, add light, and try again.", self.f_small, self.panel.w - 2 * self.pad):
            y = self._text(line, self.f_small, MUTED, x, y)
        if best:
            self._text(f"Best guess: {best} ({confidence:.0%})", self.f_small, MUTED, x, y + int(10 * self.u))
        self.buttons([("SCAN AGAIN", "scan"), ("BACK", "b")])

    def listing(self, entries: list[Entry], cursor: int, seen) -> None:
        self._screen()
        rows = max(1, self.view.h // int(30 * self.u))
        start = max(0, min(cursor - rows // 2, len(entries) - rows))
        y = self.view.y + int(8 * self.u)
        for i in range(start, min(len(entries), start + rows)):
            e = entries[i]
            color = CYAN if i == cursor else (WHITE if seen(e.name) else MUTED)
            name = e.name if seen(e.name) else "?????"
            self._text(f"{e.number:03d}  {name}", self.f_small, color, self.view.x + int(12 * self.u), y)
            y += int(30 * self.u)
        self._panel()
        self._text("Dex list", self.f_big, INK, self.panel.x + self.pad, self.panel.y + self.pad)
        self.buttons([("OPEN", "a"), ("UP", "up"), ("DOWN", "down"), ("BACK", "b")])

    def _sprite(self, rel: str | None) -> pygame.Surface | None:
        if not rel:
            return None
        if rel not in self._sprites:
            path = self.assets / rel
            try:
                self._sprites[rel] = pygame.image.load(str(path)).convert_alpha() if path.exists() else None
            except pygame.error:
                self._sprites[rel] = None
        return self._sprites[rel]
