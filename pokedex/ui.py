"""Drawing for the handheld and laptop app. Same visual language as the phone app (web/):
enamel-red body, ink-blue viewfinder glass with a phosphor reticle, one big round scan button,
and a bone readout card with the name set in Unbounded.

Landscape layout (800x480 Pi screen; everything scales from the height):

    lens  leds                                   3 of 9 seen
    +------------------------+  +-------------------------+
    |  viewfinder glass      |  |  readout card           |
    |      reticle           |  |  No. 001   New entry    |
    +------------------------+  |  Dog                    |
      World   ( Scan )  Camera  |  ...         [buttons]  |
                                +-------------------------+
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pygame

from pokedex.dex import Entry

SHELL = (200, 16, 46)
SHELL_DEEP = (142, 11, 33)
SHELL_HI = (232, 50, 76)
GLASS = (14, 26, 43)
PHOSPHOR = (94, 230, 255)
BONE = (238, 241, 236)
BEZEL = (247, 247, 242)
INK = (22, 32, 44)
INK_SOFT = (91, 102, 115)
RULE = (217, 222, 216)
CHIP = (221, 227, 220)
AMBER = (255, 181, 71)
WHITE = (255, 255, 255)
DISABLED = (227, 232, 226)

TYPE_COLORS = {
    "normal": (168, 167, 122), "fire": (238, 129, 48), "water": (99, 144, 240),
    "electric": (247, 208, 44), "grass": (122, 199, 76), "ice": (150, 217, 214),
    "fighting": (194, 46, 40), "poison": (163, 62, 161), "ground": (226, 191, 101),
    "flying": (169, 143, 243), "psychic": (249, 85, 135), "bug": (166, 185, 26),
    "rock": (182, 161, 54), "ghost": (115, 87, 151), "dragon": (111, 53, 252),
    "dark": (112, 87, 70), "steel": (183, 183, 206), "fairy": (214, 133, 173),
}

FONT_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"


def _font(name: str, size: int) -> pygame.font.Font:
    path = FONT_DIR / name
    try:
        return pygame.font.Font(str(path) if path.exists() else None, size)
    except Exception:
        return pygame.font.Font(None, size)


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


def _circle(surf, color, center, r, width=0):
    """Antialiased circle where pygame offers it."""
    try:
        from pygame import gfxdraw

        x, y, r = int(center[0]), int(center[1]), int(r)
        if width == 0:
            gfxdraw.filled_circle(surf, x, y, r, color)
            gfxdraw.aacircle(surf, x, y, r, color)
        else:
            for i in range(width):
                gfxdraw.aacircle(surf, x, y, r - i, color)
    except Exception:
        pygame.draw.circle(surf, color, (int(center[0]), int(center[1])), int(r), width)


class UI:
    def __init__(self, surface: pygame.Surface, assets_dir: Path):
        self.s = surface
        self.w, self.h = surface.get_size()
        self.assets = assets_dir
        u = self.h / 480
        self.u = u
        self.display_xl = _font("Unbounded-800.ttf", int(46 * u))
        self.display_l = _font("Unbounded-800.ttf", int(30 * u))
        self.display_m = _font("Unbounded-700.ttf", int(19 * u))
        self.display_s = _font("Unbounded-500.ttf", int(14 * u))
        self.text = _font("AtkinsonHyperlegible-Regular.ttf", int(17 * u))
        self.text_b = _font("AtkinsonHyperlegible-Bold.ttf", int(17 * u))
        self.text_s = _font("AtkinsonHyperlegible-Regular.ttf", int(14 * u))
        self.text_sb = _font("AtkinsonHyperlegible-Bold.ttf", int(14 * u))
        self.pad = int(16 * u)

        pad, top = self.pad, int(64 * u)
        left_w = int(self.w * 0.53)
        controls_h = int(118 * u)
        self.view = pygame.Rect(pad, top, left_w - pad, self.h - top - controls_h)
        self.controls = pygame.Rect(pad, self.view.bottom, left_w - pad, controls_h)
        self.panel = pygame.Rect(left_w + pad, top, self.w - left_w - 2 * pad, self.h - top - pad)
        self.hits: list[tuple[pygame.Rect, str]] = []
        self._sprites: dict[str, pygame.Surface | None] = {}
        self._body = self._make_body()

    # ---------- helpers ----------
    def _make_body(self) -> pygame.Surface:
        """Enamel body with a soft highlight, rendered once."""
        body = pygame.Surface((self.w, self.h))
        body.fill(SHELL)
        hi = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        for i in range(12):
            a = int(10 * (12 - i) / 12)
            pygame.draw.ellipse(hi, (*SHELL_HI, a), (-self.w * 0.2 + i * 18, -self.h * 0.5 + i * 10,
                                                     self.w * 0.9 - i * 36, self.h * 0.9 - i * 20))
        body.blit(hi, (0, 0))
        pygame.draw.rect(body, SHELL_DEEP, (0, self.h - int(6 * self.u), self.w, int(6 * self.u)))
        return body

    def _text(self, text: str, font, color, x: int, y: int) -> int:
        img = font.render(text, True, color)
        self.s.blit(img, (x, y))
        return y + img.get_height()

    def _para(self, text: str, font, color, x: int, y: int, width: int, max_lines: int = 8, leading: float = 1.3) -> int:
        step = int(font.get_height() * leading)
        lines = wrap(text, font, width)
        if len(lines) > max_lines:  # show that text was cut, never stop mid-sentence silently
            lines = lines[:max_lines]
            last = lines[-1]
            while last and font.size(last + "…")[0] > width:
                last = last.rsplit(" ", 1)[0] if " " in last else last[:-1]
            lines[-1] = last.rstrip(",.;:") + "…"
        for line in lines:
            self.s.blit(font.render(line, True, color), (x, y))
            y += step
        return y

    def begin_frame(self) -> None:
        self.hits = []

    def hit_test(self, pos) -> str | None:
        for rect, action in reversed(self.hits):
            if rect.collidepoint(pos):
                return action
        return None

    # ---------- body, header, controls ----------
    def chrome(self, mode: str, seen: int, total: int, t: float, lights: bool = False) -> None:
        s, u = self.s, self.u
        s.blit(self._body, (0, 0))
        # lens light
        c = (int(36 * u), int(32 * u))
        _circle(s, BEZEL, c, 24 * u)
        glow = 1.0 if lights else 0.82 + 0.18 * math.sin(t * 2)
        _circle(s, tuple(int(v * glow) for v in (10, 126, 160)), c, 19 * u)
        _circle(s, tuple(min(255, int(v * glow)) for v in PHOSPHOR), (c[0] - 2 * u, c[1] - 2 * u), 14 * u)
        _circle(s, (232, 251, 255), (c[0] - 7 * u, c[1] - 7 * u), 4.5 * u)
        for i, col in enumerate([(255, 90, 95), (255, 210, 63), (61, 220, 132)]):
            _circle(s, col, (int((76 + 16 * i) * u), int(16 * u)), 5 * u)
        # seen counter pill (opens the dex)
        num = self.display_m.render(str(seen), True, WHITE)
        of = self.text_s.render(f"of {total} seen", True, (255, 225, 230))
        w = num.get_width() + of.get_width() + int(34 * u)
        pill = pygame.Rect(self.w - self.pad - w, int(12 * u), w, int(40 * u))
        shade = pygame.Surface(pill.size, pygame.SRCALPHA)
        pygame.draw.rect(shade, (0, 0, 0, 46), shade.get_rect(), border_radius=pill.h // 2)
        s.blit(shade, pill.topleft)
        s.blit(num, (pill.x + int(14 * u), pill.centery - num.get_height() // 2))
        s.blit(of, (pill.x + int(20 * u) + num.get_width(), pill.centery - of.get_height() // 2 + int(2 * u)))
        self.hits.append((pill, "right"))
        self._mode = mode
        self._scanning = lights

    def _glass(self) -> None:
        u = self.u
        bezel = self.view.inflate(int(12 * u), int(12 * u))
        pygame.draw.rect(self.s, SHELL_DEEP, bezel.move(0, int(3 * u)), border_radius=int(22 * u))
        pygame.draw.rect(self.s, BEZEL, bezel, border_radius=int(22 * u))
        pygame.draw.rect(self.s, GLASS, self.view, border_radius=int(16 * u))

    def _frame(self, frame: np.ndarray | None) -> None:
        self._glass()
        if frame is None:
            return
        surf = pygame.surfarray.make_surface(np.ascontiguousarray(frame.swapaxes(0, 1)))
        fw, fh = surf.get_size()
        k = max(self.view.w / fw, self.view.h / fh)
        surf = pygame.transform.smoothscale(surf, (int(fw * k), int(fh * k)))
        area = pygame.Rect(0, 0, self.view.w, self.view.h)
        area.center = surf.get_rect().center
        # rounded corners: draw into a masked surface
        clip = pygame.Surface(self.view.size, pygame.SRCALPHA)
        clip.blit(surf, (0, 0), area)
        mask = pygame.Surface(self.view.size, pygame.SRCALPHA)
        pygame.draw.rect(mask, (255, 255, 255, 255), mask.get_rect(), border_radius=int(16 * self.u))
        clip.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
        self.s.blit(clip, self.view.topleft)

    def _reticle(self, dim: bool = False) -> None:
        u = self.u
        r = self.view.inflate(-int(self.view.w * 0.36), -int(self.view.h * 0.30))
        L, wd = int(26 * u), max(2, int(3 * u))
        col = tuple(int(c * (0.55 if dim else 1)) for c in PHOSPHOR)
        for (x, y, dx, dy) in [(r.left, r.top, 1, 1), (r.right, r.top, -1, 1), (r.left, r.bottom, 1, -1), (r.right, r.bottom, -1, -1)]:
            pygame.draw.line(self.s, col, (x, y), (x + dx * L, y), wd)
            pygame.draw.line(self.s, col, (x, y), (x, y + dy * L), wd)

    def _controls(self, t: float = 0.0) -> None:
        """World / Scan / Camera under the viewfinder. The scan button is the one loud thing."""
        s, u = self.s, self.u
        cx, cy = self.controls.centerx, self.controls.centery - int(2 * u)
        R = int(46 * u)
        _circle(s, SHELL_DEEP, (cx, cy + 5 * u), R)
        _circle(s, GLASS, (cx, cy), R)
        if self._scanning:  # dashed ring turning while the model runs
            n = 18
            for i in range(n):
                a0 = t * 4 + i * 2 * math.pi / n
                p0 = (cx + math.cos(a0) * (R - 9 * u), cy + math.sin(a0) * (R - 9 * u))
                p1 = (cx + math.cos(a0 + 0.17) * (R - 9 * u), cy + math.sin(a0 + 0.17) * (R - 9 * u))
                pygame.draw.line(s, PHOSPHOR, p0, p1, max(2, int(3 * u)))
        else:
            _circle(s, PHOSPHOR, (cx, cy), R - 8 * u, width=max(2, int(3 * u)))
        lbl = self.display_m.render("Scan", True, WHITE)
        s.blit(lbl, lbl.get_rect(center=(cx, cy)))
        self.hits.append((pygame.Rect(cx - R, cy - R, 2 * R, 2 * R), "scan"))

        for label, action, x in [("Pokémon" if self._mode == "pokemon" else "World", "mode", self.controls.x + self.controls.w * 0.15),
                                 ("Camera", "camera", self.controls.x + self.controls.w * 0.85)]:
            self._side_icon(action, (int(x), cy - int(12 * u)))
            img = self.text_s.render(label, True, WHITE)
            s.blit(img, img.get_rect(center=(int(x), cy + int(20 * u))))
            self.hits.append((pygame.Rect(int(x) - int(44 * u), cy - int(36 * u), int(88 * u), int(72 * u)), action))

    def _side_icon(self, kind: str, c) -> None:
        u, s, w = self.u, self.s, max(2, int(2 * self.u))
        r = int(12 * u)
        if kind == "mode":  # globe
            pygame.draw.circle(s, WHITE, c, r, w)
            pygame.draw.line(s, WHITE, (c[0] - r, c[1]), (c[0] + r, c[1]), w)
            pygame.draw.ellipse(s, WHITE, (c[0] - r // 2, c[1] - r, r, 2 * r), w)
        else:  # two curved arrows: switch camera
            pygame.draw.arc(s, WHITE, (c[0] - r, c[1] - r, 2 * r, 2 * r), 0.3, 2.6, w)
            pygame.draw.arc(s, WHITE, (c[0] - r, c[1] - r, 2 * r, 2 * r), 3.45, 5.75, w)

    # ---------- readout card ----------
    def _card(self) -> pygame.Rect:
        u = self.u
        pygame.draw.rect(self.s, SHELL_DEEP, self.panel.move(0, int(4 * u)), border_radius=int(22 * u))
        pygame.draw.rect(self.s, BONE, self.panel, border_radius=int(22 * u))
        return self.panel.inflate(-2 * int(20 * u), -2 * int(18 * u))

    def buttons(self, items: list[tuple[str, str]]) -> None:
        """Card buttons along the bottom. First = primary (red), the rest quiet."""
        u = self.u
        inner = self.panel.inflate(-2 * int(20 * u), -2 * int(18 * u))
        h, gap = int(44 * u), int(10 * u)
        y = inner.bottom - h
        weights = [2 if i == 0 else 1 for i in range(len(items))]
        unit = (inner.w - gap * (len(items) - 1)) / sum(weights)
        x = inner.x
        for i, ((label, action), wt) in enumerate(zip(items, weights)):
            rect = pygame.Rect(int(x), y, int(unit * wt), h)
            if i == 0:
                pygame.draw.rect(self.s, SHELL_DEEP, rect.move(0, int(4 * u)), border_radius=int(14 * u))
                pygame.draw.rect(self.s, SHELL, rect, border_radius=int(14 * u))
                color = WHITE
            else:
                pygame.draw.rect(self.s, (201, 207, 201), rect, max(2, int(2 * u)), border_radius=int(14 * u))
                color = INK
            font = self.text_b if self.text_b.size(label)[0] <= rect.w - int(12 * u) else self.text_sb
            img = font.render(label, True, color)
            self.s.blit(img, img.get_rect(center=rect.center))
            self.hits.append((rect, action))
            x += rect.w + gap

    # ---------- screens ----------
    def live(self, frame, note: str = "") -> None:
        self._frame(frame)
        self._reticle()
        self.hits.append((self.view.copy(), "scan"))
        self._controls()
        c = self._card()
        y = self._text("Ready", self.display_l, INK, c.x, c.y) + int(10 * self.u)
        y = self._para("Point the lens at an animal or an object, then press Scan.", self.text, INK_SOFT, c.x, y, c.w)
        if note:
            y += int(8 * self.u)
            pygame.draw.rect(self.s, AMBER, (c.x, y + int(2 * self.u), int(4 * self.u), int(self.text_s.get_height() * 1.3 * min(5, len(wrap(note, self.text_s, c.w - int(14 * self.u)))))))
            self._para(note, self.text_s, INK, c.x + int(14 * self.u), y, c.w - int(14 * self.u), max_lines=5)
        self.buttons([("Open dex", "right")])

    def scanning(self, frame, t: float) -> None:
        self._frame(frame)
        self._reticle()
        v = self.view
        # phosphor sweep: a bright line with a fading trail above it
        y = v.y + int(((t * 0.9) % 1.0) * v.h)
        trail = pygame.Surface((v.w, int(60 * self.u)), pygame.SRCALPHA)
        for i in range(trail.get_height()):
            a = int(90 * i / trail.get_height())
            pygame.draw.line(trail, (*PHOSPHOR, a), (0, i), (v.w, i))
        top = max(v.y, y - trail.get_height())
        self.s.blit(trail, (v.x, top), pygame.Rect(0, trail.get_height() - (y - top), v.w, y - top))
        pygame.draw.line(self.s, PHOSPHOR, (v.x, y), (v.right - 1, y), max(2, int(3 * self.u)))
        self._controls(t)
        c = self._card()
        dots = "." * (1 + int(t * 3) % 3)
        self._text(f"Scanning{dots}", self.display_l, INK, c.x, c.y)

    def entry(self, e: Entry, frame, confidence: float | None, new: bool, page: int) -> None:
        sprite = self._sprite(e.sprite)
        self._frame(frame if sprite is None else None)
        if sprite is not None:
            k = min(self.view.w / sprite.get_width(), self.view.h / sprite.get_height()) * 0.78
            img = pygame.transform.scale(sprite, (int(sprite.get_width() * k), int(sprite.get_height() * k)))
            self.s.blit(img, img.get_rect(center=self.view.center))
        else:
            self._reticle(dim=True)
        self._controls()

        u = self.u
        c = self._card()
        x, y = c.x, c.y
        no = self.display_s.render(f"No. {e.number:03d}", True, INK_SOFT)
        self.s.blit(no, (x, y))
        if new:
            tag = self.text_sb.render("New entry", True, (58, 36, 0))
            r = tag.get_rect().inflate(int(18 * u), int(6 * u))
            r.midleft = (x + no.get_width() + int(10 * u), y + no.get_height() // 2)
            pygame.draw.rect(self.s, AMBER, r, border_radius=r.h // 2)
            self.s.blit(tag, tag.get_rect(center=r.center))
        y += no.get_height() + int(2 * u)
        name_font = self.display_xl if self.display_xl.size(e.name)[0] <= c.w else self.display_l
        y = self._text(e.name, name_font, INK, x, y)
        if e.category:
            y = self._text(e.category, self.text, INK_SOFT, x, y + int(2 * u))
        y += int(10 * u)

        tx = x
        for t in e.types:
            lbl = self.text_sb.render(t.capitalize(), True, WHITE)
            chip = lbl.get_rect().inflate(int(20 * u), int(8 * u))
            chip.topleft = (tx, y)
            pygame.draw.rect(self.s, TYPE_COLORS.get(t.lower(), INK_SOFT), chip, border_radius=chip.h // 2)
            self.s.blit(lbl, lbl.get_rect(center=chip.center))
            tx = chip.right + int(8 * u)
        if e.types:
            y += int(36 * u)

        # stats row with rules, like the phone sheet
        cells = [("Height", f"{e.height_m:g} m" if e.height_m is not None else "Unknown"),
                 ("Weight", f"{e.weight_kg:g} kg" if e.weight_kg is not None else "Unknown"),
                 ("Match", f"{confidence:.0%}" if confidence is not None else "Logged")]
        row_h = int(50 * u)
        pygame.draw.line(self.s, RULE, (x, y), (c.right, y), max(1, int(2 * u)))
        pygame.draw.line(self.s, RULE, (x, y + row_h), (c.right, y + row_h), max(1, int(2 * u)))
        cw = c.w / 3
        for i, (k, v) in enumerate(cells):
            cx = int(x + i * cw) + (int(12 * u) if i else 0)
            if i:
                pygame.draw.line(self.s, RULE, (int(x + i * cw), y), (int(x + i * cw), y + row_h), max(1, int(2 * u)))
            self.s.blit(self.text_s.render(k, True, INK_SOFT), (cx, y + int(6 * u)))
            self.s.blit(self.display_m.render(v, True, INK), (cx, y + int(24 * u)))
        y += row_h + int(10 * u)

        lines_room = max(1, (c.bottom - int(56 * u) - y) // int(self.text.get_height() * 1.3))
        if page == 1 and e.evolutions:
            y = self._text("Evolution line", self.text_b, INK, x, y) + int(6 * u)
            self._evolutions(e, x, y, c.w)
        else:
            self._para(e.description, self.text, INK, x, y, c.w, max_lines=lines_room)
        actions = [("Scan again", "scan")]
        if e.evolutions and len(e.evolutions) > 1:
            actions.append(("Details" if page == 1 else "Evolves", "right"))
        actions.append(("Back", "b"))
        self.buttons(actions)

    def _evolutions(self, e: Entry, x: int, y: int, width: int) -> None:
        u = self.u
        tx, ty = x, y
        for name in e.evolutions:
            current = name.lower() == e.name.lower()
            lbl = (self.text_sb if current else self.text_s).render(name, True, WHITE if current else INK)
            chip = lbl.get_rect().inflate(int(22 * u), int(12 * u))
            if tx + chip.w > x + width:
                tx, ty = x, ty + chip.h + int(8 * u)
            chip.topleft = (tx, ty)
            pygame.draw.rect(self.s, INK if current else CHIP, chip, border_radius=int(12 * u))
            self.s.blit(lbl, lbl.get_rect(center=chip.center))
            tx = chip.right + int(8 * u)

    def unknown(self, frame, best: str | None, confidence: float | None) -> None:
        self._frame(frame)
        self._reticle(dim=True)
        self._controls()
        c = self._card()
        y = self._text("No match yet", self.display_l, INK, c.x, c.y) + int(10 * self.u)
        if best:
            msg = (f"Closest guess was {best.split(',')[0]} at {confidence:.0%}, which isn't in this dex. "
                   "Fill the frame with the subject, add light, and scan again.")
        else:
            msg = "Nothing recognized. Fill the frame with the subject, add light, and scan again."
        self._para(msg, self.text, INK, c.x, y, c.w)
        self.buttons([("Scan again", "scan"), ("Back", "b")])

    def listing(self, entries: list[Entry], cursor: int, seen) -> None:
        """Dex grid in the viewfinder area, details on the card."""
        u = self.u
        self._glass()
        cols, gap = 3, int(10 * u)
        area = self.view.inflate(-int(20 * u), -int(20 * u))
        cell = (area.w - gap * (cols - 1)) // cols
        cell_h = int(cell * 0.62)
        rows_visible = max(1, (area.h + gap) // (cell_h + gap))
        first_row = max(0, cursor // cols - rows_visible + 1)
        for i, e in enumerate(entries):
            row = i // cols - first_row
            if row < 0 or row >= rows_visible:
                continue
            r = pygame.Rect(area.x + (i % cols) * (cell + gap), area.y + row * (cell_h + gap), cell, cell_h)
            is_seen = seen(e.name)
            pygame.draw.rect(self.s, WHITE if is_seen else (34, 48, 66), r, border_radius=int(12 * u))
            if i == cursor:
                pygame.draw.rect(self.s, PHOSPHOR, r.inflate(int(6 * u), int(6 * u)), max(2, int(3 * u)), border_radius=int(14 * u))
            no = self.display_s.render(f"{e.number:03d}", True, INK_SOFT if is_seen else (110, 125, 145))
            self.s.blit(no, no.get_rect(center=(r.centerx, r.centery - int(10 * u))))
            nm_text = e.name if is_seen else "Not seen"
            nm_font = self.text_sb if self.text_sb.size(nm_text)[0] <= r.w - int(10 * u) else self.text_s
            nm = nm_font.render(nm_text, True, INK if is_seen else (110, 125, 145))
            self.s.blit(nm, nm.get_rect(center=(r.centerx, r.centery + int(12 * u))))
            self.hits.append((r, f"pick:{i}"))
        self._controls()
        c = self._card()
        y = self._text("Dex", self.display_l, INK, c.x, c.y) + int(10 * u)
        n_seen = sum(1 for e in entries if seen(e.name))
        y = self._para(f"{n_seen} of {len(entries)} seen. Scan new things to fill the grid.", self.text, INK_SOFT, c.x, y, c.w)
        if entries:
            e = entries[cursor]
            y += int(12 * u)
            self._text(f"No. {e.number:03d}", self.display_s, INK_SOFT, c.x, y)
            self._text(e.name if seen(e.name) else "Not seen yet", self.display_m, INK, c.x, y + int(22 * u))
        self.buttons([("Open", "a"), ("Up", "up"), ("Down", "down"), ("Back", "b")])

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
