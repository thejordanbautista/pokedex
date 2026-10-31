"""Input: keyboard on a laptop, GPIO buttons on the Pi. Both become the same Action events."""

from __future__ import annotations

import enum
import logging
import os
import subprocess

import pygame

log = logging.getLogger(__name__)

ACTION_EVENT = pygame.USEREVENT + 1


class Action(enum.Enum):
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    A = "a"
    B = "b"
    SCAN = "scan"
    MODE = "mode"
    QUIT = "quit"


KEYMAP = {
    pygame.K_UP: Action.UP,
    pygame.K_DOWN: Action.DOWN,
    pygame.K_LEFT: Action.LEFT,
    pygame.K_RIGHT: Action.RIGHT,
    pygame.K_z: Action.A,
    pygame.K_x: Action.B,
    pygame.K_ESCAPE: Action.B,
    pygame.K_SPACE: Action.SCAN,
    pygame.K_RETURN: Action.SCAN,
    pygame.K_m: Action.MODE,
    pygame.K_q: Action.QUIT,
}


def from_event(event: pygame.event.Event) -> Action | None:
    if event.type == pygame.QUIT:
        return Action.QUIT
    if event.type == pygame.KEYDOWN:
        return KEYMAP.get(event.key)
    if event.type == ACTION_EVENT:
        return event.action
    return None


def post(action: Action) -> None:
    pygame.event.post(pygame.event.Event(ACTION_EVENT, action=action))


class GPIOButtons:
    """Physical buttons wired pin -> GND. Hold B to shut the Pi down safely."""

    def __init__(self, pins, shutdown_hold_seconds: float = 3.0):
        from gpiozero import LED, Button

        self.buttons = []
        for action in (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT, Action.A, Action.SCAN):
            btn = Button(getattr(pins, action.value), pull_up=True, bounce_time=0.03)
            btn.when_pressed = lambda a=action: post(a)
            self.buttons.append(btn)

        b = Button(pins.b, pull_up=True, bounce_time=0.03, hold_time=shutdown_hold_seconds)
        b.when_released = self._b_released
        b.when_held = self._shutdown
        self._b_held = False
        self.buttons.append(b)

        self.leds = LED(pins.leds)
        log.info("GPIO buttons ready")

    def _b_released(self) -> None:
        if not self._b_held:
            post(Action.B)

    def _shutdown(self) -> None:
        self._b_held = True
        log.warning("B held: shutting down")
        post(Action.QUIT)
        if os.environ.get("POKEDEX_NO_SHUTDOWN") != "1":
            subprocess.Popen(["sudo", "shutdown", "-h", "now"])

    def lights(self, on: bool) -> None:
        self.leds.on() if on else self.leds.off()


class NoLights:
    def lights(self, on: bool) -> None: ...
