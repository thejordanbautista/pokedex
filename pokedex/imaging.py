"""Small image helpers: brightness measurement and low-light boost."""

from __future__ import annotations

import numpy as np

DIM = 60  # mean brightness (0-255) below this gets boosted


def brightness(frame: np.ndarray) -> float:
    """Mean brightness, sampled on a coarse grid so it's cheap to run every frame."""
    return float(frame[::8, ::8].mean())


def boost_low_light(frame: np.ndarray, target: float = 110.0) -> np.ndarray:
    """Brighten a dim frame with gain + gamma so dark rooms are still usable.

    Frames that are already bright enough, or almost fully black (nothing to recover),
    come back unchanged.
    """
    b = brightness(frame)
    if b >= DIM or b < 2:
        return frame
    x = frame.astype(np.float32) / 255.0
    gamma = float(np.clip(np.log(target / 255.0) / np.log(max(b, 1.0) / 255.0), 0.4, 1.0))
    x = np.power(x, gamma)
    gain = target / max(float(x[::8, ::8].mean()) * 255.0, 1.0)
    return np.clip(x * min(gain, 4.0) * 255.0, 0, 255).astype(np.uint8)
