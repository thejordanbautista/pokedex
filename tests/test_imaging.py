import numpy as np

from pokedex import imaging


def test_dim_frame_is_brightened():
    rng = np.random.default_rng(0)
    dim = rng.integers(0, 60, (480, 640, 3), dtype=np.uint8)  # mean ~30: a dim room
    out = imaging.boost_low_light(dim)
    assert out.dtype == np.uint8 and out.shape == dim.shape
    assert 80 < imaging.brightness(out) < 150
    # order of pixels is kept: brighter in = brighter out
    assert out[dim > 50].mean() > out[dim < 10].mean()


def test_bright_and_black_frames_unchanged():
    bright = np.full((48, 64, 3), 120, np.uint8)
    black = np.zeros((48, 64, 3), np.uint8)
    assert imaging.boost_low_light(bright) is bright
    assert imaging.boost_low_light(black) is black  # nothing to recover; don't amplify noise
