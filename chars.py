"""Character glyphs (A-Z, 0-9) rendered as small binary bitmaps.

Each glyph is a GRID x GRID array in [0, 1] with ink = 1 and background = 0.
The networks work with bipolar vectors (+1 ink, -1 background); use
``to_bipolar`` / ``from_bipolar`` to convert.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

GRID = 24
N_PIXELS = GRID * GRID
CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# Consolas Bold first: its slashed zero keeps "0" distinct from "O".
_FONT_PATHS = (
    r"C:\Windows\Fonts\consolab.ttf",
    r"C:\Windows\Fonts\arialbd.ttf",
)
_SUPERSAMPLE = 160  # render large, then box-downsample for clean strokes


@lru_cache(maxsize=1)
def _font() -> ImageFont.ImageFont:
    for path in _FONT_PATHS:
        if Path(path).exists():
            return ImageFont.truetype(path, _SUPERSAMPLE)
    return ImageFont.load_default(_SUPERSAMPLE)


def render_char(ch: str, grid: int = GRID, margin: int = 1) -> np.ndarray:
    """Render one character, scaled to fit the grid with aspect ratio kept."""
    size = _SUPERSAMPLE * 2
    canvas = Image.new("L", (size, size), 0)
    ImageDraw.Draw(canvas).text((size // 4, size // 8), ch, fill=255, font=_font())
    glyph = canvas.crop(canvas.getbbox())

    inner = grid - 2 * margin
    w, h = glyph.size
    scale = inner / max(w, h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    small = glyph.resize((nw, nh), Image.Resampling.BOX)

    out = Image.new("L", (grid, grid), 0)
    out.paste(small, ((grid - nw) // 2, (grid - nh) // 2))
    arr = np.asarray(out, dtype=float) / 255.0
    return (arr >= 0.5).astype(float)


@lru_cache(maxsize=1)
def _all_glyphs() -> np.ndarray:
    return np.stack([render_char(c) for c in CHARSET])


def glyphs(chars: str = CHARSET) -> np.ndarray:
    """Return an array of shape (len(chars), GRID, GRID) with values in {0, 1}."""
    table = _all_glyphs()
    return np.stack([table[CHARSET.index(c)] for c in chars])


def to_bipolar(img: np.ndarray) -> np.ndarray:
    """Threshold [0,1] image(s) at 0.5 and flatten to bipolar vectors (+1 / -1)."""
    img = np.asarray(img, dtype=float)
    flat = img.reshape(img.shape[0], -1) if img.ndim == 3 else img.reshape(-1)
    return np.where(flat >= 0.5, 1.0, -1.0)


def from_bipolar(vec: np.ndarray, grid: int = GRID) -> np.ndarray:
    """Bipolar vector(s) back to {0, 1} image(s)."""
    vec = np.asarray(vec, dtype=float)
    img = (vec > 0).astype(float)
    return img.reshape(-1, grid, grid) if vec.ndim == 2 else img.reshape(grid, grid)
