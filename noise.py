"""Grayscale noise models. All functions keep values in [0, 1].

Noise is applied to the clean {0,1} glyph as a grayscale image, so the fuzzy
stage has real intensity information to work with (bit-flip noise on a binary
image would leave nothing for fuzzification to act on).
"""
import numpy as np

KINDS = ("gaussian", "saltpepper", "faded")

# A faded scan: grey paper (0.3) and weak ink (0.6) instead of 0 and 1.
FADED_BACKGROUND = 0.3
FADED_INK = 0.6


def gaussian(img: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Additive Gaussian noise with standard deviation ``sigma``."""
    return np.clip(img + rng.normal(0.0, sigma, img.shape), 0.0, 1.0)


def salt_pepper(img: np.ndarray, density: float, rng: np.random.Generator) -> np.ndarray:
    """Replace a fraction ``density`` of pixels with pure white (1) or black (0)."""
    out = np.array(img, dtype=float, copy=True)
    hit = rng.random(img.shape) < density
    salt = rng.random(img.shape) < 0.5
    out[hit] = salt[hit].astype(float)
    return out


def faded(img: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Low-contrast scan (grey paper, weak ink) plus Gaussian noise with std ``sigma``."""
    dim = FADED_BACKGROUND + (FADED_INK - FADED_BACKGROUND) * np.asarray(img, dtype=float)
    return gaussian(dim, sigma, rng)


def add_noise(img: np.ndarray, kind: str, level: float, rng: np.random.Generator) -> np.ndarray:
    """Dispatch by name: 'gaussian' (level = sigma), 'saltpepper' (level = density),
    or 'faded' (level = sigma of the noise on a low-contrast image)."""
    if kind == "gaussian":
        return gaussian(img, level, rng)
    if kind == "saltpepper":
        return salt_pepper(img, level, rng)
    if kind == "faded":
        return faded(img, level, rng)
    raise ValueError(f"unknown noise kind {kind!r}; expected one of {KINDS}")


def mixed(img: np.ndarray, sigma: float, density: float, rng: np.random.Generator) -> np.ndarray:
    """Gaussian followed by salt-and-pepper (used by the GUI)."""
    return salt_pepper(gaussian(img, sigma, rng), density, rng)
