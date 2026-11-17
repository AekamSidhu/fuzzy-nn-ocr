"""Checkpoint: does fuzzy enhancement reduce pixel errors before any network sees the image?

Run:  python checkpoint.py
Prints the pixel error rate (fraction of pixels wrong after thresholding at 0.5)
for plain thresholding, several fuzzy configurations, and a 3x3 median filter,
across noise levels.
"""
import numpy as np

import fuzzy
from chars import glyphs
from noise import add_noise

SEED = 7
TRIALS = 20
LEVELS = {
    "gaussian": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    "saltpepper": [0.05, 0.1, 0.2, 0.3, 0.4],
}


def median3(img: np.ndarray) -> np.ndarray:
    """3x3 median filter (non-fuzzy reference)."""
    pad = [(0, 0)] * (img.ndim - 2) + [(1, 1), (1, 1)]
    p = np.pad(img, pad, mode="reflect")
    h, w = img.shape[-2:]
    stack = [p[..., dy : dy + h, dx : dx + w] for dy in range(3) for dx in range(3)]
    return np.median(np.stack(stack), axis=0)


METHODS = {
    "threshold": lambda x: x,
    "rule x1": lambda x: fuzzy.rule_filter(x),
    "rule x2": lambda x: fuzzy.rule_filter(fuzzy.rule_filter(x)),
    "INT->rule": lambda x: fuzzy.rule_filter(fuzzy.intensify(x)),
    "enhance()": fuzzy.enhance,
    "median3": median3,
}


def pixel_error(clean: np.ndarray, processed: np.ndarray) -> float:
    return float(((processed >= 0.5) != (clean >= 0.5)).mean())


def ink_changes(clean: np.ndarray, processed: np.ndarray) -> tuple[float, float]:
    """Mean ink pixels lost (stroke -> background) and added (background -> stroke) per image."""
    ink, out = clean >= 0.5, processed >= 0.5
    axes = tuple(range(1, clean.ndim))
    lost = (ink & ~out).sum(axis=axes).mean()
    added = (~ink & out).sum(axis=axes).mean()
    return float(lost), float(added)


def run() -> tuple[dict, dict]:
    rng = np.random.default_rng(SEED)
    clean = np.repeat(glyphs(), TRIALS, axis=0)
    errors, inks = {}, {}
    for kind, levels in LEVELS.items():
        for level in levels:
            noisy = add_noise(clean, kind, level, rng)
            errors[(kind, level)] = {name: pixel_error(clean, fn(noisy)) for name, fn in METHODS.items()}
            inks[(kind, level)] = {
                "threshold": ink_changes(clean, noisy),
                "enhance()": ink_changes(clean, fuzzy.enhance(noisy)),
            }
    return errors, inks


if __name__ == "__main__":
    errors, inks = run()
    names = list(METHODS)
    print("Pixel error rate")
    print(f"{'noise':>18} | " + " | ".join(f"{n:>11}" for n in names))
    for (kind, level), row in errors.items():
        print(f"{kind:>12} {level:>5} | " + " | ".join(f"{row[n]:>11.4f}" for n in names))
    print("\nInk pixels lost / added per image")
    print(f"{'noise':>18} | {'threshold':>15} | {'fuzzy enhance':>15}")
    for (kind, level), row in inks.items():
        t, f = row["threshold"], row["enhance()"]
        print(f"{kind:>12} {level:>5} | {t[0]:>6.1f} / {t[1]:<6.1f} | {f[0]:>6.1f} / {f[1]:<6.1f}")
