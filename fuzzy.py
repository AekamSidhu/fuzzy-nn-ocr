"""Fuzzy image enhancement (syllabus Units 4 and 5), written from scratch in numpy.

Pipeline for each noisy grayscale image in [0, 1]:

0. Fuzzification of the image (Unit 4): each pixel's membership in the fuzzy
   set "ink" is computed from the image's own intensity range (5th to 95th
   percentile mapped to 0..1). A clean black-on-white image is unchanged; a
   faded, low-contrast scan is stretched to the full range.
1. INT contrast intensification (Unit 4, an operation on fuzzy sets): the
   membership values are pushed away from 0.5.
2. A Mamdani fuzzy inference system (Unit 5) that filters each pixel using two
   inputs: its own intensity ``p`` and the mean of its 8 neighbours ``m``.
   * Fuzzification: triangular / trapezoidal membership functions low,
     medium, high on both inputs (0 = paper, 1 = ink).
   * 9 IF-THEN rules (see RULES), AND = min.
   * Implication = min (clipping), aggregation = max.
   * Defuzzification = centroid over a discretized output universe.

Everything is vectorized over images and pixels; there are no per-pixel loops.
"""
import numpy as np

LABELS = ("low", "medium", "high")  # value 0 = paper, 1 = ink
UNIVERSE = np.linspace(0.0, 1.0, 101)  # discretized output universe


# ----------------------------------------------------------------------------
# Membership functions
# ----------------------------------------------------------------------------
def trapmf(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """Trapezoid: 0 below a, rises to 1 at b, flat to c, falls to 0 at d.

    a == b or c == d gives a shoulder (vertical edge) instead of a ramp.
    """
    x = np.asarray(x, dtype=float)
    rise = np.ones_like(x) if b == a else np.clip((x - a) / (b - a), 0.0, 1.0)
    fall = np.ones_like(x) if d == c else np.clip((d - x) / (d - c), 0.0, 1.0)
    mu = np.minimum(rise, fall)
    if a == b:
        mu = np.where(x < a, 0.0, mu)
    if c == d:
        mu = np.where(x > d, 0.0, mu)
    return mu


def trimf(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Triangle with feet at a and c and peak at b."""
    return trapmf(x, a, b, b, c)


# Input fuzzy sets for the pixel intensity p.
INPUT_MF = {
    "low": (0.0, 0.0, 0.2, 0.45),
    "medium": (0.2, 0.45, 0.55, 0.8),
    "high": (0.55, 0.8, 1.0, 1.0),
}
# Input fuzzy sets for the neighbourhood mean m (fraction of the 8 neighbours that are ink).
# With ~3 px strokes on the 24x24 grid: a pixel inside a stroke sees mostly ink
# (m near 1), a stroke-edge pixel about 5/8, background beside a stroke about 3/8,
# and open background about 0. "low" ends at 0.4 so isolated specks are removed;
# "high" starts at 0.7 so holes inside strokes are filled without filling the
# concave corners of a glyph. Breakpoints were tuned by grid search on a separate
# seed, with an objective that weights lost ink 1.5x more than added ink.
NEIGHBOUR_MF = {
    "low": (0.0, 0.0, 0.2, 0.4),
    "medium": (0.2, 0.4, 0.7, 0.9),
    "high": (0.7, 0.9, 1.0, 1.0),
}
# Output fuzzy sets for the filtered intensity.
OUTPUT_MF = {
    "low": (0.0, 0.0, 0.1, 0.35),
    "medium": (0.3, 0.5, 0.5, 0.7),
    "high": (0.65, 0.9, 1.0, 1.0),
}

# RULES[p_label][m_label] -> output label.
#   p = pixel value, m = mean of the 8 neighbours (0 = paper, 1 = ink).
RULES = {
    "low":    {"low": "low", "medium": "low",    "high": "high"},  # hole inside a stroke -> fill with ink
    "medium": {"low": "low", "medium": "medium", "high": "high"},  # unsure pixel follows its neighbours
    "high":   {"low": "low", "medium": "high",   "high": "high"},  # isolated speck on paper -> remove
}


def fuzzify(x: np.ndarray, mf: dict = INPUT_MF) -> np.ndarray:
    """Degrees of membership of x in each label. Shape (3, *x.shape)."""
    return np.stack([trapmf(x, *mf[lbl]) for lbl in LABELS])


_OUT_CURVES = np.stack([trapmf(UNIVERSE, *OUTPUT_MF[lbl]) for lbl in LABELS])  # (3, U)
_RULE_IDX = np.array([[LABELS.index(RULES[p][m]) for m in LABELS] for p in LABELS])  # (3, 3)


# ----------------------------------------------------------------------------
# Operators
# ----------------------------------------------------------------------------
def fuzzify_image(img: np.ndarray, lo_pct: float = 5.0, hi_pct: float = 95.0) -> np.ndarray:
    """Membership of each pixel in "ink", from the image's own intensity range.

    Works on one image (H, W) or a batch (N, H, W); each image is scaled separately.
    """
    img = np.asarray(img, dtype=float)
    lo = np.percentile(img, lo_pct, axis=(-2, -1), keepdims=True)
    hi = np.percentile(img, hi_pct, axis=(-2, -1), keepdims=True)
    span = hi - lo
    stretched = np.clip((img - lo) / np.maximum(span, 1e-6), 0.0, 1.0)
    return np.where(span < 1e-3, img, stretched)  # a flat image has no range to stretch


def intensify(mu: np.ndarray) -> np.ndarray:
    """INT contrast-intensification operator: 2mu^2 if mu <= 0.5 else 1 - 2(1-mu)^2."""
    mu = np.asarray(mu, dtype=float)
    return np.where(mu <= 0.5, 2.0 * mu**2, 1.0 - 2.0 * (1.0 - mu) ** 2)


def neighbour_mean(img: np.ndarray) -> np.ndarray:
    """Mean of the 8 neighbours of each pixel (edge pixels use reflection padding)."""
    pad = [(0, 0)] * (img.ndim - 2) + [(1, 1), (1, 1)]
    p = np.pad(img, pad, mode="reflect")
    h, w = img.shape[-2:]
    total = sum(
        p[..., 1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if (dy, dx) != (0, 0)
    )
    return total / 8.0


def infer(p: np.ndarray, m: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """Mamdani inference + centroid defuzzification for arrays of (p, m) pairs."""
    p = np.asarray(p, dtype=float)
    m = np.asarray(m, dtype=float)
    shape = p.shape
    p, m = p.reshape(-1), m.reshape(-1)
    out = np.empty_like(p)

    for start in range(0, p.size, chunk):
        sl = slice(start, start + chunk)
        mu_p = fuzzify(p[sl])  # (3, n)
        mu_m = fuzzify(m[sl], NEIGHBOUR_MF)  # (3, n)
        # Firing strength of every rule: AND = min.  (3, 3, n)
        fire = np.minimum(mu_p[:, None, :], mu_m[None, :, :])
        # Strength per output label = max over rules concluding that label.
        strength = np.zeros((3, fire.shape[-1]))
        for k in range(3):
            strength[k] = fire[_RULE_IDX == k].max(axis=0)
        # Clip each output set (min implication), aggregate with max.  (n, U)
        agg = np.minimum(strength.T[:, :, None], _OUT_CURVES[None, :, :]).max(axis=1)
        area = agg.sum(axis=1)
        centroid = (agg * UNIVERSE).sum(axis=1) / np.where(area > 0, area, 1.0)
        out[sl] = np.where(area > 0, centroid, p[sl])
    return out.reshape(shape)


def rule_filter(img: np.ndarray) -> np.ndarray:
    """One pass of the fuzzy rule-based pixel filter."""
    return infer(img, neighbour_mean(img))


def enhance(img: np.ndarray, passes: int = 1, use_int: bool = True, adaptive: bool = True) -> np.ndarray:
    """Full fuzzy enhancement: fuzzification, INT, then the rule-based filter.

    Accepts a single image (H, W) or a batch (N, H, W) with values in [0, 1].
    """
    return enhance_stages(img, passes, use_int, adaptive)["filtered"]


def enhance_stages(img: np.ndarray, passes: int = 1, use_int: bool = True, adaptive: bool = True) -> dict:
    """Same as ``enhance`` but returns every intermediate image (for plots and the GUI)."""
    out = {"input": np.asarray(img, dtype=float)}
    out["fuzzified"] = fuzzify_image(out["input"]) if adaptive else out["input"]
    out["intensified"] = intensify(out["fuzzified"]) if use_int else out["fuzzified"]
    filtered = out["intensified"]
    for _ in range(passes):
        filtered = rule_filter(filtered)
    out["filtered"] = filtered
    return out


def control_surface(n: int = 41) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Output of the inference system over a grid of (p, m), for plotting."""
    axis = np.linspace(0.0, 1.0, n)
    P, M = np.meshgrid(axis, axis)
    return P, M, infer(P, M)


def rule_table() -> list[tuple[str, str, str]]:
    """The rule base as (p_label, m_label, output_label) triples."""
    return [(p, m, RULES[p][m]) for p in LABELS for m in LABELS]
