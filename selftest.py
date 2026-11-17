"""Correctness checks for every component.  Run:  python selftest.py

Prints PASS/FAIL per check and exits with status 1 if anything fails.
No test framework needed.
"""
import sys

import numpy as np

import fuzzy
import models as M
from chars import CHARSET, glyphs, to_bipolar
from hamming import Maxnet
from noise import add_noise

CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


# --- fuzzy stage ---------------------------------------------------------------
@check
def membership_functions_are_valid():
    x = np.linspace(0, 1, 1001)
    for mf in (fuzzy.INPUT_MF, fuzzy.NEIGHBOUR_MF, fuzzy.OUTPUT_MF):
        mu = fuzzy.fuzzify(x, mf)
        assert mu.min() >= 0 and mu.max() <= 1, "membership outside [0, 1]"
        assert (mu.max(axis=0) > 0).all(), "some input value belongs to no fuzzy set"
    assert fuzzy.trapmf(0.5, 0, 0.4, 0.6, 1) == 1.0
    assert fuzzy.trimf(0.25, 0, 0.5, 1) == 0.5


@check
def centroid_of_symmetric_set_is_its_centre():
    # p and m both fully "medium" -> only the medium/medium rule fires -> symmetric triangle at 0.5
    out = fuzzy.infer(np.array([0.5]), np.array([0.55]))
    assert abs(out[0] - 0.5) < 1e-6, out


@check
def int_operator_properties():
    mu = np.linspace(0, 1, 101)
    out = fuzzy.intensify(mu)
    assert np.allclose(fuzzy.intensify(np.array([0.0, 0.5, 1.0])), [0.0, 0.5, 1.0])
    assert np.all(np.diff(out) >= 0), "INT must be monotone"
    assert np.all(out[mu < 0.5] <= mu[mu < 0.5]) and np.all(out[mu > 0.5] >= mu[mu > 0.5]), "INT must increase contrast"


@check
def fuzzification_leaves_clean_glyphs_unchanged():
    g = glyphs()
    assert np.array_equal(fuzzy.fuzzify_image(g), g)


@check
def enhancement_preserves_clean_glyphs():
    g = glyphs()
    wrong = ((fuzzy.enhance(g) >= 0.5) != (g >= 0.5)).mean()
    assert wrong < 0.005, f"{wrong:.3%} of pixels changed on clean input"


@check
def fuzzy_reduces_pixel_error_at_moderate_noise():
    rng = np.random.default_rng(1)
    clean = np.repeat(glyphs(), 5, axis=0)
    for kind, level in (("gaussian", 0.3), ("saltpepper", 0.1), ("faded", 0.2)):
        noisy = add_noise(clean, kind, level, rng)
        thr = ((noisy >= 0.5) != (clean >= 0.5)).mean()
        fz = ((fuzzy.enhance(noisy) >= 0.5) != (clean >= 0.5)).mean()
        assert fz < thr, f"{kind} {level}: fuzzy {fz:.3f} vs threshold {thr:.3f}"


# --- networks ------------------------------------------------------------------
MODELS = M.build()
FULL = to_bipolar(glyphs(CHARSET))
MEM = to_bipolar(glyphs(MODELS.memory_chars))


@check
def every_network_recognises_clean_inputs():
    idx = np.arange(len(CHARSET))
    for pipe in M.PIPELINES:
        for net, pred in M.predict_full(MODELS, FULL, pipe).items():
            assert np.array_equal(pred, idx), f"{net} ({pipe}) misses clean glyphs"
    for net, pred in M.predict_memory(MODELS, MEM).items():
        assert np.array_equal(pred, np.arange(len(MEM))), f"{net} misses clean glyphs"


@check
def stored_patterns_are_fixed_points():
    assert all(MODELS.hopfield.is_fixed_point(p) for p in FULL), "projection Hopfield"
    assert all(MODELS.hopfield_hebb.is_fixed_point(p) for p in MEM), "Hebbian Hopfield"


@check
def hopfield_weights_symmetric_zero_diagonal():
    for net in (MODELS.hopfield, MODELS.hopfield_hebb):
        assert np.allclose(net.W, net.W.T) and np.all(np.diag(net.W) == 0)


@check
def hopfield_energy_never_increases():
    rng = np.random.default_rng(2)
    for net in (MODELS.hopfield, MODELS.hopfield_hebb):
        for ch in "AK8":
            start = to_bipolar(add_noise(glyphs(ch)[0], "gaussian", 0.6, rng))
            _, hist = net.recall(start, seed=3, record=True)
            energy = np.array([e for _, e in hist])
            assert np.all(np.diff(energy) <= 1e-9), f"energy rose during recall of {ch}"


@check
def maxnet_leaves_exactly_one_winner():
    rng = np.random.default_rng(4)
    acts = rng.integers(0, 577, size=(200, 36)).astype(float)
    acts[:5, :2] = 300.0  # exact ties
    winners, iters, final = Maxnet(36).run(acts)
    assert np.all((final > 0).sum(axis=1) == 1), "Maxnet did not converge to a single winner"
    assert np.array_equal(winners[5:], acts[5:].argmax(axis=1)), "winner is not the largest input"


@check
def hamming_scores_equal_agreement_counts():
    x = FULL[3]
    agreements = (FULL == x).sum(axis=1)
    assert np.allclose(MODELS.hamming.agreements(x)[0], agreements)


@check
def bam_pairs_stable_both_directions():
    assert MODELS.bam.is_stable()


@check
def perceptrons_converged():
    for pipe, p in MODELS.perceptrons.items():
        assert p.errors_per_epoch[-1] == 0, f"{pipe} perceptron still makes training errors"


def main() -> int:
    failed = 0
    for fn in CHECKS:
        name = fn.__name__.replace("_", " ")
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {name}: {exc}")
    print(f"\n{len(CHECKS) - failed}/{len(CHECKS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
