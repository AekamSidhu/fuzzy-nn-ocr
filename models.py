"""Build and train every network with fixed seeds.

Shared by experiments.py, selftest.py and gui.py so all three use identical models.

Two preprocessing pipelines are compared throughout:
  "threshold": noisy image -> threshold at 0.5 -> bipolar
  "fuzzy":     noisy image -> fuzzy.enhance -> threshold at 0.5 -> bipolar
The perceptron is the only network trained on noisy data, so it gets one copy
per pipeline, each trained on the same noisy images passed through its own
pipeline (no train/test mismatch). The other networks store the clean glyphs only.
"""
from dataclasses import dataclass

import numpy as np

import fuzzy
from bam import BAM, hadamard_codes
from chars import CHARSET, N_PIXELS, glyphs, to_bipolar
from hamming import HammingNet
from hopfield import Hopfield
from noise import add_noise
from perceptron import Perceptron
from subset import memory_subset

SEED = 2026
PIPELINES = ("threshold", "fuzzy")

TRAIN_NOISE = (("gaussian", 0.2), ("gaussian", 0.3), ("saltpepper", 0.05), ("saltpepper", 0.1))
TRAIN_COPIES = 10


def preprocess(noisy: np.ndarray, pipeline: str) -> np.ndarray:
    """Grayscale noisy image(s) -> bipolar network input for the given pipeline."""
    if pipeline == "threshold":
        return to_bipolar(noisy)
    if pipeline == "fuzzy":
        return to_bipolar(fuzzy.enhance(noisy))
    raise ValueError(f"unknown pipeline {pipeline!r}; expected one of {PIPELINES}")


@dataclass
class Models:
    chars: str               # all 36 characters
    memory_chars: str        # small set stored by the Hebbian memories
    perceptrons: dict        # pipeline -> Perceptron
    hamming: HammingNet
    hopfield: Hopfield       # projection rule, all 36 characters
    hopfield_hebb: Hopfield  # Hebbian rule, memory_chars only
    bam: BAM                 # Hebbian, memory_chars only


def perceptron_training_set(pipeline: str, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Clean glyphs plus noisy copies, binarised by ``pipeline``.

    The same seed gives the same noisy images for both pipelines.
    """
    rng = np.random.default_rng(seed)
    clean = glyphs()
    X = [to_bipolar(clean)]
    y = [np.arange(len(CHARSET))]
    for kind, level in TRAIN_NOISE:
        reps = np.repeat(clean, TRAIN_COPIES, axis=0)
        X.append(preprocess(add_noise(reps, kind, level, rng), pipeline))
        y.append(np.repeat(np.arange(len(CHARSET)), TRAIN_COPIES))
    return np.concatenate(X), np.concatenate(y)


def build(seed: int = SEED) -> Models:
    full = to_bipolar(glyphs(CHARSET))
    mem_chars = memory_subset()
    mem = to_bipolar(glyphs(mem_chars))

    perceptrons = {}
    for pipeline in PIPELINES:
        X, y = perceptron_training_set(pipeline, seed)
        perceptrons[pipeline] = Perceptron(N_PIXELS, len(CHARSET)).fit(X, y, epochs=50, seed=seed)

    return Models(
        chars=CHARSET,
        memory_chars=mem_chars,
        perceptrons=perceptrons,
        hamming=HammingNet(full),
        hopfield=Hopfield(N_PIXELS).store(full, rule="projection"),
        hopfield_hebb=Hopfield(N_PIXELS).store(mem, rule="hebbian"),
        bam=BAM(hadamard_codes(len(mem_chars))).store(mem),
    )


def predict_full(models: Models, X: np.ndarray, pipeline: str, seed: int = 0) -> dict[str, np.ndarray]:
    """Class indices into CHARSET for the networks that know all 36 characters.

    Hopfield returns -1 when recall ends in a spurious state.
    """
    return {
        "Perceptron": models.perceptrons[pipeline].predict(X),
        "Hamming+Maxnet": models.hamming.predict(X),
        "Hopfield (projection)": models.hopfield.match(models.hopfield.recall(X, seed=seed)),
    }


def predict_memory(models: Models, X: np.ndarray, seed: int = 0) -> dict[str, np.ndarray]:
    """Class indices into memory_chars for the Hebbian memories."""
    return {
        "Hopfield (Hebbian)": models.hopfield_hebb.match(models.hopfield_hebb.recall(X, seed=seed)),
        "BAM": models.bam.classify(X),
    }
