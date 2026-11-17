"""Choose which characters the Hebbian associative memories store.

Hebbian memories hold only a few patterns: roughly 0.138 n for random
patterns, and far fewer for glyphs, which share a large common background and
overlapping strokes. On these glyphs no 5-character set is fully stable.
We pick mutually distant characters (greedy max-min Hamming distance) and keep
the largest prefix whose patterns are all stable:

* Hebbian Hopfield: every pattern is a fixed point.
* BAM: every (image, code) pair is stable in both directions.

The projection-rule Hopfield is not limited this way and stores all 36.
"""
import numpy as np

from bam import BAM, hadamard_codes
from chars import CHARSET, glyphs, to_bipolar
from hopfield import Hopfield


def greedy_order(chars: str = CHARSET) -> str:
    """Order characters so each next one maximises its minimum distance to those before."""
    B = to_bipolar(glyphs(chars))
    D = (B[:, None, :] != B[None, :, :]).sum(axis=2)
    order = [int(D.sum(axis=1).argmax())]
    while len(order) < len(chars):
        rest = [i for i in range(len(chars)) if i not in order]
        order.append(max(rest, key=lambda i: D[i, order].min()))
    return "".join(chars[i] for i in order)


def hebbian_hopfield_stable(chars: str) -> bool:
    P = to_bipolar(glyphs(chars))
    net = Hopfield(P.shape[1]).store(P, rule="hebbian")
    return all(net.is_fixed_point(p) for p in P)


def bam_stable(chars: str) -> bool:
    P = to_bipolar(glyphs(chars))
    return BAM(hadamard_codes(len(chars))).store(P).is_stable()


def memory_subset(max_size: int = 8) -> str:
    """Largest greedy prefix (<= max_size) that BAM and Hebbian Hopfield both store stably."""
    order = greedy_order()
    best = order[:1]
    for k in range(2, max_size + 1):
        if bam_stable(order[:k]) and hebbian_hopfield_stable(order[:k]):
            best = order[:k]
        else:
            break
    return best


def fixed_point_count(chars: str, rule: str) -> int:
    """How many of ``chars`` are fixed points when all are stored with ``rule``."""
    P = to_bipolar(glyphs(chars))
    net = Hopfield(P.shape[1]).store(P, rule=rule)
    return int(sum(net.is_fixed_point(p) for p in P))
