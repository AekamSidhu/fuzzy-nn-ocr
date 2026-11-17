"""Hamming net + Maxnet (syllabus Unit 3).

Hamming net: weights are half the stored bipolar templates and the bias is n/2,
so each output unit's net input equals the number of pixels on which the input
agrees with its template (n minus the Hamming distance).

Maxnet: a fully connected competitive layer with self-weight 1 and mutual
inhibition -eps (0 < eps < 1/m). It iterates
    a_j <- f(a_j - eps * sum_{k != j} a_k),  f(x) = max(x, 0)
until only one unit stays positive: the winner.
"""
import numpy as np


class Maxnet:
    def __init__(self, n_units: int, eps: float | None = None):
        self.m = n_units
        self.eps = eps if eps is not None else 1.0 / (2 * n_units)
        if not 0 < self.eps < 1.0 / n_units:
            raise ValueError("Maxnet needs 0 < eps < 1/m")

    def run(self, activations: np.ndarray, max_iter: int = 5000):
        """Return (winner index per row, iterations used per row, final activations)."""
        A = np.array(np.atleast_2d(activations), dtype=float, copy=True)
        # Exact ties would never separate; break them by a tiny, index-based offset
        # (lower index wins) so the competition is deterministic.
        A = A - np.arange(self.m) * 1e-9 * (A.max(axis=1, keepdims=True) + 1.0)
        A = np.maximum(A, 0.0)
        iters = np.zeros(len(A), dtype=int)
        for _ in range(max_iter):
            active = (A > 0).sum(axis=1) > 1
            if not active.any():
                break
            total = A[active].sum(axis=1, keepdims=True)
            A[active] = np.maximum(A[active] - self.eps * (total - A[active]), 0.0)
            iters[active] += 1
        return A.argmax(axis=1), iters, A


class HammingNet:
    def __init__(self, templates: np.ndarray):
        self.E = np.atleast_2d(np.asarray(templates, dtype=float))  # (m, n) bipolar
        self.W = self.E.T / 2.0
        self.b = self.E.shape[1] / 2.0
        self.maxnet = Maxnet(self.E.shape[0])

    def agreements(self, X: np.ndarray) -> np.ndarray:
        """Net input of each Hamming unit = n - HammingDistance(x, template)."""
        return self.b + np.atleast_2d(X) @ self.W

    def predict(self, X: np.ndarray, return_iters: bool = False):
        winners, iters, _ = self.maxnet.run(self.agreements(X))
        return (winners, iters) if return_iters else winners
