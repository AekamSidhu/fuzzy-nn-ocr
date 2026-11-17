"""Bidirectional Associative Memory (syllabus Unit 3).

Hetero-associative memory between an image layer X (n = 256 units) and a code
layer Y (p units). Storage is Hebbian, W = sum_k x_k^T y_k. Recall alternates
    y <- sign(x W),   x <- sign(y W^T)
(a zero net input leaves the unit unchanged) until neither layer changes.
Class codes are rows of a Sylvester-Hadamard matrix, which are mutually
orthogonal, so stored codes are as far apart as possible.
"""
import numpy as np


def hadamard_codes(n_codes: int, length: int = 16) -> np.ndarray:
    """``n_codes`` mutually orthogonal bipolar codes (skips the all-ones row)."""
    H = np.array([[1.0]])
    while H.shape[0] < length:
        H = np.block([[H, H], [H, -H]])
    if n_codes > length - 1:
        raise ValueError(f"at most {length - 1} codes of length {length}")
    return H[1 : n_codes + 1]


def _sign_keep(h: np.ndarray, prev: np.ndarray) -> np.ndarray:
    return np.where(h > 0, 1.0, np.where(h < 0, -1.0, prev))


class BAM:
    def __init__(self, codes: np.ndarray):
        self.codes = np.atleast_2d(np.asarray(codes, dtype=float))  # (k, p)
        self.W = np.zeros((0, self.codes.shape[1]))
        self.patterns = np.empty((0, 0))

    def store(self, patterns: np.ndarray) -> "BAM":
        """Store pairs (patterns[k], codes[k]) with the Hebbian outer-product rule."""
        X = np.atleast_2d(np.asarray(patterns, dtype=float))
        if len(X) != len(self.codes):
            raise ValueError("need exactly one code per stored pattern")
        self.patterns = X
        self.W = X.T @ self.codes
        return self

    def recall(self, X: np.ndarray, max_iter: int = 50):
        """Recall from images X (B, n). Returns (x_final, y_final, iterations)."""
        X = np.array(np.atleast_2d(X), dtype=float, copy=True)
        Y = np.zeros((len(X), self.codes.shape[1]))
        it = 0
        for it in range(1, max_iter + 1):
            Y_new = _sign_keep(X @ self.W, Y)
            X_new = _sign_keep(Y_new @ self.W.T, X)
            done = np.array_equal(Y_new, Y) and np.array_equal(X_new, X)
            X, Y = X_new, Y_new
            if done:
                break
        return X, Y, it

    def classify(self, X: np.ndarray) -> np.ndarray:
        """Class index whose code is nearest (Hamming) to the recalled Y."""
        _, Y, _ = self.recall(X)
        return (Y @ self.codes.T).argmax(axis=1)

    def is_stable(self) -> bool:
        """Every stored pair is a fixed point in both directions."""
        y = _sign_keep(self.patterns @ self.W, np.zeros_like(self.codes))
        x = _sign_keep(self.codes @ self.W.T, np.zeros_like(self.patterns))
        return bool(np.array_equal(y, self.codes) and np.array_equal(x, self.patterns))
