"""Multi-class perceptron (syllabus Unit 1).

One bipolar output unit per class. Net input y_in = b + x.w. Training uses
the perceptron learning rule: for each output unit whose response differs from
its target t (+1 for the true class, -1 otherwise), w += lr * t * x and
b += lr * t. Classification picks the unit with the largest net input.
"""
import numpy as np


class Perceptron:
    def __init__(self, n_inputs: int, n_classes: int, lr: float = 1.0, theta: float = 0.0):
        self.w = np.zeros((n_inputs, n_classes))
        self.b = np.zeros(n_classes)
        self.lr = lr
        self.theta = theta
        self.epochs_run = 0
        self.errors_per_epoch: list[int] = []

    def _activate(self, y_in: np.ndarray) -> np.ndarray:
        """Bipolar step with an undecided band [-theta, theta] -> 0."""
        return np.where(y_in > self.theta, 1.0, np.where(y_in < -self.theta, -1.0, 0.0))

    def fit(self, X: np.ndarray, labels: np.ndarray, epochs: int = 50, seed: int = 0) -> "Perceptron":
        """Train on bipolar inputs X (N, n) with integer class labels (N,)."""
        rng = np.random.default_rng(seed)
        n_classes = self.w.shape[1]
        targets = -np.ones((len(X), n_classes))
        targets[np.arange(len(X)), labels] = 1.0

        for _ in range(epochs):
            errors = 0
            for i in rng.permutation(len(X)):
                x, t = X[i], targets[i]
                y = self._activate(self.b + x @ self.w)
                wrong = y != t
                if wrong.any():
                    errors += 1
                    self.w[:, wrong] += self.lr * np.outer(x, t[wrong])
                    self.b[wrong] += self.lr * t[wrong]
            self.epochs_run += 1
            self.errors_per_epoch.append(errors)
            if errors == 0:
                break
        return self

    def net_input(self, X: np.ndarray) -> np.ndarray:
        return self.b + np.atleast_2d(X) @ self.w

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Class index with the largest net input, for each row of X."""
        return self.net_input(X).argmax(axis=1)
