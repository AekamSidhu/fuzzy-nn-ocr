"""Discrete Hopfield network (syllabus Unit 2).

* Storage, two rules (both give a symmetric weight matrix with zero diagonal):
  - "hebbian": the syllabus outer-product rule, W = sum_k p_k p_k^T.
  - "projection": the pseudo-inverse rule, W = P^+ P (an extension beyond the
    syllabus). It decorrelates the stored patterns, so strongly overlapping
    glyphs can still be fixed points, which the Hebbian rule cannot manage
    beyond about 4 glyphs.
* Recall: asynchronous updates in random order, s_i <- sign(sum_j w_ij s_j);
  a zero net input leaves the neuron unchanged.
* Energy: E = -1/2 s^T W s, which never increases under asynchronous updates
  (true for any symmetric W with zero diagonal, so it holds for both rules).
"""
import numpy as np

RULES = ("hebbian", "projection")


class Hopfield:
    def __init__(self, n_neurons: int):
        self.n = n_neurons
        self.W = np.zeros((n_neurons, n_neurons))
        self.patterns = np.empty((0, n_neurons))
        self.rule = None

    def store(self, patterns: np.ndarray, rule: str = "hebbian") -> "Hopfield":
        """Store bipolar patterns (k, n) with the chosen learning rule."""
        P = np.atleast_2d(np.asarray(patterns, dtype=float))
        if rule == "hebbian":
            W = P.T @ P
        elif rule == "projection":
            W = np.linalg.pinv(P) @ P
            W = (W + W.T) / 2.0  # remove floating-point asymmetry
        else:
            raise ValueError(f"unknown rule {rule!r}; expected one of {RULES}")
        np.fill_diagonal(W, 0.0)
        self.W = W
        self.rule = rule
        self.patterns = P
        return self

    def energy(self, s: np.ndarray) -> np.ndarray:
        s = np.atleast_2d(s)
        return -0.5 * np.einsum("bi,ij,bj->b", s, self.W, s)

    def is_fixed_point(self, s: np.ndarray) -> bool:
        h = self.W @ s
        return bool(np.all((h * s > 0) | (h == 0)))

    def recall(
        self,
        states: np.ndarray,
        max_sweeps: int = 30,
        seed: int = 0,
        record: bool = False,
    ):
        """Run asynchronous recall on one state (n,) or a batch (B, n).

        Each sweep visits every neuron once in a random order (shared across the
        batch). Stops when a whole sweep changes nothing.

        Returns final states, or (final states, history) if ``record`` is True.
        History is a list of (state copy, energy) after every single-neuron
        update for the first state in the batch (for animation / energy plots).
        """
        rng = np.random.default_rng(seed)
        S = np.array(np.atleast_2d(states), dtype=float, copy=True)
        history = [(S[0].copy(), float(self.energy(S[0])[0]))] if record else None

        for _ in range(max_sweeps):
            changed = False
            for i in rng.permutation(self.n):
                h = S @ self.W[:, i]
                new = np.where(h > 0, 1.0, np.where(h < 0, -1.0, S[:, i]))
                if np.any(new != S[:, i]):
                    changed = True
                    S[:, i] = new
                if record:
                    history.append((S[0].copy(), float(self.energy(S[0])[0])))
            if not changed:
                break

        out = S if np.ndim(states) == 2 else S[0]
        return (out, history) if record else out

    def match(self, states: np.ndarray) -> np.ndarray:
        """Index of the stored pattern each state equals exactly, or -1 (spurious)."""
        S = np.atleast_2d(states)
        eq = (S[:, None, :] == self.patterns[None, :, :]).all(axis=2)
        return np.where(eq.any(axis=1), eq.argmax(axis=1), -1)
