"""Reference-style implementation of Versatile Elliptic Basis Function (VEBF).

This version follows the VEBF rules used by the reference implementation:
- Initial node width: one global average pairwise distance vector per training batch.
- New node: center at the incoming sample, zero covariance, identity eigenvectors,
  n = 1, and the class's initialized width.
- Growth test: tentative center + tentative covariance/eigenvectors are used with
  the current node width.
- Width update: only after n > N0 (N0 = 2), expand each axis by the projected
  center displacement.
- Merge: merge same-class nodes when either center is inside the other's
  hyperellipsoid (psi <= theta); merged width is sqrt(2*pi*|lambda|).
- Prediction: minimum hyperellipsoidal function value over all nodes.

The public train(x, y) method is retained as a one-sample convenience wrapper.
For paper-style initialization, prefer fit(X, y) for the initial batch and then
partial_fit(X_chunk, y_chunk) for subsequent streaming updates.
"""

import math
import random
from itertools import chain

import numpy as np
from scipy.spatial.distance import pdist


class Node:
    """A VEBF hyperellipsoidal node."""

    def __init__(self, dimension, label, A, n=1, M=None, S=None, U=None, eps=1e-10):
        self.dim = dimension
        self.label = label
        self.A = np.asarray(A, dtype=float).copy()
        self.eps = float(eps)
        self.n = int(n)
        self.M = np.zeros(dimension, dtype=float) if M is None else np.asarray(M, dtype=float).copy()
        self.S = np.zeros((dimension, dimension), dtype=float) if S is None else np.asarray(S, dtype=float).copy()
        # U stores eigenvectors as columns.  This is equivalent to the reference
        # implementation's row-wise eig_component representation.
        self.U = np.eye(dimension, dtype=float) if U is None else np.asarray(U, dtype=float).copy()

    def __str__(self):
        return f"A node covering {self.n} points."


class VEBF:
    """Versatile Elliptic Basis Function classifier following the reference rules.

    Parameters
    ----------
    dimension : int
        Input dimensionality.
    merge_parameter : float
        Equivalent to the reference implementation's theta: overlap threshold
        used during same-class node merging.  Default is 0.
    eps : float
        Small numerical floor added when evaluating ellipsoids/eigenvalues.
    delta : float
        Scaling factor for the initial pairwise-distance width.  Default is 1.
    """

    # VEBF paper constant: width expansion starts only after the node has more
    # than two samples.
    N0 = 2

    def __init__(self, dimension, merge_parameter=0.0, eps=1e-10, delta=1.0):
        self.dim = int(dimension)
        self.theta = float(merge_parameter)
        self.eps = float(eps)
        self.delta = float(delta)

        self.nodes = {}
        self.init_width = {}
        self.classes_ = np.array([], dtype=object)

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _as_2d(X, dim=None):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2:
            raise ValueError(f"X must have shape (n_samples, n_features), got {X.shape}")
        if dim is not None and X.shape[1] != dim:
            raise ValueError(f"Expected {dim} features, got {X.shape[1]}")
        return X

    def average_pairwise_distance(self, X):
        """Reference VEBF initial-width formula.

        The reference implementation uses

            2 * delta / n^2 * sum_{i<j} ||x_i - x_j||_2

        and copies the resulting scalar into every axis.
        """
        X = self._as_2d(X, self.dim)
        n = len(X)
        distance = 0.0 if n < 2 else float(np.sum(pdist(X)))
        width = (2.0 * self.delta / (n ** 2)) * distance if n > 0 else 0.0
        return np.full(self.dim, width, dtype=float)

    def surface_single(self, x, U, A, k=1.0):
        """Reference L2 hyperellipsoidal function for one point."""
        x = np.asarray(x, dtype=float)
        U = np.asarray(U, dtype=float)
        A = np.asarray(A, dtype=float)
        x_local = U.T @ x
        margin = A + self.eps
        return np.linalg.norm(x_local / margin, ord=2) ** 2 - (k ** 2)

    def surface_multi(self, xs, Us, As, ks=None):
        """Reference L2 hyperellipsoidal function for point/node pairs."""
        xs = np.asarray(xs, dtype=float)
        Us = np.asarray(Us, dtype=float)
        As = np.asarray(As, dtype=float)

        if xs.ndim == 1:
            xs = xs[None, :]
        if Us.ndim != 3 or As.ndim != 2:
            raise ValueError("Us must have shape (n,d,d), As must have shape (n,d)")

        if ks is None:
            ks = np.ones(len(xs), dtype=float)
        else:
            ks = np.asarray(ks, dtype=float)
            if ks.ndim == 0:
                ks = np.full(len(xs), float(ks))

        # U has eigenvectors as columns, so U.T @ x is the local coordinate.
        xs_local = np.einsum("nij,nj->ni", Us.swapaxes(1, 2), xs)
        margin = As + self.eps
        return np.sum((xs_local / margin) ** 2, axis=1) - ks ** 2

    def _compute_sorted_eigencomponent(self, cov):
        """Match the reference eigendecomposition convention.

        Eigenvectors are returned as columns here, while the reference source
        stores them as rows.  The ordering and non-positive-eigenvalue handling
        follow the reference implementation.
        """
        cov = np.asarray(cov, dtype=float)
        eigen_value, eigen_vector = np.linalg.eig(cov)

        sort_idx = np.argsort(eigen_value)[::-1]
        eigen_value = eigen_value[sort_idx].real
        eigen_vector = eigen_vector[:, sort_idx].real

        first_zero_index = np.searchsorted(-eigen_value, 0)
        if first_zero_index < len(eigen_value):
            # Same lower-bound rule as the reference source.
            previous = eigen_value[first_zero_index - 1] if first_zero_index > 0 else eigen_value[-1]
            lowest_lambda = min(previous / 2.0, self.eps)
            eigen_value[first_zero_index:] = lowest_lambda

        return eigen_value, eigen_vector

    # ------------------------------------------------------------------
    # Initialization / public training API
    # ------------------------------------------------------------------
    def width_init(self, X, y):
        """Initialize widths for classes appearing in a batch."""
        X = self._as_2d(X, self.dim)
        y = np.asarray(y)
        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of samples")

        new_classes = set(np.unique(y)) - set(self.init_width.keys())
        if new_classes:
            average_width = self.average_pairwise_distance(X)
            for label in new_classes:
                self.init_width[label] = average_width.copy()

    def _create_new_node(self, x, label):
        """Reference VEBF new-node initialization."""
        if label not in self.init_width:
            raise RuntimeError(
                "Initial width for this class is not initialized. "
                "Call fit/partial_fit with an initial batch before train(), "
                "or provide the class in a width_init() call."
            )

        x = np.asarray(x, dtype=float)
        return Node(
            dimension=self.dim,
            label=label,
            A=self.init_width[label],
            n=1,
            M=x,
            # The VEBF paper/reference starts a new node with zero covariance.
            S=np.zeros((self.dim, self.dim), dtype=float),
            U=np.eye(self.dim, dtype=float),
            eps=self.eps,
        )

    def fit(self, X, y, reset=True):
        """Fit on an initial batch, then process samples in streaming order.

        This mirrors the reference implementation's fit behavior: initial widths
        are computed from the complete batch before samples are consumed.
        """
        X = self._as_2d(X, self.dim)
        y = np.asarray(y)
        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of samples")

        if reset:
            self.nodes = {}
            self.init_width = {}
            self.classes_ = np.array([], dtype=object)

        self.width_init(X, y)
        for x_i, y_i in zip(X, y):
            self._train_one(x_i, y_i)

        self._set_classes()
        return self

    def partial_fit(self, X, y):
        """Incrementally process a chunk without resetting existing state."""
        X = self._as_2d(X, self.dim)
        y = np.asarray(y)
        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of samples")

        # The reference implementation initializes widths from the current
        # chunk for any classes that have not been seen before.
        self.width_init(X, y)
        for x_i, y_i in zip(X, y):
            self._train_one(x_i, y_i)

        self._set_classes()
        return self

    def train(self, x, y):
        """Backward-compatible one-sample training method.

        For exact reference-style initial-width behavior, initialize the model
        with fit() (or a multi-sample partial_fit()) first.  This method then
        processes one incoming sample at a time.
        """
        x = np.asarray(x, dtype=float)
        if x.ndim != 1 or x.shape[0] != self.dim:
            raise ValueError(f"x must have shape ({self.dim},)")
        self._train_one(x, y)
        self._set_classes()
        return self

    # ------------------------------------------------------------------
    # Online update
    # ------------------------------------------------------------------
    def _find_shortest(self, x, label):
        nodes = self.nodes[label]
        return min(nodes, key=lambda node: np.linalg.norm(node.M - x))

    def _train_one(self, x, label):
        x = np.asarray(x, dtype=float)
        if label not in self.nodes:
            self.nodes[label] = [self._create_new_node(x, label)]
            return

        # VEBF chooses the nearest center within the true class.
        node = self._find_shortest(x, label)

        n = node.n
        M = node.M
        S = node.S

        # Tentative center.
        M_new = (n * M + x) / (n + 1)

        # Tentative covariance, equivalent to the reference recurrence.
        xx = np.outer(x, x)
        MM = np.outer(M, M)
        MnewMnew = np.outer(M_new, M_new)
        S_new = (
            n * S + xx - MM
        ) / (n + 1) - MnewMnew + MM

        # Reference growth test uses the tentative covariance/eigenvectors but
        # the current node width.
        eigen_value, eigen_vector = self._compute_sorted_eigencomponent(S_new)
        psi_temp = self.surface_single(x - M_new, eigen_vector, node.A)

        if psi_temp > 0:
            # Incoming point cannot be reasonably learned by this node: create
            # a new same-class node and let it participate in merging.
            latest = self._create_new_node(x, label)
            self.nodes[label].append(latest)
            self._merge_node(latest)
            return

        # Update accepted into the current node.
        n_new = n + 1

        # Reference VEBF width update: no expansion until n > N0.
        if n_new > self.N0:
            displacement = M_new - M
            new_A = node.A + np.abs(eigen_vector.T @ displacement)
        else:
            new_A = node.A.copy()

        node.M = M_new
        node.U = eigen_vector
        node.S = S_new
        node.A = new_A
        node.n = n_new

        # The reference implementation performs at most one merge attempt for
        # the updated/created node.
        self._merge_node(node)

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------
    def _merge_node(self, alpha):
        """Reference same-class merge rule; at most one merge per update."""
        nodes = self.nodes[alpha.label]
        if len(nodes) <= 1:
            return

        for beta in nodes:
            if beta is alpha:
                continue

            # Reference merge criterion: either center lies inside the other
            # node's ellipsoid.
            psi_alpha = self.surface_single(alpha.M - beta.M, beta.U, beta.A)
            psi_beta = self.surface_single(beta.M - alpha.M, alpha.U, alpha.A)

            if psi_alpha <= self.theta or psi_beta <= self.theta:
                n_gamma = alpha.n + beta.n
                M_gamma = (alpha.n * alpha.M + beta.n * beta.M) / n_gamma

                displacement = alpha.M - beta.M
                S_gamma = (
                    alpha.n * alpha.S
                    + beta.n * beta.S
                    + (alpha.n * beta.n / n_gamma) * np.outer(displacement, displacement)
                ) / n_gamma

                eigen_value, eigen_vector = self._compute_sorted_eigencomponent(S_gamma)
                A_gamma = np.sqrt(2.0 * math.pi * np.abs(eigen_value))

                beta.n = n_gamma
                beta.M = M_gamma
                beta.S = S_gamma
                beta.U = eigen_vector
                beta.A = A_gamma

                nodes.remove(alpha)
                return

    # ------------------------------------------------------------------
    # Prediction / utilities
    # ------------------------------------------------------------------
    def predict(self, X):
        X = self._as_2d(X, self.dim)
        nodes = self.get_nodes()
        if not nodes:
            raise AssertionError("Can't predict with empty VEBF.")

        centers = np.asarray([node.M for node in nodes])
        Us = np.asarray([node.U for node in nodes])
        As = np.asarray([node.A for node in nodes])

        distances = np.empty((len(X), len(nodes)), dtype=float)
        for i, x in enumerate(X):
            distances[i] = self.surface_multi(
                (x[None, :] - centers),
                Us,
                As,
            )

        nearest = np.argmin(distances, axis=1)
        labels = np.asarray([node.label for node in nodes], dtype=object)
        return labels[nearest]

    def get_nodes(self):
        return list(chain.from_iterable(self.nodes.values()))

    def find_shortest(self, x, y):
        return self._find_shortest(np.asarray(x, dtype=float), y)

    def find_shortest_all(self, x, nodes=None):
        x = np.asarray(x, dtype=float)
        nodes = self.get_nodes() if nodes is None else nodes
        return min(nodes, key=lambda node: np.linalg.norm(node.M - x))

    def _set_classes(self):
        self.classes_ = np.asarray(list(self.nodes.keys()))

    def __str__(self):
        total = sum(len(group) for group in self.nodes.values())
        string = f"A vebf with {total} nodes.\n"
        for label, nodes in self.nodes.items():
            for node in nodes:
                string += f"Label {label}: {node}\n"
        return string


# ----------------------------------------------------------------------
# Small self-checks for the reference rules.
# ----------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(7)
    X0 = np.vstack([
        rng.normal(loc=(-2.0, 0.0), scale=0.25, size=(30, 2)),
        rng.normal(loc=(2.0, 0.0), scale=0.25, size=(30, 2)),
    ])
    y0 = np.array([0] * 30 + [1] * 30)

    model = VEBF(dimension=2)
    model.fit(X0, y0)

    pred = model.predict(X0)
    accuracy = float(np.mean(pred == y0))

    # Initial width check against the exact reference formula.
    expected_width = np.full(2, 2.0 * np.sum(pdist(X0)) / (len(X0) ** 2))
    for width in model.init_width.values():
        assert np.allclose(width, expected_width)

    # Basic geometry/state invariants.
    assert all(node.n >= 1 for node in model.get_nodes())
    assert all(node.A.shape == (2,) for node in model.get_nodes())
    assert all(node.U.shape == (2, 2) for node in model.get_nodes())
    assert model.classes_.size == 2

    print(model)
    print(f"Reference-rule self-check accuracy: {accuracy:.4f}")
