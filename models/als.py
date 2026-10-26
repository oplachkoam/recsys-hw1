import numpy as np
import scipy.sparse as sp


class ImplicitALS:
    def __init__(
        self,
        factors=64,
        reg=10.0,
        alpha=20.0,
        iterations=15,
        cg_steps=3,
        seed=0,
    ):
        self.factors = factors
        self.reg = reg
        self.alpha = alpha
        self.iterations = iterations
        self.cg_steps = cg_steps
        self.seed = seed

    def _apply(self, W, Y, YtY, P):
        rows, cols = W.nonzero()
        dots = (P[rows] * Y[cols]).sum(1) * W.data
        M = sp.csr_matrix((dots, (rows, cols)), shape=W.shape)
        return P @ YtY + M @ Y

    def _solve(self, W, Y, X0):
        YtY = Y.T @ Y + self.reg * np.eye(self.factors, dtype=np.float32)
        B = W.copy()
        B.data = B.data + 1.0
        b = B @ Y
        x = X0.copy()
        r = b - self._apply(W, Y, YtY, x)
        p = r.copy()
        rs = (r * r).sum(1)
        for _ in range(self.cg_steps):
            Ap = self._apply(W, Y, YtY, p)
            a = rs / np.maximum((p * Ap).sum(1), 1e-12)
            x += a[:, None] * p
            r -= a[:, None] * Ap
            rs_new = (r * r).sum(1)
            p = r + (rs_new / np.maximum(rs, 1e-12))[:, None] * p
            rs = rs_new
        return x

    def fit(self, X):
        rng = np.random.default_rng(self.seed)
        W = X.tocsr().astype(np.float32).copy()
        W.data = self.alpha * W.data
        Wt = W.T.tocsr()
        n_users, n_items = X.shape
        self.U = (rng.standard_normal((n_users, self.factors)) * 0.01).astype(
            np.float32
        )
        self.V = (rng.standard_normal((n_items, self.factors)) * 0.01).astype(
            np.float32
        )
        for _ in range(self.iterations):
            self.U = self._solve(W, self.V, self.U)
            self.V = self._solve(Wt, self.U, self.V)
        return self

    def predict(self, users):
        return self.U[users] @ self.V.T
