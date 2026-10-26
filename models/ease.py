import numpy as np


class EASE:
    def __init__(self, reg=500.0):
        self.reg = reg

    def fit(self, X):
        X = X.tocsr().astype(np.float32).copy()
        X.data[:] = 1.0
        self.X = X
        G = (X.T @ X).toarray().astype(np.float64)
        G[np.diag_indices_from(G)] += self.reg
        P = np.linalg.inv(G)
        B = -P / np.diag(P)[None, :]
        B[np.diag_indices_from(B)] = 0.0
        self.B = B.astype(np.float32)
        return self

    def predict(self, users):
        return self.X[users] @ self.B
