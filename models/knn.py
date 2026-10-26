import numpy as np
import scipy.sparse as sp


class ItemKNN:
    def __init__(self, k=50, shrinkage=0.0):
        self.k = k
        self.shrinkage = shrinkage

    def fit(self, X):
        X = X.tocsr().astype(np.float32).copy()
        X.data[:] = 1.0
        self.X = X
        norms = np.sqrt(np.asarray(X.sum(axis=0)).ravel())
        G = (X.T @ X).tocoo()
        denom = norms[G.row] * norms[G.col] + self.shrinkage
        sim = G.data / np.maximum(denom, 1e-12)
        sim[G.row == G.col] = 0.0
        S = sp.csr_matrix((sim, (G.row, G.col)), shape=G.shape)
        self.S = self._topk_rows(S, self.k).T.tocsr()
        return self

    @staticmethod
    def _topk_rows(S, k):
        S = S.tocsr()
        rows, cols, vals = [], [], []
        for i in range(S.shape[0]):
            lo, hi = S.indptr[i], S.indptr[i + 1]
            if hi - lo > k:
                idx = np.argpartition(-S.data[lo:hi], k)[:k] + lo
            else:
                idx = np.arange(lo, hi)
            rows.append(np.full(len(idx), i))
            cols.append(S.indices[idx])
            vals.append(S.data[idx])
        rows, cols, vals = map(np.concatenate, (rows, cols, vals))
        return sp.csr_matrix((vals, (rows, cols)), shape=S.shape)

    def predict(self, users):
        return (self.X[users] @ self.S).toarray()
