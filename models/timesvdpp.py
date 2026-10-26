import numpy as np
import scipy.sparse as sp
import torch


def to_torch_sparse(X):
    X = X.tocoo()
    idx = torch.from_numpy(np.vstack([X.row, X.col]).astype(np.int64))
    return torch.sparse_coo_tensor(
        idx, torch.from_numpy(X.data.astype(np.float32)), X.shape
    ).coalesce()


class TimeSVDpp:
    def __init__(
        self,
        factors=32,
        lr=0.005,
        reg=0.02,
        reg_alpha=1.0,
        epochs=20,
        n_bins=30,
        beta=0.4,
        batch_size=4096,
        seed=0,
    ):
        self.factors = factors
        self.lr = lr
        self.reg = reg
        self.reg_alpha = reg_alpha
        self.epochs = epochs
        self.n_bins = n_bins
        self.beta = beta
        self.batch_size = batch_size
        self.seed = seed

    def _dev(self, users, days, clip=False):
        d = days - self.t_mean[users]
        if clip:
            d = np.clip(d, -self.t_span[users], self.t_span[users])
        return np.sign(d) * np.abs(d) ** self.beta

    def _bin(self, days):
        b = (days - self.t_min) / (self.t_max - self.t_min + 1e-9) * self.n_bins
        return np.clip(b.astype(np.int64), 0, self.n_bins - 1)

    def _forward(self, u, i, dev, bins):
        imp = torch.sparse.mm(to_torch_sparse(self.N[u]), self.Y)
        pu = self.P[u] + imp
        pred = self.mu + self.bu[u] + self.bi[i] + (self.Q[i] * pu).sum(1)
        return pred + self.au[u] * dev + self.bit[i, bins]

    def fit(self, X, D):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        n_users, n_items = X.shape
        X = X.tocsr().copy()
        X.sort_indices()
        D = D.tocsr().copy()
        D.sort_indices()
        coo = X.tocoo()
        users, items = coo.row.astype(np.int64), coo.col.astype(np.int64)
        ratings = coo.data.astype(np.float32)
        days = D.tocoo().data.astype(np.float64)
        self.t_min, self.t_max = days.min(), days.max()
        self.t_mean = np.zeros(n_users)
        np.add.at(self.t_mean, users, days)
        cnt = np.bincount(users, minlength=n_users)
        self.t_mean = self.t_mean / np.maximum(cnt, 1)
        self.t_span = np.zeros(n_users)
        np.maximum.at(self.t_span, users, np.abs(days - self.t_mean[users]))
        N = X.tocsr().astype(np.float32).copy()
        N.data[:] = 1.0
        norm = 1.0 / np.sqrt(np.maximum(np.asarray(N.sum(1)).ravel(), 1))
        self.N = sp.diags(norm.astype(np.float32)) @ N
        self.N = self.N.tocsr()
        self.mu = float(ratings.mean())
        f = self.factors
        self.bu = torch.nn.Parameter(torch.zeros(n_users))
        self.bi = torch.nn.Parameter(torch.zeros(n_items))
        self.au = torch.nn.Parameter(torch.zeros(n_users))
        self.bit = torch.nn.Parameter(torch.zeros(n_items, self.n_bins))
        self.P = torch.nn.Parameter(torch.randn(n_users, f) * 0.01)
        self.Q = torch.nn.Parameter(torch.randn(n_items, f) * 0.01)
        self.Y = torch.nn.Parameter(torch.randn(n_items, f) * 0.01)
        params = [self.bu, self.bi, self.au, self.bit, self.P, self.Q, self.Y]
        opt = torch.optim.Adam(params, lr=self.lr)
        dev_all = self._dev(users, days).astype(np.float32)
        bin_all = self._bin(days)
        r = torch.from_numpy(ratings)
        n = len(users)
        self.history = []
        for epoch in range(self.epochs):
            perm = rng.permutation(n)
            total = 0.0
            for s in range(0, n, self.batch_size):
                b = perm[s : s + self.batch_size]
                u, i = users[b], items[b]
                pred = self._forward(
                    u,
                    i,
                    torch.from_numpy(dev_all[b]),
                    torch.from_numpy(bin_all[b]),
                )
                err = ((pred - r[b]) ** 2).mean()
                penalty = (
                    self.bu[u] ** 2
                    + self.bi[i] ** 2
                    + self.bit[i, bin_all[b]] ** 2
                    + (self.P[u] ** 2).sum(1)
                    + (self.Q[i] ** 2).sum(1)
                ).mean()
                penalty = penalty + (self.Y**2).sum() / n_items
                loss = (
                    err
                    + self.reg * penalty
                    + self.reg_alpha * (self.au[u] ** 2).mean()
                )
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += err.item() * len(b)
            self.history.append(np.sqrt(total / n))
        return self

    def predict(self, users, day):
        users = np.asarray(users, dtype=np.int64)
        with torch.no_grad():
            imp = torch.sparse.mm(to_torch_sparse(self.N[users]), self.Y)
            pu = (self.P[users] + imp).numpy()
            scores = (
                self.mu
                + self.bu[users].numpy()[:, None]
                + self.bi.numpy()[None, :]
                + pu @ self.Q.numpy().T
            )
            dev = self._dev(
                users, np.full(len(users), float(day)), clip=True
            ).astype(np.float32)
            b = self._bin(np.array([float(day)]))[0]
            scores = (
                scores
                + (self.au[users].numpy() * dev)[:, None]
                + self.bit[:, b].numpy()[None, :]
            )
        return scores
