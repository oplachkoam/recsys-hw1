import numpy as np
import torch


def to_torch_sparse(X):
    X = X.tocoo()
    idx = torch.from_numpy(np.vstack([X.row, X.col]).astype(np.int64))
    return torch.sparse_coo_tensor(
        idx, torch.from_numpy(X.data.astype(np.float32)), X.shape
    ).coalesce()


class FM:
    def __init__(
        self, factors=32, lr=0.005, reg=1e-4, epochs=15, batch_size=4096, seed=0
    ):
        self.factors = factors
        self.lr = lr
        self.reg = reg
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed

    def _embed(self, F, V, w):
        S = torch.sparse.mm(F, V)
        lin = torch.sparse.mm(F, w).squeeze(1)
        F2 = torch.sparse_coo_tensor(F.indices(), F.values() ** 2, F.shape)
        Q = torch.sparse.mm(F2, V**2).sum(1)
        return S, lin, Q

    def _score(self, Su, linu, Qu, Si, lini, Qi):
        return (
            self.w0
            + linu
            + lini
            + 0.5 * ((Su + Si) ** 2).sum(1)
            - 0.5 * (Qu + Qi)
        )

    def fit(self, X, user_features, item_features):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.UF = user_features.tocsr().astype(np.float32)
        self.IF = item_features.tocsr().astype(np.float32)
        n_users, n_items = X.shape
        self.n_items = n_items
        coo = X.tocoo()
        users, items = coo.row.astype(np.int64), coo.col.astype(np.int64)
        known = X.tocsr()
        nu, ni = self.UF.shape[1], self.IF.shape[1]
        self.Vu = torch.nn.Parameter(torch.randn(nu, self.factors) * 0.01)
        self.Vi = torch.nn.Parameter(torch.randn(ni, self.factors) * 0.01)
        self.wu = torch.nn.Parameter(torch.zeros(nu, 1))
        self.wi = torch.nn.Parameter(torch.zeros(ni, 1))
        self.w0 = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.Adam(
            [self.Vu, self.Vi, self.wu, self.wi, self.w0], lr=self.lr
        )
        n = len(users)
        self.history = []
        for epoch in range(self.epochs):
            perm = rng.permutation(n)
            negs = rng.integers(0, n_items, size=n)
            clash = np.asarray(known[users, negs]).ravel() > 0
            while clash.any():
                negs[clash] = rng.integers(0, n_items, size=clash.sum())
                clash = np.asarray(known[users, negs]).ravel() > 0
            total = 0.0
            for s in range(0, n, self.batch_size):
                b = perm[s : s + self.batch_size]
                u, i, j = users[b], items[b], negs[b]
                Fu = to_torch_sparse(self.UF[u])
                Fi = to_torch_sparse(self.IF[i])
                Fj = to_torch_sparse(self.IF[j])
                Su, linu, Qu = self._embed(Fu, self.Vu, self.wu)
                Si, lini, Qi = self._embed(Fi, self.Vi, self.wi)
                Sj, linj, Qj = self._embed(Fj, self.Vi, self.wi)
                x_ui = self._score(Su, linu, Qu, Si, lini, Qi)
                x_uj = self._score(Su, linu, Qu, Sj, linj, Qj)
                loss = -torch.nn.functional.logsigmoid(x_ui - x_uj).mean()
                loss = loss + self.reg * (Qu + Qi + Qj).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(b)
            self.history.append(total / n)
        with torch.no_grad():
            Si, lini, Qi = self._embed(
                to_torch_sparse(self.IF), self.Vi, self.wi
            )
            self.item_S = Si.numpy()
            self.item_bias = (lini + 0.5 * (Si**2).sum(1) - 0.5 * Qi).numpy()
        return self

    def predict(self, users):
        with torch.no_grad():
            Su, linu, Qu = self._embed(
                to_torch_sparse(self.UF[users]), self.Vu, self.wu
            )
            user_bias = (linu + 0.5 * (Su**2).sum(1) - 0.5 * Qu).numpy()
            return (
                Su.numpy() @ self.item_S.T
                + user_bias[:, None]
                + self.item_bias[None, :]
            )
