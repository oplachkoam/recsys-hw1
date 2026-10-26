import numpy as np
import scipy.sparse as sp


def build_matrix(df, n_users, n_items, values=None):
    vals = (
        np.ones(len(df), dtype=np.float32)
        if values is None
        else df[values].to_numpy(dtype=np.float32)
    )
    return sp.csr_matrix(
        (vals, (df["uid"].to_numpy(), df["iid"].to_numpy())),
        shape=(n_users, n_items),
    )


def temporal_split(df, t1, t2):
    train = df[df["dt"] < t1]
    val = df[(df["dt"] >= t1) & (df["dt"] < t2)]
    test = df[df["dt"] >= t2]
    return train, val, test


def ground_truth(eval_df, train_df, pos_threshold=4):
    known_users = set(train_df["uid"].unique())
    known_items = set(train_df["iid"].unique())
    d = eval_df[
        eval_df["uid"].isin(known_users) & eval_df["iid"].isin(known_items)
    ]
    d = d[d["rating"] >= pos_threshold]
    return (
        d.groupby("uid")["iid"]
        .apply(lambda s: np.array(sorted(set(s))))
        .to_dict()
    )


def recommend(model, users, known, k=10, batch_size=2048, **kwargs):
    item_counts = np.asarray(known.sum(axis=0)).ravel()
    dead = item_counts == 0
    recs = np.zeros((len(users), k), dtype=np.int64)
    for s in range(0, len(users), batch_size):
        batch = users[s : s + batch_size]
        scores = np.asarray(model.predict(batch, **kwargs), dtype=np.float32)
        sub = known[batch].tocoo()
        scores[sub.row, sub.col] = -np.inf
        scores[:, dead] = -np.inf
        part = np.argpartition(-scores, k, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(scores, part, axis=1), axis=1)
        recs[s : s + batch_size] = np.take_along_axis(part, order, axis=1)
    return recs


def evaluate(recs, users, truth, k=10):
    rec, ndcg = [], []
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    for row, u in zip(recs, users):
        gt = truth[u]
        hits = np.isin(row[:k], gt)
        rec.append(hits.sum() / len(gt))
        idcg = discounts[: min(len(gt), k)].sum()
        ndcg.append((hits * discounts).sum() / idcg)
    return {"recall@10": float(np.mean(rec)), "ndcg@10": float(np.mean(ndcg))}


def run_eval(model, truth, known, k=10, **kwargs):
    users = np.array(sorted(truth))
    recs = recommend(model, users, known, k, **kwargs)
    return evaluate(recs, users, truth, k)


class Popularity:
    def fit(self, X):
        self.scores = np.asarray(X.getnnz(axis=0), dtype=np.float32)
        return self

    def predict(self, users):
        return np.tile(self.scores, (len(users), 1))
