"""Sparse item-item recommendations and temporal, customer-level evaluation."""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


class Recommender:
    def __init__(self, neighbors=40):
        self.neighbors = neighbors

    def fit(self, sales, rules):
        pairs = sales[["CustomerID", "StockCode"]].drop_duplicates()
        self.items = np.array(sorted(pairs.StockCode.unique()))
        self.users = np.array(sorted(pairs.CustomerID.unique()))
        self.ii = {v: i for i, v in enumerate(self.items)}
        self.ui = {v: i for i, v in enumerate(self.users)}
        self.matrix = csr_matrix((np.ones(len(pairs)), (pairs.CustomerID.map(self.ui), pairs.StockCode.map(self.ii))),
                                 shape=(len(self.users), len(self.items)))
        self.popularity = np.asarray(self.matrix.sum(axis=0)).ravel()
        self.popularity /= max(self.popularity.max(), 1)
        # Retain only top neighbors. Query in batches to avoid a full dense item-item matrix.
        knn = NearestNeighbors(metric="cosine", algorithm="brute", n_jobs=2).fit(self.matrix.T)
        rows, cols, values = [], [], []
        for start in range(0, len(self.items), 128):
            distances, indices = knn.kneighbors(self.matrix.T.tocsr()[start:start+128],
                                                n_neighbors=min(self.neighbors+1, len(self.items)))
            for offset, (ds, js) in enumerate(zip(distances, indices)):
                for distance, j in zip(ds, js):
                    if j != start + offset and distance < 1:
                        rows.append(start + offset); cols.append(j); values.append(1-distance)
        self.similarity = csr_matrix((values, (rows, cols)), shape=(len(self.items), len(self.items)))
        self.rules = [(set(row.antecedents), set(row.consequents), float(row.confidence * np.log1p(row.lift)))
                      for row in rules.itertuples()]
        return self

    def components(self, customer):
        if customer not in self.ui:
            return np.zeros(len(self.items)), np.zeros(len(self.items))
        history = self.matrix[self.ui[customer]]
        cf = (history @ self.similarity).toarray().ravel()
        seen = set(self.items[history.indices])
        rule = np.zeros(len(self.items))
        for antecedent, consequent, weight in self.rules:
            if antecedent <= seen:
                for item in consequent:
                    if item in self.ii:
                        rule[self.ii[item]] += weight
        for scores in (cf, rule):
            if scores.max() > 0:
                scores /= scores.max()
        return cf, rule

    def recommend(self, customer, k=10, mode="hybrid", alpha=.5, exclude_seen=False):
        if not 0 <= alpha <= 1 or k < 1:
            raise ValueError("Invalid alpha or k.")
        if mode not in {"popularity", "item_cf", "rules", "hybrid"}:
            raise ValueError(f"Unknown recommender {mode}")
        cf, rule = self.components(customer) if mode != "popularity" else (None, None)
        score = {"popularity": lambda: self.popularity.copy(), "item_cf": lambda: cf,
                 "rules": lambda: rule, "hybrid": lambda: alpha * cf + (1-alpha) * rule}[mode]()
        # Use popularity explicitly when there is no personalized evidence.
        if not np.any(score):
            score = self.popularity.copy()
        score = score + 1e-8 * self.popularity
        eligible = np.ones(len(self.items), dtype=bool)
        if exclude_seen and customer in self.ui:
            eligible[self.matrix[self.ui[customer]].indices] = False
        ids = np.flatnonzero(eligible)
        order = ids[np.lexsort((self.items[ids], -score[ids]))[:k]]
        return [(self.items[j], float(score[j])) for j in order]


def evaluate(model, future, k=10, alpha=.5, modes=("popularity", "rules", "item_cf", "hybrid"), seed=42):
    """All test customers, including unseen users/items. Repeat purchases are eligible."""
    truths = future.groupby("CustomerID").StockCode.agg(set)
    rows, summaries = [], []
    for mode in modes:
        recommended = set()
        for customer, truth in truths.items():
            prediction = [p for p, _ in model.recommend(customer, k, mode, alpha)]
            recommended.update(prediction)
            hit = np.array([p in truth for p in prediction], dtype=float)
            dcg = float((hit / np.log2(np.arange(len(hit)) + 2)).sum())
            ideal = float((1 / np.log2(np.arange(min(k, len(truth))) + 2)).sum())
            rows.append({"model": mode, "CustomerID": customer, "cold_user": customer not in model.ui,
                         "precision_at_k": hit.sum()/k, "recall_at_k": hit.sum()/len(truth),
                         "ndcg_at_k": dcg/ideal if ideal else 0., "hit_at_k": float(hit.any()),
                         "candidate_recall_ceiling": len(truth & set(model.items))/len(truth)})
        part = pd.DataFrame([r for r in rows if r["model"] == mode])
        if part.empty:
            raise ValueError("No future purchases available for recommendation evaluation.")
        for cohort, group in [("all", part), ("warm", part.loc[~part.cold_user]), ("cold", part.loc[part.cold_user])]:
            if group.empty:
                continue
            values = group.ndcg_at_k.to_numpy()
            rng = np.random.default_rng(seed)
            boots = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(500)]
            summaries.append({"model": mode, "cohort": cohort, "customers": len(group), "k": k,
                **group[["precision_at_k", "recall_at_k", "ndcg_at_k", "hit_at_k", "candidate_recall_ceiling"]].mean().to_dict(),
                "ndcg_ci_low": np.quantile(boots, .025), "ndcg_ci_high": np.quantile(boots, .975),
                "catalog_coverage_all_users": len(recommended)/len(model.items)})
    return pd.DataFrame(summaries), pd.DataFrame(rows)
