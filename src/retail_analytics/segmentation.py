"""RFM segmentation with model selection and seed stability diagnostics."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from .data import features


def segment(sales, cutoff, seed=42):
    rfm = features(sales, cutoff)[["Recency", "Frequency", "Monetary"]]
    if len(rfm) < 5:
        raise ValueError("At least five customers are needed for segmentation.")
    scaler = StandardScaler()
    z = scaler.fit_transform(np.log1p(rfm))
    rows, models = [], {}
    for k in range(2, min(8, len(rfm) - 1) + 1):
        if k >= len(np.unique(z, axis=0)):
            break
        m = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(z)
        alternative = KMeans(n_clusters=k, n_init=20, random_state=seed + 1).fit(z)
        rows.append({"k": k, "silhouette": silhouette_score(z, m.labels_, sample_size=min(2000, len(z)), random_state=seed),
                     "davies_bouldin": davies_bouldin_score(z, m.labels_), "inertia": m.inertia_,
                     "seed_stability_ari": adjusted_rand_score(m.labels_, alternative.labels_)})
        models[k] = m
    if not rows:
        raise ValueError("Not enough distinct RFM profiles to cluster.")
    diagnostics = pd.DataFrame(rows)
    k = int(diagnostics.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0].k)
    model = models[k]
    rfm["Cluster"] = model.labels_
    # Stable display ordering by median monetary value, not arbitrary premium labels.
    order = rfm.groupby("Cluster").Monetary.median().sort_values().index
    mapping = {int(old): f"Segment {i+1}" for i, old in enumerate(order)}
    rfm["Segment"] = rfm.Cluster.map(mapping)
    rfm["AsOfDate"] = str(pd.Timestamp(cutoff).date())
    profiles = rfm.groupby("Segment").agg(Customers=("Monetary", "size"),
        MedianRecency=("Recency", "median"), MedianFrequency=("Frequency", "median"),
        MedianMonetary=("Monetary", "median"), TotalMonetary=("Monetary", "sum")).reset_index()
    return rfm.reset_index(), profiles, diagnostics, {"scaler": scaler, "kmeans": model, "labels": mapping}
