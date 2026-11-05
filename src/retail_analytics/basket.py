"""Bounded sparse Apriori and FP-Growth, with matched search spaces."""
from time import perf_counter
import pandas as pd
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder


def mine(sales, min_support=.02, max_products=300, max_len=3):
    counts = sales.drop_duplicates(["InvoiceNo", "StockCode"]).StockCode.value_counts()
    catalog = set(counts.head(max_products).index)
    # Keep empty baskets in the denominator after catalog restriction.
    baskets = sales.groupby("InvoiceNo").StockCode.agg(lambda x: sorted(set(x) & catalog)).tolist()
    if not baskets or not catalog:
        raise ValueError("No baskets to mine.")
    enc = TransactionEncoder()
    matrix = enc.fit(baskets).transform(baskets, sparse=True)
    frame = pd.DataFrame.sparse.from_spmatrix(matrix.astype("int8"), columns=enc.columns_).astype(pd.SparseDtype(bool, False))
    timings, itemsets = [], {}
    for name, algorithm in [("apriori", apriori), ("fpgrowth", fpgrowth)]:
        start = perf_counter()
        args = {"low_memory": True} if name == "apriori" else {}
        itemsets[name] = algorithm(frame, min_support=min_support, use_colnames=True, max_len=max_len, **args)
        timings.append({"algorithm": name, "seconds": perf_counter()-start,
                        "itemsets": len(itemsets[name]), "baskets": len(baskets), "catalog_size": len(catalog)})
    a, b = itemsets["apriori"], itemsets["fpgrowth"]
    if set(a.itemsets) != set(b.itemsets):
        raise AssertionError("Apriori/FP-Growth itemsets disagree.")
    rules = (association_rules(a, metric="confidence", min_threshold=.2, num_itemsets=len(baskets),
                                return_metrics=["antecedent support", "consequent support", "support", "confidence", "lift", "leverage"])
             if any(a.itemsets.map(len).gt(1)) else pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"]))
    rules = rules.loc[rules.lift.gt(1)].sort_values(["lift", "confidence"], ascending=False).reset_index(drop=True)
    return rules, a, pd.DataFrame(timings)


def serializable_rules(rules):
    out = rules.copy()
    for c in ["antecedents", "consequents"]:
        out[c] = out[c].map(lambda v: " | ".join(sorted(v)))
    return out
