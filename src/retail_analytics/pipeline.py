"""Orchestration. Invoke from Colab; real data is never needed by unit tests."""
import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import importlib.metadata
import json
import platform
import subprocess
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .data import download, sha256, clean, purchases, SOURCE
from .segmentation import segment
from .basket import mine, serializable_rules
from .recommend import Recommender, evaluate
from .predict import experiment


@dataclass
class Config:
    seed: int = 42
    rec_validation_start: str = "2011-09-01"
    rec_test_start: str = "2011-10-01"
    rec_test_end: str = "2011-11-01"
    prediction_train_dates: tuple = ("2011-07-01", "2011-08-01", "2011-09-01")
    prediction_validation_date: str = "2011-10-01"
    prediction_test_date: str = "2011-11-01"
    min_support: float = .02
    max_basket_products: int = 300
    max_itemset_length: int = 3
    recommendation_k: int = 10
    neighbors: int = 40


def json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def csv_write(out, name, frame):
    frame.to_csv(Path(out) / f"{name}.csv", index=False)


def export_powerbi(ledger, segments, current, recommendations, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    fact = ledger[["InvoiceNo", "StockCode", "CustomerID", "Country", "Date", "Quantity", "UnitPrice", "Revenue", "IsReturn"]].copy()
    fact.insert(0, "LineID", np.arange(1, len(fact)+1))
    products = ledger.sort_values("InvoiceDate").drop_duplicates("StockCode", keep="last")[["StockCode", "Description"]]
    customer_ids = ledger[["CustomerID"]].drop_duplicates()
    customers = customer_ids.merge(segments[["CustomerID", "Segment", "Recency", "Frequency", "Monetary", "AsOfDate"]], how="left", on="CustomerID")
    customers = customers.merge(current[["CustomerID", "RepeatProbability30d"]], how="left", on="CustomerID")
    customers["Segment"] = customers.Segment.fillna("Not modeled")
    calendar = pd.DataFrame({"Date": pd.date_range(ledger.InvoiceDate.min().normalize(), ledger.InvoiceDate.max().normalize())})
    calendar["Year"] = calendar.Date.dt.year
    calendar["MonthNumber"] = calendar.Date.dt.month
    calendar["Month"] = calendar.Date.dt.strftime("%b")
    calendar["YearMonth"] = calendar.Date.dt.strftime("%Y-%m")
    calendar["Date"] = calendar.Date.dt.strftime("%Y-%m-%d")
    tables = {"FactSales": fact, "DimProduct": products, "DimCustomer": customers,
              "DimCountry": ledger[["Country"]].drop_duplicates(), "DimDate": calendar,
              "FactRecommendations": recommendations}
    for name, frame in tables.items():
        csv_write(out, name, frame)
    csv_write(out, "DataDictionary", pd.DataFrame([{"Table": n, "Column": c, "PandasType": str(f[c].dtype)}
                                                   for n, f in tables.items() for c in f.columns]))
    return tables


def figures(out, ledger, diagnostics, profiles, metrics, prediction):
    figdir = Path(out)/"figures"; figdir.mkdir(exist_ok=True)
    def save(name):
        plt.tight_layout(); plt.savefig(figdir/f"{name}.png", dpi=160); plt.close()
    ledger.assign(Month=ledger.InvoiceDate.dt.to_period("M").astype(str)).groupby("Month").Revenue.sum().plot(
        figsize=(10,4), marker="o", title="Net merchandise revenue (partial months included)", ylabel="GBP")
    save("monthly_revenue")
    diagnostics.plot(x="k", y="silhouette", marker="o", legend=False, title="RFM cluster selection")
    save("cluster_selection")
    profiles.plot.bar(x="Segment", y="MedianMonetary", legend=False, title="Segment profiles", ylabel="GBP")
    save("segment_profiles")
    metrics.loc[metrics.cohort.eq("all")].plot.bar(x="model", y="ndcg_at_k", legend=False, title="Temporal test recommendation NDCG@K")
    save("recommendation_comparison")
    plt.figure(figsize=(5,5)); plt.plot([0,1], [0,1], "--", color="gray")
    plt.plot(prediction["calibration"].mean_probability, prediction["calibration"].observed_fraction, "o-")
    plt.xlabel("Mean predicted probability"); plt.ylabel("Observed repeat-purchase fraction"); plt.title("Test calibration (quantile bins)")
    save("prediction_calibration")
    prediction["importance"].sort_values("ap_drop_mean").plot.barh(x="feature", y="ap_drop_mean", legend=False,
        xerr=prediction["importance"].sort_values("ap_drop_mean").ap_drop_std, title="Held-out permutation importance", xlabel="Average precision decrease")
    save("permutation_importance")


def run(input_path, output_dir, config=None):
    cfg = config or Config()
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output directory is not empty. Use a new run directory to preserve provenance.")
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    manifest = {"status": "running", "created_utc": datetime.now(timezone.utc).isoformat(),
                "config": asdict(cfg), "source": SOURCE, "input_sha256": sha256(input_path),
                "python": platform.python_version(),
                "packages": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn", "mlxtend", "joblib"]}}
    try:
        manifest["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        manifest["git_commit"] = "unavailable"
    json_write(out/"manifest.json", manifest)
    try:
        path = Path(input_path)
        raw = pd.read_excel(path) if path.suffix.lower() == ".xlsx" else pd.read_csv(path)
        ledger, audit = clean(raw)
        if ledger.empty:
            raise ValueError("No valid merchandise transactions remain after cleaning.")
        sales = purchases(ledger)
        # Last day might be partial; do not use it to certify complete label horizons.
        observed_until = ledger.InvoiceDate.max().normalize()
        asof = observed_until + pd.Timedelta(days=1)
        audit.update(first_timestamp=str(ledger.InvoiceDate.min()), last_timestamp=str(ledger.InvoiceDate.max()),
                     modeling_purchase_rows=len(sales), purchase_customers=sales.CustomerID.nunique())
        json_write(out/"data_quality.json", audit)
        print("Cleaning complete", flush=True)
        segments, profiles, diagnostics, cluster_model = segment(sales, asof, cfg.seed)
        for name, table in [("customer_segments", segments), ("segment_profiles", profiles), ("cluster_diagnostics", diagnostics)]:
            csv_write(out, name, table)
        joblib.dump(cluster_model, out/"segmentation.joblib")
        val_start, test_start, test_end = map(pd.Timestamp, [cfg.rec_validation_start, cfg.rec_test_start, cfg.rec_test_end])
        if not sales.InvoiceDate.min() < val_start < test_start < test_end <= observed_until:
            raise ValueError("Recommendation dates do not fit available history.")
        train = sales.loc[sales.InvoiceDate.lt(val_start)]
        validation = sales.loc[sales.InvoiceDate.ge(val_start) & sales.InvoiceDate.lt(test_start)]
        test = sales.loc[sales.InvoiceDate.ge(test_start) & sales.InvoiceDate.lt(test_end)]
        mining_args = dict(min_support=cfg.min_support, max_products=cfg.max_basket_products, max_len=cfg.max_itemset_length)
        rules, _, _ = mine(train, **mining_args)
        rec = Recommender(cfg.neighbors).fit(train, rules)
        alpha_rows = []
        for alpha in [0., .25, .5, .75, 1.]:
            summary, _ = evaluate(rec, validation, cfg.recommendation_k, alpha, modes=("hybrid",), seed=cfg.seed)
            alpha_rows.append({"alpha": alpha, "validation_ndcg": float(summary.loc[summary.cohort.eq("all"), "ndcg_at_k"].iloc[0])})
        alpha_results = pd.DataFrame(alpha_rows)
        alpha = float(alpha_results.sort_values(["validation_ndcg", "alpha"], ascending=[False, True]).iloc[0].alpha)
        csv_write(out, "recommendation_validation", alpha_results)
        development = sales.loc[sales.InvoiceDate.lt(test_start)]
        rules, _, _ = mine(development, **mining_args)
        rec = Recommender(cfg.neighbors).fit(development, rules)
        metrics, per_customer = evaluate(rec, test, cfg.recommendation_k, alpha, seed=cfg.seed)
        csv_write(out, "recommendation_test_metrics", metrics)
        csv_write(out, "recommendation_test_per_customer", per_customer)
        joblib.dump(rec, out/"recommendation_evaluation.joblib")
        print("Recommendation evaluation complete", flush=True)
        prediction = experiment(sales, observed_until, cfg.prediction_train_dates,
                                cfg.prediction_validation_date, cfg.prediction_test_date, cfg.seed)
        for name in ["validation", "test", "calibration", "importance", "predictions", "current"]:
            csv_write(out, f"prediction_{name}", prediction[name])
        json_write(out/"prediction_metadata.json", prediction["metadata"])
        joblib.dump(prediction["evaluation_model"], out/"prediction_evaluation.joblib")
        joblib.dump(prediction["production_model"], out/"prediction_production.joblib")
        # Score production at the next midnight using all observed history, including partial last day.
        from .data import features, FEATURES
        current = features(sales, asof)
        current["RepeatProbability30d"] = prediction["production_model"].predict_proba(current[FEATURES])[:, 1]
        current["AsOfDate"] = str(asof.date())
        current = current.reset_index()
        csv_write(out, "prediction_current", current)
        print("Prediction evaluation complete", flush=True)
        rules, itemsets, timings = mine(sales, **mining_args)
        csv_write(out, "association_rules", serializable_rules(rules))
        itemsets["itemsets"] = itemsets.itemsets.map(lambda x: " | ".join(sorted(x)))
        csv_write(out, "frequent_itemsets", itemsets)
        csv_write(out, "mining_benchmark", timings)
        production_rec = Recommender(cfg.neighbors).fit(sales, rules)
        joblib.dump(production_rec, out/"recommendation_production.joblib")
        recommendation_rows = []
        for customer in production_rec.users:
            for rank, (item, score) in enumerate(production_rec.recommend(customer, cfg.recommendation_k, alpha=alpha), 1):
                recommendation_rows.append({"CustomerID": customer, "StockCode": item, "Rank": rank,
                                            "Score": score, "AsOfDate": str(asof.date())})
        recommendations = pd.DataFrame(recommendation_rows)
        tables = export_powerbi(ledger, segments, current, recommendations, out/"powerbi")
        figures(out, ledger, diagnostics, profiles, metrics, prediction)
        manifest.update(status="complete", elapsed_seconds=time.perf_counter()-start,
                        asof_date=str(asof.date()), selected_hybrid_alpha=alpha,
                        powerbi_rows={n: len(t) for n, t in tables.items()})
        # Hash artifacts for traceability; manifest itself is intentionally excluded.
        manifest["artifact_sha256"] = {str(p.relative_to(out)): sha256(p) for p in sorted(out.rglob("*")) if p.is_file() and p.name != "manifest.json"}
        json_write(out/"manifest.json", manifest)
        (out/"environment.txt").write_text(subprocess.check_output(["python", "-m", "pip", "freeze"], text=True))
        print(f"Completed: {out}", flush=True)
        return manifest
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        json_write(out/"manifest.json", manifest)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config")
    args = parser.parse_args()
    cfg = Config(**json.loads(Path(args.config).read_text())) if args.config else Config()
    run(args.input, args.output, cfg)


if __name__ == "__main__":
    main()
