"""Predict a purchase in the next 30 days, not permanent churn."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, precision_score, recall_score, f1_score
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from .data import FEATURES, snapshot, features


def scores(y, p, threshold=.5):
    two = len(np.unique(y)) == 2
    predicted = p >= threshold
    return {"pr_auc_ap": average_precision_score(y, p) if two else np.nan,
            "roc_auc": roc_auc_score(y, p) if two else np.nan,
            "brier": brier_score_loss(y, p), "f1": f1_score(y, predicted, zero_division=0),
            "precision": precision_score(y, predicted, zero_division=0),
            "recall": recall_score(y, predicted, zero_division=0), "prevalence": float(np.mean(y))}


def experiment(sales, observed_until, train_dates, validation_date, test_date, seed=42):
    horizon = pd.Timedelta(days=30)
    dates = [pd.Timestamp(x) for x in train_dates]
    val_date, test_date = pd.Timestamp(validation_date), pd.Timestamp(test_date)
    if not dates or max(dates)+horizon > val_date or val_date+horizon > test_date:
        raise ValueError("Training/validation label horizons must end before the next partition.")
    train = pd.concat([snapshot(sales, t, observed_until) for t in dates])
    val = snapshot(sales, val_date, observed_until)
    test = snapshot(sales, test_date, observed_until)
    if any(d.empty or d.RepeatPurchase.nunique() < 2 for d in (train, val, test)):
        raise ValueError("Each prediction partition needs customers from both target classes.")
    models = {
        "prevalence": DummyClassifier(strategy="prior"),
        "logistic": make_pipeline(FunctionTransformer(np.log1p, feature_names_out="one-to-one"), StandardScaler(),
                                   LogisticRegression(max_iter=2000, random_state=seed)),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=15,
                                    l2_regularization=2, early_stopping=False, random_state=seed)}
    validation_rows, fitted = [], {}
    for name, model in models.items():
        fitted[name] = clone(model).fit(train[FEATURES], train.RepeatPurchase)
        p = fitted[name].predict_proba(val[FEATURES])[:, 1]
        validation_rows.append({"model": name, **scores(val.RepeatPurchase, p)})
    validation = pd.DataFrame(validation_rows)
    winner = validation.sort_values(["pr_auc_ap", "brier"], ascending=[False, True]).iloc[0].model
    p_val = fitted[winner].predict_proba(val[FEATURES])[:, 1]
    grid = np.linspace(.05, .95, 37)
    threshold = float(max(grid, key=lambda t: f1_score(val.RepeatPurchase, p_val >= t, zero_division=0)))
    # Freeze model choice and threshold before touching the test month.
    development = pd.concat([train, val])
    test_rows, final_models = [], {}
    for name, model in models.items():
        final_models[name] = clone(model).fit(development[FEATURES], development.RepeatPurchase)
        p = final_models[name].predict_proba(test[FEATURES])[:, 1]
        test_rows.append({"model": name, "selected_on_validation": name == winner,
                          **scores(test.RepeatPurchase, p, threshold if name == winner else .5)})
    model = final_models[winner]
    probabilities = model.predict_proba(test[FEATURES])[:, 1]
    fraction, mean_prob = calibration_curve(test.RepeatPurchase, probabilities, n_bins=10, strategy="quantile")
    calibration = pd.DataFrame({"mean_probability": mean_prob, "observed_fraction": fraction})
    importance = permutation_importance(model, test[FEATURES], test.RepeatPurchase,
                                       scoring="average_precision", n_repeats=5, random_state=seed)
    explanation = pd.DataFrame({"feature": FEATURES, "ap_drop_mean": importance.importances_mean,
                                "ap_drop_std": importance.importances_std})
    predictions = test.reset_index()[["CustomerID", "SnapshotDate", "RepeatPurchase"]]
    predictions["Probability"] = probabilities
    predictions["PredictedRepeat"] = (probabilities >= threshold).astype(int)
    # Customer bootstrap on one test snapshot; descriptive CI, not model-selection uncertainty.
    rng = np.random.default_rng(seed)
    bootstrap = []
    for _ in range(500):
        idx = rng.integers(0, len(test), len(test))
        y = test.RepeatPurchase.to_numpy()[idx]
        if len(np.unique(y)) == 2:
            bootstrap.append(average_precision_score(y, probabilities[idx]))
    metadata = {"selected_model": winner, "threshold": threshold, "target": "purchase_within_30_days",
                "test_ap_ci_low": float(np.quantile(bootstrap, .025)),
                "test_ap_ci_high": float(np.quantile(bootstrap, .975)),
                "train_rows": len(train), "validation_rows": len(val), "test_rows": len(test)}
    # Deployment refit includes test labels only AFTER evaluation and saves separate scores.
    production = clone(models[winner]).fit(pd.concat([development, test])[FEATURES],
                                          pd.concat([development, test]).RepeatPurchase)
    current = features(sales, observed_until)
    current["RepeatProbability30d"] = production.predict_proba(current[FEATURES])[:, 1]
    current["AsOfDate"] = str(pd.Timestamp(observed_until).date())
    return {"validation": validation, "test": pd.DataFrame(test_rows), "calibration": calibration,
            "importance": explanation, "predictions": predictions, "current": current.reset_index(),
            "metadata": metadata, "evaluation_model": model, "production_model": production}
