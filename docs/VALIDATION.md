# Validation record

Verification date: 2026-11-05.

## Completed

- 10 automated tests passed in 7.53 seconds on artificial transactions.
- Full synthetic integration run exercised cleaning, K-Means selection, Apriori/FP-Growth, all recommendation baselines, chronological predictive modeling, model serialization, six chart exports and Power BI tables.
- Tested schema rejection, invalid numerical values, returns and anonymous customers, invoice-based frequency, invariance of past features to future records, exclusive label-window boundaries, rejection of censored labels, cold-start fallback, seen-item exclusion, hand-calculated ranking metrics, overlapping label-horizon rejection, dimension-key integrity, output-overwrite prevention and failure manifests.
- Notebook validated against nbformat; no precomputed outputs or fabricated results are embedded.
- All Python modules and notebook code cells parsed successfully.

Environment: Python 3.12.14; numpy 2.3.5; pandas 2.2.3; scipy 1.17.0; scikit-learn 1.8.0; mlxtend 0.23.4; openpyxl 3.1.5; matplotlib 3.10.8; joblib 1.5.3. Tests capped numerical library threads at two.

## Real-data Colab run subsequently verified

The user supplied the completed Colab results archive on 2026-11-05. Its manifest reports successful execution on the original 541,909-row input under Python 3.13.18. All 35 recorded artifact hashes matched. Customer/product dimension keys were unique and covered the sales and recommendation fact keys. All 4,334 modeled customers had exactly ten unique product recommendations.

The pipeline recorded 153.84 seconds; this excludes installation and acquisition. Aggregate results and figures are preserved in `results/2026-11-05/`. The original manifest has no Git commit because the run preceded repository publication. This verifies the supplied output artifacts; it does not independently observe or rerun the user's Colab session.

## Remaining acceptance steps

- Assemble and validate the actual Power BI Desktop dashboard with the complete Colab exports.
- Confirm department-specific assessment requirements with the project guide.
- Assess generalization beyond these historical test months before claiming broader performance.

No causal business uplift or live deployment is established by offline metrics.
