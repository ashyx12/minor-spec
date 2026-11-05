# Submission and viva checklist

This is a suggested review checklist, not an official institutional rubric. Obtain the actual rubric from your guide.

## Map from the supplied report

| Report requirement | Implemented location | What you must show |
|---|---|---|
| Data collection and preprocessing | data.py + Colab acquisition cell | Provenance, hash, audit, handling of returns/missing IDs |
| RFM/K-Means | segmentation.py | K selection, profiles, stability, correct frequency definition |
| Apriori/support/confidence/lift | basket.py | Meaningful rules with honest restrictions |
| Purchase-pattern recommendations | recommend.py | Temporal test against popularity and component baselines |
| Power BI visuals | pipeline.py exports + powerbi/ | Actual PBIX and screenshots created from your run |
| Predictive title | predict.py (extension) | Defined 30-day target, labels, validation and test evidence |

Real-time APIs, deep learning, sentiment, mobile support and chatbots are listed as future work in the report. They are not prerequisites for demonstrating the stated objectives and are not implemented. The available dataset cannot support several of them without new data.

## Before submission

- Run the full notebook in Colab on the original data; save manifest, environment and output ZIP.
- Inspect cleaning counts and verify merchandise/duplicate assumptions.
- Report all baselines, including negative results. Explain why AP and NDCG are appropriate.
- Populate a results table directly from CSVs; never substitute synthetic scores.
- Build and check the PBIX using the supplied guide. Include five pages or a justified smaller set covering the original objectives.
- Replace the original report's unsupported “improves customer experience” claims with measured offline findings and limitations.
- Expand the literature review with verified, relevant recent work; confirm bibliographic details against publisher/author sources. Existing references in the report have not been comprehensively verified by this implementation.
- State contributions as an integrated, evaluated implementation—not invention of K-Means, Apriori or hybrid recommendation.
- Add architecture, reproducibility procedure, data/model limitations, screenshots, results and a link to the actual repository.
- Reproduce the run from a fresh Colab session and preserve your original evidence before changing settings.

## Suggested results table (fill only after a real run)

| Question | Evidence file | Discussion |
|---|---|---|
| Which K and why? | cluster_diagnostics.csv | Silhouette and domain interpretation; stability limits |
| Are rules useful? | association_rules.csv | Support vs rare-item high lift; catalog restrictions |
| Is hybrid better? | recommendation_test_metrics.csv | Compare all four models; validation alpha and cold users |
| Can repeat purchase be predicted? | prediction_test.csv | Compare AP to prevalence and Brier to baseline |
| Are probabilities reliable? | prediction_calibration.csv | Under/overconfidence; do not equate ranking with calibration |
| What drives predictions? | prediction_importance.csv | Predictive association, not causation |

## Viva questions to prepare

Why do returns stay in the BI ledger but leave RFM training? Why does frequency count invoices? What information is available at a snapshot? Why is random train/test splitting unsuitable here? What does a cold-start customer receive? Why allow previously purchased products? What does candidate-recall ceiling mean? How do you select K without target labels? Why is no-purchase in 30 days not permanent churn? Can high lift prove a promotion will increase revenue? Why can a popularity baseline win? How would you test temporal stability or real uplift?

## Scope of validation

Unit and integration tests use artificial transactions only. They test correctness and artifacts, not project performance. Real-data Colab execution, actual Power BI rendering and institutional approval remain separate acceptance steps.
