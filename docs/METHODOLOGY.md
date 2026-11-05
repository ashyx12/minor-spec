# Methodology and evaluation protocol

## Research questions

1. What distinct purchase-value and activity profiles emerge from RFM?
2. What product associations recur, and how do Apriori and FP-Growth compare under identical restrictions?
3. Do personalized rankings outperform a popularity baseline on later purchases?
4. Can prior transaction features predict whether an existing customer buys again within 30 days?

These are offline questions. They do not establish causal revenue uplift or improved engagement.

## Data populations

The merchandise ledger retains signed returns and anonymous purchases for BI. Modeling uses positive-price, non-return merchandise transactions with known customer IDs. Exact duplicate source rows are removed under an explicit, imperfect duplicate policy. Large orders are flagged, not discarded. RFM uses log1p to reduce skew without changing revenue totals. Identifiers are strings. Frequency is distinct invoices, not line count.

Feature cutoff is exclusive: only timestamps `< cutoff` enter features. Recency and tenure use fractional days; monetary value is gross positive purchase value, not lifetime profit or net ledger revenue. Country is descriptive and is not used by prediction models. Customer ID is never a model feature.

## Segmentation

Full-history, latest-as-of segmentation is descriptive. Transform R/F/M with log1p and StandardScaler; try K=2..8 when enough distinct profiles exist. Select highest silhouette (sample at most 2,000 customers), with lower K breaking ties. Report inertia, Davies–Bouldin and adjusted Rand agreement between two seeds. This seed check is not temporal stability. Cluster numbers are ordered by median monetary value; semantic labels such as “premium” need inspection. No supervised “cluster accuracy” exists here. The selected clustering is not used as a predictive feature, avoiding leakage from full-history clusters.

## Basket mining

Baskets are invoice-level product presence. Support denominator includes every modeling invoice, even when restricting the catalog leaves a basket empty. Select the 300 most frequent products in the applicable training partition; default support .02, confidence .20, lift >1, maximum itemset length 3. Apriori uses low-memory mode. FP-Growth uses the identical basket matrix, support and maximum length. Itemset membership is asserted equal. Benchmark times are single-run wall-clock observations, not a general complexity result.

Restrictions can miss niche associations and must be reported. Associations do not imply causation. No statistical significance correction is claimed. Full-history displayed rules are exploratory; evaluation rules are independently trained only before each cutoff.

## Recommendation experiment

| Stage | Fit history | Evaluation purchases |
|---|---|---|
| Validation | Before 2011-09-01 | September 2011 |
| Final test | Before 2011-10-01 | October 2011 |
| Production export | All available purchases | No measured future outcome |

Candidate catalog comes only from fit history. Binary customer-item interactions produce cosine item similarity; keep up to 40 non-self neighbors. Sum neighbor scores across history. Rule score sums confidence × log(1+lift) where every antecedent item is in customer history. Normalize each component by its maximum for that customer. Hybrid is alpha×CF + (1-alpha)×rules. Select alpha from {0,.25,.5,.75,1} by validation all-customer NDCG@10, then freeze it. A tiny popularity tie-break/fallback applies to all models. Unseen customers receive popularity recommendations.

Previously purchased items remain eligible: this task is future-purchase ranking including replenishment, not exclusively novel-item discovery. All unique products bought by a customer in the evaluation window are relevant. Unseen future products stay in the recall and ideal-DCG denominators, so catalog limitations cannot artificially improve scores. Candidate-recall ceiling makes that limitation visible. Test customers include unseen customers; separate warm/cold summaries are exported. Customers with no purchase in the evaluation window are not in ranking metrics, a limitation of this protocol.

For hits h at ranks r: Precision@K = hits/K; Recall@K = hits/number of relevant products; DCG = sum(h/log2(r+1)); NDCG = DCG/ideal DCG; Hit@K indicates any hit. Metrics are macro averaged by customer. Coverage is unique recommended products divided by training catalog size (reported for all evaluation users). 500 customer bootstrap resamples estimate fixed-model mean-NDCG intervals. Intervals do not demonstrate statistical superiority; use paired customer comparisons if making a significance claim.

## Repeat-purchase prediction

Target: at least one qualifying purchase during `[snapshot, snapshot+30 days)`, among customers known before the snapshot. This is not permanent churn. Features: recency, distinct invoice frequency, monetary value, tenure, distinct products, average order value. Each comes only from earlier purchases.

| Partition | Snapshot dates | Label horizon |
|---|---|---|
| Training | July 1, August 1, September 1, 2011 | 30 days after each |
| Validation | October 1, 2011 | 30 days after |
| Test | November 1, 2011 | 30 days after |

Each partition's labels end before the next partition starts. The same customer may appear across snapshots, matching forecasting for known customers; this is not evaluation of generalization to unseen identities. Repeated training customers are correlated. Censored horizons are rejected using the start of the final observed calendar day as a conservative complete-data boundary; continuity of earlier source coverage is assumed.

Compare training prevalence, log1p/scaled logistic regression and constrained histogram gradient boosting. All transformers fit on the relevant training partition. Disable random early-stopping validation in gradient boosting. Select model by validation average precision (Brier as tie-break); choose its threshold by validation F1. Refit candidate models on train+validation, evaluate on untouched test, and save all baseline results. The threshold remains fixed after refit, which can change calibration. AP is a precision-recall ranking summary, not trapezoidal PR area; exported column is explicitly `pr_auc_ap`. Also report ROC-AUC, Brier, F1, precision, recall and prevalence. Calibration curves and held-out permutation importance are diagnostic; do not tune against them. AP intervals use customer bootstrap of the single test snapshot.

Finally refit the selected model including test labels for production use, only after saving evaluation results. Production scores use all observed feature history at the next midnight after the last transaction day. Saved evaluation and production models are distinct. Production rankings/scores are not evaluated outcomes. The binary probability is not calibrated by an additional fitted calibrator; assess its reliability from Brier and calibration curves.

## Reproducibility and resource choices

CPU implementation, seeded random components, pinned core packages, bounded mining and neighbor queries, source SHA-256 and artifact hashes. Dependency pins correspond to the synthetic verification environment; the exact full Colab environment is captured at execution. Drive output protects finished stages from runtime loss, but automatic checkpoint resume is not implemented. Snapshot dates are configurable through a JSON file accepted by the runner; changing data requires reviewing cutoffs and complete label horizons.

## Known limitations and defensible extensions

One historical retailer, wholesaler effects, seasonality, unknown missingness, no inventory/exposure logs, no margin or demographic labels. Single test months can be unstable; no claim of live 2026 performance. Global chronological evaluation handles future leakage but does not address all selection biases. Repeat the study across earlier rolling cutoffs as a documented extension; do not silently tune using the designated test months. Compare an alternate catalog cap/support and duplicate policy in separate runs if resource budgets permit. Real business uplift requires a properly designed live experiment, outside this dataset's capabilities.
