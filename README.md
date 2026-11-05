# Predictive E-Commerce Behavioral Analytics

A Colab-first implementation of customer segmentation, market basket analysis, recommendations, and repeat-purchase prediction, with Power BI exports.

**Status:** implemented and exercised on the original dataset in Colab. [Verified real-data results](results/2026-11-05/README.md) and [validation record](docs/VALIDATION.md) are included. The Power BI Desktop dashboard remains to be assembled. This is an academic project, not a deployed retail service.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ashyx12/minor-spec/blob/main/notebooks/01_colab_end_to_end.ipynb)

## Start here — no local Python required

1. Click **Open in Colab** above, or clone/download this repository.
2. Open [Google Colab](https://colab.research.google.com/), choose **File → Upload notebook**, and select `notebooks/01_colab_end_to_end.ipynb`. Alternatively, open it via Colab's GitHub tab after uploading the repository.
3. The first code cell already sets `REPO_URL` to this repository. If you have not created the repo yet, leave it blank and upload the supplied repository ZIP when prompted. Private repositories can also use ZIP upload without embedding tokens.
4. Use a standard CPU runtime, execute the notebook in order, and mount Drive when prompted. All dataset processing, training and evaluation happen in Colab. GPU is unnecessary.
5. Download the generated results ZIP. Build the dashboard in Power BI Desktop using `powerbi/BUILD_GUIDE.md` and `powerbi/measures.dax`.

Direct notebook: https://colab.research.google.com/github/ashyx12/minor-spec/blob/main/notebooks/01_colab_end_to_end.ipynb

## What is implemented

| Component | Implementation | Evidence produced by a run |
|---|---|---|
| Data preparation | Schema validation, missing-ID policy, duplicates, return ledger, non-merchandise exclusions, outlier flags | Data-quality JSON, source SHA-256 |
| RFM segmentation | log1p + scaling, K-Means k=2..8, silhouette selection | Profiles, inertia, Davies–Bouldin, seed stability |
| Basket analysis | Sparse Apriori and FP-Growth on matched search spaces | Support/confidence/lift rules, itemsets, timings |
| Recommendations | Popularity, association rules, item-item cosine, validated hybrid weight | Precision/Recall/NDCG/Hit@10, coverage, warm/cold cohorts, bootstrap intervals |
| Predictive extension | 30-day repeat purchase; prevalence, logistic and gradient boosting baselines | Chronological validation/test metrics, calibration, permutation importance |
| BI handoff | Fact/dimension CSVs, recommendation table, data dictionary | Six figures, Power BI import function, DAX, build guide |
| Reproducibility | Dependency pins, seeded models, config, hashes, failure manifests | Saved models, environment record, synthetic tests and CI |

## Repository map

- `notebooks/01_colab_end_to_end.ipynb`: complete user workflow.
- `src/retail_analytics/`: reusable processing, models, evaluation and exports.
- `tests/`: synthetic-only correctness and integration tests; no dataset download.
- `docs/METHODOLOGY.md`: precise targets, splits, metrics, assumptions and limitations.
- `docs/DATA_CARD.md`: source, license and cleaning decisions.
- `docs/PROJECT_REVIEW.md`: report mapping, assessment checklist and viva prompts.
- `powerbi/`: model relationships, import helper, measures and dashboard design.

## Outputs

Every run has an immutable directory containing `manifest.json`, `data_quality.json`, CSV metrics, model files, `figures/`, and `powerbi/`. A failed run records `status=failed`; create a fresh run directory to retry. Do not confuse full-history production exports with held-out test predictions. `environment.txt` captures the actual runtime; pins are not a guarantee that future Colab runtimes remain compatible.

## Scope and academic honesty

The attached report specifies transactions, RFM/K-Means, Apriori, recommendations and Power BI. This repository implements those computational components and adds evaluation and an explicitly predictive task. A `.pbix` dashboard must still be assembled in Power BI Desktop. Real-data result summaries are included; no `.pbix` or faculty acceptance is claimed.

There is no universal “2026 minor specialization project” rubric. The additions aim to make the work reproducible, technically justified and assessable. Confirm your department's requirements, execute the experiments, interpret failures, complete the dashboard, and rewrite the report using actual results. Do not claim novelty merely from combining established methods.

The source is a historical UK gift retailer (2010–2011), not 2026 consumer behavior. There are no verified product categories, clickstream, reviews, ratings, profit margins or demographic attributes. Those cannot be inferred as ground truth. Sentiment analysis, real-time serving, mobile apps and chatbots from the report's future-work list remain future work, as they require different data or scope.

## Sources

- D. Chen (2015), *Online Retail*, UCI, DOI [10.24432/C5BW33](https://doi.org/10.24432/C5BW33), CC BY 4.0. [Source](https://archive.ics.uci.edu/dataset/352/online+retail).
- [scikit-learn: leakage and preprocessing](https://scikit-learn.org/stable/common_pitfalls.html).
- [mlxtend Apriori](https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/) and [association rules](https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/).
- [Microsoft: Power BI star schema](https://learn.microsoft.com/power-bi/guidance/star-schema).
- [Colab resource and runtime limitations](https://research.google.com/colaboratory/faq.html).

The code is provided without choosing a repository license on your behalf. Add a code license if you intend to grant reuse rights. Dataset attribution/license is separate.
