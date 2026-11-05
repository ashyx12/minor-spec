# Data card

**Source:** Daqing Chen, Online Retail, UCI Machine Learning Repository (2015), DOI 10.24432/C5BW33. Original source: https://archive.ics.uci.edu/dataset/352/online+retail. License: CC BY 4.0; retain attribution when distributing derived data.

UCI describes 541,909 transaction lines from a UK non-store retailer between 2010-12-01 and 2011-12-09; many customers are wholesalers. Prices are GBP. The repository downloads the original workbook in Colab. The source count is a catalog description, not a claim that a run has been verified against it. Actual row counts, missingness and hashes are logged.

Required columns: InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country. Source XLSX timestamps are naive local business timestamps; no timezone conversion is invented. For CSV substitutions use ISO timestamps. Customer IDs are string identifiers; missing IDs become UNKNOWN in BI and are excluded from modeling. Missing descriptions/countries receive explicit labels. Missing dates/keys, nonfinite quantities/prices, negative prices and zero quantities are rejected.

Exact duplicate rows are removed before normalization. This may remove legitimate identical invoice lines, so discuss this assumption. Cancellation-prefix invoices or negative quantities are treated as returns; quantities are normalized to negative. Zero-price merchandise stays in the ledger but not in paid-purchase models. Merchandise stock codes must match five digits followed by optional letters; adjustment, bank-charge, postage and other codes are excluded. This is a declared heuristic, not a verified product taxonomy. Review excluded counts and adapt the rule if changing sources.

Revenue is quantity × unit price for the retained merchandise ledger. It is not total company revenue, profit, tax-adjusted revenue, or audited accounting. Returns cannot reliably be matched to original orders and are not allocated back to their original period. Net revenue is booked on the return timestamp. Anonymous activity remains in totals; do not mix ledger customer counts with modeled-customer counts.

Outliers above the observed 99th percentile of quantity and price are counted, not trimmed. No manually fabricated categories, ratings, reviews, age or gender appear in this project. Product description uses the last observed value for the latest BI product dimension; it is not used in predictive training.

Public customer IDs are pseudonymous, not proof of anonymity. Keep row-level exports and trained artifacts out of the public Git repository by default. The source report contains personal contact details; it is intentionally not bundled with this repository.
