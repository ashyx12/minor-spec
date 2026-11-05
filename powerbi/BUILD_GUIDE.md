# Build the Power BI dashboard

The computational pipeline runs in Colab. Desktop imports finished tables and defines relationships/measures; it does not train models. A PBIX file is not included because it must be authored and validated in Power BI Desktop with your actual run outputs.

## Import

1. Extract your completed Colab results ZIP to a stable directory.
2. In Power BI Desktop, Get Data → Text/CSV for each table under the run's `powerbi/` directory. Alternatively use `ImportCsv.pq` with a Text parameter `ExportFolder` and one `LoadCsv("Table.csv")` query per table.
3. Name queries exactly as the CSV basenames. Do not load DataDictionary into the analytic model unless needed for documentation.
4. Disable automatic type detection for IDs or explicitly set InvoiceNo, StockCode, CustomerID, Country and Segment to Text. Set Date/AsOfDate to Date, IsReturn to True/False, LineID/Rank/Year/MonthNumber to Whole number, Quantity/UnitPrice/Revenue and RFM values to Decimal number, probabilities/Score to Decimal number. Source numbers use dot decimals; choose English (United Kingdom) locale when needed.
5. Check one-sided keys for uniqueness. The exporter supplies unique dimension keys and all fact keys should be covered. Use single-direction filtering from dimensions to facts.

## Relationships

| One side | Many side | Notes |
|---|---|---|
| DimDate.Date | FactSales.Date | Active, one-to-many; mark DimDate as date table |
| DimProduct.StockCode | FactSales.StockCode | Active, one-to-many |
| DimCustomer.CustomerID | FactSales.CustomerID | Active, one-to-many |
| DimCountry.Country | FactSales.Country | Active, one-to-many |
| DimCustomer.CustomerID | FactRecommendations.CustomerID | Active, one-to-many |
| DimProduct.StockCode | FactRecommendations.StockCode | Active, one-to-many |

Do not link the fact tables directly. Do not enable bidirectional filtering. Recommendation AsOfDate is a snapshot label, not a sales date; leave it disconnected from DimDate. Current segment/probability describe the latest snapshot and do not change with a historical sales date slicer. Country is an invoice-level dimension, not a guaranteed fixed customer residence.

Add each measure in `measures.dax` separately. Format revenue/AOV as GBP, rates/probabilities as percentages and counts as integers. Sort DimDate.Month by MonthNumber; use YearMonth on monthly trend axes.

## Five report pages

| Page | Visuals | Filters and interpretation |
|---|---|---|
| Sales overview | Net revenue, gross revenue, return value, order count, AOV; monthly trend; top products | Date and country slicers; label December 2011 as partial |
| Customer segments | Customer count by Segment, median RFM profile table, recency vs monetary scatter | Latest AsOfDate in title; segment filters historical sales by current membership |
| Basket insights | Rule table showing antecedents/consequents/support/confidence/lift; benchmark chart | Import association_rules and mining_benchmark as disconnected tables; rules are exploratory |
| Recommendations | Customer selector, ranked products/descriptions, score; test baseline comparison | FactRecommendations rank ascending, top 10; import recommendation_test_metrics as disconnected; select cohort=all |
| Repeat-purchase model | Current probability histogram/customer table; test AP/ROC/Brier comparison; calibration plot | Import prediction_test and prediction_calibration as disconnected; title states 30-day repeat purchase, not permanent churn |

You can import customer_segments and segment_profiles as disconnected explanatory tables if you need their exact diagnostics, but do not add redundant relationships. Disconnected metric tables do not respond to sales filters; show that in page captions. Prediction test metrics and current probabilities come from different model stages. Recommendation scores are not probabilities.

For calibration, place mean_probability on X and observed_fraction on Y. Add the saved PNG as an alternative when a native visual is inconvenient. For the customer scatter, use CustomerID as detail and do not sum per-customer RFM across unrelated tables.

## Acceptance checks

- FactSales row count matches manifest.powerbi_rows.FactSales; sum Revenue equals the retained ledger total, not the raw workbook total.
- Net Merchandise Revenue = Gross Purchase Revenue − Return Value under identical filters.
- A single customer selection shows at most 10 recommendations, with unique product IDs/ranks.
- Dates are actual Date columns; monthly ordering is chronological.
- Date filters affect sales; they do not silently imply that current segment/risk values were known historically.
- Unknown customers are excluded from Known Purchasing Customers but their merchandise revenue remains included.
- No unsupported demographic, margin, profit, uplift, review-sentiment or live-service claims appear.
- Publish only after reviewing exports; no automatic Power BI service publication is performed by this project.
