"""Source acquisition, explicit cleaning policy and point-in-time features."""
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO
import hashlib
import numpy as np
import pandas as pd

SOURCE = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
REQUIRED = ["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID", "Country"]
FEATURES = ["Recency", "Frequency", "Monetary", "Tenure", "Products", "AverageOrderValue"]


def download(destination):
    """Download the original workbook; cached files are hashed by the runner."""
    p = Path(destination)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        with urlopen(SOURCE, timeout=120) as response:
            payload = response.read()
        with ZipFile(BytesIO(payload)) as archive:
            matches = [n for n in archive.namelist() if n.endswith(".xlsx")]
            if len(matches) != 1:
                raise ValueError("Expected exactly one UCI workbook; inspect source manually.")
            p.write_bytes(archive.read(matches[0]))
    return p


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clean(raw):
    missing = set(REQUIRED) - set(raw.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    d = raw[REQUIRED].copy()
    audit = {"raw_rows": len(d), "missing_by_column": d.isna().sum().to_dict()}
    # Exact duplicate policy is deliberate: identical lines may be genuine; see data card.
    audit["exact_duplicates_removed"] = int(d.duplicated().sum())
    d = d.drop_duplicates().copy()
    for c in ["InvoiceNo", "StockCode", "CustomerID", "Country", "Description"]:
        d[c] = d[c].astype("string").str.strip().replace("", pd.NA)
    d["CustomerID"] = d.CustomerID.str.replace(r"\.0$", "", regex=True)
    d["InvoiceDate"] = pd.to_datetime(d.InvoiceDate, errors="coerce")
    for c in ["Quantity", "UnitPrice"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    invalid = (d[["InvoiceNo", "StockCode", "InvoiceDate"]].isna().any(axis=1)
               | ~np.isfinite(d.Quantity) | ~np.isfinite(d.UnitPrice)
               | d.UnitPrice.lt(0) | d.Quantity.eq(0))
    audit["invalid_rows_removed"] = int(invalid.sum())
    d = d.loc[~invalid].copy()
    # Known adjustment/postage codes are excluded from merchandise analytics.
    merchandise = d.StockCode.str.fullmatch(r"\d{5}[A-Za-z]*", na=False)
    audit["non_merchandise_rows_removed"] = int((~merchandise).sum())
    d = d.loc[merchandise].copy()
    d["Country"] = d.Country.fillna("Unknown")
    d["Description"] = d.Description.fillna("Unknown product")
    d["CustomerID"] = d.CustomerID.fillna("UNKNOWN")
    d["IsReturn"] = d.InvoiceNo.str.upper().str.startswith("C") | d.Quantity.lt(0)
    # Normalize inconsistent positive-quantity cancellation records to negative values.
    d["Quantity"] = np.where(d.IsReturn, -d.Quantity.abs(), d.Quantity)
    d["Revenue"] = (d.Quantity * d.UnitPrice).astype(float)
    d["Date"] = d.InvoiceDate.dt.strftime("%Y-%m-%d")
    audit.update(retained_rows=len(d), return_rows=int(d.IsReturn.sum()),
                 zero_price_rows=int(d.UnitPrice.eq(0).sum()),
                 anonymous_rows=int(d.CustomerID.eq("UNKNOWN").sum()))
    # Audit flags, not automatic deletion: wholesalers can be legitimate large orders.
    audit["quantity_above_99pct_rows"] = int(d.Quantity.gt(d.Quantity.quantile(.99)).sum())
    audit["price_above_99pct_rows"] = int(d.UnitPrice.gt(d.UnitPrice.quantile(.99)).sum())
    d = d.sort_values(["InvoiceDate", "InvoiceNo", "StockCode"]).reset_index(drop=True)
    return d, audit


def purchases(ledger):
    return ledger.loc[~ledger.IsReturn & ledger.UnitPrice.gt(0) & ledger.CustomerID.ne("UNKNOWN")].copy()


def features(sales, cutoff):
    """All features use strictly earlier transactions; frequency counts invoices."""
    cutoff = pd.Timestamp(cutoff)
    d = sales.loc[sales.InvoiceDate.lt(cutoff)]
    g = d.groupby("CustomerID")
    x = g.agg(Last=("InvoiceDate", "max"), First=("InvoiceDate", "min"),
              Frequency=("InvoiceNo", "nunique"), Monetary=("Revenue", "sum"),
              Products=("StockCode", "nunique"))
    x["Recency"] = (cutoff - x.Last).dt.total_seconds() / 86400
    x["Tenure"] = (cutoff - x.First).dt.total_seconds() / 86400
    x["AverageOrderValue"] = x.Monetary / x.Frequency
    return x[FEATURES]


def snapshot(sales, cutoff, observed_until, horizon=30):
    cutoff = pd.Timestamp(cutoff)
    end = cutoff + pd.Timedelta(days=horizon)
    if end > pd.Timestamp(observed_until):
        raise ValueError("Incomplete label horizon: cannot label censored customers.")
    x = features(sales, cutoff)
    future = sales.loc[sales.InvoiceDate.ge(cutoff) & sales.InvoiceDate.lt(end)]
    x["RepeatPurchase"] = x.index.isin(future.CustomerID).astype(int)
    x["SnapshotDate"] = cutoff
    return x
