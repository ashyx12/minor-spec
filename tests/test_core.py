import numpy as np
import pandas as pd
import pytest
from retail_analytics.data import clean, purchases, features, snapshot
from retail_analytics.basket import mine
from retail_analytics.recommend import Recommender, evaluate
from retail_analytics.predict import experiment


def raw_rows():
    return pd.DataFrame([
        ['1','10000','A',2,'2011-01-01',5,1,'UK'],
        ['1','10001','B',1,'2011-01-01',3,1,'UK'],
        ['C2','10000','A',-1,'2011-01-02',5,1,'UK'],
        ['3','10000','A',1,'2011-01-03',5,None,'UK'],
        ['4','POST','Postage',1,'2011-01-03',2,1,'UK'],
        ['5','10002','C',0,'2011-01-03',2,1,'UK'],
    ], columns=['InvoiceNo','StockCode','Description','Quantity','InvoiceDate','UnitPrice','CustomerID','Country'])


def synthetic_transactions(seed=41):
    """Artificial data only; never report its model scores as project evidence."""
    rng = np.random.default_rng(seed)
    rows = []; invoice = 0
    for day in pd.date_range('2010-12-01', '2011-12-09'):
        for customer in range(1, 51):
            probability = .008 + (customer % 10)*.006
            if rng.random() < probability:
                invoice += 1
                group = customer % 4
                items = {10000+group, 10010+group, int(rng.integers(10000,10040))}
                for item in items:
                    rows.append([str(invoice),str(item),f'Product {item}',int(rng.integers(1,5)),
                                 day + pd.Timedelta(hours=12),float(item%9+1),str(customer),'UK'])
    return pd.DataFrame(rows, columns=raw_rows().columns)


def test_cleaning_ledger_and_identity():
    raw = raw_rows()
    raw = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
    ledger, audit = clean(raw)
    assert audit['exact_duplicates_removed'] == 1
    assert audit['non_merchandise_rows_removed'] == 1
    assert ledger.Revenue.sum() == 13
    assert len(purchases(ledger)) == 2
    assert set(ledger.CustomerID) == {'1','UNKNOWN'}


def test_schema_and_invalid_numbers():
    with pytest.raises(ValueError, match='Missing columns'):
        clean(pd.DataFrame())
    raw = raw_rows()
    raw['UnitPrice'] = raw.UnitPrice.astype(float)
    raw.loc[0, 'UnitPrice'] = np.inf
    ledger, audit = clean(raw)
    assert audit['invalid_rows_removed'] == 2
    assert np.isfinite(ledger.Revenue).all()


def test_point_in_time_frequency_and_future_invariance():
    sales = purchases(clean(raw_rows())[0])
    x = features(sales, '2011-01-02')
    assert x.loc['1', 'Frequency'] == 1
    assert x.loc['1', 'Monetary'] == 13
    future = sales.copy(); future['InvoiceDate'] = pd.Timestamp('2012-01-01'); future['Revenue'] = 1e9
    pd.testing.assert_frame_equal(x, features(pd.concat([sales, future]), '2011-01-02'))


def test_label_window_and_censoring():
    sales = purchases(clean(raw_rows())[0])
    with pytest.raises(ValueError, match='Incomplete'):
        snapshot(sales, '2011-01-02', '2011-01-10')
    future = sales.iloc[[0]].copy(); future['InvoiceDate'] = pd.Timestamp('2011-02-01')
    combined = pd.concat([sales, future])
    assert snapshot(combined, '2011-01-02', '2011-02-01').RepeatPurchase.iloc[0] == 0
    future['InvoiceDate'] = pd.Timestamp('2011-01-31')
    assert snapshot(pd.concat([sales, future]), '2011-01-02', '2011-02-01').RepeatPurchase.iloc[0] == 1


def test_empty_rules_and_popularity_fallback():
    sales = purchases(clean(raw_rows())[0])
    rules, _, _ = mine(sales, min_support=1.)
    assert rules.empty  # lift = 1 is filtered
    model = Recommender().fit(sales, rules)
    assert model.recommend('unknown', 2) == model.recommend('unknown', 2, mode='popularity')
    assert model.recommend('1', 2, exclude_seen=True) == []


def test_hand_calculated_ranking_metrics():
    sales = purchases(clean(raw_rows())[0])
    rules, _, _ = mine(sales)
    model = Recommender().fit(sales, rules)
    future = sales.iloc[[0]].copy()
    metrics, _ = evaluate(model, future, k=2, modes=('popularity',))
    row = metrics.loc[metrics.cohort.eq('all')].iloc[0]
    assert row.precision_at_k == .5
    assert row.recall_at_k == 1
    assert row.ndcg_at_k == 1


def test_snapshot_overlap_rejected():
    sales = purchases(clean(raw_rows())[0])
    with pytest.raises(ValueError, match='horizons'):
        experiment(sales, '2012-01-01', ['2011-01-01'], '2011-01-15', '2011-02-15')
