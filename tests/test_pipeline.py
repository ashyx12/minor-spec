import json
from pathlib import Path
import nbformat
import pandas as pd
import pytest
from retail_analytics.pipeline import run, Config
from test_core import synthetic_transactions


def test_colab_notebook_valid():
    nb = nbformat.read(Path(__file__).parents[1]/'notebooks'/'01_colab_end_to_end.ipynb', as_version=4)
    nbformat.validate(nb)
    assert all(not c.get('outputs') for c in nb.cells if c.cell_type == 'code')


def test_full_pipeline_synthetic_only(tmp_path):
    raw = tmp_path/'synthetic.csv'; synthetic_transactions().to_csv(raw, index=False)
    output = tmp_path/'run'
    manifest = run(raw, output, Config(max_basket_products=30, min_support=.05))
    assert manifest['status'] == 'complete'
    facts = pd.read_csv(output/'powerbi/FactSales.csv', dtype={'CustomerID':str, 'StockCode':str})
    customers = pd.read_csv(output/'powerbi/DimCustomer.csv', dtype={'CustomerID':str})
    products = pd.read_csv(output/'powerbi/DimProduct.csv', dtype={'StockCode':str})
    assert customers.CustomerID.is_unique and products.StockCode.is_unique
    assert set(facts.CustomerID) <= set(customers.CustomerID)
    assert set(facts.StockCode) <= set(products.StockCode)
    assert facts.LineID.is_unique
    assert customers.RepeatProbability30d.between(0,1).all()
    metrics = pd.read_csv(output/'recommendation_test_metrics.csv')
    assert set(metrics.model) == {'popularity','rules','item_cf','hybrid'}
    assert metrics.ndcg_at_k.between(0,1).all()
    assert len(list((output/'figures').glob('*.png'))) == 6
    with pytest.raises(ValueError, match='not empty'):
        run(raw, output)


def test_failed_run_records_status(tmp_path):
    raw = tmp_path/'invalid.csv'; pd.DataFrame({'bad':[1]}).to_csv(raw,index=False)
    with pytest.raises(ValueError, match='Missing columns'):
        run(raw, tmp_path/'failed')
    assert json.loads((tmp_path/'failed/manifest.json').read_text())['status'] == 'failed'
