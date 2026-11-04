"""Tests for the prediction models and the stored results of brewery_prediction.ipynb.

Run from the project root:  python -m pytest tests -q
"""
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import prediction_core as pc  # noqa: E402
import quality_core as qc  # noqa: E402


@pytest.fixture(scope="module")
def raw():
    with sqlite3.connect(ROOT / "data" / "raw_breweries.db") as con:
        return qc.harmonize_states(qc.apply_corrections(pd.read_sql_query("SELECT * FROM raw_breweries", con)))


@pytest.fixture(scope="module")
def model(raw):
    return pc.PlaceTypeModel(alpha=0.05, beta=20).fit(pc.operating(raw))


def test_operating_excludes_planning_and_closed(raw):
    assert not pc.operating(raw)["brewery_type"].isin(["planning", "closed"]).any()


def test_good_turing_add_one():
    assert pc.good_turing_new([1, 1, 3]) == pytest.approx((2 + 1) / (5 + 1))
    assert pc.good_turing_new([4, 5]) > 0


@pytest.mark.parametrize("level,parent", [("country", ()), ("state", ("United States",)),
                                          ("city", ("United States", "California")), ("city", ("Germany", "Bayern"))])
def test_place_distributions_are_probabilities(model, level, parent):
    d = model.place_distribution(level, parent)
    assert d.sum() == pytest.approx(1.0) and (d > 0).all() and pc.NEW_PLACE in d.index
    t = model.place_given_type(level, "brewpub", parent)
    assert t.sum() == pytest.approx(1.0)


def test_type_shrinkage_moves_toward_the_region(model):
    city = model.type_distribution("United States", "California", "San Diego")
    state = model.type_distribution("United States", "California")
    own = model.t["city"].loc[("United States", "California", "San Diego")]
    own = own / own.sum()
    for t in ["micro", "brewpub"]:
        assert min(own[t], state[t]) - 1e-9 <= city[t] <= max(own[t], state[t]) + 1e-9
    assert model.type_distribution("Germany").index[0] == "brewpub"


def test_stored_results_beat_baselines():
    with sqlite3.connect(ROOT / "data" / "prediction_results.db") as con:
        summ = pd.read_sql_query("SELECT * FROM summary", con)
        params = dict(pd.read_sql_query("SELECT key, value FROM params", con).itertuples(index=False))
        ptest = pd.read_sql_query("SELECT * FROM planning_test", con).set_index("test")
    assert (summ["log_loss"] < summ["baseline_log_loss"]).all()
    assert float(params["bandwidth_km"]) in (10, 20, 30, 40, 50, 60, 80)
    assert ptest.loc["state", "log_loss"] < ptest.loc["state", "log_loss_uniform"]
    assert abs(ptest.loc["new_city", "predicted_%"] - ptest.loc["new_city", "observed_%"]) < 10


def test_kde_grid_shares_sum_to_one(raw):
    g = pc.operating(raw).dropna(subset=["latitude", "longitude"])
    g = g[g[pc.STATE] == "Colorado"]
    grid = pc.kde_grid(pc.kde_fit(g["latitude"], g["longitude"], 30), (37, 41), (-109, -102), n=30)
    assert grid["share"].sum() == pytest.approx(1.0) and (grid["share"] >= 0).all()
