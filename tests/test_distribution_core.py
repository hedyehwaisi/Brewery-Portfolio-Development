"""Regression tests for the frequency, hierarchy and brewery-type calculations.

Run from the project root:  python -m pytest tests -q
They pin the facts quoted on the Brewery Types and Countries/States/Cities pages.
"""
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import distribution_core as dc  # noqa: E402
import quality_core as qc  # noqa: E402


def load(db, table):
    with sqlite3.connect(ROOT / "data" / db) as con:
        return qc.harmonize_states(pd.read_sql_query(f'SELECT * FROM "{table}"', con))


@pytest.fixture(scope="module")
def all_records():
    return load("analysis_all_records.db", "analysis_all_records")


@pytest.fixture(scope="module")
def raw():
    return load("raw_breweries.db", "raw_breweries")


def test_counts_are_complete_and_ranked(all_records):
    for level in dc.LEVELS:
        c = dc.counts(all_records, level)
        assert c["records"].sum() == len(all_records)
        assert c["records"].is_monotonic_decreasing and c["rank"].tolist() == list(range(1, len(c) + 1))


def test_top_within_shares_sum_to_100_for_full_lists(all_records):
    t = dc.top_within(all_records, "country", "state", 1000)
    sums = t.groupby("country")["share_of_parent_%"].sum()
    assert ((sums - 100).abs() < 1e-9).all()


def test_type_counts_cover_every_record(all_records):
    t = dc.type_counts(all_records)
    assert t["records"].sum() == len(all_records) and abs(t["share_%"].sum() - 100) < 1e-9
    assert set(dc.TYPE_DEFINITIONS) <= set(t["brewery_type"])


def test_country_labelling_conventions(all_records):
    mix = dc.type_mix(all_records, "country", places=["Germany", "Australia", "Belgium"])
    assert mix.loc["Germany", "brewpub"] > 85 and mix.loc["Australia", "micro"] > 95 and mix.loc["Belgium", "micro"] > 90


def test_dominant_type_margin(all_records):
    d = dc.dominant_types(all_records, "state", min_records=30)
    assert (d["margin_pp"] >= 0).all()
    assert d.set_index("place").loc["Bayern, Germany", "dominant_type"] == "brewpub"


def test_type_profiles_split_on_closed_records(all_records):
    assign, prof, sil = dc.type_profiles(all_records)
    assert prof["closed"].max() - prof["closed"].min() > 10


def test_contract_breweries_sit_next_to_each_other(all_records):
    us = all_records[all_records["country"] == "United States"]
    co = dc.type_colocation(us, permutations=99, exclude_same_point=True).set_index("brewery_type")
    assert co.loc["contract", "ratio"] > 2 and co.loc["contract", "p_value"] < 0.05


def test_geocoding_drops_texas(all_records, raw):
    shift = dc.rank_shift(all_records, raw, "state", n=10).set_index("place")
    assert shift.loc["Texas, United States", "rank_reference"] <= 10
    assert shift.loc["Texas, United States", "rank_dataset"] > 10


def test_city_level_is_a_flat_long_tail(all_records):
    city = dc.concentration_stats(dc.counts(all_records, "city")["records"])
    state = dc.concentration_stats(dc.counts(all_records, "state")["records"])
    assert city["single_record_groups_%"] > 60 and city["zipf_exponent"] < state["zipf_exponent"]


def test_type_b_reasons(all_records, raw):
    k = dc.city_state_kinds(all_records, n_states=10, n_cities=10, reference=raw).set_index("city")
    assert k.loc["Toronto", "share_of_state_%"] == 100 and k.loc["Toronto", "share_of_state_in_reference_%"] < 50
    assert k.loc["San Diego", "city_type"].startswith("A")
