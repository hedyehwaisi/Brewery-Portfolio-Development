"""Regression tests for the data-quality calculations.

Run from the project root:  python -m pytest tests -q
They pin the facts the dashboard text relies on (duplicate pairs, shared points,
redundant columns, harmonized labels), so a change in the data or the logic
shows up here before it shows up as a wrong sentence on a page.
"""
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import quality_core as qc  # noqa: E402

FILES = {"raw": ("raw_breweries.db", "raw_breweries"),
         "all_records": ("analysis_all_records.db", "analysis_all_records"),
         "unique_records": ("analysis_unique_records.db", "analysis_unique_records")}


@pytest.fixture(scope="module")
def frames():
    out = {}
    for name, (db, tbl) in FILES.items():
        with sqlite3.connect(ROOT / "data" / db) as con:
            out[name] = pd.read_sql_query(f'SELECT * FROM "{tbl}"', con)
    return out


def test_same_columns_in_every_dataset(frames):
    ov = qc.column_overview(frames)
    assert len(ov) == 16
    never_null = set(ov.loc[ov["complete_in_every_dataset"], "column"])
    assert never_null == {"id", "name", "brewery_type", "city", "state_province", "country", "state"}


def test_redundant_columns(frames):
    pairs = {tuple(sorted(p)) for p in qc.identical_columns(frames["raw"])[["column", "identical_to"]].values}
    assert pairs == {("address_1", "street"), ("state", "state_province")}


@pytest.mark.parametrize("name", ["all_records", "unique_records"])
def test_near_duplicates_are_six_belgian_pairs(frames, name):
    nd = qc.near_duplicate_pairs(frames[name])
    assert len(nd) == 12 and nd["pair"].nunique() == 6
    assert set(nd["country"]) == {"Belgium"}
    assert (nd.drop_duplicates("pair")["pair_distance_km"] > 1).sum() == 1  # Liefmans only


def test_shared_coordinates(frames):
    sc = qc.shared_coordinate_records(frames["all_records"])
    assert len(sc) == 800 and sc["location_id"].nunique() == 336
    kinds = sc.drop_duplicates("location_id")["stack_kind"].value_counts()
    assert kinds["same street address"] == 299 and kinds["different street addresses"] == 37


def test_harmonization_keeps_raw_and_merges_regions(frames):
    h = qc.harmonize_states(frames["raw"])
    assert h["state_province"].equals(frames["raw"]["state_province"])
    be = h[h["country"] == "Belgium"]
    assert be["state_province"].nunique() == 7
    assert set(be["state_province_harmonized"]) == {"Flanders", "Wallonia", "Brussels Capital Region"}
    assert "MIssouri" not in set(h["state_province_harmonized"])
    frontier = h[h["name"] == "Frontier Beer Company"].iloc[0]
    assert frontier["state_province"] == "Gauteng" and frontier["state_province_harmonized"] == "Western Cape"


def test_coordinate_missing_rate_totals(frames):
    for col in ["brewery_type", "country", "state_province", "city"]:
        t = qc.coordinate_missing_rate(frames["raw"], col)
        assert t["total_records"].sum() == 11959
        assert t["records_missing_coordinates"].sum() == 2468
        assert t["missing_coordinate_percentage"].is_monotonic_decreasing


def test_coordinate_missingness_is_not_random(frames):
    res = qc.missingness_association(frames["raw"], "coordinates", "brewery_type")
    assert res["p_value"] < 0.001 and res["cramers_v"] >= 0.3
    assert qc.missingness_association(frames["all_records"], "coordinates", "brewery_type") is None


def test_shared_value_location_is_a_real_place(frames):
    g = qc.shared_value_groups(frames["all_records"], "phone")
    d = frames["all_records"]
    places = set(d["city"] + ", " + d["state_province"] + ", " + d["country"])
    assert g["top_location"].isin(places).all()


def test_coordinate_corrections_log_and_application(frames):
    log = qc.load_corrections()
    assert log["action"].value_counts().to_dict() == {"flagged only": 18, "swapped back": 8,
                                                      "sign flipped back": 1, "copied from identical record": 1}
    fixed = qc.apply_corrections(frames["all_records"])
    assert fixed["latitude"].min() > -60
    mescan = fixed[fixed["name"] == "Mescan Brewery"].iloc[0]
    assert mescan["longitude"] < 0 and mescan["coordinate_status"] == "corrected: coordinate sign flipped back"
    assert fixed["coordinate_status"].value_counts()["flagged: far from its city"] == 18
    liefmans = fixed[fixed["name"] == "Liefmans"][["latitude", "longitude"]].drop_duplicates()
    assert len(liefmans) == 1  # both records now share the twin's position
    raw = frames["raw"]
    assert pd.to_numeric(raw["latitude"]).min() < -80  # the database itself is unchanged


@pytest.mark.parametrize("name,url,expected", [
    ("Ballast Point Brewing Company", "https://ballastpoint.com", "full name in domain"),
    ("Stöttner", "http://www.stoettner.de", "full name in domain"),
    ("Granite City Food & Brewery", "http://www.gcfb.net", "initials in domain"),
    ("Daft Badger Brewing", "https://www.facebook.com/daftbadgerbrewing", "social-media page"),
    ("Anything", None, "no website")])
def test_name_website_link(name, url, expected):
    assert qc.name_website_link(name, url) == expected
