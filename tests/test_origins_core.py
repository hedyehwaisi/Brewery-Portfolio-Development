"""Regression tests for the reference places and the origin-based distance calculations.

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
import origins_core as oc  # noqa: E402
import quality_core as qc  # noqa: E402

SWAPPED = {"299656e1-ccc9-4659-8fe8-e62b86be62eb", "a6e8b0d5-d8ba-4206-bd8b-31c1e8cf1061",
           "fc1a8b7d-4870-408a-8810-3dd7720df531", "fc2985d7-f666-4922-bb61-819ddf0c8f88",
           "7ab0e9c9-ffb3-44e1-91fb-99dc7fd4959d", "1eb023fa-2183-4dbb-84a8-bfbc70a8abf2",
           "bcc9f869-4952-4137-a502-5ac716b0a360", "d2aeedba-6cfe-493d-b393-f32ae8986490"}


@pytest.fixture(scope="module")
def ref():
    return oc.load_reference()


@pytest.fixture(scope="module")
def data(ref):
    with sqlite3.connect(ROOT / "data" / "analysis_all_records.db") as con:
        df = qc.harmonize_states(pd.read_sql_query("SELECT * FROM analysis_all_records", con))
    for c in ["latitude", "longitude"]:
        df[c] = pd.to_numeric(df[c])
    return oc.add_origins(df, ref)


def test_reference_coverage(ref):
    assert ref["country"]["capital"].notna().all()
    assert ref["state"]["capital"].notna().sum() >= 207
    c = ref["city"]
    assert c.loc[c["city_latitude"].notna(), "raw_records"].sum() / c["raw_records"].sum() > 0.95
    assert set(ref["sources"]["license"]) >= {"CC BY 4.0"}


@pytest.mark.parametrize("country,state,capital", [
    ("Germany", "Bayern", "Munich"), ("Austria", "Niederösterreich", "Sankt Pölten"),
    ("Netherlands", "Zuid-Holland", "The Hague"), ("United States", "California", "Sacramento"),
    ("Australia", "VIC", "Melbourne")])
def test_known_state_capitals(ref, country, state, capital):
    s = ref["state"].set_index(["country", oc.STATE_COL])
    assert s.loc[(country, state), "capital"] == capital


def test_country_capitals(ref):
    c = ref["country"].set_index("country")["capital"]
    assert c["United States"] == "Washington" and c["Scotland"] == "Edinburgh" and c["England"] == "London"


def test_distances_are_valid(data):
    for col in oc.DIST.values():
        x = data[col].dropna()
        assert (x >= 0).all()
    assert data[oc.DIST[("city", "mean")]].median() < 2


def test_dispersion_radius_shares_are_monotonic(data):
    d = oc.dispersion(data, "state", "mean")
    shares = d[["within_10km_%", "within_25km_%", "within_50km_%", "within_100km_%"]].to_numpy()
    assert (np.diff(shares, axis=1) >= -1e-9).all()


def test_interpret_ratio():
    assert oc.interpret_ratio(1.05)[0] == "mean ≈ median"
    assert oc.interpret_ratio(1.5)[0] == "mean > median"
    assert oc.interpret_ratio(0.7)[0] == "mean < median"
    assert oc.interpret_ratio(np.nan)[0] == "undefined"


def test_coordinate_check_finds_the_eight_swaps(data, ref):
    chk = oc.coordinate_consistency(data, ref)
    found = set(chk.loc[chk["check"] == "latitude/longitude swapped", "id"])
    assert found == SWAPPED
    assert (chk.loc[chk["id"].isin(SWAPPED), "km_to_city_center_if_swapped"] < 5).all()


def test_coordinate_check_finds_the_sign_error(data, ref):
    chk = oc.coordinate_consistency(data, ref)
    sign = chk[chk["check"] == "sign error"]
    assert list(sign["name"]) == ["Mescan Brewery"]  # Kilsallagh, Mayo: longitude stored as +9.73
    assert sign["sign_error_in"].iloc[0] == "longitude"
    assert sign["km_to_reference_if_sign_reversed"].iloc[0] < 50


def test_capital_roles(data, ref):
    central, _ = oc.central_cities(data, ref)
    c = central.set_index(oc.STATE_COL)
    assert c.loc["Colorado", "capital_role"] == "capital is the main brewery city"
    assert c.loc["California", "largest_city"] == "San Diego"
