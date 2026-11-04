"""data/dashboard_stats.db is current: built from today's input files and equal to a fresh computation.

If these fail after a data change, rerun brewery_dashboard_statistics.ipynb.
"""
import sqlite3

import pandas as pd
import pytest

import distribution_core as dc
import origins_core as oc
import quality_core as qc
import stats_store as ss

eq = pd.testing.assert_frame_equal


@pytest.fixture(scope="module")
def T():
    with sqlite3.connect(f"file:{ss.DB}?mode=ro", uri=True) as con:
        return {name: pd.read_sql_query(f"SELECT * FROM {name}", con)
                for name in ["missingness", "spearman", "type_association", "type_profile_states", "type_profile_mix",
                             "type_profile_silhouette", "type_colocation", "inputs"]}


@pytest.fixture(scope="module")
def prepared():
    return {name: ss.load_prepared(name) for name in qc.DATASET_FILES}


def test_built_from_the_current_input_files(T):
    eq(T["inputs"], ss.file_hashes())


@pytest.mark.parametrize("dataset", list(qc.DATASET_FILES))
def test_missingness_matches_fresh(T, dataset):
    d = ss.load_stored(dataset)
    fresh = pd.DataFrame([t for t in (qc.missingness_association(d, c, by) for c in ss.missing_targets(d)
                                      for by in qc.CATEGORY_COLUMNS) if t])
    eq(ss.missingness(T["missingness"], dataset), fresh)


def test_type_association_matches_fresh(T, prepared):
    for dataset in prepared:
        for mn in [5, 30, 100]:
            fresh = pd.DataFrame([dc.type_association(prepared[dataset], lv, min_records=mn) for lv in ss.LEVELS])
            eq(ss.type_association(T["type_association"], dataset, mn), fresh)
    assert set(T["type_association"]["min_records"]) == set(ss.TYPE_MIN_RECORDS)


@pytest.mark.parametrize("dataset,country,mn", [("raw", "United States", 30), ("all_records", "United States", 5),
                                                ("unique_records", "United States", 100), ("raw", "Germany", 10),
                                                ("all_records", "Belgium", 5), ("raw", "Australia", 60)])
def test_type_profiles_match_fresh(T, prepared, dataset, country, mn):
    stored = ss.type_profiles(T["type_profile_states"], T["type_profile_mix"], T["type_profile_silhouette"],
                              dataset, country, mn)
    try:
        fresh = dc.type_profiles(prepared[dataset], country=country, min_records=mn)
    except Exception:
        fresh = None
    assert (stored is None) == (fresh is None)
    if fresh is not None:
        for a, b in zip(stored, fresh):
            eq(a, b, check_dtype=False, check_index_type=False)  # SciPy labels groups as int32


def test_us_profiles_exist_for_every_slider_value(T):
    us = T["type_profile_states"].query("country == 'United States'")
    for dataset in qc.DATASET_FILES:
        assert set(us[us["dataset"] == dataset]["min_records"]) == set(ss.TYPE_MIN_RECORDS)


@pytest.mark.parametrize("exclude", [False, True])
def test_colocation_matches_fresh(T, prepared, exclude):
    for dataset in ["raw", "all_records"]:
        for country in ["United States", "Germany"]:
            d = prepared[dataset]
            fresh = dc.type_colocation(d[d["country"] == country], permutations=ss.COLOCATION_PERMUTATIONS,
                                       exclude_same_point=exclude)
            eq(ss.type_colocation(T["type_colocation"], dataset, country, exclude), fresh)


@pytest.mark.parametrize("dataset", ss.ANALYSIS_DATASETS)
def test_spearman_matches_fresh(T, dataset):
    ref = oc.load_reference()
    d = oc.flag_outliers(oc.add_origins(ss.load_prepared(dataset), ref), "state", "mean")
    stored = ss.spearman(T["spearman"], dataset)
    eq(oc.central_cities(d, ref, spearman=stored)[0], oc.central_cities(d, ref)[0])
    one = d[(d["country"] == "United States") & (d[oc.STATE_COL] == "Colorado")]
    eq(oc.central_cities(one, ref, spearman=stored)[0], oc.central_cities(one, ref)[0])
