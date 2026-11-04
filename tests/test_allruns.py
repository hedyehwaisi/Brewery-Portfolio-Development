"""Integrity tests for the stored results of brewery_clustering_all_runs.ipynb (allruns_* tables).

Run from the project root:  python -m pytest tests -q
They fail if the notebook output is incomplete, if stored labels and stored metrics disagree, or if the stored
record order no longer matches the analysis tables.
"""
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cluster_analysis as ca  # noqa: E402
import clustering_store as cs  # noqa: E402

pytestmark = pytest.mark.skipif(bool(cs.allruns_missing()), reason="allruns tables not generated yet")
DATASETS = list(cs.DATASETS)


def test_every_table_verified():
    assert cs.allruns_verification()["verified"].astype(bool).all()


@pytest.mark.parametrize("dataset", DATASETS)
def test_full_grids_are_stored(dataset):
    runs = cs.allruns_runs(dataset)
    counts = runs["algorithm"].value_counts().to_dict()
    assert counts == {"DBSCAN": len(ca.EPS_KM) * len(ca.MIN_SAMPLES),
                      "HDBSCAN": len(ca.MIN_CLUSTER_SIZE) * (len(ca.MIN_SAMPLES) + 1),
                      "OPTICS": len(ca.MIN_SAMPLES) * len(ca.OPTICS_XI)}
    assert runs["run_id"].is_unique


@pytest.mark.parametrize("dataset", DATASETS)
def test_record_order_matches_analysis_table(dataset):
    rec = cs.allruns_records(dataset)
    with sqlite3.connect(ROOT / "data" / f"analysis_{dataset}.db") as con:
        ids = pd.read_sql_query(f'SELECT id FROM "analysis_{dataset}"', con)["id"]
    assert set(rec["id"]) == set(ids) and rec["position"].tolist() == list(range(len(rec)))
    assert (rec["coordinate_status"] != "as recorded").sum() == 28


@pytest.mark.parametrize("dataset", DATASETS)
def test_labels_reproduce_stored_metrics(dataset):
    runs = cs.allruns_runs(dataset)
    n = len(cs.allruns_records(dataset))
    for _, r in runs.iterrows():
        lab = cs.allruns_labels(r["run_id"])
        assert len(lab) == n
        m = ca.basic_metrics(lab)
        for k in ["cluster_count", "noise_records", "largest_cluster_size", "smallest_cluster_size"]:
            assert m[k] == int(r[k]), (r["run_id"], k)


def test_dbscan_star_matches_dbscan_on_core_points():
    for dataset in DATASETS:
        star = cs.allruns_dbscan_star(dataset)
        assert len(star) == len(ca.EPS_KM) * len(ca.MIN_SAMPLES)
        assert star["ari_core_points"].dropna().min() > 0.999


def test_hdbscan_extras_are_consistent():
    runs = cs.allruns_runs("all_records")
    for rid in runs.loc[runs["algorithm"] == "HDBSCAN", "run_id"].sample(25, random_state=0):
        lab = cs.allruns_labels(rid)
        ex = cs.allruns_hdbscan_extras(rid)
        k = lab.max() + 1
        assert len(ex["persistence"]) == k and len(ex["glosh"]) == len(lab)
        assert not np.isnan(ex["glosh"]).any()
        assert (ex["condensed"]["selected_label"] >= 0).sum() == k


def test_optics_reachability_covers_every_record():
    for ms in ca.MIN_SAMPLES:
        o = cs.allruns_optics("all_records", ms)
        assert sorted(o["position"]) == list(range(len(o)))
        assert (o["core_distance_km"] >= 0).all()
