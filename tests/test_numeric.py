"""The NumPy replacements that keep SciPy and scikit-learn out of the dashboard give the same answers.

cluster_analysis.convex_hull_area (for scipy.spatial.ConvexHull), cluster_analysis.ari (for
sklearn.metrics.adjusted_rand_score) and prediction_core.HaversineKDE (for sklearn.neighbors.KernelDensity).
"""
import math
import sqlite3

import numpy as np
import pandas as pd
import pytest
from scipy.spatial import ConvexHull, QhullError
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import KernelDensity

import cluster_analysis as ca
import clustering_store as cs
import prediction_core as pc
import quality_core as qc


def scipy_area(xy):
    try:
        return float(ConvexHull(xy).volume)
    except QhullError:
        return 0.0


def test_convex_hull_area_matches_scipy():
    rng = np.random.default_rng(0)
    for n in [3, 4, 10, 100, 2000]:
        xy = rng.normal(size=(n, 2)) * rng.uniform(0.1, 300)
        assert ca.convex_hull_area(xy) == pytest.approx(scipy_area(xy), rel=1e-12)
    square = np.array([[0, 0], [0, 2], [2, 0], [2, 2], [1, 1], [0, 1]], dtype=float)
    assert ca.convex_hull_area(square) == 4.0
    assert ca.convex_hull_area([[0, 0], [1, 1], [2, 2], [3, 3]]) == 0.0  # on a line
    assert ca.convex_hull_area([[0, 0], [0, 0], [1, 0]]) == 0.0          # two distinct points
    assert ca.convex_hull_area(np.zeros((5, 2))) == 0.0


def test_hull_area_matches_scipy_on_stored_clusters():
    rec = cs.allruns_records("all_records")
    with sqlite3.connect(qc.CORRECTIONS_FILE.parent / "analysis_all_records.db") as con:
        d = qc.prepare(pd.read_sql_query("SELECT * FROM analysis_all_records", con))
    d = d.set_index("id").loc[rec["id"]]
    lat, lon = d["latitude"].to_numpy(), d["longitude"].to_numpy()
    runs = ["all_records|DBSCAN|eps=45|ms=15", "all_records|HDBSCAN|mcs=15|ms=4", "all_records|OPTICS|ms=10|xi=0.05"]
    for run in runs:
        lab = cs.allruns_labels(run)
        for c in np.unique(lab[lab >= 0]):
            m = lab == c
            pts = np.unique(np.column_stack([lat[m], lon[m]]), axis=0)
            if len(pts) < 3:
                continue
            lat0 = np.radians(pts[:, 0].mean())
            xy = np.column_stack([np.radians(pts[:, 1] - pts[:, 1].mean()) * np.cos(lat0) * ca.R,
                                  np.radians(pts[:, 0] - pts[:, 0].mean()) * ca.R])
            assert ca.hull_area_km2(lat[m], lon[m]) == pytest.approx(scipy_area(xy), rel=1e-9, abs=1e-9)


def test_ari_matches_sklearn():
    rng = np.random.default_rng(1)
    for _ in range(200):
        n = int(rng.integers(1, 80))
        a = rng.integers(-1, int(rng.integers(1, 9)), n)
        b = rng.integers(-1, int(rng.integers(1, 9)), n)
        assert ca.ari(a, b) == pytest.approx(adjusted_rand_score(a, b), abs=1e-12)
    for a, b in [([0, 0, 0], [1, 1, 1]), ([0, 1, 2], [3, 4, 5]), ([0, 0, 1], [0, 1, 1]), ([7], [7])]:
        assert ca.ari(a, b) == pytest.approx(adjusted_rand_score(a, b), abs=1e-12)
    a, b = cs.allruns_labels("all_records|DBSCAN|eps=45|ms=15"), cs.allruns_labels("all_records|DBSCAN|eps=50|ms=15")
    assert ca.ari(a, b) == pytest.approx(adjusted_rand_score(a, b), abs=1e-12)


@pytest.fixture(scope="module")
def points():
    rng = np.random.default_rng(2)
    centers = np.radians([[40.0, -105.0], [47.6, -122.3], [50.8, 4.4]])
    train = np.vstack([c + rng.normal(scale=0.01, size=(300, 2)) for c in centers])
    near = np.vstack([c + rng.normal(scale=0.01, size=(50, 2)) for c in centers])
    return train, near


def test_kde_matches_sklearn_near_the_data(points):
    train, near = points
    for bw_km in [10, 30, 80]:
        h = bw_km / pc.R
        ref = KernelDensity(bandwidth=h, metric="haversine", kernel="gaussian").fit(train).score_samples(near)
        assert pc.HaversineKDE(h).fit(train).score_samples(near) == pytest.approx(ref, abs=1e-8)


def test_kde_is_exact_far_from_the_data(points):
    """scikit-learn's ball tree drifts here; the direct sum must equal an independent log-sum-exp."""
    train, _ = points
    h = 5 / pc.R
    q = np.radians([[37.0, -99.0]])
    z = []
    for lat, lon in train:
        s = (math.sin((q[0, 0] - lat) / 2) ** 2
             + math.cos(q[0, 0]) * math.cos(lat) * math.sin((q[0, 1] - lon) / 2) ** 2)
        z.append(-0.5 * (2 * math.asin(math.sqrt(min(1.0, s))) / h) ** 2)
    m = max(z)
    assert math.sqrt(-2 * m) * 5 > 200  # the nearest training point is more than 200 km away
    exact = m + math.log(math.fsum(math.exp(v - m) for v in z)) - (math.log(2 * math.pi) + 2 * math.log(h)
                                                                    + math.log(len(train)))
    assert pc.HaversineKDE(h).fit(train).score_samples(q)[0] == pytest.approx(exact, rel=1e-9)
