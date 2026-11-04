"""Notebook logic (geographic analysis + DBSCAN/HDBSCAN) as plain functions, so the dashboard can recompute
every table directly from the SQLite databases. Pure pandas / numpy (no Streamlit); the clustering functions import
scikit-learn when called, so the dashboard, which uses only the geographic helpers, never loads it."""
import numpy as np
import pandas as pd

R = 6371.0088
STATE = ["country", "state_province"]
CITY = ["country", "state_province", "city"]

# ---- parameter lists from the clustering notebook (de-duplicated) ----
EPS_KM = sorted({25, 30, 35, 40, 45, 50, 55, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 300, 280, 320, 400, 480, 500, 520})
MIN_SAMPLES = sorted({4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 32})
MIN_CLUSTER = sorted({5, 10, 15, 20, 30, 50, 25, 35, 40, 45, 55, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 300, 280, 320, 400, 480, 500, 520})
H_MIN_SAMPLES = [None] + MIN_SAMPLES
CANDIDATES = [("DBSCAN", "DBSCAN_A", dict(eps_km=40, min_samples=15)), ("DBSCAN", "DBSCAN_B", dict(eps_km=45, min_samples=15)),
              ("HDBSCAN", "HDBSCAN_A", dict(min_cluster_size=15, min_samples=4)),
              ("HDBSCAN", "HDBSCAN_B", dict(min_cluster_size=20, min_samples=4)),
              ("HDBSCAN", "HDBSCAN_C", dict(min_cluster_size=15, min_samples=10))]
SELECTED = {"DBSCAN": dict(eps_km=45, min_samples=15), "HDBSCAN": dict(min_cluster_size=15, min_samples=4)}


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# ======================= geographic analysis =======================
def coverage(df):
    lat, lon = df["latitude"].notna(), df["longitude"].notna()
    both = lat & lon
    return {"total_records": len(df), "both_coordinates_present": int(both.sum()),
            "records_missing_latitude": int((~lat).sum()), "records_missing_longitude": int((~lon).sum()),
            "records_missing_either": int((~both).sum()), "record_coverage_pct": both.mean() * 100}


def extent(df):
    return {"latitude_min": df.latitude.min(), "latitude_max": df.latitude.max(),
            "longitude_min": df.longitude.min(), "longitude_max": df.longitude.max(),
            "north_south_span_deg": df.latitude.max() - df.latitude.min(),
            "east_west_span_deg": df.longitude.max() - df.longitude.min(),
            "invalid_latitude": int((~df.latitude.between(-90, 90)).sum()),
            "invalid_longitude": int((~df.longitude.between(-180, 180)).sum()),
            "swap_candidates": int((~df.latitude.between(-90, 90) & df.longitude.between(-90, 90)).sum()),
            "center_latitude": df.latitude.mean(), "center_longitude": df.longitude.mean()}


def centers(df):
    out = []
    for level, cols in {"country": ["country"], "state_province": STATE, "city": CITY}.items():
        g = df.groupby(cols, dropna=False).agg(record_count=("latitude", "size"), center_latitude=("latitude", "mean"),
                                               center_longitude=("longitude", "mean")).reset_index()
        out.append(g.assign(level=level))
    return pd.concat(out, ignore_index=True)


def distances(df):
    d = df.copy()
    d["distance_from_overall_center_km"] = haversine_km(d.latitude, d.longitude, d.latitude.mean(), d.longitude.mean())
    for name, cols in (("state", STATE), ("city", CITY)):
        c = d.groupby(cols, dropna=False)[["latitude", "longitude"]].transform("mean")
        d[f"distance_from_{name}_center_km"] = haversine_km(d.latitude, d.longitude, c.latitude, c.longitude)
    return d


def dispersion(d):
    g = d.groupby(STATE, dropna=False)["distance_from_state_center_km"]
    s = g.agg(record_count="count", mean_km="mean", median_km="median", max_km="max")
    s["p25_km"], s["p75_km"], s["p90_km"] = g.quantile(.25), g.quantile(.75), g.quantile(.90)
    s = s.reset_index()
    s["iqr_km"] = s.p75_km - s.p25_km
    s["mean_to_median_ratio"] = s.mean_km / s.median_km.replace(0, np.nan)
    return s


def flag_outliers(d):
    g = d.groupby(STATE, dropna=False)["distance_from_state_center_km"]
    q1, q3 = g.transform(lambda x: x.quantile(.25)), g.transform(lambda x: x.quantile(.75))
    d = d.copy()
    d["outlier_threshold_km"] = q3 + 1.5 * (q3 - q1)
    d["geographic_outlier"] = d["distance_from_state_center_km"] > d["outlier_threshold_km"]
    return d


def outlier_summary(d):
    s = d.groupby(STATE, dropna=False).agg(record_count=("geographic_outlier", "size"),
                                            geographic_outlier_count=("geographic_outlier", "sum"),
                                            maximum_distance_km=("distance_from_state_center_km", "max")).reset_index()
    s["geographic_outlier_pct"] = s.geographic_outlier_count / s.record_count * 100
    return s


# ======================= clustering =======================
def xy(df):
    return np.radians(df[["latitude", "longitude"]].to_numpy())


def k_distance(df, samples=MIN_SAMPLES):
    from sklearn.neighbors import NearestNeighbors
    X, parts = xy(df), []
    for ms in samples:
        dist, _ = NearestNeighbors(n_neighbors=ms, metric="haversine").fit(X).kneighbors(X)
        v = np.sort(dist[:, -1] * R)
        parts.append(pd.DataFrame({"min_samples": ms, "rank_pct": np.linspace(0, 100, len(v)), "k_distance_km": v}).iloc[::5])
    return pd.concat(parts, ignore_index=True)


def metrics(labels):
    n, noise = len(labels), int((labels == -1).sum())
    s = pd.Series(labels[labels != -1]).value_counts()
    big = int(s.max()) if len(s) else 0
    return dict(cluster_count=len(s), clustered_records=n - noise, clustered_share_pct=(n - noise) / n * 100,
                noise_records=noise, noise_share_pct=noise / n * 100, largest_cluster_size=big,
                largest_cluster_share_pct=big / n * 100, smallest_cluster_size=int(s.min()) if len(s) else 0,
                median_cluster_size=float(s.median()) if len(s) else np.nan)


def diagnose(r):
    w = []
    if r["cluster_count"] < 2: w.append("fewer_than_two_clusters")
    if r["noise_share_pct"] >= 80: w.append("very_high_noise")
    if r["largest_cluster_share_pct"] >= 80: w.append("dominant_cluster")
    if r["cluster_count"] >= 100: w.append("many_clusters")
    if pd.isna(r["median_cluster_size"]): w.append("no_detected_clusters")
    return "; ".join(w) or "review_candidate"


def fit(df, algorithm, **p):
    from sklearn.cluster import DBSCAN, HDBSCAN
    X = xy(df)
    if algorithm == "DBSCAN":
        return DBSCAN(eps=p["eps_km"] / R, min_samples=p["min_samples"], metric="haversine").fit_predict(X)
    return HDBSCAN(min_cluster_size=p["min_cluster_size"], min_samples=p["min_samples"], metric="haversine", copy=True).fit_predict(X)


def dbscan_grid(df, eps=EPS_KM, samples=MIN_SAMPLES):
    rows = [dict(algorithm="DBSCAN", eps_km=e, min_cluster_size=np.nan, min_samples=m,
                 **metrics(fit(df, "DBSCAN", eps_km=e, min_samples=m))) for e in eps for m in samples if len(df) >= m]
    out = pd.DataFrame(rows)
    out["diagnostic"] = out.apply(diagnose, axis=1)
    return out


def hdbscan_grid(df, sizes=MIN_CLUSTER, samples=H_MIN_SAMPLES):
    rows = [dict(algorithm="HDBSCAN", eps_km=np.nan, min_cluster_size=s, min_samples=m if m is not None else np.nan,
                 **metrics(fit(df, "HDBSCAN", min_cluster_size=s, min_samples=m))) for s in sizes for m in samples]
    out = pd.DataFrame(rows)
    out["diagnostic"] = out.apply(diagnose, axis=1)
    return out


def candidate_table(df):
    rows = [dict(algorithm=a, candidate=c, eps_km=p.get("eps_km", np.nan), min_cluster_size=p.get("min_cluster_size", np.nan),
                 min_samples=p["min_samples"], total_records=len(df), **metrics(fit(df, a, **p))) for a, c, p in CANDIDATES]
    return pd.DataFrame(rows)


def selected_models(df):
    """{algorithm: dataframe with a `cluster` column} for the selected parameters."""
    return {a: df.assign(cluster=fit(df, a, **p)) for a, p in SELECTED.items()}


def cluster_summaries(lab):
    n, cl, noise = len(lab), lab[lab.cluster != -1], lab[lab.cluster == -1]
    sizes = cl.cluster.value_counts().rename_axis("cluster").reset_index(name="cluster_size")
    sizes["cluster_share_pct"] = sizes.cluster_size / n * 100
    sizes["share_of_clustered_pct"] = sizes.cluster_size / max(len(cl), 1) * 100
    geo = cl.groupby("cluster").agg(cluster_size=("cluster", "size"), center_latitude=("latitude", "mean"),
                                    center_longitude=("longitude", "mean"), min_latitude=("latitude", "min"),
                                    max_latitude=("latitude", "max"), min_longitude=("longitude", "min"),
                                    max_longitude=("longitude", "max")).reset_index()
    c = geo.set_index("cluster")
    dist = haversine_km(cl.latitude, cl.longitude, cl.cluster.map(c.center_latitude), cl.cluster.map(c.center_longitude))
    geo["max_distance_from_center_km"] = dist.groupby(cl.cluster).max().reindex(geo.cluster).values
    by_region = (noise.groupby(STATE, dropna=False).size().reset_index(name="noise_records")
                 .sort_values("noise_records", ascending=False))
    cols = [c for c in ["name", "city", "state_province", "country", "latitude", "longitude"] if c in noise]
    return sizes, geo.sort_values("cluster_size", ascending=False), by_region, noise[cols]


# ======================= interpretation helpers =======================
def _mode(s):
    s = s.dropna()
    return s.value_counts().index[0] if len(s) else None


def describe_clusters(lab):
    """One row per cluster with a human-readable name (dominant city/state), size, spread and place counts."""
    cl = lab[lab.cluster != -1]
    if cl.empty:
        return pd.DataFrame()
    g = cl.groupby("cluster")
    d = g.agg(size=("cluster", "size"), cities=("city", "nunique"), states=("state_province", "nunique"),
              countries=("country", "nunique"), center_latitude=("latitude", "mean"),
              center_longitude=("longitude", "mean")).reset_index()
    for col, new in [("city", "top_city"), ("state_province", "top_state"), ("country", "top_country")]:
        d[new] = d["cluster"].map(g[col].agg(_mode))
    c = d.set_index("cluster")
    dist = haversine_km(cl.latitude, cl.longitude, cl.cluster.map(c.center_latitude), cl.cluster.map(c.center_longitude))
    d["radius_km"] = dist.groupby(cl.cluster).max().reindex(d.cluster).values
    d["share_of_all_%"] = d["size"] / len(lab) * 100
    extra = (d.cities - 1).clip(lower=0)
    d["label"] = ("#" + d.cluster.astype(str) + " · " + d.top_city.astype(str) + ", " + d.top_state.astype(str)
                  + np.where(extra > 0, " (+" + extra.astype(str) + " cities)", ""))
    return d.sort_values("size", ascending=False).reset_index(drop=True)


def agreement(a, h):
    """Compare two labelings of the same records (rows must align)."""
    from sklearn.metrics import adjusted_rand_score
    ca, ch = a.cluster.to_numpy() != -1, h.cluster.to_numpy() != -1
    cat = np.select([ca & ch, ca & ~ch, ~ca & ch], ["Clustered by both", "Only DBSCAN", "Only HDBSCAN"], "Noise in both")
    return cat, float(adjusted_rand_score(a.cluster, h.cluster))


def noise_rate_by_country(lab):
    g = lab.assign(noise=lab.cluster == -1).groupby("country").agg(total=("noise", "size"), noise=("noise", "sum")).reset_index()
    g["noise_rate_%"] = g.noise / g.total * 100
    return g.sort_values("total", ascending=False)
