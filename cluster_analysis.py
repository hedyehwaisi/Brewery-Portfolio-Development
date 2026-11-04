"""Clustering of brewery coordinates over full parameter grids, and the geographic reading of every cluster.

Used by brewery_clustering_all_runs.ipynb (which fits every run and stores the results) and by the dashboard (which
only reads stored labels and computes cluster tables for the run being viewed). No Streamlit. The fitting functions
import scikit-learn and hdbscan when called; everything the dashboard uses (cluster tables, hulls, ARI, stability)
is plain numpy / pandas, so the running app never loads SciPy or scikit-learn.

Conventions
- Coordinates go in as radians [latitude, longitude] and distances use the Haversine metric; every radius is given
  in kilometres and converted with the mean Earth radius R.
- `min_samples` always follows scikit-learn: the point itself counts. The `hdbscan` library does not count it, so
  it receives min_samples - 1 (checked in the notebook: same labels up to ties).
- One `eps` for the whole map. A per-state eps would redefine "dense" from state to state, the way calling people
  "tall" above 170 cm in one city and above 190 cm in another makes "50% tall" mean two different things.
"""
import zlib

import numpy as np
import pandas as pd

R = 6371.0088
EPS_KM = [25, 30, 35, 40, 45, 50, 55, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 280, 300, 320, 400, 480, 500, 520]
MIN_SAMPLES = [4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 32]
MIN_CLUSTER_SIZE = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 280,
                    300, 320, 400, 480, 500, 520]
OPTICS_XI = [0.01, 0.05, 0.1]
OPTICS_MAX_EPS_KM = 520
CITY = ["country", "state_province_harmonized", "city"]
STATE = ["country", "state_province_harmonized"]
COMPASS = ["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"]


# ------------------------------------------------------------------ storage
def encode(arr, dtype):
    return zlib.compress(np.ascontiguousarray(arr, dtype=dtype).tobytes(), 9)


def decode(blob, dtype):
    return np.frombuffer(zlib.decompress(blob), dtype=dtype)


def run_id(dataset, algorithm, eps_km=None, min_cluster_size=None, min_samples=None, xi=None):
    parts = [dataset, algorithm]
    if eps_km is not None and not pd.isna(eps_km):
        parts.append(f"eps={eps_km:g}")
    if min_cluster_size is not None and not pd.isna(min_cluster_size):
        parts.append(f"mcs={int(min_cluster_size)}")
    parts.append("ms=default" if min_samples is None or pd.isna(min_samples) else f"ms={int(min_samples)}")
    if xi is not None and not pd.isna(xi):
        parts.append(f"xi={xi:g}")
    return "|".join(parts)


# ------------------------------------------------------------------ fitting
def radians(df):
    return np.radians(df[["latitude", "longitude"]].to_numpy(dtype=float))


def fit_dbscan(X, eps_km, min_samples):
    from sklearn.cluster import DBSCAN
    m = DBSCAN(eps=eps_km / R, min_samples=min_samples, metric="haversine", algorithm="ball_tree").fit(X)
    core = np.zeros(len(X), bool)
    core[m.core_sample_indices_] = True
    return m.labels_, core


def hdbscan_tree(X, min_samples, n_jobs=8):
    """Single-linkage tree of mutual reachability distances. It depends only on min_samples (scikit-learn
    convention), so one tree serves every min_cluster_size."""
    import hdbscan
    # approx_min_span_tree=False: the library's default approximate tree breaks the exact link to DBSCAN (a flat cut
    # of the exact tree reproduces DBSCAN on its core points; the approximate one did not in 156 of 528 settings)
    return hdbscan.HDBSCAN(min_cluster_size=2, min_samples=max(int(min_samples) - 1, 1), metric="haversine",
                           approx_min_span_tree=False, core_dist_n_jobs=n_jobs).fit(X)._single_linkage_tree


def hdbscan_from_tree(X, tree, min_cluster_size):
    """Labels, membership probabilities, cluster persistence (stability), condensed tree and GLOSH outlier scores
    for one min_cluster_size, extracted from a prebuilt tree (identical to a full fit)."""
    from hdbscan.hdbscan_ import _tree_to_labels
    from hdbscan._hdbscan_tree import outlier_scores
    labels, prob, stab, condensed, _ = _tree_to_labels(X, tree, min_cluster_size=int(min_cluster_size))
    glosh = outlier_scores(condensed) if len(condensed) else np.zeros(len(X))
    # records stacked on identical coordinates leave at an infinite lambda, which makes GLOSH 0/0; they sit at the
    # densest possible point of their cluster, so their outlier score is 0
    glosh = np.nan_to_num(np.asarray(glosh, float), nan=0.0)
    return labels, prob, np.asarray(stab, float), condensed, glosh


def condensed_clusters(condensed, labels):
    """Cluster-level skeleton of a condensed tree: one row per cluster node with its parent, birth and death
    (as distances in km), size at birth, stability and whether it was selected. Enough to draw the tree."""
    ct = pd.DataFrame(condensed)
    if ct.empty:
        return pd.DataFrame()
    nodes = ct[ct["child_size"] > 1].rename(columns={"child": "node", "lambda_val": "lambda_birth",
                                                     "child_size": "size_at_birth"})
    root = int(ct["parent"].min())
    death = ct.groupby("parent")["lambda_val"].max()
    rows = pd.concat([pd.DataFrame({"node": [root], "parent": [-1], "lambda_birth": [0.0],
                                    "size_at_birth": [int(len(labels))]}),
                      nodes[["node", "parent", "lambda_birth", "size_at_birth"]]], ignore_index=True)
    rows["lambda_death"] = rows["node"].map(death).fillna(rows["lambda_birth"])
    # stability = sum over points/children leaving the node of (lambda leave - lambda birth) * size
    birth = rows.set_index("node")["lambda_birth"]
    contrib = (ct["lambda_val"] - ct["parent"].map(birth)) * ct["child_size"]
    rows["stability"] = rows["node"].map(contrib.groupby(ct["parent"]).sum()).fillna(0.0)
    with np.errstate(divide="ignore"):
        rows["birth_km"] = np.where(rows["lambda_birth"] > 0, R / rows["lambda_birth"], np.inf)
        rows["death_km"] = np.where(rows["lambda_death"] > 0, R / rows["lambda_death"], np.inf)
    # which node became which final cluster: hdbscan numbers the selected nodes in increasing node-id order, and a
    # cluster's node is the deepest common ancestor of the nodes its points fall out of
    parent = dict(zip(rows["node"], rows["parent"]))
    leaf_parent = ct[ct["child_size"] == 1].set_index("child")["parent"]
    rows["selected_label"] = -1
    for lab in np.unique(labels[labels >= 0]):
        nodes = leaf_parent.reindex(np.flatnonzero(labels == lab)).dropna().astype(int).unique()
        common = None
        for n in nodes:
            chain, x = [], int(n)
            while x != -1:
                chain.append(x)
                x = int(parent.get(x, -1))
            common = chain if common is None else [c for c in chain if c in set(common)]
        if common:
            rows.loc[rows["node"] == common[0], "selected_label"] = int(lab)
    return rows


def fit_optics(X, min_samples, max_eps_km=OPTICS_MAX_EPS_KM):
    # single process: n_jobs=-1 was 30x slower here (process start-up dominates for 9,488 points)
    from sklearn.cluster import OPTICS
    return OPTICS(min_samples=int(min_samples), metric="haversine", max_eps=max_eps_km / R).fit(X)


def optics_labels(model, min_samples, xi):
    from sklearn.cluster import cluster_optics_xi
    labels, _ = cluster_optics_xi(reachability=model.reachability_, predecessor=model.predecessor_,
                                  ordering=model.ordering_, min_samples=int(min_samples), xi=xi)
    return labels


def dbscan_star_cut(tree, eps_km, min_cluster_size=2):
    """Flat cut of the HDBSCAN single-linkage tree at eps: DBSCAN* (DBSCAN without border points)."""
    from hdbscan.plots import SingleLinkageTree
    return SingleLinkageTree(tree).get_clusters(eps_km / R, min_cluster_size=min_cluster_size)


# ------------------------------------------------------------------ metrics
def basic_metrics(labels):
    labels = np.asarray(labels)
    n, noise = len(labels), int((labels == -1).sum())
    s = pd.Series(labels[labels != -1]).value_counts()
    big = int(s.max()) if len(s) else 0
    return {"cluster_count": int(len(s)), "clustered_records": n - noise, "clustered_share_pct": (n - noise) / n * 100,
            "noise_records": noise, "noise_share_pct": noise / n * 100, "largest_cluster_size": big,
            "largest_cluster_share_pct": big / n * 100, "smallest_cluster_size": int(s.min()) if len(s) else 0,
            "median_cluster_size": float(s.median()) if len(s) else np.nan}


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def bearing(lat1, lon1, lat2, lon2):
    """Compass direction (8 points) of point 2 seen from point 1."""
    lat1, lon1, lat2, lon2 = map(np.radians, (np.asarray(lat1, float), np.asarray(lon1, float),
                                              np.asarray(lat2, float), np.asarray(lon2, float)))
    x = np.sin(lon2 - lon1) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon2 - lon1)
    deg = (np.degrees(np.arctan2(x, y)) + 360) % 360
    return np.array(COMPASS)[((deg + 22.5) // 45).astype(int) % 8]


def convex_hull_area(xy):
    """Area of the convex hull of 2-D points: Andrew's monotone chain, then the shoelace formula. 0 for fewer than
    3 distinct points or points on a line. Matches SciPy's ConvexHull(xy).volume (tests/test_numeric.py)."""
    pts = np.unique(np.asarray(xy, dtype=float), axis=0).tolist()  # sorted by x, then y
    if len(pts) < 3:
        return 0.0

    def half(points):
        h = []
        for x, y in points:
            while len(h) >= 2 and ((h[-1][0] - h[-2][0]) * (y - h[-2][1])
                                   - (h[-1][1] - h[-2][1]) * (x - h[-2][0])) <= 0:
                h.pop()
            h.append((x, y))
        return h

    hull = np.array(half(pts)[:-1] + half(pts[::-1])[:-1])
    if len(hull) < 3:
        return 0.0
    x, y = hull[:, 0], hull[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


def hull_area_km2(lat, lon):
    """Area of the convex hull of the points (local equirectangular projection around their mean). 0 when the
    points are fewer than 3 distinct positions or lie on a line."""
    pts = np.unique(np.column_stack([lat, lon]), axis=0)
    if len(pts) < 3:
        return 0.0
    lat0 = np.radians(pts[:, 0].mean())
    xy = np.column_stack([np.radians(pts[:, 1] - pts[:, 1].mean()) * np.cos(lat0) * R,
                          np.radians(pts[:, 0] - pts[:, 0].mean()) * R])
    return convex_hull_area(xy)


def hubs(df, min_records=10):
    """Brewery hubs: cities with at least `min_records` breweries, with their brewery mean position."""
    h = df.groupby(CITY).agg(hub_records=("id", "size"), hub_latitude=("latitude", "mean"),
                             hub_longitude=("longitude", "mean")).reset_index()
    h = h[h["hub_records"] >= min_records].reset_index(drop=True)
    h["hub"] = h["city"] + ", " + h["state_province_harmonized"]
    return h


def cluster_table(df, labels, hub_min_records=10, hub_min_inside=3, hub_min_sep_km=100, city_centers=None,
                  state_centers=None, state_capitals=None):
    """One row per cluster: size, center, bounds, spread (median, P90 and maximum distance to the center), convex
    hull area and density, how many countries/states/cities it spans and which dominates, brewery-type mix, the
    nearest city, its position relative to its dominant state's brewery center and capital, and whether it joins
    two or more distinct brewery hubs (cities with >= `hub_min_records` breweries, each contributing >=
    `hub_min_inside` records, more than `hub_min_sep_km` apart)."""
    d = df.assign(cluster=np.asarray(labels))
    c = d[d["cluster"] != -1]
    if c.empty:
        return pd.DataFrame()
    g = c.groupby("cluster")
    t = g.agg(size=("id", "size"), center_latitude=("latitude", "mean"), center_longitude=("longitude", "mean"),
              min_latitude=("latitude", "min"), max_latitude=("latitude", "max"),
              min_longitude=("longitude", "min"), max_longitude=("longitude", "max"),
              countries=("country", "nunique"), states=("state_province_harmonized", "nunique"),
              cities=("city", "nunique"))
    dist = haversine(c["latitude"], c["longitude"], c["cluster"].map(t["center_latitude"]),
                     c["cluster"].map(t["center_longitude"]))
    dg = pd.Series(dist.to_numpy(), index=c.index).groupby(c["cluster"])
    t["median_km"], t["p90_km"], t["max_km"] = dg.median(), dg.quantile(.9), dg.max()
    t["hull_area_km2"] = [hull_area_km2(x["latitude"].to_numpy(), x["longitude"].to_numpy()) for _, x in g]
    t["density_per_1000km2"] = np.where(t["hull_area_km2"] > 0, t["size"] / t["hull_area_km2"] * 1000, np.nan)
    t["share_of_all_%"] = t["size"] / len(d) * 100
    for col, name in [("country", "country"), ("state_province_harmonized", "state"), ("city", "city")]:
        vc = c.groupby(["cluster", col]).size().rename("n").reset_index().sort_values(["cluster", "n"],
                                                                                    ascending=[True, False])
        top = vc.drop_duplicates("cluster").set_index("cluster")
        t[f"dominant_{name}"] = top[col]
        t[f"dominant_{name}_share_%"] = top["n"] / t["size"] * 100
    st_of = (c.groupby(["cluster"] + STATE).size().rename("n").reset_index()
             .sort_values(["cluster", "n"], ascending=[True, False]).drop_duplicates("cluster").set_index("cluster"))
    tv = c.groupby(["cluster", "brewery_type"]).size().rename("n").reset_index().sort_values(
        ["cluster", "n"], ascending=[True, False])
    t["dominant_type"] = tv.drop_duplicates("cluster").set_index("cluster")["brewery_type"]
    t["dominant_type_share_%"] = tv.drop_duplicates("cluster").set_index("cluster")["n"] / t["size"] * 100
    t["type_mix"] = tv.groupby("cluster").apply(
        lambda x: ", ".join(f"{a} {b}" for a, b in zip(x["brewery_type"], x["n"])), include_groups=False)
    # nearest city: the dataset city (brewery mean position) closest to the cluster center
    cc = city_centers if city_centers is not None else (
        df.groupby(CITY)[["latitude", "longitude"]].mean().reset_index())
    lat_c, lon_c = cc["latitude"].to_numpy(), cc["longitude"].to_numpy()
    near = [int(np.argmin(haversine(lat_c, lon_c, la, lo))) for la, lo in zip(t["center_latitude"], t["center_longitude"])]
    t["nearest_city"] = (cc["city"].iloc[near] + ", " + cc["state_province_harmonized"].iloc[near]).to_numpy()
    t["nearest_city_km"] = haversine(lat_c[near], lon_c[near], t["center_latitude"], t["center_longitude"]).to_numpy()
    # position relative to the dominant state's brewery center (origin 1) and capital (origin 2)
    keys = list(zip(st_of["country"], st_of["state_province_harmonized"]))
    sc = state_centers if state_centers is not None else df.groupby(STATE)[["latitude", "longitude"]].mean()
    s_lat = np.array([sc["latitude"].get(k, np.nan) for k in keys])
    s_lon = np.array([sc["longitude"].get(k, np.nan) for k in keys])
    t["km_from_state_center"] = haversine(s_lat, s_lon, t["center_latitude"], t["center_longitude"]).to_numpy()
    t["direction_from_state_center"] = bearing(s_lat, s_lon, t["center_latitude"], t["center_longitude"])
    if state_capitals is not None:
        cap = state_capitals.set_index(STATE)
        c_lat = np.array([cap["capital_latitude"].get(k, np.nan) for k in keys], dtype=float)
        c_lon = np.array([cap["capital_longitude"].get(k, np.nan) for k in keys], dtype=float)
        t["state_capital"] = [cap["capital"].get(k) for k in keys]
        t["km_from_state_capital"] = haversine(c_lat, c_lon, t["center_latitude"], t["center_longitude"]).to_numpy()
        t["direction_from_state_capital"] = np.where(np.isnan(c_lat), None,
                                                     bearing(np.nan_to_num(c_lat), np.nan_to_num(c_lon),
                                                             t["center_latitude"], t["center_longitude"]))
    # distinct hubs merged into one cluster
    h = hubs(df, hub_min_records)
    t["hubs_inside"], t["hub_names"], t["max_hub_separation_km"] = 0, "", 0.0
    if len(h):
        inside = c.merge(h, on=CITY).groupby(["cluster", "hub"]).agg(
            n=("id", "size"), la=("hub_latitude", "first"), lo=("hub_longitude", "first")).reset_index()
        inside = inside[inside["n"] >= hub_min_inside]
        for cl, x in inside.groupby("cluster"):
            t.loc[cl, "hubs_inside"] = len(x)
            t.loc[cl, "hub_names"] = "; ".join(x.sort_values("n", ascending=False)["hub"])
            if len(x) > 1:
                la, lo = x["la"].to_numpy(), x["lo"].to_numpy()
                t.loc[cl, "max_hub_separation_km"] = float(haversine(la[:, None], lo[:, None], la[None, :],
                                                                     lo[None, :]).max())
    t["merges_distinct_hubs"] = (t["hubs_inside"] >= 2) & (t["max_hub_separation_km"] > hub_min_sep_km)
    place = (c.groupby(["cluster"] + CITY).size().rename("n").reset_index()
             .sort_values(["cluster", "n"], ascending=[True, False]).drop_duplicates("cluster").set_index("cluster"))
    t["label"] = ("#" + t.index.astype(str) + " · " + place["city"].astype(str) + ", "
                  + place["state_province_harmonized"].astype(str))
    return t.reset_index().sort_values("size", ascending=False).reset_index(drop=True)


def run_geo_summary(t, n_records):
    """Run-level summary of a cluster table, for comparing runs."""
    if t is None or t.empty:
        return {"median_cluster_median_km": np.nan, "median_cluster_p90_km": np.nan, "max_cluster_max_km": np.nan,
                "largest_cluster_max_km": np.nan, "median_density_per_1000km2": np.nan,
                "clusters_over_1_country": 0, "clusters_over_1_state": 0, "records_in_multi_state_clusters_%": 0.0,
                "clusters_merging_hubs": 0, "records_in_hub_merging_clusters_%": 0.0, "densest_cluster": None,
                "sparsest_cluster": None, "largest_cluster": None}
    dens = t.dropna(subset=["density_per_1000km2"])
    dens = dens[dens["size"] >= 10]
    return {"median_cluster_median_km": float(t["median_km"].median()),
            "median_cluster_p90_km": float(t["p90_km"].median()),
            "max_cluster_max_km": float(t["max_km"].max()),
            "largest_cluster_max_km": float(t.iloc[0]["max_km"]),
            "median_density_per_1000km2": float(dens["density_per_1000km2"].median()) if len(dens) else np.nan,
            "clusters_over_1_country": int((t["countries"] > 1).sum()),
            "clusters_over_1_state": int((t["states"] > 1).sum()),
            "records_in_multi_state_clusters_%": float(t.loc[t["states"] > 1, "size"].sum() / n_records * 100),
            "clusters_merging_hubs": int(t["merges_distinct_hubs"].sum()),
            "records_in_hub_merging_clusters_%": float(t.loc[t["merges_distinct_hubs"], "size"].sum() / n_records * 100),
            "densest_cluster": dens.loc[dens["density_per_1000km2"].idxmax(), "label"] if len(dens) else None,
            "sparsest_cluster": dens.loc[dens["density_per_1000km2"].idxmin(), "label"] if len(dens) else None,
            "largest_cluster": t.iloc[0]["label"]}


# ------------------------------------------------------------------ stability
def neighbours(runs, axes):
    """For every run, the runs one grid step away along each parameter axis (other parameters equal).
    `axes` = list of (column, sorted values). Returns {run_id: [neighbour run_ids]}."""
    idx = {tuple(r[c] for c, _ in axes): r["run_id"] for _, r in runs.iterrows()}
    out = {}
    for _, r in runs.iterrows():
        key = [r[c] for c, _ in axes]
        nb = []
        for i, (c, values) in enumerate(axes):
            if key[i] not in values:
                continue
            j = values.index(key[i])
            for jj in (j - 1, j + 1):
                if 0 <= jj < len(values):
                    k2 = list(key)
                    k2[i] = values[jj]
                    if tuple(k2) in idx:
                        nb.append(idx[tuple(k2)])
        out[r["run_id"]] = nb
    return out


def cluster_recurrence(a, b):
    """For every cluster of labelling `a`, the best Jaccard overlap with any cluster of labelling `b`
    (noise excluded). 1 = the same group of breweries reappears exactly."""
    a, b = np.asarray(a), np.asarray(b)
    ok = a != -1
    if not ok.any():
        return pd.Series(dtype=float)
    pairs = pd.DataFrame({"a": a[ok], "b": b[ok]})
    inter = pairs[pairs["b"] != -1].groupby(["a", "b"]).size().rename("inter").reset_index()
    size_a = pd.Series(a[ok]).value_counts()
    size_b = pd.Series(b[b != -1]).value_counts()
    inter["jaccard"] = inter["inter"] / (inter["a"].map(size_a) + inter["b"].map(size_b) - inter["inter"])
    best = inter.groupby("a")["jaccard"].max()
    return best.reindex(size_a.index).fillna(0.0)


def ari(a, b):
    """Adjusted Rand index of two labellings, from the pair-confusion counts, with scikit-learn's definition and
    special case (identical partitions give 1.0). Matches scikit-learn's adjusted_rand_score (tests/test_numeric.py)."""
    _, ai = np.unique(np.asarray(a), return_inverse=True)
    _, bi = np.unique(np.asarray(b), return_inverse=True)
    n = ai.size
    cont = np.zeros((ai.max() + 1, bi.max() + 1), dtype=np.int64)
    np.add.at(cont, (ai, bi), 1)
    sum_sq = int((cont ** 2).sum())
    tp = sum_sq - n
    fp = int((cont.sum(axis=0) ** 2).sum()) - sum_sq
    fn = int((cont.sum(axis=1) ** 2).sum()) - sum_sq
    tn = n * n - fp - fn - sum_sq
    if fn == 0 and fp == 0:
        return 1.0
    return 2.0 * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))


def stability(labels_by_run, nb, min_cluster_records=1):
    """Mean ARI with neighbouring runs, median cluster recurrence (best Jaccard, averaged over neighbours) and the
    share of clusters that recur with Jaccard >= 0.5 on average."""
    rows = []
    for rid, nbs in nb.items():
        lab = labels_by_run[rid]
        if not nbs:
            rows.append({"run_id": rid, "neighbours": 0, "mean_neighbour_ari": np.nan,
                         "median_cluster_recurrence": np.nan, "stable_clusters_%": np.nan})
            continue
        aris = [ari(lab, labels_by_run[n]) for n in nbs]
        rec = pd.concat([cluster_recurrence(lab, labels_by_run[n]) for n in nbs], axis=1).mean(axis=1)
        rows.append({"run_id": rid, "neighbours": len(nbs), "mean_neighbour_ari": float(np.mean(aris)),
                     "median_cluster_recurrence": float(rec.median()) if len(rec) else np.nan,
                     "stable_clusters_%": float((rec >= 0.5).mean() * 100) if len(rec) else np.nan})
    return pd.DataFrame(rows)
