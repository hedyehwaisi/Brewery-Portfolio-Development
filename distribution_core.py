"""Frequency, hierarchy and brewery-type calculations (country -> state/province -> city).

Pure pandas / numpy (no Streamlit), shared by the dashboard pages and notebooks. The three statistical tests
(type_association, type_profiles, type_colocation) import SciPy / scikit-learn when called; the dashboard reads their
results from data/dashboard_stats.db (see stats_store) instead.
State-level work uses `state_province_harmonized` (see quality_core.harmonize_states), so a region written in
several languages is counted once. Cities are always keyed by country + state + city, because the same city
name exists in several states and countries.
"""
import numpy as np
import pandas as pd

STATE_COL = "state_province_harmonized"
LEVELS = {"country": ["country"], "state": ["country", STATE_COL], "city": ["country", STATE_COL, "city"]}

# Official definitions from https://www.openbrewerydb.org/documentation (by_type), checked 2026-10-30.
TYPE_DEFINITIONS = {
    "micro": "Most craft breweries. For example, Samuel Adams is still considered a micro brewery.",
    "nano": "An extremely small brewery which typically only distributes locally.",
    "regional": "A regional location of an expanded brewery, e.g. Sierra Nevada's Asheville, NC location.",
    "brewpub": "A beer-focused restaurant or restaurant/bar with a brewery on-premise.",
    "large": "A very large brewery, likely not for visitors, e.g. Miller-Coors (deprecated).",
    "planning": "A brewery in planning or not yet opened to the public.",
    "bar": "A bar. No brewery equipment on premise (deprecated).",
    "contract": "A brewery that uses another brewery's equipment.",
    "proprietor": "Similar to contract brewing but refers more to a brewery incubator.",
    "closed": "A location which has been closed.",
}


def place_label(df, level):
    """Readable name of each record's place at `level`."""
    if level == "country":
        return df["country"].astype(str)
    if level == "state":
        return df[STATE_COL].astype(str) + ", " + df["country"].astype(str)
    return df["city"].astype(str) + ", " + df[STATE_COL].astype(str) + ", " + df["country"].astype(str)


def counts(df, level):
    """Records per place at `level`, with share of all records and rank (1 = most records)."""
    c = df.groupby(LEVELS[level]).size().rename("records").reset_index()
    c["place"] = place_label(c, level)
    c["share_%"] = c["records"] / len(df) * 100
    c = c.sort_values(["records", "place"], ascending=[False, True]).reset_index(drop=True)
    c["rank"] = np.arange(1, len(c) + 1)
    return c


def top_within(df, parent, child, n):
    """Top `n` child places inside every parent place (e.g. states per country), with each child's share of
    its parent and its rank inside the parent."""
    c = counts(df, child)
    keys = LEVELS[parent]
    c["parent_records"] = c.groupby(keys)["records"].transform("sum")
    c["share_of_parent_%"] = c["records"] / c["parent_records"] * 100
    c["rank_in_parent"] = c.groupby(keys).cumcount() + 1
    c["places_in_parent"] = c.groupby(keys)["records"].transform("size")
    return c[c["rank_in_parent"] <= n].reset_index(drop=True)


# ------------------------------------------------------------------ brewery types
def type_counts(df):
    t = df["brewery_type"].value_counts().rename_axis("brewery_type").reset_index(name="records")
    t["share_%"] = t["records"] / len(df) * 100
    t["cumulative_%"] = t["share_%"].cumsum()
    t["definition"] = t["brewery_type"].map(TYPE_DEFINITIONS).fillna("Not defined in the API documentation.")
    return t


def type_mix(df, level, places=None):
    """Share (%) of each brewery type inside each place. `places` limits and orders the rows."""
    lab = place_label(df, level)
    ct = pd.crosstab(lab, df["brewery_type"])
    if places is not None:
        ct = ct.reindex(places).dropna(how="all").fillna(0)
    return ct.div(ct.sum(axis=1), axis=0) * 100


def location_quotient(df, level, min_records=30):
    """Location quotient: a type's share in a place divided by its share in the whole dataset.
    1 = the national mix, 2 = twice as common as overall, 0 = absent. Places below `min_records` are dropped."""
    lab = place_label(df, level)
    ct = pd.crosstab(lab, df["brewery_type"])
    ct = ct[ct.sum(axis=1) >= min_records]
    overall = df["brewery_type"].value_counts(normalize=True)
    return ct.div(ct.sum(axis=1), axis=0).div(overall[ct.columns], axis=1)


def type_association(df, level, min_records=30):
    """Chi-square test: is the type mix the same everywhere? Cramér's V gives the strength (0-1)."""
    from scipy.stats import chi2_contingency
    lab = place_label(df, level)
    ct = pd.crosstab(lab, df["brewery_type"])
    ct = ct[ct.sum(axis=1) >= min_records]
    ct = ct.loc[:, ct.sum() > 0]
    if min(ct.shape) < 2:  # fewer than two places (or types) reach min_records: nothing to test
        return {"level": level, "places_tested": int(ct.shape[0]), "records": int(ct.values.sum()), "chi2": np.nan,
                "dof": 0, "p_value": np.nan, "cramers_v": np.nan}
    chi2, p, dof, _ = chi2_contingency(ct)
    v = float(np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1))))
    return {"level": level, "places_tested": int(ct.shape[0]), "records": int(ct.values.sum()), "chi2": float(chi2),
            "dof": int(dof), "p_value": float(p), "cramers_v": v}


def dominant_types(df, level, min_records=1):
    """Most common type in every place, its share, the runner-up and the margin between them."""
    lab = place_label(df, level)
    ct = pd.crosstab(lab, df["brewery_type"])
    ct = ct[ct.sum(axis=1) >= min_records]
    shares = ct.div(ct.sum(axis=1), axis=0) * 100
    order = np.argsort(-shares.values, axis=1)
    cols = shares.columns.to_numpy()
    out = pd.DataFrame({
        "place": shares.index, "records": ct.sum(axis=1).values,
        "dominant_type": cols[order[:, 0]], "dominant_share_%": np.take_along_axis(shares.values, order[:, :1], 1)[:, 0],
        "runner_up": cols[order[:, 1]], "runner_up_share_%": np.take_along_axis(shares.values, order[:, 1:2], 1)[:, 0],
    })
    out["margin_pp"] = out["dominant_share_%"] - out["runner_up_share_%"]
    keys = df.assign(place=lab).drop_duplicates("place").set_index("place")[LEVELS[level]]
    return out.join(keys, on="place").sort_values("records", ascending=False).reset_index(drop=True)


def type_profiles(df, country="United States", min_records=30, k=None, k_range=range(2, 7)):
    """Group the states of one country by their brewery-type mix.
    Each state is a vector of type shares; square roots of the shares (Hellinger geometry) keep small types from
    being swamped by micro/brewpub; Ward hierarchical clustering builds the groups. Without `k`, the number of
    groups with the best silhouette score is used. Returns (assignments, group profiles, silhouette per k)."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from sklearn.metrics import silhouette_score
    d = df[df["country"] == country]
    ct = pd.crosstab(d[STATE_COL], d["brewery_type"])
    ct = ct[ct.sum(axis=1) >= min_records]
    shares = ct.div(ct.sum(axis=1), axis=0)
    X = np.sqrt(shares.values)
    Z = linkage(X, "ward")
    sil = pd.DataFrame([{"k": kk, "silhouette": silhouette_score(X, fcluster(Z, kk, "maxclust"))}
                        for kk in k_range if kk < len(X)])
    k = k or int(sil.loc[sil["silhouette"].idxmax(), "k"])
    labels = fcluster(Z, k, "maxclust")
    assign = pd.DataFrame({STATE_COL: shares.index, "profile": labels, "records": ct.sum(axis=1).values})
    profile = (shares * 100).groupby(labels).mean()
    profile.insert(0, "states", pd.Series(labels).value_counts().sort_index().values)
    profile.index.name = "profile"
    return assign, profile, sil


def type_colocation(df, permutations=200, seed=0, exclude_same_point=False):
    """Do breweries of the same type sit next to each other more often than chance?
    For every record we find its nearest other brewery (Haversine). The observed share of same-type neighbours is
    compared with permutations that shuffle types *within each state*, which keeps every state's type mix and
    every location fixed; only the pairing of type and position is randomised. `exclude_same_point` skips
    neighbours at distance 0 (records stacked on one coordinate)."""
    from sklearn.neighbors import BallTree
    d = df.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
    X = np.radians(d[["latitude", "longitude"]].to_numpy(dtype=float))
    dist, idx = BallTree(X, metric="haversine").query(X, k=min(25, len(d)))
    if exclude_same_point:
        pos = np.argmax(dist > 0, axis=1)
        keep = dist[np.arange(len(d)), pos] > 0
        nn = idx[np.arange(len(d)), pos]
    else:
        keep = np.ones(len(d), bool)
        nn = np.where(idx[:, 0] == np.arange(len(d)), idx[:, 1], idx[:, 0])
    types = d["brewery_type"].to_numpy()
    groups = [np.asarray(v) for v in d.groupby(LEVELS["state"]).indices.values()]
    rng = np.random.default_rng(seed)
    cats = pd.unique(types)
    obs = {t: float((types[nn][keep & (types == t)] == t).mean()) for t in cats}
    obs_all = float((types[nn][keep] == types[keep]).mean())
    sims = {t: [] for t in cats}
    sims_all = []
    for _ in range(permutations):
        p = types.copy()
        for g in groups:
            p[g] = p[rng.permutation(g)]
        sims_all.append((p[nn][keep] == p[keep]).mean())
        for t in cats:
            m = keep & (p == t)
            sims[t].append((p[nn][m] == t).mean() if m.any() else np.nan)
    rows = [{"brewery_type": "all types", "records": int(keep.sum()), "observed_%": obs_all * 100,
             "expected_%": float(np.mean(sims_all)) * 100,
             "p_value": (np.sum(np.array(sims_all) >= obs_all) + 1) / (permutations + 1)}]
    for t in cats:
        s = np.array(sims[t], dtype=float)
        rows.append({"brewery_type": t, "records": int((keep & (types == t)).sum()), "observed_%": obs[t] * 100,
                     "expected_%": float(np.nanmean(s)) * 100,
                     "p_value": (np.sum(s >= obs[t]) + 1) / (permutations + 1)})
    out = pd.DataFrame(rows)
    out["ratio"] = out["observed_%"] / out["expected_%"].replace(0, np.nan)
    return out.sort_values("records", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------------ hierarchy: states and cities
def cities_of_top_states(df, n_states=10, coverage=50.0, n_top_cities=10):
    """For each of the `n_states` largest states: every city with its share of the state and the cumulative
    share. `needed_for_coverage` marks the smallest set of cities that together hold `coverage` % of the
    state's breweries, so a state with many cities shows more of them. `in_top_cities` flags cities that are
    also among the `n_top_cities` largest cities overall. Returns (city detail, one-row-per-state summary)."""
    st = counts(df, "state").head(n_states)
    ct = counts(df, "city")
    top_city_places = set(ct.head(n_top_cities)["place"])
    ct = ct.merge(st[LEVELS["state"] + ["records", "rank"]].rename(columns={"records": "state_records",
                                                                            "rank": "state_rank"}),
                  on=LEVELS["state"])
    ct = ct.sort_values(["state_rank", "records", "city"], ascending=[True, False, True])
    ct["share_of_state_%"] = ct["records"] / ct["state_records"] * 100
    ct["cumulative_share_%"] = ct.groupby("state_rank")["share_of_state_%"].cumsum()
    ct["needed_for_coverage"] = (ct["cumulative_share_%"] - ct["share_of_state_%"]) < coverage
    ct["in_top_cities"] = ct["place"].isin(top_city_places)
    ct["rank_in_state"] = ct.groupby("state_rank").cumcount() + 1
    summ = ct.groupby(["state_rank"] + LEVELS["state"]).agg(
        records=("state_records", "first"), cities=("city", "size"),
        cities_for_coverage=("needed_for_coverage", "sum"),
        top_city=("city", "first"), top_city_share_pct=("share_of_state_%", "first"),
        cities_in_overall_top=("in_top_cities", "sum")).reset_index()
    summ["cities_for_coverage_pct_of_cities"] = summ["cities_for_coverage"] / summ["cities"] * 100
    return ct.reset_index(drop=True), summ


def city_state_kinds(df, n_states=10, n_cities=20, n_countries=5, reference=None):
    """The `n_cities` largest cities with their state's size and rank, the city's share of its state, and
    Type A (state among the `n_states` largest) or Type B (state outside them).
    `reference` (another dataset, e.g. raw) adds the city's share of its state there, to spot shares inflated
    by missing records."""
    st = counts(df, "state")
    co = counts(df, "country")
    ct = counts(df, "city").head(n_cities)
    ct = ct.merge(st[LEVELS["state"] + ["records", "rank"]].rename(columns={"records": "state_records",
                                                                            "rank": "state_rank"}),
                  on=LEVELS["state"])
    ct = ct.merge(co[["country", "rank"]].rename(columns={"rank": "country_rank"}), on="country")
    ct["share_of_state_%"] = ct["records"] / ct["state_records"] * 100
    ct["city_type"] = np.where(ct["state_rank"] <= n_states, "A: top state", "B: outside top states")
    ct["state_in_top_countries"] = ct["country_rank"] <= n_countries
    if reference is not None:
        rc = counts(reference, "city").set_index("place")["records"]
        rs = counts(reference, "state").set_index(LEVELS["state"])["records"]
        ct["share_of_state_in_reference_%"] = [
            rc.get(p, np.nan) / rs.get((c, s), np.nan) * 100
            for p, c, s in zip(ct["place"], ct["country"], ct[STATE_COL])]
    return ct.sort_values("rank").reset_index(drop=True)


def concentration_stats(values):
    """How unequal is a set of group sizes (e.g. breweries per city)? Gini (0 = equal, 1 = one group holds
    everything), HHI (sum of squared shares) and its effective number of equal-sized groups, the share of the
    largest groups, the share of single-record groups, and the Zipf exponent: the slope of log(size) against
    log(rank) over groups with at least 3 records (about 1 = classic rank-size law)."""
    x = np.sort(np.asarray(values, dtype=float))
    n, total = len(x), x.sum()
    gini = float(2 * np.sum(np.arange(1, n + 1) * x) / (n * total) - (n + 1) / n)
    shares = x / total
    hhi = float(np.sum(shares ** 2))
    desc = x[::-1]
    big = desc[desc >= 3]
    slope = float(-np.polyfit(np.log(np.arange(1, len(big) + 1)), np.log(big), 1)[0]) if len(big) >= 5 else np.nan
    return {"groups": n, "records": int(total), "mean": float(x.mean()), "median": float(np.median(x)),
            "p90": float(np.percentile(x, 90)), "max": int(x.max()),
            "single_record_groups_%": float((x == 1).mean() * 100),
            "top_1_share_%": float(desc[:1].sum() / total * 100),
            "top_10_share_%": float(desc[:10].sum() / total * 100),
            "gini": gini, "hhi": hhi, "effective_groups": 1 / hhi, "zipf_exponent": slope}


def lorenz(values):
    """Lorenz curve: cumulative share of groups (smallest first) vs cumulative share of records."""
    x = np.sort(np.asarray(values, dtype=float))
    return pd.DataFrame({"share_of_groups_%": np.r_[0, np.arange(1, len(x) + 1) / len(x)] * 100,
                         "share_of_records_%": np.r_[0, np.cumsum(x) / x.sum()] * 100})


def rank_shift(df, reference, level, n=15):
    """Ranks of the `n` largest places in `reference` (e.g. raw) next to their ranks in `df`."""
    r = counts(reference, level)[["place", "records", "rank"]].head(n)
    d = counts(df, level)[["place", "records", "rank"]]
    out = r.merge(d, on="place", how="left", suffixes=("_reference", "_dataset"))
    out["records_kept_%"] = out["records_dataset"] / out["records_reference"] * 100
    out["rank_change"] = out["rank_reference"] - out["rank_dataset"]
    return out
