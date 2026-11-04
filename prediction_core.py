"""Where would a new brewery open, and what type would it be?

Three complementary answers, all estimated from the Open Brewery DB snapshot (no opening dates exist, so the models
assume new breweries follow the geography of today's operating ones, and test that assumption on the breweries that
are still in planning):

1. Place and type probabilities (PlaceTypeModel)
   P(country) -> P(state | country) -> P(city | state), each with additive smoothing `alpha` and a Good-Turing share
   for "a place with no brewery yet" (based on the number of places in the parent that hold exactly one brewery:
   Good and Turing's estimate of the probability that the next observation is new). The type of a brewery is predicted with
   hierarchical shrinkage: a city's type mix is pulled toward its state's, the state's toward its country's, and the
   country's toward the global mix, each with `beta` pseudo-records. Knowing the type, Bayes' rule gives
   P(place | type) proportional to P(place) x P(type | place).
2. Spatial density (kde_*): a Gaussian kernel density of brewery positions on the sphere (Haversine), its bandwidth
   chosen by held-out log-likelihood.
3. The pipeline (pipeline_by_state, planning_test): breweries recorded as `planning` have not opened yet; they show
   where openings are being prepared and serve as an out-of-sample test of model 1.

`planning` and `closed` records are excluded from training: a new brewery is neither. Pure numpy / pandas.
"""
import numpy as np
import pandas as pd

R = 6371.0088
STATE = "state_province_harmonized"
LEVELS = {"country": ["country"], "state": ["country", STATE], "city": ["country", STATE, "city"]}
NEW_PLACE = "(a place with no brewery yet)"
EXCLUDED_TYPES = ("planning", "closed")


def operating(df):
    """Records that describe an existing, working brewery (not planning, not closed)."""
    return df[~df["brewery_type"].isin(EXCLUDED_TYPES)]


def good_turing_new(child_counts):
    """Good-Turing probability that the next record falls in a child with no record yet, N1 / N, in its add-one
    form (N1 + 1) / (N + 1) so that it is never exactly zero (a parent whose places all hold two or more records
    can still gain a new one)."""
    n = float(np.sum(child_counts))
    return float((np.sum(np.asarray(child_counts) == 1) + 1) / (n + 1))


class PlaceTypeModel:
    def __init__(self, alpha=0.5, beta=10.0):
        self.alpha, self.beta = float(alpha), float(beta)

    def fit(self, df):
        self.types = sorted(df["brewery_type"].unique())
        self.n = {lv: df.groupby(keys).size() for lv, keys in LEVELS.items()}
        self.t = {lv: pd.crosstab([df[k] for k in keys], df["brewery_type"]).reindex(columns=self.types, fill_value=0)
                  for lv, keys in LEVELS.items()}
        g = df["brewery_type"].value_counts().reindex(self.types, fill_value=0)
        self.t_global = (g + 1) / (g.sum() + len(self.types))
        return self

    # ---------------------------------------------------------------- places
    def _children(self, level, parent):
        s = self.n[level]
        if level == "country":
            return s
        sub = s.xs(tuple(parent), level=tuple(range(len(parent)))) if len(parent) else s
        return sub

    def place_distribution(self, level, parent=()):
        """P(child | parent) over the places that have breweries, plus NEW_PLACE (Good-Turing)."""
        counts = self._children(level, parent)
        if isinstance(counts.index, pd.MultiIndex):
            counts.index = counts.index.map(lambda k: k[-1] if isinstance(k, tuple) else k)
        p_new = min(good_turing_new(counts.to_numpy()), 0.95)
        share = (counts + self.alpha) / (counts.sum() + self.alpha * len(counts))
        out = pd.concat([share * (1 - p_new), pd.Series({NEW_PLACE: p_new})])
        return out.sort_values(ascending=False)

    # ---------------------------------------------------------------- types
    def type_distribution(self, country=None, state=None, city=None):
        """P(type | place) with hierarchical shrinkage toward the parent place's mix."""
        p = self.t_global.copy()
        for level, key in [("country", (country,)), ("state", (country, state)), ("city", (country, state, city))]:
            if key[-1] is None:
                break
            k = key[0] if level == "country" else key
            row = self.t[level].loc[k] if k in self.t[level].index else pd.Series(0, index=self.types)
            p = (row + self.beta * p) / (row.sum() + self.beta)
        return p.sort_values(ascending=False)

    def place_given_type(self, level, brewery_type, parent=()):
        """P(child | parent, type) proportional to P(child | parent) x P(type | child). NEW_PLACE uses the parent's
        type mix."""
        prior = self.place_distribution(level, parent)
        names = list(parent)
        lik = {}
        for child in prior.index:
            if child == NEW_PLACE:
                lik[child] = self.type_distribution(*(names + [None] * (3 - len(names))))[brewery_type] \
                    if names else self.t_global[brewery_type]
            else:
                key = names + [child]
                lik[child] = self.type_distribution(*(key + [None] * (3 - len(key))))[brewery_type]
        post = prior * pd.Series(lik)
        return (post / post.sum()).sort_values(ascending=False)


# ---------------------------------------------------------------- evaluation
def folds(n, k=5, seed=0):
    idx = np.random.default_rng(seed).permutation(n)
    return [idx[i::k] for i in range(k)]


def evaluate_places(train, test, level, alpha):
    """Held-out performance of P(child | true parent) at `level` (state or city): log-loss, top-1 and top-3
    accuracy, and the same for a uniform baseline over the parent's places (+ one 'new place' option)."""
    m = PlaceTypeModel(alpha=alpha).fit(train)
    keys = LEVELS[level]
    rows = []
    cache = {}
    for parent, grp in test.groupby(keys[:-1]):
        parent = parent if isinstance(parent, tuple) else (parent,)
        try:
            dist = cache.setdefault(parent, m.place_distribution(level, parent))
        except KeyError:  # parent unseen in training
            continue
        rank = dist.rank(ascending=False, method="first")
        k_obs = len(dist) - 1
        for child in grp[keys[-1]]:
            target = child if child in dist.index else NEW_PLACE
            rows.append({"p": float(dist[target]), "rank": float(rank[target]), "p_uniform": 1.0 / (k_obs + 1)})
    r = pd.DataFrame(rows)
    return {"level": level, "alpha": alpha, "records": len(r), "log_loss": float(-np.log(r["p"]).mean()),
            "top1_%": float((r["rank"] <= 1).mean() * 100), "top3_%": float((r["rank"] <= 3).mean() * 100),
            "log_loss_uniform": float(-np.log(r["p_uniform"]).mean())}


def evaluate_types(train, test, beta):
    """Held-out performance of P(type | city) with shrinkage, against the global type mix and against the state's
    own mix (no city information)."""
    m = PlaceTypeModel(beta=beta).fit(train)
    rows = []
    for (c, s, ci), grp in test.groupby(LEVELS["city"]):
        p_city = m.type_distribution(c, s, ci)
        p_state = m.type_distribution(c, s)
        for t in grp["brewery_type"]:
            if t not in m.types:
                continue
            rows.append({"p": p_city[t], "p_state": p_state[t], "p_global": m.t_global[t],
                         "hit": p_city.index[0] == t, "hit_state": p_state.index[0] == t,
                         "hit_global": m.t_global.idxmax() == t})
    r = pd.DataFrame(rows)
    return {"beta": beta, "records": len(r), "log_loss": float(-np.log(r["p"]).mean()),
            "log_loss_state_only": float(-np.log(r["p_state"]).mean()),
            "log_loss_global": float(-np.log(r["p_global"]).mean()), "top1_%": float(r["hit"].mean() * 100),
            "top1_state_only_%": float(r["hit_state"].mean() * 100),
            "top1_global_%": float(r["hit_global"].mean() * 100)}


def cross_validate(df, alphas, betas, k=5, seed=0):
    """k-fold cross-validation of the place model (state | country, city | state) for every alpha and of the type
    model (type | city) for every beta. Returns (place results, type results), averaged over folds."""
    df = df.reset_index(drop=True)
    place, types = [], []
    for f in folds(len(df), k, seed):
        test, train = df.iloc[f], df.drop(index=f)
        for a in alphas:
            for lv in ("state", "city"):
                place.append(evaluate_places(train, test, lv, a))
        for b in betas:
            types.append(evaluate_types(train, test, b))
    place = pd.DataFrame(place).groupby(["level", "alpha"]).mean(numeric_only=True).reset_index()
    types = pd.DataFrame(types).groupby("beta").mean(numeric_only=True).reset_index()
    return place, types


# ---------------------------------------------------------------- the pipeline (planning breweries)
def pipeline_by_state(df, country="United States"):
    """Operating, planning and closed records per state with each state's share of operating and of planning
    breweries; lift = planning share / operating share (> 1: more openings in preparation than its current size
    suggests)."""
    d = df[df["country"] == country]
    t = pd.DataFrame({"operating": operating(d).groupby(STATE).size(),
                      "planning": d[d["brewery_type"] == "planning"].groupby(STATE).size(),
                      "closed": d[d["brewery_type"] == "closed"].groupby(STATE).size()}).fillna(0).astype(int)
    t["operating_share_%"] = t["operating"] / t["operating"].sum() * 100
    t["planning_share_%"] = t["planning"] / max(t["planning"].sum(), 1) * 100
    t["lift"] = t["planning_share_%"] / t["operating_share_%"].replace(0, np.nan)
    t["planning_per_100_operating"] = t["planning"] / t["operating"].replace(0, np.nan) * 100
    t["closed_per_100_operating"] = t["closed"] / t["operating"].replace(0, np.nan) * 100
    return t.reset_index().rename(columns={"index": STATE}).sort_values("planning", ascending=False)


def planning_test(df, alpha, country="United States"):
    """Train the place model on operating breweries and score it on the planning ones (future openings): state and
    city log-loss and top-k accuracy against a uniform baseline, and the Good-Turing prediction of 'new city' share
    against the share of planning breweries that really are in a city without an operating brewery."""
    d = df[df["country"] == country]
    train, test = operating(d), d[d["brewery_type"] == "planning"]
    out = {lv: evaluate_places(train, test, lv, alpha) for lv in ("state", "city")}
    m = PlaceTypeModel(alpha=alpha).fit(train)
    pred_new, obs_new = [], []
    for (c, s), grp in test.groupby(LEVELS["state"]):
        try:
            dist = m.place_distribution("city", (c, s))
        except KeyError:
            continue
        pred_new += [dist[NEW_PLACE]] * len(grp)
        obs_new += [ci not in dist.index for ci in grp["city"]]
    out["new_city"] = {"predicted_%": float(np.mean(pred_new) * 100), "observed_%": float(np.mean(obs_new) * 100),
                       "records": len(obs_new)}
    return out


# ---------------------------------------------------------------- spatial density
class HaversineKDE:
    """Gaussian kernel density on the sphere with the Haversine distance, summed exactly over every training point.

    The normalisation is scikit-learn's for KernelDensity(metric="haversine", kernel="gaussian"), so results agree
    with it wherever its ball tree is exact. Far from the data at small bandwidths the tree is not: at 5 km it was off
    by tens to hundreds of log units for points 90-130 km from the nearest brewery, which also bent the cross-validated
    likelihood of the smallest bandwidths. Computing the sum directly (in chunks, to bound memory) avoids that and
    keeps scikit-learn out of the dashboard."""

    def __init__(self, bandwidth):
        self.bandwidth = float(bandwidth)  # radians

    def fit(self, X):
        self.X_ = np.asarray(X, dtype=float)
        lat, lon = self.X_[:, 0], self.X_[:, 1]
        # half-angle sines and cosines, so the Haversine terms below need products instead of a sine per pair
        self._half = (np.sin(lat / 2), np.cos(lat / 2), np.sin(lon / 2), np.cos(lon / 2), np.cos(lat))
        return self

    def score_samples(self, X, chunk=25):
        """Log density (per steradian) at each row of X = radians [latitude, longitude]. Rows go through in chunks:
        25 query points against ~5,600 breweries keep the temporary arrays at a few MB (200 needed about 50 MB at the
        peak) at the same speed."""
        X = np.asarray(X, dtype=float)
        s_lat, c_lat, s_lon, c_lon, cos_lat = self._half
        h, dim = self.bandwidth, self.X_.shape[1]
        log_norm = 0.5 * dim * np.log(2 * np.pi) + dim * np.log(h) + np.log(len(self.X_))
        out = np.empty(len(X))
        for s in range(0, len(X), chunk):
            la, lo = X[s:s + chunk, :1], X[s:s + chunk, 1:]
            d_lat = np.sin(la / 2) * c_lat - np.cos(la / 2) * s_lat  # sin((la - lat) / 2)
            d_lon = np.sin(lo / 2) * c_lon - np.cos(lo / 2) * s_lon  # sin((lo - lon) / 2)
            a = d_lat * d_lat + np.cos(la) * cos_lat * d_lon * d_lon
            z = -0.5 * (2 * np.arcsin(np.sqrt(np.clip(a, 0, 1))) / h) ** 2
            m = z.max(axis=1, keepdims=True)
            out[s:s + chunk] = m[:, 0] + np.log(np.exp(z - m).sum(axis=1)) - log_norm
        return out


def kde_fit(lat, lon, bandwidth_km):
    X = np.radians(np.column_stack([lat, lon]))
    return HaversineKDE(bandwidth_km / R).fit(X)


def kde_score(model, lat, lon):
    """Mean log density (per steradian) of the given points under the model."""
    return float(model.score_samples(np.radians(np.column_stack([lat, lon]))).mean())


def kde_cv(lat, lon, bandwidths_km, k=5, seed=0):
    """Mean held-out log-likelihood per point for every bandwidth (higher = better)."""
    lat, lon = np.asarray(lat), np.asarray(lon)
    rows = []
    for bw in bandwidths_km:
        scores = []
        for f in folds(len(lat), k, seed):
            mask = np.ones(len(lat), bool)
            mask[f] = False
            scores.append(kde_score(kde_fit(lat[mask], lon[mask], bw), lat[f], lon[f]))
        rows.append({"bandwidth_km": bw, "heldout_log_likelihood": float(np.mean(scores))})
    return pd.DataFrame(rows)


def kde_grid(model, lat_range, lon_range, n=120):
    """Density on a regular latitude/longitude grid, normalised so the cells of the grid sum to 1: the share of
    the next brewery expected in each cell, if it falls inside this region."""
    la = np.linspace(*lat_range, n)
    lo = np.linspace(*lon_range, n)
    LA, LO = np.meshgrid(la, lo, indexing="ij")
    dens = np.exp(model.score_samples(np.radians(np.column_stack([LA.ravel(), LO.ravel()]))))
    cell_area = np.cos(np.radians(LA.ravel()))  # cells shrink toward the poles
    w = dens * cell_area
    return pd.DataFrame({"latitude": LA.ravel(), "longitude": LO.ravel(), "share": w / w.sum()})
