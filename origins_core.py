"""Distances from two kinds of origin, at country, state/province and city level.

Origin 1 - brewery distribution center: the mean latitude/longitude of the place's own breweries. It tells whether
           the breweries gather around their own middle or spread out.
Origin 2 - reference point of the place, from data/reference_places.db (GeoNames, see tools/build_reference_places.py):
           national capital (country), state/province capital (state), GeoNames city center (city). It tells how far
           breweries are from the place people would call its center.

All distances are great-circle (Haversine) kilometres. States are keyed by country + state_province_harmonized and
cities by country + state + city. Pure pandas / numpy (no Streamlit); spearman_by_state imports SciPy when called,
and the dashboard reads its results from data/dashboard_stats.db instead.
"""
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from geographic_utils import haversine_km

REF_DB = Path(__file__).resolve().parent / "data" / "reference_places.db"
STATE_COL = "state_province_harmonized"
KEYS = {"country": ["country"], "state": ["country", STATE_COL], "city": ["country", STATE_COL, "city"]}
DIST = {("country", "mean"): "dist_country_mean_center_km", ("country", "reference"): "dist_country_capital_km",
        ("state", "mean"): "dist_state_mean_center_km", ("state", "reference"): "dist_state_capital_km",
        ("city", "mean"): "dist_city_mean_center_km", ("city", "reference"): "dist_city_center_km"}
ORIGIN_NAME = {"mean": {"country": "country brewery center", "state": "state brewery center",
                        "city": "city brewery center"},
               "reference": {"country": "national capital", "state": "state/province capital",
                             "city": "city center (GeoNames)"}}


def load_reference(path=REF_DB):
    """{'country': country_capitals, 'state': state_capitals, 'city': city_centers} as DataFrames."""
    with sqlite3.connect(path) as con:
        return {"country": pd.read_sql_query("SELECT * FROM country_capitals", con),
                "state": pd.read_sql_query("SELECT * FROM state_capitals", con),
                "city": pd.read_sql_query("SELECT * FROM city_centers", con),
                "sources": pd.read_sql_query("SELECT * FROM sources", con)}


def reference_points(ref, level):
    """Origin-2 point per place: columns = level keys + ref_name, ref_latitude, ref_longitude."""
    if level == "country":
        t = ref["country"].rename(columns={"capital": "ref_name", "capital_latitude": "ref_latitude",
                                           "capital_longitude": "ref_longitude"})
    elif level == "state":
        t = ref["state"].rename(columns={"capital": "ref_name", "capital_latitude": "ref_latitude",
                                         "capital_longitude": "ref_longitude"})
    else:
        t = ref["city"].rename(columns={"geonames_name": "ref_name", "city_latitude": "ref_latitude",
                                        "city_longitude": "ref_longitude"})
    return t[KEYS[level] + ["ref_name", "ref_latitude", "ref_longitude"]]


def add_origins(df, ref):
    """Copy of a geocoded dataset with both origins and the distance to each, at every level, plus the distance
    from the overall dataset center."""
    d = df.dropna(subset=["latitude", "longitude"]).copy()
    d["dist_overall_center_km"] = haversine_km(d["latitude"], d["longitude"], d["latitude"].mean(),
                                               d["longitude"].mean())
    for level, keys in KEYS.items():
        c = d.groupby(keys)[["latitude", "longitude"]].transform("mean")
        d[f"{level}_mean_latitude"], d[f"{level}_mean_longitude"] = c["latitude"], c["longitude"]
        d[DIST[(level, "mean")]] = haversine_km(d["latitude"], d["longitude"], c["latitude"], c["longitude"])
        r = reference_points(ref, level).rename(columns={"ref_name": f"{level}_reference",
                                                         "ref_latitude": f"{level}_ref_latitude",
                                                         "ref_longitude": f"{level}_ref_longitude"})
        d = d.merge(r, on=keys, how="left")
        d[DIST[(level, "reference")]] = haversine_km(d["latitude"], d["longitude"], d[f"{level}_ref_latitude"],
                                                     d[f"{level}_ref_longitude"])
    return d


def dispersion(d, level, origin, radii=(10, 25, 50, 100)):
    """Spread of each place's breweries around one origin: count, mean, median, quartiles, P90, max, IQR,
    mean-to-median ratio, and the share of breweries within each radius."""
    col = DIST[(level, origin)]
    x = d.dropna(subset=[col])
    g = x.groupby(KEYS[level])[col]
    s = g.agg(record_count="count", mean_km="mean", median_km="median", max_km="max")
    s["p25_km"], s["p75_km"], s["p90_km"] = g.quantile(.25), g.quantile(.75), g.quantile(.90)
    s["iqr_km"] = s["p75_km"] - s["p25_km"]
    s["mean_to_median_ratio"] = s["mean_km"] / s["median_km"].replace(0, np.nan)
    for r in radii:
        s[f"within_{r}km_%"] = x[col].le(r).groupby([x[k] for k in KEYS[level]]).mean() * 100
    return s.reset_index()


def interpret_ratio(ratio, tolerance=0.1):
    """Automatic reading of mean/median of distance-to-center. Returns (class, sentence)."""
    if pd.isna(ratio):
        return ("undefined", "The median distance is 0: at least half of the breweries sit exactly on the center "
                             "point, so the ratio cannot be computed.")
    if abs(ratio - 1) <= tolerance:
        return ("mean ≈ median", f"Mean and median almost agree (ratio {ratio:.2f}): distances are fairly symmetric, "
                                 "so the median is a good summary of how far a typical brewery sits from the center.")
    if ratio > 1:
        return ("mean > median", f"The mean is {ratio:.2f}× the median: most breweries sit fairly close to the "
                                 "center and a minority far away stretches the distribution into a long tail.")
    return ("mean < median", f"The mean is only {ratio:.2f}× the median: most breweries sit at similar, larger "
                             "distances (several hubs or a ring around the center) while a smaller group very close "
                             "to the center pulls the mean down.")


def add_interpretation(disp, tolerance=0.1):
    out = disp.copy()
    res = [interpret_ratio(r, tolerance) for r in out["mean_to_median_ratio"]]
    out["shape"], out["interpretation"] = [a for a, _ in res], [b for _, b in res]
    return out


def flag_outliers(d, level="state", origin="mean"):
    """Q3 + 1.5 x IQR rule on the distance to the place's origin, computed separately for every place."""
    col = DIST[(level, origin)]
    out = d.copy()
    g = out.groupby(KEYS[level])[col]
    q1, q3 = g.transform(lambda x: x.quantile(.25)), g.transform(lambda x: x.quantile(.75))
    out["outlier_threshold_km"] = q3 + 1.5 * (q3 - q1)
    out["geographic_outlier"] = out[col] > out["outlier_threshold_km"]
    return out


def outlier_summary(d, level="state", origin="mean"):
    col = DIST[(level, origin)]
    s = d.groupby(KEYS[level]).agg(record_count=("geographic_outlier", "size"),
                                   geographic_outlier_count=("geographic_outlier", "sum"),
                                   maximum_distance_km=(col, "max")).reset_index()
    s["geographic_outlier_pct"] = s["geographic_outlier_count"] / s["record_count"] * 100
    return s


def city_table(d, ref):
    """One row per city: records, share of its state, the city's brewery mean position and its distance to the
    state brewery center (origin 1) and to the state capital (origin 2); whether the city is the capital."""
    c = d.groupby(KEYS["city"]).agg(records=("id", "size"), latitude=("latitude", "mean"),
                                    longitude=("longitude", "mean")).reset_index()
    st = d.groupby(KEYS["state"]).agg(state_records=("id", "size"), state_lat=("latitude", "mean"),
                                      state_lon=("longitude", "mean")).reset_index()
    c = c.merge(st, on=KEYS["state"])
    cap = ref["state"][KEYS["state"] + ["capital", "capital_latitude", "capital_longitude", "geonameid"]] \
        .rename(columns={"geonameid": "capital_geonameid"})
    c = c.merge(cap, on=KEYS["state"], how="left")
    cid = ref["city"][KEYS["city"] + ["geonameid"]]
    c = c.merge(cid, on=KEYS["city"], how="left")
    c["share_of_state_%"] = c["records"] / c["state_records"] * 100
    c["rank_in_state"] = c.groupby(KEYS["state"])["records"].rank(method="min", ascending=False).astype(int)
    c["km_to_state_center"] = haversine_km(c["latitude"], c["longitude"], c["state_lat"], c["state_lon"])
    c["km_to_state_capital"] = haversine_km(c["latitude"], c["longitude"], c["capital_latitude"],
                                            c["capital_longitude"])
    c["is_capital"] = c["geonameid"].notna() & (c["geonameid"] == c["capital_geonameid"])
    return c


def spearman_by_state(c, min_cities=5):
    """Spearman correlation, per state, between a city's distance from the state brewery center (origin 1) or the
    capital (origin 2) and the city's share of the state. `c` is a city_table; states with fewer than `min_cities`
    cities, or with every city holding the same share, get NaN."""
    from scipy.stats import spearmanr
    rows = []
    for key, g in c.groupby(KEYS["state"]):
        row = dict(zip(KEYS["state"], key))
        for origin, col in [("center", "km_to_state_center"), ("capital", "km_to_state_capital")]:
            ok = g.dropna(subset=[col])
            if len(ok) >= min_cities and ok["share_of_state_%"].nunique() > 1:
                rho, p = spearmanr(ok[col], ok["share_of_state_%"])
                row[f"rho_distance_to_{origin}_vs_share"], row[f"p_{origin}"] = float(rho), float(p)
            else:
                row[f"rho_distance_to_{origin}_vs_share"], row[f"p_{origin}"] = np.nan, np.nan
        rows.append(row)
    return pd.DataFrame(rows, columns=KEYS["state"] + ["rho_distance_to_center_vs_share", "p_center",
                                                     "rho_distance_to_capital_vs_share", "p_capital"])


def central_cities(d, ref, states=None, min_cities=5, spearman=None):
    """For each state: the city holding the largest share, the city nearest to the state's brewery center, the
    capital's rank and share, and Spearman correlations between a city's distance from the state center (origin 1)
    or capital (origin 2) and its share of the state. A negative rho means cities near the center hold more of the
    state's breweries. `spearman` passes precomputed correlations (a spearman_by_state table, as the dashboard does
    from data/dashboard_stats.db); without it they are computed here."""
    c = city_table(d, ref)
    if states is not None:
        c = c.merge(states[KEYS["state"]].drop_duplicates(), on=KEYS["state"])
    rows = []
    for key, g in c.groupby(KEYS["state"]):
        top = g.sort_values("records", ascending=False).iloc[0]
        near = g.sort_values("km_to_state_center").iloc[0]
        capg = g[g["is_capital"]]
        row = dict(zip(KEYS["state"], key))
        row.update({"records": int(g["state_records"].iat[0]), "cities": len(g),
                    "largest_city": top["city"], "largest_city_share_%": top["share_of_state_%"],
                    "largest_city_km_to_center": top["km_to_state_center"],
                    "city_nearest_center": near["city"], "nearest_city_share_%": near["share_of_state_%"],
                    "capital": g["capital"].iat[0],
                    "capital_records": int(capg["records"].sum()),
                    "capital_share_%": float(capg["share_of_state_%"].sum()),
                    "capital_rank_in_state": int(capg["rank_in_state"].min()) if len(capg) else np.nan})
        rows.append(row)
    sp = spearman_by_state(c, min_cities) if spearman is None else spearman
    out = (pd.DataFrame(rows).merge(sp, on=KEYS["state"], how="left")
           .sort_values("records", ascending=False).reset_index(drop=True))
    out["capital_role"] = np.select(
        [out["capital"].isna(), out["capital_records"] == 0, out["capital_rank_in_state"] == 1,
         out["capital_rank_in_state"] <= 3],
        ["no capital on record", "capital has no brewery", "capital is the main brewery city",
         "capital is a secondary brewery city (rank 2-3)"], "capital is a minor brewery city")
    return out, c


def capital_vs_center(d, level="state", radius=50):
    """Per place: median distance and share within `radius` km for origin 1 (brewery center) and origin 2
    (capital or city center), and the distance between the two origins."""
    m, r = DIST[(level, "mean")], DIST[(level, "reference")]
    x = d.dropna(subset=[r])
    keys = [x[k] for k in KEYS[level]]
    out = pd.DataFrame({
        "records": x.groupby(keys).size(),
        "reference": x.groupby(keys)[f"{level}_reference"].first(),
        "median_km_to_brewery_center": x.groupby(keys)[m].median(),
        "median_km_to_reference": x.groupby(keys)[r].median(),
        f"within_{radius}km_of_brewery_center_%": x[m].le(radius).groupby(keys).mean() * 100,
        f"within_{radius}km_of_reference_%": x[r].le(radius).groupby(keys).mean() * 100,
    })
    lat_m, lon_m = x.groupby(keys)[f"{level}_mean_latitude"].first(), x.groupby(keys)[f"{level}_mean_longitude"].first()
    lat_r, lon_r = x.groupby(keys)[f"{level}_ref_latitude"].first(), x.groupby(keys)[f"{level}_ref_longitude"].first()
    out["km_between_origins"] = haversine_km(lat_m, lon_m, lat_r, lon_r)
    out["closer_origin"] = np.where(out["median_km_to_reference"] < out["median_km_to_brewery_center"],
                                    "reference point", "brewery center")
    return out.reset_index().sort_values("records", ascending=False).reset_index(drop=True)


def coordinate_consistency(d, ref, far_km=50, swap_km=25, swap_min_km=500, capital_km=150):
    """Check every record against its own city center (GeoNames). A record is 'swapped' when it lies more than
    `swap_min_km` from its city but within `swap_km` once latitude and longitude are exchanged, has a 'sign error'
    when it lands within `swap_km` once the sign of one coordinate is reversed, and is 'far from its city' when it
    lies more than `far_km` away. A record whose city has no reference point is checked for a sign error against
    its state capital instead (within `capital_km` once reversed); otherwise it is 'not checked'."""
    cc = ref["city"][KEYS["city"] + ["city_latitude", "city_longitude", "match_method"]]
    sc = ref["state"][KEYS["state"] + ["capital", "capital_latitude", "capital_longitude"]]
    x = d.merge(cc, on=KEYS["city"], how="left").merge(sc, on=KEYS["state"], how="left")
    x["km_to_city_center"] = haversine_km(x["latitude"], x["longitude"], x["city_latitude"], x["city_longitude"])
    x["km_to_city_center_if_swapped"] = haversine_km(x["longitude"], x["latitude"], x["city_latitude"],
                                                     x["city_longitude"])
    # reference point: the city center, or the state capital when the city has none
    has_city = x["city_latitude"].notna()
    x["reference_point"] = np.where(has_city, x["city"] + " city center", x["capital"] + " (state capital)")
    x["reference_latitude"] = x["city_latitude"].where(has_city, x["capital_latitude"])
    x["reference_longitude"] = x["city_longitude"].where(has_city, x["capital_longitude"])
    x["km_to_reference"] = haversine_km(x["latitude"], x["longitude"], x["reference_latitude"],
                                        x["reference_longitude"])
    lat_flip = haversine_km(-x["latitude"], x["longitude"], x["reference_latitude"], x["reference_longitude"])
    lon_flip = haversine_km(x["latitude"], -x["longitude"], x["reference_latitude"], x["reference_longitude"])
    x["sign_error_in"] = np.where(lon_flip <= lat_flip, "longitude", "latitude")
    x["km_to_reference_if_sign_reversed"] = np.fmin(lat_flip, lon_flip)
    sign = (x["km_to_reference"] > swap_min_km) & (
        x["km_to_reference_if_sign_reversed"] < np.where(has_city, swap_km, capital_km))
    x["check"] = np.select(
        [(x["km_to_city_center"] > swap_min_km) & (x["km_to_city_center_if_swapped"] < swap_km),
         sign,
         ~has_city,
         x["km_to_city_center"] > far_km],
        ["latitude/longitude swapped", "sign error", "not checked", "far from its city"], "consistent")
    return x.drop(columns=["capital", "capital_latitude", "capital_longitude"])


def build_corrections(raw, ref):
    """Coordinate corrections derived from the city check (run on the raw snapshot, harmonized states):
    - 'swapped back': the record is >500 km from its city and <25 km once latitude and longitude are exchanged;
    - 'sign flipped back': the record is >500 km from its city (or, without a city center, its state capital) and
      close to it once the sign of one coordinate is reversed;
    - 'copied from identical record': the record is far from its city and has a near-duplicate twin (same values
      in every column except id and coordinates) whose position agrees with the city;
    - 'flagged only': far from its city with no safe correction; the position is kept and flagged.
    Returns one row per affected record with original and corrected coordinates and the evidence."""
    geo = raw.assign(latitude=pd.to_numeric(raw["latitude"], errors="coerce"),
                     longitude=pd.to_numeric(raw["longitude"], errors="coerce")).dropna(subset=["latitude", "longitude"])
    chk = coordinate_consistency(geo, ref)
    rows = []
    twin_cols = [c for c in raw.columns if c not in ("id", "latitude", "longitude", STATE_COL)]
    for _, r in chk[chk["check"].isin(["latitude/longitude swapped", "sign error", "far from its city"])].iterrows():
        row = {"id": r["id"], "name": r["name"], "city": r["city"], "state_province": r["state_province"],
               "country": r["country"], "original_latitude": r["latitude"], "original_longitude": r["longitude"],
               "km_to_city_center_original": r["km_to_city_center"],
               "city_latitude": r["city_latitude"], "city_longitude": r["city_longitude"]}
        if r["check"] == "latitude/longitude swapped":
            row.update(corrected_latitude=r["longitude"], corrected_longitude=r["latitude"], action="swapped back",
                       evidence=f"{r['km_to_city_center']:,.0f} km from {r['city']} as stored, "
                                f"{r['km_to_city_center_if_swapped']:.1f} km with latitude and longitude exchanged")
        elif r["check"] == "sign error":
            flip_lat = r["sign_error_in"] == "latitude"
            row.update(corrected_latitude=-r["latitude"] if flip_lat else r["latitude"],
                       corrected_longitude=r["longitude"] if flip_lat else -r["longitude"], action="sign flipped back",
                       evidence=f"{r['km_to_reference']:,.0f} km from {r['reference_point']} as stored, "
                                f"{r['km_to_reference_if_sign_reversed']:.1f} km with the {r['sign_error_in']} "
                                "sign reversed")
        else:
            twins = chk[(chk["id"] != r["id"]) & (chk["check"] == "consistent")]
            for c in twin_cols:
                twins = twins[(twins[c] == r[c]) | (twins[c].isna() & pd.isna(r[c]))]
            if len(twins) == 1:
                t = twins.iloc[0]
                row.update(corrected_latitude=t["latitude"], corrected_longitude=t["longitude"],
                           action="copied from identical record",
                           evidence=f"{r['km_to_city_center']:,.0f} km from {r['city']}; its identical record "
                                    f"{t['id']} lies {t['km_to_city_center']:.1f} km from it")
            else:
                row.update(corrected_latitude=r["latitude"], corrected_longitude=r["longitude"], action="flagged only",
                           evidence=f"{r['km_to_city_center']:,.0f} km from the center of {r['city']}; no safe "
                                    "correction, position kept")
        rows.append(row)
    out = pd.DataFrame(rows)
    out["km_to_city_center_corrected"] = haversine_km(out["corrected_latitude"], out["corrected_longitude"],
                                                      out["city_latitude"], out["city_longitude"])
    out = out.drop(columns=["city_latitude", "city_longitude"])
    return out.sort_values(["action", "km_to_city_center_original"], ascending=[False, False]).reset_index(drop=True)
