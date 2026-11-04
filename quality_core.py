"""Data-quality and record-identity calculations shared by the dashboard pages and notebooks.

Pure pandas / numpy (no Streamlit), so every table can be rebuilt and tested outside the app. The chi-square test
(missingness_association) imports SciPy when called; the dashboard reads its results from data/dashboard_stats.db.
`frames` arguments are dicts such as {"raw": df, "all_records": df, "unique_records": df} holding the
untouched SQLite tables (no "Unknown" filling, so real nulls stay visible).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from geographic_utils import haversine_km

CATEGORY_COLUMNS = ["brewery_type", "country", "state_province", "city"]
# the three dataset versions as (database file, table name)
DATASET_FILES = {"raw": ("raw_breweries.db", "raw_breweries"),
                 "all_records": ("analysis_all_records.db", "analysis_all_records"),
                 "unique_records": ("analysis_unique_records.db", "analysis_unique_records")}

# Region labels that name the same place in different languages or spellings. The raw `state_province` column is
# kept untouched; `state_province_harmonized` carries one label per real region for state-level analysis.
STATE_HARMONIZATION = {
    ("Belgium", "Vlaanderen"): "Flanders",
    ("Belgium", "Région flamande"): "Flanders",
    ("Belgium", "Flanders"): "Flanders",
    ("Belgium", "Région wallonne"): "Wallonia",
    ("Belgium", "Wallonia"): "Wallonia",
    ("Belgium", "Waals Gewest"): "Wallonia",
    ("United States", "MIssouri"): "Missouri",
}

# Single records whose region label contradicts their own city and position (record id -> harmonized label).
STATE_RECORD_FIXES = {
    # Frontier Beer Company, Salt River (a Cape Town suburb): labelled Gauteng; its point is 4 km from Cape Town,
    # the Western Cape capital, and 1,259 km from the Gauteng capital.
    "cd265508-e5a6-43a6-8fc0-e5eaf84ca940": "Western Cape",
}


def harmonize_states(df):
    """Copy of `df` with `state_province_harmonized` added (raw `state_province` untouched)."""
    keys = list(zip(df["country"], df["state_province"]))
    labels = [STATE_HARMONIZATION.get(k, k[1]) for k in keys]
    if "id" in df:
        labels = [STATE_RECORD_FIXES.get(i, label) for i, label in zip(df["id"], labels)]
    return df.assign(state_province_harmonized=labels)


def prepare(df, corrected=True):
    """A loaded table ready for analysis: numeric coordinates, the coordinate corrections log applied (`corrected`)
    and `state_province_harmonized` added. common.harmonized and stats_store use the same steps."""
    df = df.copy()
    for c in ["latitude", "longitude"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if corrected:
        df = apply_corrections(df)
    return harmonize_states(df)


def harmonization_table(df):
    """Every raw label that the harmonization changes, with its record count."""
    h = harmonize_states(df)
    changed = h[h["state_province"] != h["state_province_harmonized"]]
    out = (changed.groupby(["country", "state_province", "state_province_harmonized"]).size()
           .reset_index(name="records"))
    kept = (h[h["state_province_harmonized"].isin(out["state_province_harmonized"])]
            .groupby(["country", "state_province_harmonized"]).size().rename("records_after_harmonization"))
    return out.join(kept, on=["country", "state_province_harmonized"])


# ------------------------------------------------------------------ structure
def column_overview(frames):
    """One row per column: is it present, and how many non-null / null values does each dataset hold."""
    cols = list(dict.fromkeys(c for d in frames.values() for c in d.columns))
    rows = []
    for c in cols:
        row = {"column": c}
        for name, d in frames.items():
            if c in d.columns:
                nn = int(d[c].notna().sum())
                row[f"{name}_non_null"], row[f"{name}_null"] = nn, len(d) - nn
            else:
                row[f"{name}_non_null"] = row[f"{name}_null"] = None
        row["complete_in_every_dataset"] = all(row[f"{n}_null"] == 0 for n in frames)
        rows.append(row)
    return pd.DataFrame(rows)


def unique_values(frames):
    """Distinct values per column and dataset, as a count and as a share of that dataset's records.
    `records_sharing_a_value` counts records whose value occurs at least twice (nulls excluded)."""
    rows = []
    for name, d in frames.items():
        for c in d.columns:
            vc = d[c].value_counts(dropna=True)
            rows.append({"dataset": name, "column": c, "records": len(d), "non_null": int(d[c].notna().sum()),
                         "unique_values": int(len(vc)), "unique_%": len(vc) / len(d) * 100,
                         "values_used_more_than_once": int((vc > 1).sum()),
                         "records_sharing_a_value": int(vc[vc > 1].sum())})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ shared values
def shared_value_groups(df, col):
    """Values of `col` held by two or more records, with how far each group spreads over countries,
    states/provinces, cities and brewery types. Sorted by group size."""
    s = df[df[col].notna() & df.duplicated(col, keep=False)]
    if s.empty:
        return pd.DataFrame()
    s = s.assign(_place=s["city"].astype(str) + ", " + s["state_province"].astype(str) + ", " + s["country"].astype(str))
    g = s.groupby(col).agg(records=("id", "size"), countries=("country", "nunique"),
                           states=("state_province", "nunique"), cities=("city", "nunique"),
                           brewery_types=("brewery_type", "nunique"),
                           dominant_type=("brewery_type", lambda x: x.value_counts().index[0]),
                           top_location=("_place", lambda x: x.value_counts().index[0]))
    return g.reset_index().sort_values(["records", col], ascending=[False, True]).reset_index(drop=True)


def composition_lift(subset, full, col):
    """Share of each category of `col` in `subset` versus in `full`. lift > 1 means over-represented."""
    t = pd.DataFrame({"records": subset[col].value_counts(), "share_%": subset[col].value_counts(normalize=True) * 100,
                      "overall_share_%": full[col].value_counts(normalize=True) * 100}).fillna(0)
    t["lift"] = t["share_%"] / t["overall_share_%"].replace(0, np.nan)
    return t.rename_axis(col).reset_index().sort_values("records", ascending=False)


# ------------------------------------------------------------------ missing values
def coordinate_missing_rate(df, column):
    """Missing-coordinate rate within each category of `column` (as specified for the project)."""
    work = df.copy()
    work[column] = work[column].astype("string").fillna("(Missing)")

    work["has_missing_coordinates"] = (
        work["latitude"].isna() | work["longitude"].isna()
    )

    result = (
        work.groupby(column, dropna=False)
        .agg(
            total_records=("id", "size"),
            records_missing_coordinates=("has_missing_coordinates", "sum"),
        )
        .reset_index()
    )

    result["missing_coordinate_percentage"] = (
        result["records_missing_coordinates"]
        / result["total_records"]
        * 100
    )

    return result.sort_values(
        ["missing_coordinate_percentage", "total_records"],
        ascending=[False, False],
    ).reset_index(drop=True)


def missing_rate_by(df, target, by):
    """For each category of `by`: records, records with `target` missing, and the missing rate (%).
    `target` is a column name, or "coordinates" for latitude-or-longitude missing."""
    flag = (df["latitude"].isna() | df["longitude"].isna()) if target == "coordinates" else df[target].isna()
    g = (pd.DataFrame({by: df[by].astype("string").fillna("(Missing)"), "missing": flag})
         .groupby(by).agg(records=("missing", "size"), missing=("missing", "sum")).reset_index())
    g["missing_%"] = g["missing"] / g["records"] * 100
    g["share_of_all_missing_%"] = g["missing"] / max(int(g["missing"].sum()), 1) * 100
    return g.sort_values(["missing", "records"], ascending=False).reset_index(drop=True)


def missingness_association(df, target, by, min_records=30):
    """Is missingness of `target` independent of category `by`? Chi-square test of independence on the
    category x missing table. Categories with fewer than `min_records` records are pooled into one
    "other" group so the expected counts stay usable. Cramér's V (0-1) measures how strong the link is.
    Returns None when the test is not possible (nothing missing, everything missing, or one category)."""
    from scipy.stats import chi2_contingency
    flag = (df["latitude"].isna() | df["longitude"].isna()) if target == "coordinates" else df[target].isna()
    if flag.all() or not flag.any():
        return None
    cat = df[by].astype("string").fillna("(Missing)")
    size = cat.map(cat.value_counts())
    cat = cat.where(size >= min_records, f"other ({by} with < {min_records} records)")
    table = pd.crosstab(cat, flag)
    if table.shape[0] < 2 or table.shape[1] < 2:
        return None
    chi2, p, dof, expected = chi2_contingency(table)
    v = float(np.sqrt(chi2 / (table.values.sum() * (min(table.shape) - 1))))
    strength = "negligible" if v < 0.1 else "weak" if v < 0.3 else "moderate" if v < 0.5 else "strong"
    if p < 0.001 and v >= 0.1:
        verdict = f"not random: depends on {by}"
    elif p < 0.001:
        verdict = "detectable but very weak link"
    else:
        verdict = "consistent with random"
    return {"column": target, "by": by, "categories_tested": int(table.shape[0]), "records": int(table.values.sum()),
            "missing": int(flag.sum()), "chi2": float(chi2), "dof": int(dof), "p_value": float(p),
            "cramers_v": v, "strength": strength, "verdict": verdict}


def co_missingness(df, columns=None):
    """P(column B missing | column A missing) next to the overall rate of B missing. A ratio well above 1
    means the two fields tend to be missing together (one gap travels with the other)."""
    flags = pd.DataFrame({c: df[c].isna() for c in (columns or df.columns)})
    flags["coordinates"] = df["latitude"].isna() | df["longitude"].isna()
    flags = flags.drop(columns=[c for c in ("latitude", "longitude") if c in flags])
    flags = flags.loc[:, flags.any() & ~flags.all()]
    rows = []
    for a in flags:
        for b in flags:
            if a == b:
                continue
            base = flags[b].mean()
            cond = flags.loc[flags[a], b].mean()
            rows.append({"if_missing": a, "then_missing": b, "records_missing_a": int(flags[a].sum()),
                         "p_b_given_a_%": cond * 100, "p_b_overall_%": base * 100,
                         "ratio": cond / base if base else np.nan})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ record identity
def near_duplicate_pairs(df):
    """Records identical in every column except `id`, latitude and longitude (same brewery, two positions).
    Adds `pair`, `pair_distance_km` (Haversine distance between the two positions) and `records_in_group`."""
    cols = [c for c in df.columns if c not in ("id", "latitude", "longitude")]
    nd = df[df.duplicated(cols, keep=False)].copy()
    if nd.empty:
        return nd
    nd["pair"] = nd.groupby(cols, dropna=False).ngroup() + 1
    nd["records_in_group"] = nd.groupby("pair")["id"].transform("size")
    lat = pd.to_numeric(nd["latitude"], errors="coerce")
    lon = pd.to_numeric(nd["longitude"], errors="coerce")
    first = nd.assign(lat=lat, lon=lon).groupby("pair")[["lat", "lon"]].transform("first")
    last = nd.assign(lat=lat, lon=lon).groupby("pair")[["lat", "lon"]].transform("last")
    nd["pair_distance_km"] = haversine_km(first["lat"], first["lon"], last["lat"], last["lon"])
    return nd.sort_values(["pair_distance_km", "pair"], ascending=[False, True])


def shared_coordinate_records(df):
    """Every record whose exact (latitude, longitude) is shared with at least one other record.
    `stack_size` = records at that point; `stack_kind` separates points where all records give the same
    street address (shared premises) from points where the addresses differ (several addresses geocoded
    to one point, e.g. a city or postcode centroid)."""
    s = df[df.duplicated(["latitude", "longitude"], keep=False)].copy()
    if s.empty:
        return s
    g = s.groupby(["latitude", "longitude"])
    s["location_id"] = g.ngroup() + 1
    s["stack_size"] = g["id"].transform("size")
    s["addresses_at_point"] = g["address_1"].transform(lambda x: x.nunique(dropna=False))
    s["stack_kind"] = np.where(s["addresses_at_point"] == 1, "same street address", "different street addresses")
    return s.sort_values(["stack_size", "location_id"], ascending=[False, True])


def identical_columns(df):
    """Pairs of columns whose values are identical on every record (nulls compared as equal), e.g. a field the
    source repeats under two names."""
    cols = list(df.columns)
    pairs = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            if df[a].astype("string").fillna("<null>").equals(df[b].astype("string").fillna("<null>")):
                pairs.append({"column": a, "identical_to": b, "records_compared": len(df)})
    return pd.DataFrame(pairs)


# ------------------------------------------------------------------ coordinate corrections
CORRECTIONS_FILE = Path(__file__).resolve().parent / "data" / "coordinate_corrections.csv"
STATUS = {"swapped back": "corrected: latitude/longitude swapped back",
          "sign flipped back": "corrected: coordinate sign flipped back",
          "copied from identical record": "corrected: copied from identical record",
          "flagged only": "flagged: far from its city"}
FIXED_ACTIONS = ["swapped back", "sign flipped back", "copied from identical record"]


def load_corrections(path=CORRECTIONS_FILE):
    """The coordinate corrections log written by the clustering notebook (empty if it does not exist yet)."""
    return pd.read_csv(path, dtype={"id": str}) if Path(path).exists() else pd.DataFrame(
        columns=["id", "action", "corrected_latitude", "corrected_longitude"])


def apply_corrections(df, corrections=None):
    """Copy of `df` with corrected coordinates where the log says so ('swapped back', 'sign flipped back', 'copied
    from identical record') and a `coordinate_status` column for every record. Flagged records keep their position.
    The databases themselves are never changed."""
    c = load_corrections() if corrections is None else corrections
    out = df.copy()
    for col in ["latitude", "longitude"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["coordinate_status"] = "as recorded"
    if c.empty:
        return out
    log = c.set_index("id")
    hit = out["id"].isin(log.index)
    out.loc[hit, "coordinate_status"] = out.loc[hit, "id"].map(log["action"]).map(STATUS)
    fix = hit & out["id"].map(log["action"]).isin(FIXED_ACTIONS)
    out.loc[fix, "latitude"] = out.loc[fix, "id"].map(log["corrected_latitude"]).astype(float)
    out.loc[fix, "longitude"] = out.loc[fix, "id"].map(log["corrected_longitude"]).astype(float)
    return out


# ------------------------------------------------------------------ names vs websites
_GENERIC_NAME_WORDS = {"the", "and", "of", "brewing", "brewery", "breweries", "brewers", "brewer", "brew", "brews",
                       "brewpub", "company", "co", "inc", "llc", "ltd", "gmbh", "bv", "sa", "beer", "beers", "ale", "ales",
                       "taproom", "tap", "room", "restaurant", "pub", "house", "works", "project", "craft",
                       "cerveceria", "brauerei", "brouwerij", "brasserie", "bier", "biere", "bieren", "kg", "ag",
                       "limited", "pty", "group"}
SOCIAL_HOSTS = ("facebook.com", "instagram.com", "untappd.com", "twitter.com", "x.com", "linktr.ee", "yelp.com")
NAME_LINK_ORDER = ["full name in domain", "part of the name in domain", "initials in domain",
                   "social-media page", "no visible link", "no website"]


def _ascii_variants(text):
    """Lower-case ASCII spellings of a text: accents dropped, and German umlauts written out (ü -> ue)."""
    import unicodedata
    t = str(text).lower()
    german = t.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
    return {unicodedata.normalize("NFKD", v).encode("ascii", "ignore").decode() for v in (t, german)}


def name_website_link(name, url):
    """How a brewery's name shows up in its website address. Categories, from strongest to weakest:
    'full name in domain' (the distinctive words of the name, joined, appear in the domain label, e.g.
    'Ballast Point Brewing' -> ballastpoint.com), 'part of the name in domain', 'initials in domain',
    'social-media page' (the website is a Facebook/Instagram/... page), 'no visible link', 'no website'."""
    import re
    if not isinstance(url, str) or not url.strip():
        return "no website"
    host = re.sub(r"^[a-z]+://", "", url.strip().lower()).split("/")[0].split("?")[0].split(":")[0]
    host = re.sub(r"^www\d?\.", "", host)
    if any(host == s or host.endswith("." + s) for s in SOCIAL_HOSTS):
        return "social-media page"
    parts = host.split(".")
    label = parts[-3] if len(parts) >= 3 and parts[-2] in {"co", "com", "org", "net", "ac"} else \
        parts[-2] if len(parts) >= 2 else parts[0]
    label = re.sub(r"[^a-z0-9]", "", label)
    best = "no visible link"
    for variant in _ascii_variants(name):
        words = re.findall(r"[a-z0-9]+", variant.replace("&", " and "))
        distinctive = [w for w in words if w not in _GENERIC_NAME_WORDS] or words
        if distinctive and "".join(distinctive) in label:
            return "full name in domain"
        if any(len(w) >= 4 and w in label for w in distinctive):
            best = "part of the name in domain"
        initials = "".join(w[0] for w in words if w not in {"the", "and", "of"})
        if best == "no visible link" and len(initials) >= 2 and initials in label:
            best = "initials in domain"
    return best


def name_website_links(df):
    """`name_website_link` for every record."""
    return pd.Series([name_website_link(n, u) for n, u in zip(df["name"], df["website_url"])], index=df.index,
                     name="name_website_link")
