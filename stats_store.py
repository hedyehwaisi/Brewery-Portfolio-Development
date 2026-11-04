"""Statistics the dashboard shows that need SciPy or scikit-learn, computed once and stored.

brewery_dashboard_statistics.ipynb calls `build()`, which runs these tests on exactly the inputs the pages use and
writes them to data/dashboard_stats.db. The pages only read those tables, so the running app never imports SciPy or
scikit-learn (about 80 MB less memory per server process). Rebuild after any change to the data, the coordinate
corrections or the reference places; tests/test_stats_store.py checks that the stored tables still match a fresh
computation and that the input files are unchanged.

Tables
- missingness: quality_core.missingness_association for every dataset x column with gaps x grouping (Data Quality)
- spearman: origins_core.spearman_by_state for each analysis dataset (Geographic Analysis, capitals tab)
- type_association: distribution_core.type_association per dataset x level x minimum records (Brewery Types)
- type_profile_states, type_profile_mix, type_profile_silhouette: distribution_core.type_profiles per dataset x
  country x minimum records (Brewery Types)
- type_colocation: distribution_core.type_colocation per dataset x country x same-point option (Brewery Types)
- inputs: SHA-256 of every input file at build time
"""
import hashlib
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

import distribution_core as dc
import origins_core as oc
import quality_core as qc

DATA = Path(__file__).resolve().parent / "data"
DB_FILE = "dashboard_stats.db"
DB = DATA / DB_FILE
ANALYSIS_DATASETS = ["all_records", "unique_records"]
LEVELS = ["country", "state", "city"]
TYPE_MIN_RECORDS = range(5, 101)       # the Brewery Types page slider
COLOCATION_PERMUTATIONS = 199
COLOCATION_MIN_GEOCODED = 50           # countries with fewer geocoded records are not tested
INPUT_FILES = [src[0] for src in qc.DATASET_FILES.values()] + ["coordinate_corrections.csv", "reference_places.db"]


# ------------------------------------------------------------------ inputs (the same steps as the pages)
def load_stored(name):
    """A dataset exactly as stored (Data Quality page)."""
    db, table = qc.DATASET_FILES[name]
    with sqlite3.connect(f"file:{DATA / db}?mode=ro", uri=True) as con:
        return pd.read_sql_query(f'SELECT * FROM "{table}"', con)


def load_prepared(name):
    """A dataset with corrected coordinates and harmonized states (common.harmonized)."""
    return qc.prepare(load_stored(name))


def missing_targets(d):
    """Columns with gaps that the Data Quality page tests, in its order."""
    coords = ["coordinates"] if (d["latitude"].isna() | d["longitude"].isna()).any() else []
    return coords + [c for c in d.columns if c not in ("latitude", "longitude") and d[c].isna().any()]


def colocation_countries(d):
    """Countries with enough geocoded records for the co-location test."""
    n = d.dropna(subset=["latitude", "longitude"])["country"].value_counts()
    return n[n >= COLOCATION_MIN_GEOCODED].index.tolist()


def file_hashes():
    return pd.DataFrame([{"file": f, "sha256": hashlib.sha256((DATA / f).read_bytes()).hexdigest()}
                         for f in INPUT_FILES])


# ------------------------------------------------------------------ builders
def build_missingness():
    rows = []
    for name in qc.DATASET_FILES:
        d = load_stored(name)
        for c in missing_targets(d):
            for by in qc.CATEGORY_COLUMNS:
                t = qc.missingness_association(d, c, by)
                if t:
                    rows.append({"dataset": name, **t})
    return pd.DataFrame(rows).rename_axis("row").reset_index()


def build_spearman(ref):
    parts = []
    for name in ANALYSIS_DATASETS:
        d = oc.flag_outliers(oc.add_origins(load_prepared(name), ref), "state", "mean")
        parts.append(oc.spearman_by_state(oc.city_table(d, ref)).assign(dataset=name))
    return pd.concat(parts, ignore_index=True)


def build_type_statistics(prepared):
    """type_association, type_profiles and type_colocation for every setting the Brewery Types page offers."""
    assoc, states, mix, sil, coloc = [], [], [], [], []
    for name, df in prepared.items():
        for mn in TYPE_MIN_RECORDS:
            assoc += [{"dataset": name, "min_records": mn, **dc.type_association(df, lv, min_records=mn)}
                      for lv in LEVELS]
        for country in dc.counts(df, "country")["place"]:
            for mn in TYPE_MIN_RECORDS:
                try:
                    a, p, s = dc.type_profiles(df, country=country, min_records=mn)
                except Exception:  # too few states to group: the page says so
                    continue
                key = {"dataset": name, "country": country, "min_records": mn}
                states.append(a.assign(**key))
                types = [c for c in p.columns if c != "states"]
                mix.append(p.reset_index().melt(id_vars=["profile", "states"], value_vars=types,
                                                var_name="brewery_type", value_name="share_%")
                           .assign(type_order=lambda x: x["brewery_type"].map(types.index), **key))
                sil.append(s.assign(**key))
        for country in colocation_countries(df):
            for excl in (False, True):
                t = dc.type_colocation(df[df["country"] == country], permutations=COLOCATION_PERMUTATIONS,
                                       exclude_same_point=excl)
                coloc.append(t.assign(dataset=name, country=country, exclude_same_point=excl))
    return {"type_association": pd.DataFrame(assoc), "type_profile_states": pd.concat(states, ignore_index=True),
            "type_profile_mix": pd.concat(mix, ignore_index=True),
            "type_profile_silhouette": pd.concat(sil, ignore_index=True),
            "type_colocation": pd.concat(coloc, ignore_index=True)}


def build(path=DB):
    """Compute every table and write data/dashboard_stats.db. Returns {table: rows}."""
    prepared = {name: load_prepared(name) for name in qc.DATASET_FILES}
    tables = {"missingness": build_missingness(), "spearman": build_spearman(oc.load_reference()),
              **build_type_statistics(prepared), "inputs": file_hashes()}
    Path(path).unlink(missing_ok=True)
    with sqlite3.connect(path) as con:
        for name, t in tables.items():
            t.to_sql(name, con, index=False)
        con.execute("VACUUM")
    return {name: len(t) for name, t in tables.items()}


# ------------------------------------------------------------------ readers (no SciPy / scikit-learn)
def missingness(tbl, dataset):
    """Rows of missingness_association for one dataset, in the page's order."""
    return tbl[tbl["dataset"] == dataset].sort_values("row").drop(columns=["row", "dataset"]).reset_index(drop=True)


def spearman(tbl, dataset):
    return tbl[tbl["dataset"] == dataset].drop(columns="dataset").reset_index(drop=True)


def type_association(tbl, dataset, min_records):
    t = tbl[(tbl["dataset"] == dataset) & (tbl["min_records"] == min_records)].drop(columns=["dataset", "min_records"])
    return t.set_index("level").loc[LEVELS].reset_index()


def type_profiles(states, mix, sil, dataset, country, min_records):
    """(assignments, group profiles, silhouette per k) as distribution_core.type_profiles returns them, or None when
    the country has too few states to group."""
    def pick(t):
        return t[(t["dataset"] == dataset) & (t["country"] == country) & (t["min_records"] == min_records)] \
            .drop(columns=["dataset", "country", "min_records"])
    a = pick(states)
    if a.empty:
        return None
    m = pick(mix).sort_values("type_order")
    prof = m.pivot(index="profile", columns="brewery_type", values="share_%")[m["brewery_type"].unique()]
    prof.columns.name = "brewery_type"
    prof.insert(0, "states", m.drop_duplicates("profile").set_index("profile")["states"].loc[prof.index].values)
    return a.reset_index(drop=True), prof, pick(sil).reset_index(drop=True)


def type_colocation(tbl, dataset, country, exclude_same_point):
    t = tbl[(tbl["dataset"] == dataset) & (tbl["country"] == country)
            & (tbl["exclude_same_point"].astype(bool) == bool(exclude_same_point))]
    return None if t.empty else t.drop(columns=["dataset", "country", "exclude_same_point"]).reset_index(drop=True)
