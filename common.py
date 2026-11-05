"""Shared Streamlit helpers: cached SQLite reads and the dataset selector."""
from functools import wraps
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
import streamlit as st

from quality_core import DATASET_FILES  # the three dataset versions as (database file, table name)

DATA = Path(__file__).resolve().parent / "data"
LABEL = {"all_records": "All records", "unique_records": "Unique records"}


def _private(obj):
    """What a caller of `shared_cache` receives: frames and series as shallow copies (pandas copy-on-write shares the
    data until either side writes, so a caller can never change the cached object), arrays as copies, and dicts,
    lists and tuples rebuilt around them. Other objects are returned as they are and must be treated as read-only."""
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        return obj.copy(deep=False)
    if isinstance(obj, np.ndarray):
        return obj.copy()
    if isinstance(obj, dict):
        return {k: _private(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_private(v) for v in obj)
    return obj


def shared_cache(func=None, *, max_entries=None, show_spinner=True):
    """`st.cache_resource` for read-only results. One copy is kept per argument set and shared by every session,
    instead of `st.cache_data`'s pickled copy plus a fresh unpickled copy per call; on a 512 MB host that difference
    decides whether the app fits. Callers get `_private` copies, so writing to a result stays local."""
    def decorate(f):
        cached = st.cache_resource(max_entries=max_entries, show_spinner=show_spinner)(f)

        @wraps(f)
        def wrapper(*args, **kwargs):
            return _private(cached(*args, **kwargs))
        wrapper.clear = cached.clear
        return wrapper
    return decorate(func) if func is not None else decorate


@shared_cache
def table(db, name):
    p = DATA / db
    if not p.exists():
        return None
    with sqlite3.connect(p) as con:
        try:
            return pd.read_sql_query(f'SELECT * FROM "{name}"', con)
        except Exception:
            return None


def stats_table(name):
    """A table of data/dashboard_stats.db: the statistics that need SciPy or scikit-learn, precomputed by
    brewery_dashboard_statistics.ipynb (see stats_store) so the app never loads either library."""
    t = table("dashboard_stats.db", name)
    if t is None:
        st.error("data/dashboard_stats.db is missing or incomplete: run brewery_dashboard_statistics.ipynb.")
        st.stop()
    return t


def pick_dataset():
    return st.sidebar.radio("Dataset", list(LABEL), format_func=LABEL.get)


# Free tile basemaps (no API key). Both draw country and state/province borders and label countries,
# states/provinces and cities as you zoom in.
BASEMAPS = {"OpenStreetMap": "open-street-map", "CARTO Positron (light grey)": "carto-positron"}

# Data and basemap credits required by the upstream licences (MIT, CC BY 4.0, ODbL, CARTO basemap terms).
# Shown in the sidebar of every page through pick_basemap(); the repository holds the full text in NOTICE.md.
CREDITS = ("Data: [Open Brewery DB](https://www.openbrewerydb.org/) (MIT) · [GeoNames](https://www.geonames.org) "
           "(CC BY 4.0) · © [OpenStreetMap](https://www.openstreetmap.org/copyright) contributors (ODbL) · "
           "© [CARTO](https://carto.com/attributions). Analysis © 2026 Hedyeh Waisi, "
           "[MIT / CC BY 4.0](https://github.com/hedyehwaisi/Brewery-Portfolio-Development).")


def pick_basemap():
    """Sidebar basemap switch. The choice is kept in session state so it follows the user across pages."""
    names = list(BASEMAPS)
    current = st.session_state.get("basemap", names[0])
    choice = st.sidebar.radio("Basemap", names, index=names.index(current), key="_basemap_widget",
                              help="OpenStreetMap shows roads, parks and water in colour; CARTO Positron is a quiet "
                                   "grey map. Markers carry a dark ring so they stay visible on either.")
    st.session_state["basemap"] = choice
    st.sidebar.caption(CREDITS)
    return BASEMAPS[choice]


@shared_cache
def analysis(key, complete=True, corrected=True):
    """An analysis dataset with numeric coordinates. `corrected` applies data/coordinate_corrections.csv (swapped
    coordinates put back, a wrong coordinate sign reversed, a misplaced duplicate fixed, far-off records flagged); the
    database is never changed."""
    import quality_core as qc
    df = table(f"analysis_{key}.db", f"analysis_{key}")
    if df is None:
        st.error(f"data/analysis_{key}.db not found."); st.stop()
    df = df.copy()
    for c in ["latitude", "longitude"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if corrected:
        df = qc.apply_corrections(df)
    return df.dropna(subset=["latitude", "longitude"]) if complete else df


DATASET_HELP = ("raw keeps every source record, including the 2,468 without coordinates, so it gives the most "
                "complete counts; all_records and unique_records keep only geocoded records (the datasets used for "
                "maps, distances and clustering).")


def pick_any_dataset(default="all_records"):
    """Sidebar choice among raw, all_records and unique_records (for analyses that do not need coordinates)."""
    names = list(DATASET_FILES)
    return st.sidebar.radio("Dataset", names, index=names.index(default), help=DATASET_HELP)


@shared_cache
def harmonized(name, corrected=True):
    """A dataset with numeric coordinates and `state_province_harmonized` added (see quality_core). `corrected`
    applies the coordinate corrections log, as in `analysis`."""
    import quality_core as qc
    df = table(*DATASET_FILES[name])
    if df is None:
        st.error(f"data/{DATASET_FILES[name][0]} not found."); st.stop()
    return qc.prepare(df, corrected=corrected)
