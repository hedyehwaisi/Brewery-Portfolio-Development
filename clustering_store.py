"""Read-only access layer for the clustering results saved by brewery_clustering_all_runs.ipynb.

The dashboard never fits DBSCAN, HDBSCAN or OPTICS itself. Every label, metric and tree it shows comes from the
allruns_* tables of ``data/clustering_results.db``, which the notebook writes and verifies in one run, so the
dashboard always shows exactly what the notebook produced and page loads stay fast.

This module is deliberately free of Streamlit and scikit-learn so it can be unit-tested and reused from scripts.
Caching is applied by the caller.
"""
from __future__ import annotations

from contextlib import closing
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
DB_PATH = DATA_DIR / "clustering_results.db"

DATASETS = {"all_records": "All records", "unique_records": "Unique records"}
ALGORITHMS = ("DBSCAN", "HDBSCAN", "OPTICS")


class ResultsUnavailable(RuntimeError):
    """The stored clustering results are missing or incomplete."""


# --------------------------------------------------------------------------
# Low-level helpers
# --------------------------------------------------------------------------
def _connect(path: Path) -> sqlite3.Connection:
    """Open the database read-only, so the dashboard can never modify it."""
    if not path.exists():
        raise ResultsUnavailable(
            f"{path.name} was not found in {path.parent}. Run the clustering "
            "notebook to generate it."
        )
    # as_uri() percent-encodes spaces etc., which SQLite URI filenames require.
    return sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)


def _query(sql: str, params=(), path: Path = DB_PATH) -> pd.DataFrame:
    with closing(_connect(path)) as con:
        return pd.read_sql_query(sql, con, params=params)


def signature(path: Path = DB_PATH) -> tuple[int, int]:
    """(mtime, size) of the results file; used as a cache key so that
    regenerating the database automatically invalidates cached results."""
    if not path.exists():
        return (0, 0)
    stat = path.stat()
    return (stat.st_mtime_ns, stat.st_size)


def table_names(path: Path = DB_PATH) -> set[str]:
    return set(_query("SELECT name FROM sqlite_master WHERE type='table'", path=path)["name"])


def missing_tables(path: Path = DB_PATH) -> list[str]:
    """Tables the dashboard needs that are not in the database."""
    return allruns_missing(path)


def _check_dataset(dataset: str) -> None:
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}; expected one of {list(DATASETS)}")


# --------------------------------------------------------------------------
# Every run (brewery_clustering_all_runs.ipynb): allruns_* tables
# --------------------------------------------------------------------------
ALLRUNS_TABLES = ("allruns_records", "allruns_runs", "allruns_labels", "allruns_hdbscan_extras", "allruns_optics",
                  "allruns_dbscan_star", "allruns_hdbscan_check", "allruns_settings", "allruns_verification")


def allruns_missing(path: Path = DB_PATH) -> list[str]:
    return [t for t in ALLRUNS_TABLES if t not in table_names(path)]


def allruns_runs(dataset: str | None = None, path: Path = DB_PATH) -> pd.DataFrame:
    """One row per stored run: parameters, cluster metrics, geography summary and stability."""
    if dataset is None:
        return _query("SELECT * FROM allruns_runs", path=path)
    _check_dataset(dataset)
    return _query("SELECT * FROM allruns_runs WHERE dataset = ?", (dataset,), path=path)


def allruns_settings(path: Path = DB_PATH) -> dict:
    return dict(_query("SELECT key, value FROM allruns_settings", path=path).itertuples(index=False))


def allruns_records(dataset: str, path: Path = DB_PATH) -> pd.DataFrame:
    """Record ids in the order of every stored label array of that dataset."""
    _check_dataset(dataset)
    return _query("SELECT position, id, coordinate_status FROM allruns_records WHERE dataset = ? ORDER BY position",
                  (dataset,), path=path)


def _blob(sql: str, params, column: str, dtype, path: Path):
    import zlib
    with closing(_connect(path)) as con:
        row = con.execute(sql, params).fetchone()
    if row is None:
        raise ResultsUnavailable(f"No stored {column} for {params}.")
    return np.frombuffer(zlib.decompress(row[0]), dtype=dtype)


def allruns_labels(run_id: str, path: Path = DB_PATH) -> np.ndarray:
    """Cluster label of every record (-1 = noise), in allruns_records order."""
    return _blob("SELECT labels FROM allruns_labels WHERE run_id = ?", (run_id,), "labels", np.int16,
                 path).astype(int)


def allruns_hdbscan_extras(run_id: str, path: Path = DB_PATH) -> dict:
    """Membership probability and GLOSH outlier score per record, persistence per cluster, and the cluster-level
    condensed tree (one row per node) of one HDBSCAN run."""
    import zlib
    with closing(_connect(path)) as con:
        row = con.execute("SELECT probabilities, glosh, persistence, condensed_columns, condensed "
                          "FROM allruns_hdbscan_extras WHERE run_id = ?", (run_id,)).fetchone()
    if row is None:
        raise ResultsUnavailable(f"{run_id} is not a stored HDBSCAN run.")
    cols = row[3].split(",")
    tree = np.frombuffer(zlib.decompress(row[4]), dtype=np.float32).reshape(-1, len(cols))
    return {"probabilities": np.frombuffer(zlib.decompress(row[0]), np.uint8) / 255.0,
            "glosh": np.frombuffer(zlib.decompress(row[1]), np.uint8) / 255.0,
            "persistence": np.frombuffer(zlib.decompress(row[2]), np.float32).astype(float),
            "condensed": pd.DataFrame(tree, columns=cols)}


def allruns_optics(dataset: str, min_samples: int, path: Path = DB_PATH) -> pd.DataFrame:
    """Reachability plot data of one OPTICS fit: records in OPTICS order with reachability and core distance (km)."""
    import zlib
    _check_dataset(dataset)
    with closing(_connect(path)) as con:
        row = con.execute("SELECT ordering, reachability_km, core_distance_km FROM allruns_optics "
                          "WHERE dataset = ? AND min_samples = ?", (dataset, int(min_samples))).fetchone()
    if row is None:
        raise ResultsUnavailable(f"No OPTICS fit stored for {dataset}, min_samples={min_samples}.")
    order = np.frombuffer(zlib.decompress(row[0]), np.int16).astype(int)
    reach = np.frombuffer(zlib.decompress(row[1]), np.float32).astype(float)
    core = np.frombuffer(zlib.decompress(row[2]), np.float32).astype(float)
    return pd.DataFrame({"order": np.arange(len(order)), "position": order, "reachability_km": reach[order],
                         "core_distance_km": core[order]})


def allruns_core_distance(dataset: str, min_samples: int, path: Path = DB_PATH) -> np.ndarray:
    """Distance to the min_samples-th neighbour of every record (allruns_records order): a local density estimate."""
    o = allruns_optics(dataset, min_samples, path)
    return o.sort_values("position")["core_distance_km"].to_numpy()


def allruns_dbscan_star(dataset: str, path: Path = DB_PATH) -> pd.DataFrame:
    _check_dataset(dataset)
    return _query("SELECT * FROM allruns_dbscan_star WHERE dataset = ?", (dataset,), path=path)


def allruns_hdbscan_check(path: Path = DB_PATH) -> pd.DataFrame:
    return _query("SELECT * FROM allruns_hdbscan_check", path=path)


def allruns_verification(path: Path = DB_PATH) -> pd.DataFrame:
    return _query("SELECT * FROM allruns_verification", path=path)
