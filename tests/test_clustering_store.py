"""Tests for the read-only clustering results layer.

Run from the project root:  python -m pytest tests -q
The stored runs themselves are checked in tests/test_allruns.py; these tests cover the access layer.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import clustering_store as cs  # noqa: E402


def test_required_tables_present():
    assert cs.missing_tables() == []


def test_old_candidate_label_tables_are_gone():
    # the per-candidate label tables of the first notebook were replaced by allruns_labels
    names = cs.table_names()
    assert not any(n.startswith(("candidate_labels_", "selected_labels_")) for n in names)


def test_signature_changes_with_the_file(tmp_path):
    f = tmp_path / "x.db"
    assert cs.signature(f) == (0, 0)
    f.write_bytes(b"abc")
    assert cs.signature(f)[1] == 3


def test_unknown_inputs_are_rejected():
    with pytest.raises(ValueError):
        cs.allruns_runs("nope")
    with pytest.raises(cs.ResultsUnavailable):
        cs.allruns_labels("all_records|DBSCAN|eps=1|ms=1")
    with pytest.raises(cs.ResultsUnavailable):
        cs.allruns_hdbscan_extras("all_records|DBSCAN|eps=45|ms=15")


def test_database_is_opened_read_only():
    with pytest.raises(Exception):
        with cs._connect(cs.DB_PATH) as con:
            con.execute("CREATE TABLE should_fail (x INTEGER)")
