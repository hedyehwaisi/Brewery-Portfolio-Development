"""Colour regression tests: categorical colours must never repeat and must stay
clearly distinguishable, whatever the number of categories."""
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import palette  # noqa: E402
from geo_plots import SATURATED_SCALE, type_style  # noqa: E402

MIN_DELTA_E = 20.0  # ΔE76 above ~20 reads as clearly different colours


@pytest.mark.parametrize("n", [2, 13, 20, 32, palette.MAX_COLORS])
def test_colours_are_unique_and_distinguishable(n):
    colours = palette.distinct_colors(n)
    assert len(set(colours)) == n
    assert palette.min_pairwise_delta_e(colours + [palette.NOISE_COLOR, palette.OTHER_COLOR]) >= MIN_DELTA_E


def test_too_many_categories_is_an_error_not_a_repeat():
    with pytest.raises(ValueError):
        palette.distinct_colors(palette.MAX_COLORS + 1)


def test_fixed_groups_keep_their_colour_and_nothing_repeats():
    groups = [f"Cluster {i}" for i in range(30)] + ["Other clusters", "Noise"]
    mapping = palette.color_map(groups, fixed={"Noise": palette.NOISE_COLOR, "Other clusters": palette.OTHER_COLOR})
    assert list(mapping) == groups
    assert mapping["Noise"] == palette.NOISE_COLOR
    assert len(set(mapping.values())) == len(groups)


@pytest.mark.parametrize("dataset", ["all_records", "unique_records"])
def test_every_real_brewery_type_gets_its_own_colour(dataset):
    with sqlite3.connect(ROOT / "data" / f"analysis_{dataset}.db") as con:
        df = pd.read_sql_query(f"SELECT brewery_type FROM analysis_{dataset}", con)
    cmap, order = type_style(df)
    assert len(cmap) == df["brewery_type"].nunique() >= 11  # more than Plotly's 10-colour default
    assert len(set(cmap.values())) == len(cmap)
    assert set(order["brewery_type"]) == set(cmap)


def test_outlier_gradient_is_visible_on_openstreetmap_tiles():
    tiles = ["#f2efe9", "#aad3df", "#c8facc", "#cdebb0", "#f7fabf", "#fcd6a4", "#d9d0c9", "#e892a2"]
    for _, stop in SATURATED_SCALE:
        # distance from this stop to every tile colour (tile-vs-tile distances are irrelevant here)
        nearest = min(palette.min_pairwise_delta_e([stop, tile]) for tile in tiles)
        assert nearest >= 30, (stop, nearest)
