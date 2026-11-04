"""Visually distinct categorical colours for the dashboard.

Plotly's built-in qualitative palettes hold 10 colours and then repeat, so any
chart with more categories (13 brewery types, up to 30 clusters) silently gets
duplicate colours. This module generates a longer sequence in which every
colour is as far as possible from all the others in CIELAB (perceptual) space,
so each category is told apart at a glance and no colour is ever reused.

Colours are limited to mid-range lightness and strong chroma so they stay
visible on both light map tiles and dark backgrounds. Two neutral greys are
reserved for "Noise" and "Other" style groups and are kept well away from the
chromatic colours.

Pure numpy; no Streamlit or plotly import, so notebooks can use it too.
"""
from __future__ import annotations

import numpy as np

# One fixed colour per dataset version, used wherever raw / all_records / unique_records are compared.
# Blue, orange, aqua: validated as a set for colour-vision deficiency (worst all-pairs CVD ΔE 9.2).
DATASET_COLORS = {"raw": "#2a78d6", "all_records": "#eb6834", "unique_records": "#1baf7a"}

NOISE_COLOR = "#2e2e2e"   # charcoal: unclustered / noise records
OTHER_COLOR = "#a8a8a8"   # light grey: grouped remainder ("Other clusters")
MAX_COLORS = 48           # chromatic colours available

_L_RANGE = (30.0, 76.0)   # avoid near-white (invisible on maps) and near-black
_MIN_CHROMA = 40.0        # avoid washed-out colours that read as grey

_D65 = np.array([0.95047, 1.0, 1.08883])
_RGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])


def _hex_to_rgb(code: str) -> np.ndarray:
    code = code.lstrip("#")
    return np.array([int(code[i:i + 2], 16) for i in (0, 2, 4)], dtype=float) / 255.0


def _rgb_to_hex(rgb: np.ndarray) -> str:
    r, g, b = (int(round(float(v) * 255)) for v in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def _to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB in [0, 1], shape (n, 3) -> CIELAB (D65), shape (n, 3)."""
    rgb = np.atleast_2d(rgb)
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = (lin @ _RGB_TO_XYZ.T) / _D65
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.stack([116 * f[:, 1] - 16,
                     500 * (f[:, 0] - f[:, 1]),
                     200 * (f[:, 1] - f[:, 2])], axis=1)


def _build_sequence(count: int) -> list[str]:
    """Greedy maximin selection: every new colour maximises its distance to the
    nearest colour already chosen (including the two reserved greys). Because
    selection is incremental, any prefix of the result is itself well spread."""
    steps = np.linspace(0, 1, 18)
    grid = np.array(np.meshgrid(steps, steps, steps)).reshape(3, -1).T
    lab = _to_lab(grid)
    chroma = np.hypot(lab[:, 1], lab[:, 2])
    keep = (lab[:, 0] >= _L_RANGE[0]) & (lab[:, 0] <= _L_RANGE[1]) & (chroma >= _MIN_CHROMA)
    grid, lab = grid[keep], lab[keep]

    chosen = [_to_lab(_hex_to_rgb(NOISE_COLOR))[0], _to_lab(_hex_to_rgb(OTHER_COLOR))[0]]
    nearest = np.min([np.linalg.norm(lab - c, axis=1) for c in chosen], axis=0)

    sequence = []
    for _ in range(count):
        idx = int(np.argmax(nearest))
        sequence.append(_rgb_to_hex(grid[idx]))
        nearest = np.minimum(nearest, np.linalg.norm(lab - lab[idx], axis=1))
    return sequence


_SEQUENCE = _build_sequence(MAX_COLORS)


def distinct_colors(n: int) -> list[str]:
    """`n` mutually distinct hex colours (no repeats)."""
    if n < 0 or n > MAX_COLORS:
        raise ValueError(f"Can supply between 0 and {MAX_COLORS} distinct colours, got {n}.")
    return _SEQUENCE[:n]


def color_map(categories, fixed: dict | None = None) -> dict:
    """Map each category to its own colour, in the order given.

    `fixed` pins specific categories to a colour (e.g. {"Noise": NOISE_COLOR});
    all other categories take successive colours from the distinct sequence.
    Pass the *full* list of categories (not a filtered subset) to keep a
    category's colour identical across charts and filter selections.
    """
    fixed = fixed or {}
    seen = list(dict.fromkeys(str(c) for c in categories))
    free = [c for c in seen if c not in fixed]
    colours = distinct_colors(len(free))
    mapping = {c: fixed[c] for c in seen if c in fixed}
    mapping.update(zip(free, colours))
    return {c: mapping[c] for c in seen}


def min_pairwise_delta_e(colors) -> float:
    """Smallest CIELAB distance (ΔE76) between any two colours. Around 2.3 is
    a just-noticeable difference; above ~20 colours are clearly different."""
    lab = _to_lab(np.array([_hex_to_rgb(c) for c in colors]))
    d = np.linalg.norm(lab[:, None, :] - lab[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    return float(d.min())
