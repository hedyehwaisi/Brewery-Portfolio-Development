"""Pure pandas/plotly helpers shared by the notebook and the dashboard pages (no Streamlit import)."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from palette import color_map

# columns that define "the same brewery identity" at one coordinate
IDENTITY_COLUMNS = ["name", "city", "state_province", "country",
                    "address_1", "postal_code", "phone", "website_url"]


def build_coordinate_details(df):
    """One row per exact (latitude, longitude) shared by >=2 records.
    `same_identity` is True when every identity column is identical (missing values count as a difference)."""
    cols = [c for c in IDENTITY_COLUMNS if c in df.columns]
    shared = df[df.duplicated(["latitude", "longitude"], keep=False)]
    rows = []
    for (lat, lon), g in shared.groupby(["latitude", "longitude"]):
        diff = [c for c in cols if g[c].nunique(dropna=False) > 1]
        rows.append({
            "latitude": lat, "longitude": lon, "record_count": len(g),
            "ids": ", ".join(g["id"].astype(str)),
            "names": " | ".join(sorted(g["name"].astype(str).unique())[:4]),
            "city": g["city"].iloc[0], "state_province": g["state_province"].iloc[0], "country": g["country"].iloc[0],
            "differing_columns": ", ".join(diff), "n_differences": len(diff),
            "same_identity": not diff})
    return pd.DataFrame(rows)


def _edges(col, width):
    lo, hi = (-90, 90) if col == "latitude" else (-180, 180)
    return np.arange(lo, hi + width, width)


def band_table(df, col, width):
    counts, edges = np.histogram(df[col].dropna(), bins=_edges(col, width))
    return pd.DataFrame({"band_start": edges[:-1], "band_end": edges[1:], "records": counts,
                         "share_%": counts / counts.sum() * 100})


def band_map(df, col, width=5, scale="Viridis", highlight=None):
    """Distribution of latitude (horizontal bands) or longitude (vertical bands) drawn ON the map:
    each band is a rectangle coloured by how many breweries fall inside it. This is not a point map."""
    t = band_table(df, col, width)
    cmax = max(int(t["records"].max()), 1)
    colors = px.colors.sample_colorscale(scale, [r / cmax for r in t["records"]])
    fig = go.Figure()
    for (_, r), c in zip(t.iterrows(), colors):
        if r.records == 0:
            continue
        a, b = r.band_start, r.band_end
        lons, lats = ([-180, 180, 180, -180, -180], [a, a, b, b, a]) if col == "latitude" \
            else ([a, b, b, a, a], [-90, -90, 90, 90, -90])
        fig.add_trace(go.Scattergeo(lon=lons, lat=lats, mode="lines", fill="toself",
                                    fillcolor=c.replace("rgb(", "rgba(").replace(")", ", 0.6)"),
                                    line=dict(width=0.4, color="white"), hoverinfo="skip", showlegend=False))
    mid = (t["band_start"] + t["band_end"]) / 2
    hover = [f"{col} {a:.0f}° to {b:.0f}°<br>{n:,} records ({s:.1f}%)"
             for a, b, n, s in zip(t.band_start, t.band_end, t.records, t["share_%"])]
    fig.add_trace(go.Scattergeo(
        lon=mid if col == "longitude" else [0] * len(t), lat=mid if col == "latitude" else [0] * len(t),
        mode="markers", text=hover, hoverinfo="text",
        marker=dict(size=9, opacity=0, color=t["records"], colorscale=scale, cmin=0, cmax=cmax,
                    colorbar=dict(title="Records", thickness=12))))
    if highlight:  # selected range outlined in red
        a, b = highlight
        lons, lats = ([-180, 180, 180, -180, -180], [a, a, b, b, a]) if col == "latitude" \
            else ([a, b, b, a, a], [-90, -90, 90, 90, -90])
        fig.add_trace(go.Scattergeo(lon=lons, lat=lats, mode="lines", line=dict(width=3, color="#d62728"),
                                    hoverinfo="skip", showlegend=False))
    top = df.groupby("country").agg(n=("latitude", "size"), la=("latitude", "mean"), lo=("longitude", "mean")).nlargest(12, "n")
    fig.add_trace(go.Scattergeo(lon=top.lo, lat=top.la, mode="text", text=top.index, textfont=dict(size=10, color="#222"),
                                hoverinfo="skip", showlegend=False))  # country names
    fig.update_geos(projection_type="natural earth", showland=True, landcolor="#f2efe9",
                    showcountries=True, countrycolor="#bbb", showocean=True, oceancolor="#eaf3f8")
    fig.update_layout(title=f"{col.title()} distribution on the map ({width}° bands)",
                      margin=dict(l=0, r=0, t=45, b=0), height=430)
    return fig


def dist_plots(df, col, width=5):
    """Histogram with box marginal (oriented to match the map axis) + ECDF."""
    nb = max(int((df[col].max() - df[col].min()) / width) + 1, 5)
    kw = dict(y=col) if col == "latitude" else dict(x=col)
    hist = px.histogram(df, nbins=nb, marginal="box", title=f"{col.title()} histogram + box plot", **kw)
    ecdf = px.ecdf(df, x=col, title=f"{col.title()} cumulative distribution (ECDF)")
    return hist, ecdf


def type_style(full, col="brewery_type"):
    """(color_discrete_map, category_orders) for a categorical column such as brewery type.
    Every category gets its own colour (see palette.py), assigned alphabetically from the FULL dataset so a
    category keeps the same colour on every page and under every filter. Draw/legend order is by frequency
    (largest first), so the rare types are drawn on top instead of being hidden under the common ones."""
    values = full[col].fillna("Unknown").astype(str)
    cmap = color_map(sorted(values.unique()))
    order = values.value_counts().index.tolist()
    return cmap, {col: order}


def core_view(d, lat="latitude", lon="longitude", trim=0.01):
    """Rows inside the central (1 - 2*trim) share of latitudes and longitudes. Used to open a map on where most
    points are, instead of zooming way out to fit a few far-away records (e.g. Alaska/Hawaii for the US)."""
    if len(d) < 20:
        return d
    la, lo = d[lat].quantile([trim, 1 - trim]), d[lon].quantile([trim, 1 - trim])
    return d[d[lat].between(*la) & d[lon].between(*lo)]


HALO_COLOR = "#151515"


def add_halo(fig, ring=3, opacity=0.85):
    """Draw a dark ring under every marker of a tile map, so a point stays visible whatever the tile colour
    underneath (pale land, roads, parks, water) and whichever basemap is chosen.
    Works for fixed marker sizes and for plotly-express `size=` columns (area or diameter mode): the halo copy
    gets the same positions and a size `ring` pixels larger in diameter, and is drawn first (underneath)."""
    halo_cls = getattr(go, "Scattermap", None) or go.Scattermapbox
    halos = []
    for t in fig.data:
        if t.type not in ("scattermap", "scattermapbox"):
            continue
        m, extra = t.marker, {}
        size = 6 if m.size is None else m.size
        if np.isscalar(size):
            size = size + ring
        else:
            s = np.asarray(size, dtype=float)
            ref = m.sizeref or 1.0
            if m.sizemode == "area":  # diameter = sqrt(size / sizeref)
                size = ref * (np.sqrt(s / ref) + ring) ** 2
            else:  # diameter = size / sizeref
                size = s + ring * ref
            extra = dict(sizeref=ref, sizemode=m.sizemode or "diameter",
                         sizemin=(m.sizemin or 0) + ring)
        halos.append(halo_cls(lat=t.lat, lon=t.lon, mode="markers", hoverinfo="skip", showlegend=False,
                              marker=dict(size=size, color=HALO_COLOR, opacity=opacity, **extra)))
    n = len(fig.data)
    fig.add_traces(halos)
    fig.data = fig.data[n:] + fig.data[:n]  # halos first = drawn underneath the coloured markers
    return fig


def point_map(d, lat="latitude", lon="longitude", title="", height=520, style="open-street-map",
              marker_size=7, view=None, halo=True, **kw):
    """Tile map with place names: both basemaps offered in the sidebar (OpenStreetMap and CARTO Positron) draw
    country and state/province borders and label countries, states/provinces and cities as you zoom.
    `view` (optional DataFrame) sets the initial centre/zoom, so all of `d` can be plotted while the map opens
    on a chosen region. `halo` rings every marker in dark grey (see `add_halo`).
    Extra keyword arguments go to plotly express (color, size, hover_name, ...)."""
    scatter, style_arg = getattr(px, "scatter_map", None), "map_style"
    if scatter is None:
        scatter, style_arg = px.scatter_mapbox, "mapbox_style"
    v = d if view is None or view.empty else view
    lat_span = max(v[lat].max() - v[lat].min(), 0.01)
    lon_span = max(v[lon].max() - v[lon].min(), 0.01)
    zoom = float(np.clip(np.log2(700 / max(lon_span, lat_span * 1.6)), 0.8, 13))
    centre = dict(lat=(v[lat].max() + v[lat].min()) / 2, lon=(v[lon].max() + v[lon].min()) / 2)
    fig = scatter(d, lat=lat, lon=lon, zoom=zoom, center=centre, title=title, height=height, **{style_arg: style}, **kw)
    if "size" not in kw:
        fig.update_traces(marker_size=marker_size)
    fig.update_layout(margin=dict(l=0, r=0, t=45 if title else 0, b=0), legend=dict(bgcolor="rgba(255,255,255,0.8)"))
    return add_halo(fig) if halo else fig


# Continuous scale for values drawn on a tile map: deep violet -> magenta -> hot pink -> vermilion. These
# saturated hues are the opposite of the pale tan / green / blue / grey basemap tiles, no stop is pale, and
# lightness rises monotonically (OKLab L 0.39 -> 0.65), so the low end never fades into the map.
SATURATED_SCALE = [[0.0, "#3d00b8"], [0.35, "#a100c8"], [0.7, "#ff0f7b"], [1.0, "#ff3b00"]]


def outlier_map(d, value_col, title="", height=640, hover_name="name", hover_data=None, style="open-street-map"):
    """Map of flagged outliers coloured by `value_col` (a distance in km) on a log scale, with larger markers
    for larger distances and a dark halo under every marker so points stay visible on any tile colour.
    Distances are heavy-tailed (median ~300 km, max ~15,000 km), so a linear scale would paint almost every
    point the same colour; the colour bar is labelled in km."""
    d = d.copy()
    d["_log"] = np.log10(d[value_col].clip(lower=1))
    lo, hi = float(d["_log"].min()), float(d["_log"].max())
    hi = hi if hi > lo else lo + 1e-6
    sizes = 10 + 14 * (d["_log"] - lo) / (hi - lo)
    hover = {c: True for c in (hover_data or []) if c != value_col}
    hover.update({value_col: ":,.0f", "_log": False})
    fig = point_map(d, color="_log", hover_name=hover_name, hover_data=hover, range_color=(lo, hi),
                    color_continuous_scale=SATURATED_SCALE, title=title, height=height, style=style, opacity=1.0,
                    halo=False)
    fig.update_traces(marker=dict(size=sizes.to_numpy(), sizemode="diameter"))
    ticks = [t for t in (1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000) if lo - 0.15 <= np.log10(t) <= hi + 0.15]
    fig.update_layout(coloraxis_colorbar=dict(title="Distance from<br>state center (km)", thickness=16,
                                              tickvals=[float(np.log10(t)) for t in ticks],
                                              ticktext=[f"{t:,}" for t in ticks]))
    return add_halo(fig, ring=6, opacity=0.9)


SWAP_COLORS = {"As stored in the dataset": "#eb6834", "Latitude and longitude swapped back": "#2a78d6"}


def pair_map(d, start, end, labels, height=520, colors=("#eb6834", "#2a78d6"), symbols=("x", "circle"),
             hover_cols=("city", "state_province", "country")):
    """Two positions per record on a globe projection, joined by a dotted line: `start` and `end` are
    (latitude column, longitude column) pairs and `labels` their legend names. A globe is used because
    Web-Mercator tile maps stop at +-85.05 degrees latitude. Country and state/province borders are drawn; each
    marker is labelled with the record's name."""
    fig = go.Figure()
    for _, r in d.iterrows():
        fig.add_trace(go.Scattergeo(lat=[r[start[0]], r[end[0]]], lon=[r[start[1]], r[end[1]]], mode="lines",
                                    hoverinfo="skip", showlegend=False,
                                    line=dict(width=1.2, color="#8a8a8a", dash="dot")))
    for label, (la, lo), color, symbol in zip(labels, [start, end], colors, symbols):
        fig.add_trace(go.Scattergeo(
            lat=d[la], lon=d[lo], mode="markers+text", name=label, text=d["name"], textposition="middle right",
            textfont=dict(size=11, color="#222"), customdata=d[list(hover_cols)].to_numpy(),
            hovertemplate=("<b>%{text}</b><br>%{customdata[0]}, %{customdata[1]}, %{customdata[2]}"
                           "<br>lat %{lat:.5f}, lon %{lon:.5f}<extra>" + label + "</extra>"),
            marker=dict(size=13, color=color, symbol=symbol, line=dict(width=1.5, color=HALO_COLOR))))
    fig.update_geos(projection_type="natural earth", showland=True, landcolor="#f2efe9", showocean=True,
                    oceancolor="#eaf3f8", showcountries=True, countrycolor="#8f8f8f", showsubunits=True,
                    subunitcolor="#bdbdbd", showlakes=False)
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=10, b=0),
                      legend=dict(orientation="h", yanchor="bottom", y=0.0, x=0.01, bgcolor="rgba(255,255,255,0.85)"))
    return fig


def swap_map(d, height=520):
    """Both positions of records whose latitude and longitude look swapped: the stored position and the position
    after swapping the two values back (`latitude_swapped`, `longitude_swapped`)."""
    return pair_map(d, ("latitude", "longitude"), ("latitude_swapped", "longitude_swapped"), list(SWAP_COLORS),
                    height=height, colors=tuple(SWAP_COLORS.values()))
