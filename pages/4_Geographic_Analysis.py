import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import analysis_core as ac
import quality_core as qc
import origins_core as oc
import stats_store
from common import pick_dataset, pick_basemap, harmonized, LABEL, DATASET_FILES, table, shared_cache, stats_table
from geo_plots import point_map, outlier_map, pair_map, SATURATED_SCALE, HALO_COLOR
from palette import DATASET_COLORS

# Explanations written in the geographic-analysis notebook, shown at the top of each tab (headings demoted).
NOTEBOOK_TEXT = {
    'extent': '#### Geographic coordinate extent\n\nThe bounding box gives a first description of the geographic footprint of the dataset:\n\n- minimum and maximum latitude describe the north–south extent\n- minimum and maximum longitude describe the east–west extent\n\nThis is descriptive rather than a measure of actual occupied area. A large bounding box does not necessarily mean breweries are evenly distributed throughout it.',
    'centers': '#### Geographic Centers\n\nA geographic center is calculated as the arithmetic mean of latitude and longitude.\n\nWe use centers at multiple geographic levels:\n\n- **overall dataset center**\n- **country center**\n- **state/province center**\n- **city center**\n\nThese centers are descriptive reference points. They are not necessarily the population center, geographic centroid of a polygon, or administrative center.\n\nFor distance calculations, the center is treated as a reference point and actual great-circle distance is calculated with the Haversine formula.',
    'largest_groups': '#### Inspect the largest geographic groups\n\nGroup centers are most useful when combined with record counts.\n\nA center based on two records should not be interpreted with the same confidence as a center based on hundreds of records. Therefore, record count is retained beside every geographic center.',
    'distance': '#### Distance from Center\n\nThe next question is not only **where the center is**, but **how far individual breweries are from that center**.\n\nFor each record we calculate:\n\n- distance from the overall dataset center\n- distance from its country center\n- distance from its state/province center\n- distance from its city center\n\nThe most useful measure for studying within-region geographic spread is usually **distance from the state/province center**, while distance from the city center describes local positioning.\n\nDistances are calculated in kilometers using the Haversine formula rather than Euclidean distance on raw latitude/longitude degrees.',
    'dispersion': '#### 7.4 Spatial Dispersion\n\nDistance distributions allow us to compare how geographically concentrated different regions are.\n\nFor each country and state/province, we summarize distance from the relevant geographic center using:\n\n- count\n- mean\n- median\n- 25th percentile\n- 75th percentile\n- 90th percentile\n- maximum\n\nInterpretation:\n\n- **low median / low upper percentiles:** breweries tend to be close to the center\n- **large mean relative to median:** the distribution may have a long geographic tail\n- **large 90th percentile:** a substantial share of breweries are spread far from the center\n- **large maximum:** there may be an extreme geographic outlier\n\nThe percentiles are important because a single maximum can exaggerate dispersion.',
    'compare_dispersion': '#### Compare spatial dispersion between states\n\nThe ranking below identifies states/provinces whose breweries are geographically more spread out around their own center.\n\nA minimum record threshold can be applied when interpreting these rankings so that a state with only a few breweries does not dominate the analysis simply because of a small sample.',
    'outliers': '#### 7.5 Geographic Outliers\n\nA brewery can be geographically unusual in two different ways:\n\n1. It can be far from the center of its state/province.\n2. It can belong to a state/province whose overall spatial distribution is unusually dispersed.\n\nThese should not be confused.\n\nFor **record-level outliers**, we use an IQR-based rule on distance from the state/province center:\n\n> outlier threshold = Q3 + 1.5 × IQR\n\nThis is a descriptive rule, not a claim that an observation is erroneous. A geographically distant brewery may be completely legitimate and may reveal an important edge of the brewery distribution.\n\nWe therefore keep the original record information so that potential outliers can be investigated rather than automatically removed.',
    'visualizations': '#### Geographic Analysis Visualizations\n\nThe following visualizations focus on interpretation rather than producing a large number of plots.\n\nThey answer three useful questions:\n\n1. How are breweries distributed by distance from their state center?\n2. Which states have the greatest spatial dispersion?\n3. Are there individual records that sit unusually far from their state center?',
}

ORIGIN_COLORS = {"brewery center (origin 1)": "#2a78d6", "capital / city center (origin 2)": "#eb6834"}
LEVEL_NAMES = {"country": "Country", "state": "State / province", "city": "City"}
STATE = oc.STATE_COL

st.set_page_config(page_title="Geographic Analysis", page_icon="🌍", layout="wide")
st.title("🌍 Geographic analysis — coverage, centers, distance, dispersion, capitals, outliers")
key = pick_dataset()
basemap = pick_basemap()
st.caption("Computed live from the analysis databases. States use harmonized labels; cities are keyed by country + "
           "state + city. Origin 1 = the mean position of a place's own breweries; origin 2 = its national capital, "
           "state/province capital or GeoNames city center (data/reference_places.db).")
with st.expander("What this page answers", expanded=False):
    st.markdown(NOTEBOOK_TEXT["visualizations"])
    st.markdown("Beyond those three, the page compares every brewery with two kinds of origin at three levels "
                "(country, state/province, city), tests whether capitals act as brewery centers, and checks every "
                "coordinate against the city it claims to be in.")


@shared_cache
def reference():
    return oc.load_reference()


@shared_cache
def build(k, corrected=True):
    """Dataset with both origins and per-state outlier flags. corrected=False uses the coordinates as recorded."""
    d = oc.add_origins(harmonized(k, corrected=corrected), reference())
    return oc.flag_outliers(d, "state", "mean")


ref = reference()
d = build(key)
d_orig = build(key, corrected=False)
chk = oc.coordinate_consistency(d_orig, ref)  # the check runs on the coordinates as recorded
log = qc.load_corrections()
other = "unique_records" if key == "all_records" else "all_records"
t = st.tabs(["Coverage & extent", "Centers: origin 1 vs origin 2", "Distance from center", "Spatial dispersion",
             "Capitals & central cities", "Outliers & coordinate checks"])

# ======================================================================= 1. coverage & extent
with t[0]:
    st.markdown(NOTEBOOK_TEXT["extent"])
    cov = pd.DataFrame({name: ac.coverage(table(*DATASET_FILES[name]).assign(
        latitude=lambda x: pd.to_numeric(x["latitude"], errors="coerce"),
        longitude=lambda x: pd.to_numeric(x["longitude"], errors="coerce"))) for name in DATASET_FILES}).T
    st.subheader("Coordinate coverage"); st.dataframe(cov.round(2), width="stretch")
    n_fixed = int(log["action"].isin(qc.FIXED_ACTIONS).sum())
    ext = pd.DataFrame({f"{key} (as recorded)": ac.extent(d_orig),
                        f"{key} (corrected, used by every analysis)": ac.extent(d)}).T
    st.subheader("Extent, validation and overall center"); st.dataframe(ext.round(4), width="stretch")
    fig = go.Figure()
    for (name, row), color in zip(ext.iterrows(), ["#eb6834", "#2a78d6"]):
        la0, la1, lo0, lo1 = row["latitude_min"], row["latitude_max"], row["longitude_min"], row["longitude_max"]
        fig.add_trace(go.Scattergeo(lat=[la0, la0, la1, la1, la0], lon=[lo0, lo1, lo1, lo0, lo0], mode="lines",
                                    name=f"Bounding box: {name}", line=dict(width=2.5, color=color)))
        fig.add_trace(go.Scattergeo(lat=[row["center_latitude"]], lon=[row["center_longitude"]], mode="markers",
                                    name=f"Overall center: {name}",
                                    marker=dict(size=12, color=color, line=dict(width=1.5, color=HALO_COLOR))))
    fig.add_trace(go.Scattergeo(lat=d["latitude"], lon=d["longitude"], mode="markers", name="Breweries",
                                marker=dict(size=3, color="#555", opacity=0.35), hoverinfo="skip"))
    fig.update_geos(projection_type="natural earth", showland=True, landcolor="#f2efe9", showocean=True,
                    oceancolor="#eaf3f8", showcountries=True, countrycolor="#8f8f8f")
    fig.update_layout(height=480, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=-0.05))
    st.plotly_chart(fig, width="stretch", config={"scrollZoom": True})

    st.divider()
    st.subheader("Questions answered in this tab")
    e0, e1 = ext.iloc[0], ext.iloc[1]
    st.markdown(f"""
**How complete are the coordinates?**\\
Raw keeps {cov.loc['raw', 'record_coverage_pct']:.1f}% of its records with both coordinates; the analysis datasets keep only those, so they are 100% complete by construction.

**How large is the footprint, and what sets its edges?**\\
As recorded, latitudes run from **{e0['latitude_min']:.2f}°** to **{e0['latitude_max']:.2f}°**. That southern edge was not a brewery at all but a record with latitude and longitude swapped (see *Outliers & coordinate checks*). With the {n_fixed} corrections applied the box runs from **{e1['latitude_min']:.2f}°** to **{e1['latitude_max']:.2f}°**, and the overall center moves {float(oc.haversine_km(e0['center_latitude'], e0['center_longitude'], e1['center_latitude'], e1['center_longitude'])):.0f} km. A bounding box is decided by its most extreme record, which is exactly why it is a poor measure of where breweries are.

**Where is the overall center?**\\
At {e1['center_latitude']:.2f}°, {e1['center_longitude']:.2f}°, in the Atlantic: the average of a US-heavy cloud and a European one. It describes the mix of the sample, not a place where breweries are.
""")

# ======================================================================= 2. centers
with t[1]:
    st.markdown(NOTEBOOK_TEXT["centers"])
    st.markdown(NOTEBOOK_TEXT["largest_groups"])
    st.markdown("**Two origins.** Next to this *brewery center* (origin 1), every place also has a *reference point* "
                "(origin 2): its national capital, its state/province capital, or its GeoNames city center. Comparing "
                "the two shows whether the breweries are centred where the place itself is centred.")
    c1, c2 = st.columns(2)
    level = c1.radio("Level", list(LEVEL_NAMES), format_func=LEVEL_NAMES.get, horizontal=True, index=1, key="cen_lv")
    n = c2.slider("Minimum records", 1, 100, 5, key="cen_n")
    keys = oc.KEYS[level]
    cen = d.groupby(keys).agg(record_count=("id", "size"), center_latitude=("latitude", "mean"),
                              center_longitude=("longitude", "mean"),
                              reference=(f"{level}_reference", "first"),
                              reference_latitude=(f"{level}_ref_latitude", "first"),
                              reference_longitude=(f"{level}_ref_longitude", "first")).reset_index()
    cen["km_between_origins"] = oc.haversine_km(cen["center_latitude"], cen["center_longitude"],
                                                cen["reference_latitude"], cen["reference_longitude"])
    cen["place"] = cen[keys[::-1]].astype(str).agg(", ".join, axis=1)
    cen = cen[cen["record_count"] >= n].sort_values("record_count", ascending=False)
    pts = pd.concat([
        cen.assign(origin="brewery center (origin 1)", lat=cen["center_latitude"], lon=cen["center_longitude"],
                   label=cen["place"]),
        cen.dropna(subset=["reference_latitude"]).assign(origin="capital / city center (origin 2)",
                                                         lat=lambda x: x["reference_latitude"],
                                                         lon=lambda x: x["reference_longitude"],
                                                         label=lambda x: x["reference"].astype(str))])
    m = point_map(pts, lat="lat", lon="lon", color="origin", size="record_count", color_discrete_map=ORIGIN_COLORS,
                  hover_name="label", hover_data={"place": True, "record_count": ":,", "km_between_origins": ":,.0f",
                                                  "lat": False, "lon": False},
                  size_max=30, opacity=0.9, height=560, style=basemap,
                  title=f"{LEVEL_NAMES[level]}: brewery center vs reference point (lines join the two; size = records)")
    line_cls = getattr(go, "Scattermap", None) or go.Scattermapbox
    lines = cen.dropna(subset=["reference_latitude"]).head(300)
    lat_l, lon_l = [], []
    for _, r in lines.iterrows():
        lat_l += [r["center_latitude"], r["reference_latitude"], None]
        lon_l += [r["center_longitude"], r["reference_longitude"], None]
    m.add_trace(line_cls(lat=lat_l, lon=lon_l, mode="lines", hoverinfo="skip", showlegend=False,
                         line=dict(width=1.2, color="#666")))
    m.update_layout(legend_title_text="")
    st.plotly_chart(m, width="stretch", config={"scrollZoom": True})
    st.dataframe(cen.drop(columns=["place"]).round(4), hide_index=True, width="stretch", height=300)
    st.caption("Centers built from few records are weak — read them with record_count.")

    st.divider()
    st.subheader("Questions answered in this tab")
    sc = d.groupby(oc.KEYS["state"]).agg(n=("id", "size"), lat=("latitude", "mean"), lon=("longitude", "mean"),
                                         ref=("state_reference", "first"), rlat=("state_ref_latitude", "first"),
                                         rlon=("state_ref_longitude", "first")).reset_index()
    sc["gap"] = oc.haversine_km(sc["lat"], sc["lon"], sc["rlat"], sc["rlon"])
    big = sc[sc["n"] >= 30].dropna(subset=["gap"])
    far = big.nlargest(3, "gap")
    near = big.nsmallest(3, "gap")
    w50 = oc.dispersion(d, "state", "mean")[oc.KEYS["state"] + ["within_50km_%", "record_count"]]
    empty = w50[(w50["record_count"] >= 100)].nsmallest(1, "within_50km_%").iloc[0]
    ct_ = oc.city_table(d[(d["country"] == empty["country"]) & (d[STATE] == empty[STATE])], ref)
    empty_town = ct_.sort_values("km_to_state_center").iloc[0]["city"]
    st.markdown(f"""
**How far apart are the two origins?**\\
For states with at least 30 records, the brewery center sits a median **{big['gap'].median():.0f} km** from the state capital. Closest: {", ".join(f"{r[STATE]} ({r['gap']:.0f} km from {r['ref']})" for _, r in near.iterrows())}. Farthest: {", ".join(f"{r[STATE]} ({r['gap']:.0f} km from {r['ref']})" for _, r in far.iterrows())}.

**What does a large gap mean?**\\
Either the capital is not where the breweries are, or the breweries form several separate groups whose average falls between them. **{empty[STATE]}** is the clearest case of the second: only **{empty['within_50km_%']:.0f}%** of its {int(empty['record_count'])} breweries lie within 50 km of its brewery center, whose nearest brewery town is {empty_town}. The mean lands between hubs, not on one. A mean center is only a good summary for a place whose breweries form one group.
""")

# ======================================================================= 3. distance from center
with t[2]:
    st.markdown(NOTEBOOK_TEXT["distance"])
    rows = []
    for lv in LEVEL_NAMES:
        for o in ["mean", "reference"]:
            x = d[oc.DIST[(lv, o)]].dropna()
            rows.append({"level": lv, "origin": oc.ORIGIN_NAME[o][lv], "records": len(x), "median_km": x.median(),
                         "p90_km": x.quantile(.9), **{f"within_{r}km_%": (x <= r).mean() * 100 for r in (5, 25, 50, 100)}})
    summary = pd.DataFrame(rows)
    st.markdown("**Scattered or concentrated? Distance to each origin at each level**")
    st.dataframe(summary.round(1), hide_index=True, width="stretch")

    c1, c2 = st.columns(2)
    lv = c1.radio("Level", list(LEVEL_NAMES), format_func=LEVEL_NAMES.get, horizontal=True, index=1, key="dist_lv")
    origin = c2.radio("Origin", ["mean", "reference"], format_func=lambda o: oc.ORIGIN_NAME[o][lv], horizontal=True,
                      key="dist_origin")
    col = oc.DIST[(lv, origin)]
    both = pd.concat([build(k)[[col]].assign(dataset=k) for k in LABEL]).dropna()
    f = px.histogram(both, x=col, color="dataset", barmode="overlay", nbins=80, log_y=True,
                     color_discrete_map=DATASET_COLORS, title=f"Distance to the {oc.ORIGIN_NAME[origin][lv]} (log count)")
    f.update_layout(xaxis_title="km", yaxis_title="records (log)")
    st.plotly_chart(f, width="stretch")
    ec = pd.concat([d[[oc.DIST[(lv, o)]]].rename(columns={oc.DIST[(lv, o)]: "km"}).assign(
        origin=oc.ORIGIN_NAME[o][lv]) for o in ["mean", "reference"]]).dropna()
    f = px.ecdf(ec, x="km", color="origin", log_x=True,
                color_discrete_sequence=list(ORIGIN_COLORS.values()),
                title=f"Share of breweries within a given distance of each {LEVEL_NAMES[lv].lower()} origin")
    f.update_layout(xaxis_title="km (log)", yaxis_title="share of breweries", legend_title_text="")
    st.plotly_chart(f, width="stretch")

    st.markdown("**One place up close**")
    places = d.groupby(oc.KEYS[lv]).size().sort_values(ascending=False)
    labels = [", ".join(map(str, k[::-1])) if isinstance(k, tuple) else str(k) for k in places.index]
    pick = st.selectbox(f"{LEVEL_NAMES[lv]}", labels, key="dist_pick")
    pk = places.index[labels.index(pick)]
    pk = pk if isinstance(pk, tuple) else (pk,)
    sub = d[np.logical_and.reduce([d[k] == v for k, v in zip(oc.KEYS[lv], pk)])]
    mm = point_map(sub, color=col, color_continuous_scale=SATURATED_SCALE, hover_name="name",
                   hover_data={"city": True, col: ":,.1f"}, marker_size=8, height=520, style=basemap,
                   title=f"{pick}: breweries coloured by distance to the {oc.ORIGIN_NAME[origin][lv]}")
    for olat, olon, name, colr in [(sub[f"{lv}_mean_latitude"].iat[0], sub[f"{lv}_mean_longitude"].iat[0],
                                    "brewery center (origin 1)", ORIGIN_COLORS["brewery center (origin 1)"]),
                                   (sub[f"{lv}_ref_latitude"].iat[0], sub[f"{lv}_ref_longitude"].iat[0],
                                    f"{sub[f'{lv}_reference'].iat[0]} (origin 2)",
                                    ORIGIN_COLORS["capital / city center (origin 2)"])]:
        if pd.notna(olat):
            mm.add_trace((getattr(go, "Scattermap", None) or go.Scattermapbox)(
                lat=[olat], lon=[olon], mode="markers", name=name,
                marker=dict(size=18, color=colr, opacity=1)))
    mm.update_layout(coloraxis_colorbar=dict(title="km", thickness=12), legend=dict(orientation="h", y=-0.02))
    st.plotly_chart(mm, width="stretch", config={"scrollZoom": True})
    st.subheader("20 farthest records from the chosen origin")
    st.dataframe(d.nlargest(20, col)[["name", "city", STATE, "country", "latitude", "longitude", col]],
                 hide_index=True, width="stretch")

    st.divider()
    st.subheader("Questions answered in this tab")
    S = summary.set_index(["level", "origin"])
    st.markdown(f"""
**Are breweries scattered or concentrated around their centers?**\\
It depends on the level. Around their **city** brewery center, half of all breweries sit within **{S.loc[('city', 'city brewery center'), 'median_km']:.1f} km** and {S.loc[('city', 'city brewery center'), 'within_5km_%']:.0f}% within 5 km: a city's breweries are tight. Around their **state** brewery center the median is **{S.loc[('state', 'state brewery center'), 'median_km']:.0f} km** and only {S.loc[('state', 'state brewery center'), 'within_50km_%']:.0f}% are within 50 km: states are scattered. Around the **country** center the median is {S.loc[('country', 'country brewery center'), 'median_km']:,.0f} km, which mostly measures the size of the United States.

**Is the official center a better summary than the brewery center?**\\
At city level the GeoNames city center is almost as close (median {S.loc[('city', 'city center (GeoNames)'), 'median_km']:.1f} km): breweries sit in their towns. At state level the capital is farther than the brewery center for the typical brewery (median {S.loc[('state', 'state/province capital'), 'median_km']:.0f} km vs {S.loc[('state', 'state brewery center'), 'median_km']:.0f} km); the *Capitals & central cities* tab shows which states break that rule.

**Why use the median and the distribution instead of the mean?**\\
The distance distributions are heavily right-skewed (see the log-count histograms): a few misplaced or far-flung records add thousands of kilometres to the mean without changing where the typical brewery is.
""")

# ======================================================================= 4. spatial dispersion
with t[3]:
    st.markdown(NOTEBOOK_TEXT["dispersion"])
    st.markdown(NOTEBOOK_TEXT["compare_dispersion"])
    c1, c2, c3, c4 = st.columns(4)
    lv = c1.radio("Level", list(LEVEL_NAMES), format_func=LEVEL_NAMES.get, index=1, key="disp_lv")
    origin = c2.radio("Origin", ["mean", "reference"], format_func=lambda o: oc.ORIGIN_NAME[o][lv], key="disp_o")
    n = c3.slider("Minimum records", 1, 50, 5, key="disp_n")
    tol = c4.slider("'Close to 1' tolerance for mean/median", 0.02, 0.30, 0.10, 0.02, key="disp_tol")
    disp = oc.add_interpretation(oc.dispersion(d, lv, origin), tol)
    disp = disp[disp["record_count"] >= n].copy()
    disp["place"] = disp[oc.KEYS[lv][::-1]].astype(str).agg(", ".join, axis=1)
    SHAPE_COLORS = {"mean > median": "#eb6834", "mean ≈ median": "#2a78d6", "mean < median": "#1baf7a",
                    "undefined": "#8a8a8a"}
    k = st.columns(4)
    for box, cls in zip(k, SHAPE_COLORS):
        box.metric(cls, int((disp["shape"] == cls).sum()))
    c1, c2 = st.columns(2)
    top = disp.nlargest(20, "median_km").sort_values("median_km")
    f = px.bar(top, x="median_km", y="place", orientation="h", color="shape", color_discrete_map=SHAPE_COLORS,
               hover_data=["record_count", "p90_km", "iqr_km", "mean_to_median_ratio"],
               title=f"Most dispersed {LEVEL_NAMES[lv].lower()} places (median km to the {oc.ORIGIN_NAME[origin][lv]})")
    f.update_layout(yaxis_title="", legend_title_text="", height=560)
    c1.plotly_chart(f, width="stretch")
    f = px.scatter(disp, x="median_km", y="p90_km", size="record_count", color="shape", hover_name="place",
                   color_discrete_map=SHAPE_COLORS, hover_data={"mean_to_median_ratio": ":.2f"}, size_max=35,
                   title="Median vs P90 (colour = automatic mean/median reading)")
    f.update_layout(legend_title_text="", height=560)
    c2.plotly_chart(f, width="stretch")
    st.dataframe(disp.drop(columns=["place"]).sort_values("median_km", ascending=False).round(2), hide_index=True,
                 width="stretch", height=360,
                 column_config={"interpretation": st.column_config.TextColumn(width="large")})

    st.markdown(f"**Sensitivity: {key} vs {other}** (the notebook's `dispersion_comparison`)")
    p = pd.concat([oc.dispersion(build(kk), lv, origin).assign(dataset=kk) for kk in LABEL])
    p = p[p["record_count"] >= n].pivot_table(index=oc.KEYS[lv], columns="dataset", values="median_km").dropna()
    p["difference_km"] = p["all_records"] - p["unique_records"]
    f = px.scatter(p.reset_index(), x="all_records", y="unique_records", hover_name=oc.KEYS[lv][-1],
                   title="Median dispersion: all_records vs unique_records")
    f.add_trace(go.Scatter(x=[0, p.values.max()], y=[0, p.values.max()], mode="lines", name="identical",
                           line=dict(color="#8a8a8a", dash="dot")))
    st.plotly_chart(f, width="stretch")
    changed = p[p["difference_km"].abs() > 1e-9]
    st.caption(f"{len(changed)} of {len(p)} places change at all between the two datasets"
               + (": " + ", ".join(f"{i[-1] if isinstance(i, tuple) else i} ({v:+.2f} km)"
                                    for i, v in changed["difference_km"].items()) if len(changed) else "") + ".")

    st.divider()
    st.subheader("Questions answered in this tab")
    sd = oc.add_interpretation(oc.dispersion(d, "state", "mean"), tol)
    sd = sd[sd["record_count"] >= 20]
    sd["place"] = sd[STATE] + ", " + sd["country"]
    most, least = sd.nlargest(3, "median_km"), sd.nsmallest(3, "median_km")
    tail = sd[sd["shape"] == "mean > median"].nlargest(3, "mean_to_median_ratio")
    before = oc.dispersion(d_orig, "state", "mean")
    cmp_ = sd.merge(before[oc.KEYS["state"] + ["median_km"]], on=oc.KEYS["state"], suffixes=("", "_recorded"))
    cmp_["change_%"] = (cmp_["median_km_recorded"] / cmp_["median_km"] - 1) * 100
    hit = cmp_[(cmp_["change_%"] >= 30) & (cmp_["median_km_recorded"] >= 50)].sort_values("change_%", ascending=False)
    flat = sd[sd["shape"] == "mean < median"]
    st.markdown(f"""
**How geographically dispersed are breweries within each state?**\\
Among states with at least 20 records, the most spread out around their own brewery center are {", ".join(f"**{r['place']}** (median {r['median_km']:.0f} km)" for _, r in most.iterrows())}; the most compact are {", ".join(f"**{r['place']}** ({r['median_km']:.0f} km)" for _, r in least.iterrows())}, which are city-states or capital regions.{(" The coordinate corrections matter here: before them, " + ", ".join(f"{r['place']} measured {r['median_km_recorded']:.0f} km instead of {r['median_km']:.0f} km" for _, r in hit.iterrows()) + ". One swapped record a continent away dragged the whole state's mean center with it, and every distance grew.") if len(hit) else ""}

**What does the mean-to-median ratio say, automatically?**\\
With a tolerance of ±{tol:.2f}: **{int((sd['shape'] == 'mean > median').sum())}** states have mean > median (a compact core with a long tail of distant breweries), **{int((sd['shape'] == 'mean ≈ median').sum())}** have mean ≈ median (symmetric spread) and **{int((sd['shape'] == 'mean < median').sum())}** have mean < median{(" (" + ", ".join(flat['place']) + ")") if len(flat) else ""}. The longest tails are in {", ".join(f"{r['place']} ({r['mean_to_median_ratio']:.2f})" for _, r in tail.iterrows())}: there, the median describes the typical brewery and the mean is pulled by a few remote ones.

**Does the dispersion depend on the deduplication choice?**\\
No. {len(changed)} of {len(p)} places change between all_records and unique_records, by fractions of a kilometre.
""")

# ======================================================================= 5. capitals & central cities
with t[4]:
    st.markdown("**Does the capital act as the brewery center?** For every state the table compares the brewery "
                "center (origin 1) with the state/province capital (origin 2): median distance to each, the share "
                "of breweries within 50 km of each, how far apart the two are, and which one is closer to the "
                "typical brewery.")
    cv = oc.capital_vs_center(d, "state")
    cvn = st.slider("Minimum records per state", 1, 100, 30, key="cap_n")
    cvs = cv[cv["records"] >= cvn]
    st.dataframe(cvs.round(1), hide_index=True, width="stretch", height=300)
    spear = stats_store.spearman(stats_table("spearman"), key)  # Spearman tests precomputed (stats_store)
    central, cities = oc.central_cities(d, ref, spearman=spear)
    central = central[central["records"] >= cvn]
    c1, c2 = st.columns(2)
    roles = central["capital_role"].value_counts().rename_axis("role").reset_index(name="states")
    f = px.bar(roles, x="states", y="role", orientation="h", text="states", color_discrete_sequence=["#2a78d6"],
               title="What role does the capital play in its state's brewing?")
    f.update_layout(yaxis=dict(autorange="reversed"), yaxis_title="", height=340)
    c1.plotly_chart(f, width="stretch")
    central["place"] = central[STATE] + ", " + central["country"]
    f = px.scatter(central, x="capital_share_%", y="largest_city_share_%", size="records", hover_name="place",
                   hover_data={"capital": True, "largest_city": True, "capital_rank_in_state": True},
                   color="capital_role", size_max=30,
                   color_discrete_sequence=["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#8a8a8a"],
                   title="Capital's share vs largest city's share of the state")
    f.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", name="capital = largest city",
                           line=dict(color="#8a8a8a", dash="dot")))
    f.update_layout(legend_title_text="", height=340, xaxis_title="capital's share (%)",
                    yaxis_title="largest city's share (%)")
    c2.plotly_chart(f, width="stretch")
    st.dataframe(central.drop(columns=["place"]).round(2), hide_index=True, width="stretch", height=320)

    st.subheader("Cities of one state: distance from the center vs share of the state")
    top_states = d.groupby(oc.KEYS["state"]).size().sort_values(ascending=False).head(20)
    labels = [f"{s}, {c}" for c, s in top_states.index]
    pick = st.selectbox("State", labels, key="cap_state")
    c_, s_ = top_states.index[labels.index(pick)]
    cs = cities[(cities["country"] == c_) & (cities[STATE] == s_)].copy()
    cs["role"] = np.where(cs["is_capital"], "capital", "other city")
    row = oc.central_cities(d[(d["country"] == c_) & (d[STATE] == s_)], ref, spearman=spear)[0].iloc[0]
    c1, c2 = st.columns(2)
    f = px.scatter(cs, x="km_to_state_center", y="share_of_state_%", size="records", color="role", hover_name="city",
                   color_discrete_map={"capital": "#eb6834", "other city": "#2a78d6"}, size_max=30,
                   title=f"{pick}: Spearman ρ = {row['rho_distance_to_center_vs_share']:.2f} "
                         f"(p = {row['p_center']:.3f})")
    f.update_layout(xaxis_title="city's distance to the state brewery center (km)",
                    yaxis_title="city's share of the state (%)", legend_title_text="", height=460)
    c1.plotly_chart(f, width="stretch")
    pts = pd.concat([
        cs.assign(kind=cs["role"], lat=cs["latitude"], lon=cs["longitude"]),
        pd.DataFrame({"city": ["brewery center"], "records": [cs["records"].max()], "kind": ["brewery center (origin 1)"],
                      "lat": [cs["state_lat"].iat[0]], "lon": [cs["state_lon"].iat[0]], "share_of_state_%": [np.nan]})])
    m = point_map(pts, lat="lat", lon="lon", color="kind", size="records", hover_name="city",
                  color_discrete_map={"capital": "#eb6834", "other city": "#2a78d6",
                                      "brewery center (origin 1)": "#1baf7a"},
                  hover_data={"share_of_state_%": ":.1f", "lat": False, "lon": False}, size_max=30, opacity=0.9,
                  height=460, style=basemap, title=f"{pick}: cities (size = breweries), capital and brewery center")
    m.update_layout(legend_title_text="")
    c2.plotly_chart(m, width="stretch", config={"scrollZoom": True})

    st.divider()
    st.subheader("Questions answered in this tab")
    rho = central.dropna(subset=["rho_distance_to_center_vs_share"])
    sig = rho[rho["p_center"] < 0.05]
    neg = sig[sig["rho_distance_to_center_vs_share"] < 0]
    pos = sig[sig["rho_distance_to_center_vs_share"] > 0]
    rc = central["capital_role"].value_counts()
    main_cap = central[central["capital_role"] == "capital is the main brewery city"]
    minor = central[central["capital_role"] == "capital is a minor brewery city"].nlargest(4, "records")
    same = central[central["largest_city"] == central["city_nearest_center"]]
    closer = cvs["closer_origin"].value_counts()
    st.markdown(f"""
**Does the state's capital appear to be an important brewery center?**\\
Often, but not as a rule. Among states with at least {cvn} records, the capital is the main brewery city in **{rc.get('capital is the main brewery city', 0)}**, a secondary one (rank 2–3) in **{rc.get('capital is a secondary brewery city (rank 2-3)', 0)}**, a minor one in **{rc.get('capital is a minor brewery city', 0)}**, and has no brewery at all in **{rc.get('capital has no brewery', 0)}**. The capital leads in places such as {", ".join(main_cap.nlargest(5, 'records')[STATE])}, and in every city-state or region drawn around its capital. Where it is minor, the brewing hub is a bigger commercial city: {", ".join(f"{r[STATE]} ({r['capital']} rank {int(r['capital_rank_in_state'])}, while {r['largest_city']} leads)" for _, r in minor.iterrows())}.

**Are breweries closer to the capital or to their own center?**\\
For **{closer.get('brewery center', 0)}** states the typical brewery is closer to the brewery center, for **{closer.get('reference point', 0)}** closer to the capital. The split is close: a capital pulls breweries when it is also a large city, and not when it is a small administrative town.

**Is there a relationship between a city's distance from the state center and its share of the state?**\\
Mostly no. Of the {len(rho)} states with enough cities, only **{len(sig)}** show a significant Spearman correlation (p < 0.05): **{len(neg)}** negative (cities nearer the center hold more breweries{", e.g. " + ", ".join(neg.nsmallest(3, 'rho_distance_to_center_vs_share')[STATE]) if len(neg) else ""}) and **{len(pos)}** positive{(" (" + ", ".join(pos[STATE]) + ")") if len(pos) else ""}. In most states the share of a city has little to do with how central it is: hubs grow where people are, not where the geometric middle is.

**Which city is the "center" of each state?**\\
Two answers: the city that holds the largest share, and the city nearest to the brewery center. They coincide in only **{len(same)}** of {len(central)} states ({", ".join(same[STATE].head(6)) or "none"}). In most states the brewery center falls in a small town between hubs, so the real center of brewing is the largest-share city in the table above.
""")

# ======================================================================= 6. outliers & coordinate checks
with t[5]:
    st.markdown(NOTEBOOK_TEXT["outliers"])
    rd, osum = d, oc.outlier_summary(d, "state", "mean")
    k = st.columns(3)
    k[0].metric("Flagged records", f"{int(rd['geographic_outlier'].sum()):,}")
    k[1].metric("Share flagged", f"{rd['geographic_outlier'].mean()*100:.1f}%")
    k[2].metric("Max distance (km)", f"{rd['dist_state_mean_center_km'].max():,.0f}")
    fl = rd[rd["geographic_outlier"]]
    if fl.empty:
        st.info("No records are flagged as geographic outliers.")
    else:
        fig = outlier_map(fl, "dist_state_mean_center_km", hover_name="name", hover_data=["city", STATE, "country"],
                          style=basemap,
                          title="Flagged outliers (Q3 + 1.5×IQR within state) — colour and size = distance from state center")
        st.plotly_chart(fig, width="stretch", config={"scrollZoom": True})
        st.caption("Colour scale is logarithmic (most outliers are a few hundred km out, a few are thousands).")
    st.dataframe(fl.sort_values("dist_state_mean_center_km", ascending=False)
                 [["name", "city", STATE, "country", "latitude", "longitude", "dist_state_mean_center_km",
                   "outlier_threshold_km"]].head(50), hide_index=True, width="stretch")
    st.subheader("Outliers by state")
    st.dataframe(osum.sort_values(["geographic_outlier_count", "geographic_outlier_pct"], ascending=False),
                 hide_index=True, width="stretch", height=260)

    st.subheader("Coordinate check against each record's own city center (coordinates as recorded)")
    st.markdown("A distance rule can only say a brewery is *far*. Comparing each record with the GeoNames center of "
                "the city it names can say *why*: a record more than 500 km from its city that lands within 25 km once "
                "latitude and longitude are exchanged has them **swapped**; one that lands within 25 km once the sign "
                "of one coordinate is reversed has a **sign error**; a record more than 50 km from its city is "
                "**far from its city** (a geocoding error or a wrong city label). When the city has no GeoNames "
                "center, the record is still tested for a sign error against its state capital (within 150 km once "
                "reversed). The check runs on the coordinates as recorded; its result is the corrections log below, "
                "which every analysis applies.")
    vc = chk["check"].value_counts()
    k = st.columns(5)
    for box, cls in zip(k, ["consistent", "latitude/longitude swapped", "sign error", "far from its city",
                            "not checked"]):
        box.metric(cls, f"{int(vc.get(cls, 0)):,}")
    flagged = chk[chk["check"].isin(["latitude/longitude swapped", "sign error", "far from its city"])].copy()
    if len(flagged):
        st.plotly_chart(pair_map(flagged, ("latitude", "longitude"), ("reference_latitude", "reference_longitude"),
                                 ["Stored position", "Reference point: the city's center (or the state capital)"],
                                 hover_cols=("city", STATE, "country")),
                        width="stretch", config={"scrollZoom": True})
        st.dataframe(flagged.sort_values("km_to_reference", ascending=False)
                     [["check", "name", "city", STATE, "country", "latitude", "longitude", "reference_point",
                       "km_to_reference", "km_to_city_center_if_swapped", "km_to_reference_if_sign_reversed",
                       "geographic_outlier"]].round(2),
                     hide_index=True, width="stretch")
    st.markdown("**Corrections log** (`data/coordinate_corrections.csv`)")
    st.dataframe(log, hide_index=True, width="stretch", height=280)
    st.caption("`swapped back`, `sign flipped back` and `copied from identical record` are applied when data is "
               "loaded; `flagged only` "
               "records keep their position and carry `coordinate_status = flagged: far from its city`. The "
               "databases themselves keep the original values.")

    st.divider()
    st.subheader("Questions answered in this tab")
    sw = flagged[flagged["check"] == "latitude/longitude swapped"]
    sg = flagged[flagged["check"] == "sign error"]
    farc = flagged[flagged["check"] == "far from its city"]
    caught = int(flagged["geographic_outlier"].sum())
    st.markdown(f"""
**Are flagged geographic outliers errors?**\\
Not automatically. The IQR rule flags **{int(rd['geographic_outlier'].sum()):,}** records ({rd['geographic_outlier'].mean()*100:.1f}%) as far from their state's brewery center; most are real breweries at the edge of large states. The rule is computed per state, so small states give unstable thresholds.

**Which coordinates are actually wrong?**\\
The city check finds **{len(sw)} records with latitude and longitude swapped**: {", ".join(f"{r['name']} ({r['city']}, {r['country']})" for _, r in sw.iterrows())}. Swapped back, each lands within {sw['km_to_city_center_if_swapped'].max():.1f} km of its own city. {" ".join(f"**{r['name']}** ({r['city']}, {r[STATE]}) has the wrong {r['sign_error_in']} sign: stored, it lies {r['km_to_reference']:,.0f} km from {r['reference_point']}; with the sign reversed, {r['km_to_reference_if_sign_reversed']:.1f} km." for _, r in sg.iterrows())} It also finds **{len(farc)} records far from the city they name**, up to {farc['km_to_city_center'].max():,.0f} km away. Only {caught} of these {len(flagged)} are caught by the IQR outlier rule; the rest look normal to a distance rule because they landed in a plausible place, just not the right one.

**What happened to them?**\\
The {int((log['action'] == 'swapped back').sum())} swapped records are swapped back, {"the sign error is reversed, " if (log['action'] == 'sign flipped back').any() else ""}and the misplaced Liefmans record takes the position of its identical twin; every page and the new clustering runs use these corrected positions. The other {int((log['action'] == 'flagged only').sum())} far-off records have no safe correction (the right position is unknown), so they keep their position and carry a flag. The IQR outlier rule above is computed on the corrected data.
""")
