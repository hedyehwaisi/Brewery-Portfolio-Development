"""Open Brewery DB — Geographic Distribution Dashboard (Streamlit).

Tabs: Overview · Pipeline & Quality · Duplicates & Identity · Missing Coordinates
      · Geography · Concentration · All vs Unique · Methodology
Run:  streamlit run app.py   (expects ./data/*.db produced by the notebook)
"""
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common import pick_basemap, table, shared_cache
from geo_plots import point_map, type_style, core_view, swap_map, SWAP_COLORS
from geographic_utils import haversine_km
from palette import DATASET_COLORS, color_map
import quality_core as qc
import distribution_core as dc
import origins_core as oc

st.set_page_config(page_title="Open Brewery DB — Geographic Analysis",
                   page_icon="🍺", layout="wide")

DATA = Path(__file__).resolve().parent / "data"
SOURCES = {
    "All records": ("analysis_all_records.db", "analysis_all_records"),
    "Unique records": ("analysis_unique_records.db", "analysis_unique_records"),
}
RAW = ("raw_breweries.db", "raw_breweries")
DATASET_TABLES = {"raw": RAW, "all_records": SOURCES["All records"],
                  "unique_records": SOURCES["Unique records"]}
CATS = ["country", "state_province", "city", "brewery_type"]

# Findings recorded in the notebook (static, documented results)
RAW_N, EXACT_DUPES, MISSING_COORD = 11959, 3, 2468
# Reverse geocoding of both positions of the Liefmans near-duplicate pair (OpenStreetMap Nominatim, 2026-10-30).
NEAR_DUP_REVERSE_GEOCODE = {
    "4d61de42-eaeb-488a-b10f-f3727e9176d9": "Meise, Halle-Vilvoorde, Vlaams-Brabant, Belgium",
    "c85ce64e-0f31-4123-bef5-ca1abc080fc9": "Oudenaarde, Oost-Vlaanderen, Belgium",
}
# Reverse geocoding of the two swapped-back coordinates (OpenStreetMap Nominatim, checked 2026-10-30).
SWAP_REVERSE_GEOCODE = {
    "7ab0e9c9-ffb3-44e1-91fb-99dc7fd4959d": "Cullman, Cullman County, Alabama, United States",
    "1eb023fa-2183-4dbb-84a8-bfbc70a8abf2": "Manchester, Capitol Planning Region, Connecticut, United States",
}
NAME_DUPES = pd.DataFrame({
    "name": ["Circle 9 Brewing", "Circle 9 Brewing", "Paulaner", "Paulaner"],
    "brewery_type": ["micro", "closed", "brewpub", "brewpub"],
    "phone": ["8582569462", "8586342537", "+49 89 5446117", "+49 89 5446117"],
    "website_url": ["ataraxiaaleworks.com", "circle9brewing.com", "paulaner.de", "paulaner-brauhaus.de"],
    "kept_in_unique": ["yes", "yes", "yes", "no"],
})


@shared_cache
def reference_places():
    return oc.load_reference()


@shared_cache
def load(file, table):
    path = DATA / file
    if not path.exists():
        return None
    with sqlite3.connect(path) as con:
        df = pd.read_sql_query(f'SELECT * FROM "{table}"', con)
    for c in CATS:
        if c in df:
            df[c] = df[c].astype("string").str.strip().replace("", pd.NA).fillna("Unknown")
    for c in ["latitude", "longitude"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return qc.apply_corrections(df)  # corrected coordinates; the database itself keeps the original values


def bar(df, x, y, title, **kw):
    fig = px.bar(df, x=x, y=y, orientation="h", title=title, text=x, **kw)
    fig.update_layout(yaxis_title="", yaxis=dict(autorange="reversed"))
    return fig


# ---------------- Sidebar ----------------
st.sidebar.header("Dashboard controls")
dataset = st.sidebar.radio("Dataset", list(SOURCES))
df_full = load(*SOURCES[dataset])
if df_full is None:
    st.error(f"Database not found in {DATA}. Run the notebook export first.")
    st.stop()
df_full = qc.harmonize_states(df_full)  # one label per region for every state-level count
STATE = "state_province_harmonized"
countries = sorted(df_full["country"].unique())
sel_c = st.sidebar.multiselect("Country", countries, default=countries)
types = sorted(df_full["brewery_type"].unique())
sel_t = st.sidebar.multiselect("Brewery type", types, default=types)
df = df_full[df_full["country"].isin(sel_c) & df_full["brewery_type"].isin(sel_t)]
st.sidebar.caption(f"{len(df):,} of {len(df_full):,} records selected")
basemap = pick_basemap()
if df.empty:
    st.warning("No records match the filters.")
    st.stop()

st.title("🍺 Open Brewery DB — Geographic Distribution")
st.caption(f"Dataset: **{dataset}** · filters apply to Overview, Geography and Concentration tabs")

tabs = st.tabs(["Overview", "Pipeline & Quality", "Duplicates & Identity",
                "Missing Coordinates", "Geography", "Concentration",
                "All vs Unique", "Methodology"])

# ---------------- 1. Overview ----------------
with tabs[0]:
    k = st.columns(5)
    k[0].metric("Records", f"{len(df):,}")
    k[1].metric("Countries", df["country"].nunique())
    k[2].metric("States / provinces", df[["country", STATE]].drop_duplicates().shape[0],
                help="Harmonized labels: a region written in several languages counts once.")
    k[3].metric("Cities", df[["country", STATE, "city"]].drop_duplicates().shape[0])
    k[4].metric("Coordinate coverage", f"{df[['latitude','longitude']].notna().all(axis=1).mean()*100:.1f}%")

    c1, c2 = st.columns(2)
    t = df["brewery_type"].value_counts().rename_axis("type").reset_index(name="records")
    t["share_%"] = (t["records"] / len(df) * 100).round(2)
    c1.plotly_chart(bar(t, "records", "type", "Brewery types", hover_data=["share_%"]), width="stretch")
    c2.plotly_chart(px.pie(t, names="type", values="records", hole=0.45, title="Type share"),
                    width="stretch")
    c3, c4 = st.columns(2)
    co = df["country"].value_counts().rename_axis("country").reset_index(name="records")
    c3.plotly_chart(bar(co, "records", "country", "Records by country (all)", log_x=True),
                    width="stretch")
    st_ = dc.counts(df, "state").head(20)
    c4.plotly_chart(bar(st_, "records", "place", "Top 20 states / provinces"), width="stretch")
    ct = dc.counts(df, "city").head(20)
    st.plotly_chart(bar(ct, "records", "place", "Top 20 cities"), width="stretch")

    # ---- top states inside each country ----
    st.subheader("Top states / provinces inside each country")
    order_c = co["country"].tolist()
    c1, c2 = st.columns([1, 3])
    pick_c = c1.selectbox("Country", order_c, key="ov_state_country")
    n_s = c1.slider("States to show", 3, 30, 10, key="ov_state_n")
    ts = dc.top_within(df, "country", "state", n_s)
    one = ts[ts["country"] == pick_c].copy()
    one["label"] = [f"{r:,} ({p:.1f}% of {pick_c})" for r, p in zip(one["records"], one["share_of_parent_%"])]
    f = px.bar(one, x="records", y=STATE, orientation="h", text="label",
               color_discrete_sequence=[DATASET_COLORS["raw"]],
               title=f"{pick_c}: {len(one)} largest of {int(one['places_in_parent'].iat[0])} states / provinces")
    f.update_traces(textposition="outside", cliponaxis=False)
    f.update_layout(yaxis=dict(autorange="reversed"), yaxis_title="", xaxis_title="Records",
                    height=max(320, 32 * len(one) + 120), margin=dict(r=140))
    c2.plotly_chart(f, width="stretch")
    with st.expander(f"Top {n_s} states / provinces of every country"):
        st.dataframe(ts[["country", STATE, "records", "share_of_parent_%", "rank_in_parent", "places_in_parent"]]
                     .round(2), hide_index=True, width="stretch")

    # ---- top cities inside a country or a state ----
    st.subheader("Top cities inside a country or a state / province")
    c1, c2 = st.columns([1, 3])
    pick_c2 = c1.selectbox("Country", order_c, key="ov_city_country")
    states_in = dc.counts(df[df["country"] == pick_c2], "state")[STATE].tolist()
    pick_s = c1.selectbox("State / province", ["All states"] + states_in, key="ov_city_state")
    n_c = c1.slider("Cities to show", 3, 30, 10, key="ov_city_n")
    if pick_s == "All states":
        tc = dc.top_within(df, "country", "city", n_c)
        tc = tc[tc["country"] == pick_c2].copy()
        parent = pick_c2
    else:
        tc = dc.top_within(df, "state", "city", n_c)
        tc = tc[(tc["country"] == pick_c2) & (tc[STATE] == pick_s)].copy()
        parent = pick_s
    tc["label"] = [f"{r:,} ({p:.1f}% of {parent})" for r, p in zip(tc["records"], tc["share_of_parent_%"])]
    tc["city_name"] = tc["city"] + ("" if pick_s != "All states" else ", " + tc[STATE])
    f = px.bar(tc, x="records", y="city_name", orientation="h", text="label",
               color_discrete_sequence=[DATASET_COLORS["raw"]],
               title=f"{parent}: {len(tc)} largest of {int(tc['places_in_parent'].iat[0]) if len(tc) else 0} cities")
    f.update_traces(textposition="outside", cliponaxis=False)
    f.update_layout(yaxis=dict(autorange="reversed"), yaxis_title="", xaxis_title="Records",
                    height=max(320, 32 * len(tc) + 120), margin=dict(r=160))
    c2.plotly_chart(f, width="stretch")
    with st.expander(f"Top {n_c} cities of every state / province"):
        st.dataframe(dc.top_within(df, "state", "city", n_c)[["country", STATE, "city", "records",
                                                             "share_of_parent_%", "rank_in_parent",
                                                             "places_in_parent"]].round(2),
                     hide_index=True, width="stretch")
    st.caption("Shares are of the parent place (country or state). States use harmonized labels, so Belgium's "
               "regions are not split by language. Cities are keyed by country + state + city.")

    st.divider()
    st.subheader("Questions answered on this page")
    top_country = df["country"].value_counts().idxmax()
    top_country_n = int(df["country"].value_counts().max())
    top_type = df["brewery_type"].value_counts().idxmax()
    top_type_n = int(df["brewery_type"].value_counts().max())
    top_state, top_city = st_.iloc[0], ct.iloc[0]
    lead = ts[ts["rank_in_parent"] == 1].merge(co, on="country", suffixes=("", "_country"))
    lead = lead.sort_values("records_country", ascending=False).head(5)
    high_text = "as in " + " and ".join(lead.loc[lead["share_of_parent_%"] >= 40, "country"]) \
        if (lead["share_of_parent_%"] >= 40).any() else "none of the five largest countries here"
    low_text = "as in " + lead.loc[lead["share_of_parent_%"].idxmin(), "country"]
    lead_text = "; ".join(f"{r['country']}: **{r[STATE]}** ({r['share_of_parent_%']:.0f}% of the country)"
                          for _, r in lead.iterrows())
    st.markdown(f"**What dominates the current selection?** The most represented country is **{top_country}** ({top_country_n:,} records); the most frequent type is **{top_type}** ({top_type_n:,}); the largest state/province group is **{top_state['place']}** ({top_state['records']:,}); and the largest city group is **{top_city['place']}** ({top_city['records']:,}). These are records in Open Brewery DB, so they describe what the database holds rather than every brewery that exists.\n\n**Which state leads inside each country, and by how much?** {lead_text}. A single region holding a large share of its country ({high_text}) means a country-level number is mostly a statement about that region; a low top share ({low_text}) means the country's records are spread over many states.\n\n**Why show both counts and shares?** A count says how much is there; a share makes groups of very different size comparable. With one country holding about two thirds of the records, every so-called global pattern is first of all an American one, and the shares make that visible.")


# ---------------- 2. Pipeline & Quality ----------------
def pipeline_details(frames):
    """Everything in the Pipeline & Quality tab below the funnel; needs all three tables."""
    raw_t, all_t, uniq_t = frames["raw"], frames["all_records"], frames["unique_records"]

    st.subheader("Dataset structure: columns and completeness")
    ov = qc.column_overview(frames)
    complete = ov.loc[ov["complete_in_every_dataset"], "column"].tolist()
    complete_after = ov.loc[(ov["raw_null"] > 0) & (ov["all_records_null"] == 0) & (ov["unique_records_null"] == 0),
                            "column"].tolist()
    twins = qc.identical_columns(raw_t)
    k = st.columns(4)
    for box, (name, d) in zip(k, frames.items()):
        box.metric(f"Columns in {name}", len(d.columns), help=f"{len(d):,} records")
    k[3].metric("Columns with no null anywhere", len(complete))
    st.dataframe(ov, hide_index=True, width="stretch",
                 column_config={"complete_in_every_dataset": st.column_config.CheckboxColumn("no null in any dataset")})
    twin_text = "; ".join(f"`{r['column']}` = `{r['identical_to']}`" for _, r in twins.iterrows())
    st.caption(f"All three datasets carry the same {len(raw_t.columns)} columns: "
               + ", ".join(f"`{c}`" for c in raw_t.columns) + ". "
               f"Never null in any dataset: {', '.join(f'`{c}`' for c in complete)}. "
               f"Complete only after cleaning: {', '.join(f'`{c}`' for c in complete_after) or 'none'}. "
               + (f"Columns that repeat another column on every raw record: {twin_text}." if len(twins) else ""))

    state_check = "🔎 see Data Quality page"
    if {"state", "state_province"} <= set(raw_t.columns):
        norm = lambda s: s.astype("string").str.strip().str.casefold().fillna("<missing>")
        n_same = int((norm(raw_t["state"]) == norm(raw_t["state_province"])).sum())
        state_check = (f"✅ identical in all {n_same:,} raw records" if n_same == len(raw_t)
                       else f"⚠️ differ in {len(raw_t) - n_same:,} records")

    st.subheader("Missing values: raw vs all_records vs unique_records")
    rows = [{"column": col, "dataset": name, "missing": int(n), "records": len(d), "missing_%": n / len(d) * 100}
            for name, d in frames.items() for col, n in d.isna().sum().items()]
    miss = pd.DataFrame(rows)
    miss = miss[miss["column"].isin(miss.groupby("column")["missing"].max().loc[lambda s: s > 0].index)]
    miss["label"] = [f"{m:,} ({p:.1f}%)" for m, p in zip(miss["missing"], miss["missing_%"])]
    order = miss[miss["dataset"] == "raw"].sort_values("missing_%", ascending=False)["column"].tolist()
    fig = px.bar(miss, x="missing_%", y="column", color="dataset", barmode="group", orientation="h", text="label",
                 color_discrete_map=DATASET_COLORS,
                 category_orders={"column": order, "dataset": list(DATASET_COLORS)},
                 hover_data={"missing": ":,", "records": ":,", "missing_%": ":.2f", "label": False})
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(xaxis_title="Missing values (% of that dataset's records)", yaxis_title="",
                      xaxis_range=[0, 125], height=34 * len(miss) + 140, legend_title_text="Dataset",
                      bargap=0.25, bargroupgap=0.08)
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Each bar reads *missing records (share of that dataset)*. Dataset sizes: raw {len(raw_t):,}, "
               f"all_records {len(all_t):,}, unique_records {len(uniq_t):,}. Columns with no missing value in any "
               "dataset are left out. Latitude and longitude drop to zero after cleaning because records without "
               "both coordinates are excluded from the analysis datasets.")

    left, right = st.columns(2)
    right.markdown("**Validation checklist**")
    right.table(pd.DataFrame({
        "check": ["No null id", "No null name/type/city/state/country", "Duplicate ids",
                  "Latitude in [-90, 90]", "Longitude in [-180, 180]",
                  "One-coordinate-only missing", "state vs state_province"],
        "result": ["✅ pass", "✅ pass", "✅ 0", "✅ 0 invalid", "✅ 0 invalid",
                   "✅ none", state_check]}))
    sel_key = "all_records" if dataset == "All records" else "unique_records"
    sel_t = frames[sel_key]
    left.markdown(f"**Missing values in `{sel_key}`** ({len(sel_t):,} records — the dataset chosen in the sidebar)")
    md = sel_t.isna().sum().rename("missing_records").to_frame()
    md["missing_%"] = (md["missing_records"] / len(sel_t) * 100).round(2)
    left.dataframe(md[md["missing_records"] > 0].sort_values("missing_records", ascending=False),
                   width="stretch")

    # ---- exact duplicates: all columns except id are identical ----
    st.subheader("The three exact-duplicate pairs (6 records)")
    dup_cols = [c for c in raw_t.columns if c != "id"]
    ex = raw_t[raw_t.duplicated(dup_cols, keep=False)].copy()
    ex.insert(0, "status", np.where(raw_t.duplicated(dup_cols, keep="first").loc[ex.index],
                                    "removed (repeat)", "kept (first occurrence)"))
    ex.insert(0, "pair", ex.groupby(dup_cols, dropna=False).ngroup() + 1)
    ex.insert(2, "in_all_records", ex["id"].isin(all_t["id"]))
    ex.insert(3, "in_unique_records", ex["id"].isin(uniq_t["id"]))
    st.dataframe(ex.sort_values(["pair", "status"]), hide_index=True, width="stretch")
    st.caption("Rule: two rows are exact duplicates when all 15 columns except `id` match, missing values included. "
               "The first occurrence in the raw table is kept (`drop_duplicates(keep='first')`); because the pair is "
               "identical, nothing but a second id is lost. Both analysis datasets are built after this step.")
    kept = ex[ex["status"].str.startswith("kept")].copy()
    for c in ["latitude", "longitude"]:
        kept[c] = pd.to_numeric(kept[c], errors="coerce")
    kept = kept.dropna(subset=["latitude", "longitude"])
    if not kept.empty:
        kept["duplicate_ids"] = kept["pair"].map(ex.groupby("pair")["id"].agg(" | ".join))
        dmap = point_map(kept, hover_name="name", hover_data=["city", "state_province", "country", "duplicate_ids"],
                         color="name", color_discrete_map=color_map(kept["name"]), marker_size=13, height=360,
                         style=basemap, title="Where the exact-duplicate pairs sit (one marker per pair)")
        dmap.update_layout(legend_title_text="Brewery")
        st.plotly_chart(dmap, width="stretch", config={"scrollZoom": True})

    # ---- the one record separating all_records from unique_records ----
    st.subheader("Why all_records has one more record than unique_records")
    extra = all_t[~all_t["id"].isin(uniq_t["id"])]
    loc_cols = [c for c in all_t.columns if c not in ("id", "website_url")]
    pair = all_t.merge(extra[loc_cols], on=loc_cols)
    pair.insert(0, "in_unique_records", pair["id"].isin(uniq_t["id"]))
    st.dataframe(pair, hide_index=True, width="stretch")
    st.caption("`unique_records` drops rows that repeat the same location identity: every column except `id` and "
               "`website_url`. This pair is the only one in the snapshot that matches on all of them.")
    extra_text = ""
    if len(extra) == 1 and len(pair) == 2:
        e = extra.iloc[0]
        k = pair[pair["in_unique_records"]].iloc[0]
        extra_text = (f"\n\n**Why does `all_records` hold one record more than `unique_records`?** `unique_records` "
                      "is built from `all_records` by dropping rows that repeat the same location identity: name, "
                      "brewery type, the address lines, city, state/province, postal code, country, phone and "
                      f"coordinates (every column except `id` and `website_url`). Exactly one pair qualifies: two "
                      f"**{e['name']}** records of type **{e['brewery_type']}** at {e['address_1']}, "
                      f"**{e['city']}, {e['state_province']}, {e['country']}**. They differ only in their id and "
                      f"website (`{k['website_url']}` vs `{e['website_url']}`). `all_records` keeps both; "
                      f"`unique_records` keeps the first and drops the `{e['website_url']}` row. The whole gap is "
                      f"therefore one {e['brewery_type']} in {e['city']}: every other country, state, city and brewery "
                      "type has identical counts in the two datasets.")

    st.divider()
    st.subheader("Questions answered on this page")
    st.markdown(f"**Where do records leave the geographic pipeline?** The documented pipeline starts with **{RAW_N:,} raw records**, removes **{EXACT_DUPES:,} excess exact duplicates**, then excludes **{MISSING_COORD:,} records without complete coordinates** from coordinate-based analysis. The retained all-records and unique-records geographic datasets contain **{RAW_N-EXACT_DUPES-MISSING_COORD:,}** and **{RAW_N-EXACT_DUPES-MISSING_COORD-1:,}** records respectively.{extra_text}\n\n**Why not impute coordinates?** Location is exactly what we study. A guessed position would create distances and clusters that are not in the data, so we leave those records out of spatial work and say so. The price is a geocoded subset that is not random: planning breweries and some states lose far more records than others (Data Quality page).\n\n**What do the validation checks catch, and what do they miss?** They catch structural problems such as duplicate ids or coordinates outside the valid range, and every check passes. They cannot catch a coordinate that is valid but wrong: the later city check found eight records with latitude and longitude swapped and 19 more far from their own city (Geographic Analysis page).")

with tabs[1]:
    st.subheader("Record attrition across the pipeline")
    stages = pd.DataFrame({
        "stage": ["Raw API records", "After removing exact duplicates",
                  "After dropping missing coordinates (all-records)", "Unique-location dataset"],
        "records": [RAW_N, RAW_N - EXACT_DUPES, RAW_N - EXACT_DUPES - MISSING_COORD,
                    RAW_N - EXACT_DUPES - MISSING_COORD - 1]})
    st.plotly_chart(px.funnel(stages, x="records", y="stage"), width="stretch")
    st.caption("2,471 records removed in total = 2,468 missing-coordinate rows + 3 exact duplicates "
               "(one row kept per pair, so no information lost).")

    # untouched tables: load() fills empty categories with "Unknown", which would hide real nulls
    frames = {name: table(*src) for name, src in DATASET_TABLES.items()}
    if any(v is None for v in frames.values()):
        st.info("raw_breweries.db, analysis_all_records.db and analysis_unique_records.db are all needed for the comparisons below.")
    else:
        pipeline_details(frames)

# ---------------- 3. Duplicates & Identity ----------------
with tabs[2]:
    sel_t = table(*SOURCES[dataset]).copy()  # untouched table of the sidebar dataset
    sel_key = "all_records" if dataset == "All records" else "unique_records"
    for c in ["latitude", "longitude"]:
        sel_t[c] = pd.to_numeric(sel_t[c], errors="coerce")

    # ---- near-duplicates: identical except id and coordinates ----
    nd = qc.harmonize_states(qc.near_duplicate_pairs(sel_t))
    st.subheader(f"Near-duplicate records: {nd['pair'].nunique() if len(nd) else 0} pairs, {len(nd)} records")
    if nd.empty:
        st.info("No near-duplicate records in this dataset.")
    else:
        countries = nd["country"].unique().tolist()
        k = st.columns(4)
        k[0].metric("Pairs", nd["pair"].nunique())
        k[1].metric("Records", len(nd))
        k[2].metric("Countries involved", ", ".join(countries))
        if len(countries) == 1:
            n_c = int((sel_t["country"] == countries[0]).sum())
            k[3].metric(f"{countries[0]} records in {sel_key}", f"{n_c:,}",
                        help=f"These {len(nd)} records are included: {len(nd) / n_c * 100:.1f}% of {countries[0]}.")
        nd["position_reverse_geocodes_to"] = nd["id"].map(NEAR_DUP_REVERSE_GEOCODE)
        lead = ["pair", "pair_distance_km", "name", "brewery_type", "city", "state_province",
                "state_province_harmonized", "country", "latitude", "longitude", "position_reverse_geocodes_to"]
        st.dataframe(nd[lead + [c for c in nd.columns if c not in lead + ["records_in_group"]]].round(6),
                     hide_index=True, width="stretch")
        st.caption("Rule: every column except `id`, `latitude` and `longitude` is identical. `state_province` is the "
                   "label as stored; `state_province_harmonized` folds the Dutch, French and English names of one "
                   "Belgian region into a single label. Both records of each pair are kept in the analysis datasets.")

        pairs = nd.drop_duplicates("pair").sort_values("pair_distance_km")
        pairs["label"] = [f"{d * 1000:,.0f} m" if d < 1 else f"{d:,.1f} km" for d in pairs["pair_distance_km"]]
        pairs["who"] = pairs["name"] + " (" + pairs["city"] + ")"
        c1, c2 = st.columns([2, 3])
        bar_fig = px.bar(pairs, x="pair_distance_km", y="who", orientation="h", log_x=True, text="label",
                         color_discrete_sequence=[DATASET_COLORS["raw"]],
                         title="Distance between the two positions of each pair (log scale)")
        bar_fig.update_traces(textposition="outside", cliponaxis=False)
        bar_fig.update_layout(xaxis_title="km (log scale)", yaxis_title="", height=360)
        c1.plotly_chart(bar_fig, width="stretch")
        pair_colors = color_map(pairs["who"])
        nd["who"] = nd["name"] + " (" + nd["city"] + ")"
        nd_map = point_map(nd, color="who", color_discrete_map=pair_colors, hover_name="name",
                           hover_data=["id", "city", "state_province_harmonized", "pair_distance_km"],
                           marker_size=11, height=430, style=basemap,
                           title="Both positions of every pair (lines join the two records of a pair)")
        line_cls = getattr(go, "Scattermap", None) or go.Scattermapbox
        for who, grp in nd.groupby("who"):
            nd_map.add_trace(line_cls(lat=grp["latitude"], lon=grp["longitude"], mode="lines", hoverinfo="skip",
                                      showlegend=False, line=dict(width=2, color=pair_colors[who])))
        nd_map.update_layout(legend_title_text="Pair")
        c2.plotly_chart(nd_map, width="stretch", config={"scrollZoom": True})
        far = pairs[pairs["pair_distance_km"] >= 1]
        near = pairs[pairs["pair_distance_km"] < 1]
        st.info(f"{len(near)} of {len(pairs)} pairs sit less than {near['pair_distance_km'].max() * 1000:,.0f} m "
                "apart: the same building geocoded twice. "
                + "".join(f"**{r['name']}** is the exception: its two positions are {r['pair_distance_km']:,.1f} km "
                          "apart. " for _, r in far.iterrows())
                + ("Reverse geocoding puts one Liefmans position in Meise (Vlaams-Brabant) and the other in "
                   "Oudenaarde, the city written in the record, so record `4d61de42…` carries the wrong coordinates "
                   "(shown here as recorded; every analysis now uses its twin's position)."
                   if set(NEAR_DUP_REVERSE_GEOCODE) <= set(nd["id"]) else ""))

    # ---- same name and location, different details ----
    st.subheader("Same name + location, different details")
    st.dataframe(NAME_DUPES, hide_index=True, width="stretch")
    st.caption("Circle 9 Brewing: different type/phone → distinct records. "
               "Paulaner: two websites only → treated as one in the unique dataset.")

    # ---- shared coordinates ----
    sc = qc.shared_coordinate_records(sel_t)
    locs = sc.drop_duplicates("location_id")
    st.subheader(f"Records sharing the exact same coordinates: {len(sc):,} records at {len(locs):,} points")
    if sc.empty:
        st.info("No two records share coordinates.")
        sc_text = ""
    else:
        kinds = locs["stack_kind"].value_counts()
        k = st.columns(4)
        k[0].metric("Shared points", f"{len(locs):,}")
        k[1].metric("Records involved", f"{len(sc):,}", help=f"{len(sc) / len(sel_t) * 100:.1f}% of {sel_key}")
        k[2].metric("Points: same street address", f"{int(kinds.get('same street address', 0)):,}")
        k[3].metric("Points: different addresses", f"{int(kinds.get('different street addresses', 0)):,}",
                    help=f"{int(sc['stack_kind'].eq('different street addresses').sum()):,} records")
        KIND_COLORS = {"same street address": "#2a78d6", "different street addresses": "#eb6834"}

        c1, c2 = st.columns(2)
        dist = locs.groupby(["stack_size", "stack_kind"]).size().reset_index(name="points")
        f = px.bar(dist, x="stack_size", y="points", color="stack_kind", text="points",
                   color_discrete_map=KIND_COLORS, title="How many records stack on one point?")
        f.update_layout(xaxis_title="Records at the point", yaxis_title="Points", legend_title_text="",
                        xaxis=dict(type="category"))
        c1.plotly_chart(f, width="stretch")
        city_tot = sel_t.groupby(["country", "city"]).size().rename("city_records")
        cs = sc.groupby(["country", "city"]).size().rename("stacked_records").to_frame().join(city_tot)
        cs["stacked_%"] = cs["stacked_records"] / cs["city_records"] * 100
        cs = cs.reset_index().nlargest(15, "stacked_records")
        cs["place"] = cs["city"] + ", " + cs["country"]
        cs["label"] = [f"{n} of {t} ({p:.0f}%)" for n, t, p in zip(cs["stacked_records"], cs["city_records"],
                                                                   cs["stacked_%"])]
        f = px.bar(cs.sort_values("stacked_records"), x="stacked_records", y="place", orientation="h", text="label",
                   color_discrete_sequence=[DATASET_COLORS["raw"]],
                   title="Cities with the most stacked records (label = share of the city's records)")
        f.update_traces(textposition="outside", cliponaxis=False)
        f.update_layout(xaxis_title="Records at shared points", yaxis_title="", height=460)
        c2.plotly_chart(f, width="stretch")

        c1, c2 = st.columns(2)
        for box, col, title in [(c1, "brewery_type", "Brewery types"), (c2, "country", "Countries")]:
            lf = qc.composition_lift(sc, sel_t, col).head(10)
            long = lf.melt(id_vars=[col, "lift", "records"], value_vars=["share_%", "overall_share_%"],
                           var_name="group", value_name="share")
            long["group"] = long["group"].map({"share_%": "Records at shared points",
                                               "overall_share_%": f"All of {sel_key}"})
            f = px.bar(long, x="share", y=col, color="group", barmode="group", orientation="h",
                       hover_data={"lift": ":.2f", "records": True},
                       color_discrete_map={"Records at shared points": "#eb6834", f"All of {sel_key}": "#2a78d6"},
                       title=f"{title}: share among stacked records vs overall")
            f.update_layout(yaxis=dict(autorange="reversed"), yaxis_title="", xaxis_title="% of records",
                            legend_title_text="", height=420)
            box.plotly_chart(f, width="stretch")

        loc_view = (sc.groupby("location_id")
                    .agg(latitude=("latitude", "first"), longitude=("longitude", "first"),
                         stack_size=("id", "size"), stack_kind=("stack_kind", "first"),
                         city=("city", "first"), state_province=("state_province", "first"),
                         country=("country", "first"), addresses=("addresses_at_point", "first"),
                         brewery_types=("brewery_type", lambda x: ", ".join(sorted(x.unique()))),
                         names=("name", lambda x: " | ".join(x.head(6)) + (" …" if len(x) > 6 else "")))
                    .reset_index())
        m = point_map(loc_view, color="stack_kind", size="stack_size", color_discrete_map=KIND_COLORS,
                      hover_name="names", hover_data=["city", "country", "stack_size", "addresses", "brewery_types"],
                      opacity=0.9, height=560, style=basemap, size_max=28,
                      title="Every shared point (size = records stacked, colour = same vs different street address)")
        m.update_layout(legend_title_text="")
        st.plotly_chart(m, width="stretch", config={"scrollZoom": True})
        big = loc_view[loc_view["stack_size"] >= 5].sort_values("stack_size", ascending=False)
        st.markdown("**Largest stacks (5 or more records at one point)**")
        st.dataframe(big.drop(columns=["location_id"]), hide_index=True, width="stretch")
        with st.expander(f"All {len(sc):,} records at shared points"):
            st.dataframe(sc, hide_index=True, width="stretch")

        top = big.iloc[0] if len(big) else loc_view.sort_values("stack_size").iloc[-1]
        top_addr = sc.loc[sc["location_id"] == top["location_id"], "address_1"].dropna()
        tl = qc.composition_lift(sc, sel_t, "brewery_type")
        over_t = tl[(tl["lift"] >= 2) & (tl["records"] >= 10)]
        cl = qc.composition_lift(sc, sel_t, "country")
        over_c = cl[(cl["lift"] >= 2) & (cl["records"] >= 10)].sort_values("lift", ascending=False)
        n_same = int(kinds.get("same street address", 0))
        n_diff_rec = int(sc["stack_kind"].eq("different street addresses").sum())
        diff_c = sc[sc["stack_kind"] == "different street addresses"]["country"].value_counts()
        sc_text = (
            f"\n\n**What are the {len(sc):,} records at shared points?** Mostly not duplicates and mostly not geocoding "
            f"errors. At **{n_same} of {len(locs)} points ({n_same / len(locs) * 100:.0f}%)** every record gives the "
            "same street address: several brands or businesses really operate from one site. The largest stack is "
            f"**{int(top['stack_size'])} records in {top['city']}, {top['country']}**"
            + (f" at {top_addr.iat[0]}" if len(top_addr) else "")
            + f", with brewery types {top['brewery_types']}: a shared brewing site, not a city centroid. Only "
            f"**{len(locs) - n_same} points ({n_diff_rec} records)** combine different street addresses, the "
            "signature of a geocoder falling back to a city or postcode center; "
            f"{diff_c.index[0]} holds {diff_c.iat[0]} of those records. These are the points that can fake local "
            "density in clustering.\n\n**Is there a pattern by type or country?** Yes. "
            + (("Over-represented types: " + ", ".join(f"**{r['brewery_type']}** ×{r['lift']:.1f}"
                                                        for _, r in over_t.iterrows())
                + " compared with their overall share. That fits how Open Brewery DB defines them: a contract "
                  "brewery uses another brewery's equipment and a planning brewery has not opened yet, so both often "
                  "list an address they share with someone else. ") if len(over_t) else "")
            + (("Over-represented countries: " + ", ".join(f"**{r['country']}** ×{r['lift']:.1f}"
                                                            for _, r in over_c.iterrows())
                + ". In those countries whole cities stack: the city chart shows what share of each city's records "
                  "sits on shared points.") if len(over_c) else ""))

    st.divider()
    st.subheader("Questions answered on this page")
    nd_text = ""
    if len(nd):
        pd_ = nd.drop_duplicates("pair")
        n_close = int((pd_["pair_distance_km"] < 1).sum())
        far_names = ", ".join(pd_.loc[pd_["pair_distance_km"] >= 1, "name"])
        nd_text = (f"\n\n**Which records look like the same brewery twice?** {nd['pair'].nunique()} pairs "
                   f"({len(nd)} records), all in {', '.join(nd['country'].unique())}, are identical in every field "
                   "except id and coordinates. They are kept because the rule for removal is full identity. The map "
                   f"and distance chart show {n_close} pairs are the same building geocoded twice"
                   + (f", while {far_names} has one misplaced position." if far_names else "."))
    st.markdown(f"**Are shared coordinates proof of duplicates?** No. Shared points mostly hold different businesses at one real address; coordinate equality alone is not a deduplication rule.{nd_text}{sc_text}\n\n**What do the same-name examples teach?** The two Circle 9 Brewing records differ in type and phone, so they are two businesses; the two Paulaner records differ only in website, so `unique_records` keeps one. Identity has to be decided field by field, never by coordinates alone.")

# ---------------- 4. Missing coordinates ----------------
with tabs[3]:
    raw = load(*RAW)
    if raw is None:
        st.info("raw_breweries.db not found.")
    else:
        raw["missing"] = raw["latitude"].isna() | raw["longitude"].isna()
        k = st.columns(3)
        k[0].metric("Raw records", f"{len(raw):,}")
        k[1].metric("Missing coordinates", f"{int(raw['missing'].sum()):,}")
        k[2].metric("Share of raw data", f"{raw['missing'].mean()*100:.1f}%")

        def rate(col, min_n=1):
            r = (raw.assign(**{col: raw[col].fillna("Unknown")}).groupby(col)
                 .agg(total=("id", "size"), missing=("missing", "sum")).reset_index())
            r["missing_%"] = (r["missing"] / r["total"] * 100).round(1)
            return r[r["total"] >= min_n].sort_values("missing_%", ascending=False)

        c1, c2 = st.columns(2)
        rt = rate("brewery_type")
        c1.plotly_chart(bar(rt, "missing_%", "brewery_type", "Missing-coordinate rate by type (%)",
                            hover_data=["total", "missing"]), width="stretch")
        rc = rate("country")
        c2.plotly_chart(bar(rc, "missing_%", "country", "Missing-coordinate rate by country (%)",
                            hover_data=["total", "missing"]), width="stretch")
        min_n = st.slider("Minimum records per state (rate view)", 1, 100, 25)
        rs = rate("state_province", min_n).head(25)
        st.plotly_chart(bar(rs, "missing_%", "state_province", f"Highest missing rates by state (≥{min_n} records)",
                            hover_data=["total", "missing"]), width="stretch")
        st.warning("**The loss is not random.** Planning breweries lose about 83% of their coordinates, Sweden 83%, "
                   "Canada 39% and the United States 28%, while Australia, Belgium and New Zealand lose none. Every "
                   "statement about geographic spread is a statement about geocoded breweries.")
        c3, c4 = st.columns(2)
        c3.markdown("**Counts by country**")
        c3.dataframe(rc.sort_values("missing", ascending=False), hide_index=True, width="stretch")
        c4.markdown("**Top cities by missing count**")
        c4.dataframe(rate("city").sort_values("missing", ascending=False).head(20),
                     hide_index=True, width="stretch")

    st.divider()
    st.subheader("Questions answered on this page")
    if raw is None:
        st.info("The raw database is unavailable, so missing-coordinate questions cannot be answered in this session.")
    else:
        missing_mask = raw["latitude"].isna() | raw["longitude"].isna()
        st.markdown(f"**How much source data lacks usable coordinates?** The raw table has **{int(missing_mask.sum()):,}** records missing at least one coordinate out of **{len(raw):,}** (**{missing_mask.mean()*100:.1f}%**). The charts separate counts from rates on purpose: a large count can simply mean a large group, while a high rate means the group really is poorly geocoded.\n\n**Is missingness evenly spread?** No. It depends on brewery type and on place, and the Data Quality page tests that formally. A missing coordinate is a gap in the record, not a missing brewery.")

# ---------------- 5. Geography ----------------
with tabs[4]:
    geo = df.dropna(subset=["latitude", "longitude"])
    # colours are assigned from the unfiltered data, so a type keeps its colour whatever the sidebar filters are
    type_cmap, type_order = type_style(df_full)
    focus = st.selectbox("Zoom to", ["All data"] + geo["country"].value_counts().index.tolist(),
                         help="Open the map on one country. Country and state/province borders and the names of "
                              "countries, states and cities are drawn by the map and appear as you zoom in.")
    view = None if focus == "All data" else core_view(geo[geo["country"] == focus])
    fig = point_map(geo, color="brewery_type", hover_name="name", hover_data=["city", "state_province", "country"],
                    color_discrete_map=type_cmap, category_orders=type_order, view=view, marker_size=6, style=basemap,
                    opacity=0.8, height=640, title="Brewery locations (zoom in for states and cities)")
    fig.update_layout(legend=dict(title="Brewery type", itemsizing="constant"))
    st.plotly_chart(fig, width="stretch", config={"scrollZoom": True})
    st.caption("Borders and place names come from the basemap chosen in the sidebar (OpenStreetMap or CARTO "
               "Positron): countries are labelled when zoomed out, state/province borders and city names appear as "
               "you zoom in. Every brewery type has its own colour, and each marker carries a dark ring so it never "
               "blends into the tiles.")
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.histogram(geo, x="latitude", nbins=60, marginal="box",
                                 title="Latitude distribution"), width="stretch")
    c2.plotly_chart(px.histogram(geo, x="longitude", nbins=60, marginal="box",
                                 title="Longitude distribution"), width="stretch")
    st.caption("Two longitude peaks: the Americas and Europe/Oceania. They mirror where the records come from, not a "
               "global law of brewing.")
    st.markdown("**Range validation**")
    st.table(pd.DataFrame({"min": geo[["latitude", "longitude"]].min(),
                           "max": geo[["latitude", "longitude"]].max(),
                           "invalid": [(~geo["latitude"].between(-90, 90)).sum(),
                                       (~geo["longitude"].between(-180, 180)).sum()]}))
    # ---- records whose latitude and longitude were swapped (from the corrections log) ----
    log = qc.load_corrections()
    swapped = log[(log["action"] == "swapped back") & log["id"].isin(df_full["id"])].merge(
        df_full[["id", "postal_code"]], on="id")
    swap_text = ""
    if len(swapped):
        st.subheader(f"{len(swapped)} records with latitude and longitude swapped (now corrected)")
        which = st.radio("Records on the map", ["The 2 records below −60° latitude",
                                                f"All {len(swapped)} swapped records"], horizontal=True,
                         key="swap_which")
        sw = swapped[swapped["original_latitude"] < -60] if which.startswith("The 2") else swapped
        sw = sw.assign(latitude=sw["original_latitude"], longitude=sw["original_longitude"],
                       latitude_swapped=sw["corrected_latitude"], longitude_swapped=sw["corrected_longitude"])
        g1, g2 = st.columns([3, 2])
        g1.plotly_chart(swap_map(sw), width="stretch", config={"scrollZoom": True})
        g1.caption(f"All {2 * len(sw)} positions on one globe: orange crosses = coordinates as recorded in the source, "
                   "blue circles = the same records with latitude and longitude swapped back (the position every "
                   "analysis now uses); dotted lines join the two positions of each brewery. A globe projection is "
                   "used because tile maps (Web Mercator) end at ±85.05° latitude and cannot draw a point recorded "
                   f"at {swapped['original_latitude'].min():.2f}°.")
        fixed = sw.assign(latitude=sw["latitude_swapped"], longitude=sw["longitude_swapped"])
        g2.plotly_chart(point_map(fixed, hover_name="name", hover_data=["city", "state_province", "country"],
                                  color_discrete_sequence=[SWAP_COLORS["Latitude and longitude swapped back"]],
                                  marker_size=13, height=520, style=basemap, text="name",
                                  title="Swapped-back positions on the street map"),
                        width="stretch", config={"scrollZoom": True})
        g2.caption("Zoom in: the tile map names the city and state under each corrected point.")
        evidence = swapped.assign(
            detected_by=np.where(swapped["original_latitude"] < -60, "latitude below -60° and city check",
                                 "city check only"),
            swapped_point_reverse_geocodes_to=swapped["id"].map(SWAP_REVERSE_GEOCODE).fillna("not checked"))
        st.dataframe(evidence[["detected_by", "name", "city", "state_province", "country", "postal_code",
                               "original_latitude", "original_longitude", "corrected_latitude", "corrected_longitude",
                               "km_to_city_center_original", "km_to_city_center_corrected",
                               "swapped_point_reverse_geocodes_to"]].round(4),
                     hide_index=True, width="stretch")
        st.caption("Evidence for each swap: as recorded, the record lies thousands of kilometres from the GeoNames "
                   "center of the city it names; with latitude and longitude exchanged it lies a few kilometres from "
                   "it. For the two records below −60° the swapped point was also reverse geocoded (OpenStreetMap "
                   "Nominatim, 2026-10-30) and returns the very city and state stored in the record. The databases "
                   "keep the original values; every page and the clustering use the corrected ones "
                   "(data/coordinate_corrections.csv).")
        ds_key = "all_records" if dataset == "All records" else "unique_records"
        rd = table("results.db", "record_distances")
        n_out = int(rd[rd["id"].isin(swapped["id"]) & (rd["dataset"] == ds_key)]["geographic_outlier"]
                    .astype(bool).sum()) if rd is not None else 0
        # where the corrected records land in the new clustering runs (labels are stored in allruns_records order)
        import clustering_store as cstore
        after = ""
        if not cstore.allruns_missing():
            order = cstore.allruns_records(ds_key).set_index("id")["position"]
            pos = order.reindex(swapped["id"]).dropna().astype(int).to_numpy()
            hits = [(rid, int((cstore.allruns_labels(f"{ds_key}|{rid}")[pos] != -1).sum()))
                    for rid in ("HDBSCAN|mcs=15|ms=4", "DBSCAN|eps=45|ms=15")]
            after = " After the correction, " + " and ".join(f"{n} of {len(pos)} sit in a cluster of the {r} run"
                                                            for r, n in hits) + "."
        by_country = swapped.groupby("country")["name"].apply(lambda x: ", ".join(x))
        swap_text = (
            f"\n\n**Are there impossible coordinates?** No value is outside the valid ranges, yet **{len(swapped)} "
            "records had latitude and longitude the wrong way round**: "
            + "; ".join(f"{c}: {n}" for c, n in by_country.items())
            + f". The first rule of the project, latitude below −60°, only caught the "
            f"{int((swapped['original_latitude'] < -60).sum())} US records, because their swapped values point to "
            "Antarctica. The others swap into plausible-looking places (Italy's records land in the Horn of Africa, "
            "Portugal's off the East African coast), so only checking each record against its own city center "
            "reveals them. Swapped back, every one of them lands within "
            f"{swapped['km_to_city_center_corrected'].max():.1f} km of the city written in the record, so the city, "
            "state and country fields were right and only the coordinate order was wrong. In the geographic results "
            f"saved before the correction, {n_out} of them were flagged as outliers, and every one was noise in the "
            "first clustering notebook's candidates. Every page and the clustering runs now use the corrected "
            "positions." + after)

    st.divider()
    st.subheader("Questions answered on this page")
    st.markdown(f"**Where are the breweries in this selection?** The map plots **{len(geo):,} records with valid latitude and longitude** after the current country and type filters, using the corrected coordinates.\n\n**What can the map and the histograms tell us, and what not?** Where the records are and where they crowd: two bands of longitude (the Americas and Europe/Oceania) and a northern mid-latitude belt. They do not adjust for population or land area, and a degree of longitude covers fewer kilometres the farther north you go.\n\n**Why do points overlap?** Most overlaps are several businesses at one street address (brewing collectives, shared sites); a few are addresses geocoded to a town center. The Duplicates & Identity tab separates the two before a dense spot is read as a hub.{swap_text}")

# ---------------- 6. Concentration ----------------
with tabs[5]:
    sc = df.groupby(["country", STATE]).size().sort_values(ascending=False)  # keyed by country + harmonized state
    c1, c2 = st.columns(2)
    topn = pd.DataFrame({"top_n_states": [5, 10, 20]})
    topn["records"] = topn["top_n_states"].apply(lambda n: sc.head(n).sum())
    topn["share_%"] = (topn["records"] / len(df) * 100).round(2)
    c1.markdown("**Share held by the top-N states / provinces**")
    c1.dataframe(topn, hide_index=True)
    q = pd.DataFrame({"metric": ["mean", "median", "min", "max", "P25", "P75", "P90"],
                      "breweries_per_state": [sc.mean(), sc.median(), sc.min(), sc.max(),
                                              sc.quantile(.25), sc.quantile(.75), sc.quantile(.9)]}).round(1)
    c2.markdown("**Distribution across states**")
    c2.dataframe(q, hide_index=True)
    st.caption("Mean ≫ median ⇒ heavy right skew: a few states dominate, most have only a handful.")
    cum = pd.DataFrame({"rank": range(1, len(sc) + 1), "cumulative_share_%": sc.cumsum().values / len(df) * 100})
    st.plotly_chart(px.line(cum, x="rank", y="cumulative_share_%",
                            title="Pareto curve — cumulative share of breweries by state rank"),
                    width="stretch")
    cs = df.groupby(["country", STATE, "city"]).size().reset_index(name="city_count")
    cs["state_total"] = cs.groupby(["country", STATE])["city_count"].transform("sum")
    cs["city_share_of_state_%"] = (cs["city_count"] / cs["state_total"] * 100).round(1)
    big = cs[cs["state_total"] >= 20].nlargest(20, "city_share_of_state_%")
    big["location"] = big["city"] + " — " + big[STATE]
    st.plotly_chart(bar(big, "city_share_of_state_%", "location",
                        "Most city-dominated states (states with ≥20 breweries)",
                        hover_data=["city_count", "state_total"]), width="stretch")

    st.divider()
    st.subheader("Questions answered on this page")
    top5_share = sc.head(5).sum() / len(df) * 100
    top10_share = sc.head(10).sum() / len(df) * 100
    st.markdown(f"**How concentrated are records across state/province groups?** The top 5 groups account for **{top5_share:.1f}%** and the top 10 for **{top10_share:.1f}%** of records in the current selection. The Pareto curve rises steeply and the mean is far above the median: a handful of states carry the dataset and most states hold only a few records.\n\n**Does a large state mean one dominant city?** No, and the two questions have to be kept apart. A state can be large because of one hub or because of many small towns; the city-share chart measures the first, for states with at least 20 records so that tiny states do not look dominated by accident. The Countries, States & Cities page follows this up city by city.\n\n**What do these counts measure?** Records in Open Brewery DB. They show where the database is dense, which tracks where breweries are, but they are not a census or a market share.")

# ---------------- 7. All vs Unique ----------------
with tabs[6]:
    A, U = (load(*SOURCES["All records"]), load(*SOURCES["Unique records"]))
    if A is None or U is None:
        st.info("Both analysis databases are required.")
    else:
        cmp_ = pd.DataFrame({"metric": ["Records", "Countries", "States/provinces", "Cities", "Brewery types"],
                             "all": [len(A)] + [A[c].nunique() for c in ["country", "state_province", "city", "brewery_type"]],
                             "unique": [len(U)] + [U[c].nunique() for c in ["country", "state_province", "city", "brewery_type"]]})
        cmp_["difference"] = cmp_["all"] - cmp_["unique"]
        st.dataframe(cmp_, hide_index=True, width="stretch")
        tc = pd.concat([A["brewery_type"].value_counts().rename("all"),
                        U["brewery_type"].value_counts().rename("unique")], axis=1).fillna(0)
        tc["difference"] = tc["all"] - tc["unique"]
        st.dataframe(tc.sort_values("all", ascending=False), width="stretch")
        sd = pd.concat([A["state_province"].value_counts().rename("all"),
                        U["state_province"].value_counts().rename("unique")], axis=1).fillna(0)
        sd["difference"] = sd["all"] - sd["unique"]
        st.markdown("**Only differences across states / countries**")
        st.dataframe(sd[sd["difference"] != 0], width="stretch")
        st.success("The two datasets differ by one record, a Paulaner brewpub in München, Bayern. No share moves by "
                   "as much as 0.01 percentage points.")

    st.divider()
    st.subheader("Questions answered on this page")
    st.markdown(f"**Does the deduplication choice change the story?** No. The tables compare records, countries, states, cities and types: the only difference is one Paulaner record, and no share moves by 0.01 percentage points.\n\n**Then why keep both versions?** Because we did not know that before checking. Keeping `unique_records` next to `all_records` turned a decision we could have argued about into one we measured, here and again in the dispersion and clustering results, where the two datasets also agree. What it cannot fix are the larger biases: missing coordinates and the US-heavy sample.")

# ---------------- 8. Methodology ----------------
with tabs[7]:
    st.markdown("""
**Data preprocessing**

The project performs data-quality checks, handles duplicate records, and uses case and whitespace normalization when comparing the `state` and `state_province` fields.

For geographic analysis, records with incomplete coordinates are excluded rather than having missing locations imputed. Spatial distances are calculated using the Haversine metric, with DBSCAN's neighbourhood radius converted from kilometres to angular distance as required.

Conventional numerical feature scaling is not described as part of the current geographic clustering workflow. This preserves the interpretation of the clustering parameters in geographic terms.

Duplicate handling, text normalization, and geographic distance conversion are treated as distinct preprocessing operations.

**Source.** Open Brewery DB API, read in pages of 200 records and stored in SQLite (`raw_breweries.db`, 11,959
records). Everything else is derived from that snapshot plus two reference sources: GeoNames (capitals and city
centers, CC BY 4.0) and, for 68 towns GeoNames lacks, OpenStreetMap Nominatim (ODbL).

**Decisions, in the order we made them**
1. Exact duplicates (all 15 columns equal except `id`): 3 pairs, first row kept.
2. Records without coordinates (2,468) are left out of spatial work, not imputed: location is what we study.
3. Two analysis datasets: `all_records` (9,488) and `unique_records` (9,487); they differ by one Paulaner record.
4. `state` duplicates `state_province` and `street` duplicates `address_1` on every record; we use `state_province`
   and `address_1`.
5. Region labels are harmonized (Belgium's Dutch, French and English names; `MIssouri`) in an extra column, and
   one record filed under the wrong province (Frontier Beer Company, Cape Town, labelled Gauteng) is moved to
   Western Cape there; the source column is kept.
6. Coordinates are checked against the GeoNames center of each record's own city (or its state capital when the
   city has no center). Eight swapped latitude/longitude pairs are swapped back, one longitude with the wrong sign
   (Mescan Brewery, Mayo, stored in northern Germany) is reversed, one misplaced Liefmans record takes its twin's
   position, and 18 records more than 50 km from their city are flagged and kept. The log is
   `data/coordinate_corrections.csv`; the databases keep the original values.
7. Records sharing a coordinate are kept: 299 of the 336 shared points are one street address used by several
   businesses, not geocoding errors.

**Where each method lives**
- Data Quality: missing, unique and shared values, chi-square tests of missingness, state labels, names vs websites.
- Geographic Analysis: centers, distances and dispersion around two origins (brewery center and capital/city
  center), coordinate checks.
- Brewery Types and Countries, States & Cities: frequencies, type mix, location quotients, type A/B cities.
- Spatial Clustering: every DBSCAN, HDBSCAN and OPTICS setting, hub merges, stability.
- Prediction: place and type probabilities, kernel density, the planning pipeline.

**Limits worth keeping in mind**
- Records without coordinates are not random (planning breweries, Texas, Vermont, Sweden lose the most), so spatial
  results describe geocoded breweries.
- The data is US-heavy (64% of all_records) and each country labels brewery types its own way.
- Counts are records in a community database, not a census of breweries or a market share.
- There are no opening dates. The prediction page states where the next brewery is most likely under the
  assumption that it resembles today's, and tests that assumption on the planning breweries.
""")


    st.divider()
    st.subheader("Questions answered on this page")
    st.markdown("""
**Why three clustering algorithms?**\\
DBSCAN is the easiest to read: one radius (`eps`, in kilometres) and one density threshold (`min_samples`). Its weak
point showed up clearly in our grid: from about 30 km a single radius chains neighbouring hubs into corridors, and at
45 km one cluster runs from Brussels to München. HDBSCAN builds the hierarchy of every radius at once and keeps the
clusters that last longest, so it separates hubs of different density; a flat cut of its tree reproduces DBSCAN
exactly on core points. OPTICS orders breweries by reachability and finds the tightest local groups.

**How did we judge a clustering?**\\
Not by one number. Every one of the 1,266 stored runs is described by its noise share, how many clusters join hubs
more than 100 km apart, how far its clusters spread, and whether its clusters reappear when a parameter moves one
step. The Spatial Clustering page weighs these openly and picks the best run of each algorithm for the weights you
choose. Silhouette scores are not used: noise labels and long, chain-shaped clusters make them misleading here.

**Which setting is "the" answer?**\\
None was fixed in advance. The first notebook's working choices (DBSCAN 45 km / 15, HDBSCAN 15 / 4) are still in the
grid; the second one merges no hubs, the first merges eight.

**Can the data say where a new brewery will open?**\\
Yes, as a probability. The Prediction page gives the most likely country, state and city and the most likely type,
validated on breweries the model never saw and on the 634 breweries in planning. It cannot say how many will open
or when, because the data has no opening dates.
""")

