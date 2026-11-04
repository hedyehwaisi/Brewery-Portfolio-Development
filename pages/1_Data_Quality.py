import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import quality_core as qc
import stats_store
from common import table, pick_basemap, DATASET_FILES, shared_cache, stats_table
from geo_plots import point_map, SATURATED_SCALE
from palette import DATASET_COLORS, color_map

st.set_page_config(page_title="Data Quality", page_icon="🧪", layout="wide")
st.title("🧪 Data Quality — missing, unique and shared values, state labels")
basemap = pick_basemap()


@shared_cache
def frames():
    """The three dataset versions exactly as stored (no filling, so real nulls stay visible)."""
    return {name: table(*src) for name, src in DATASET_FILES.items()}


@shared_cache
def shared_groups(name, col):
    return qc.shared_value_groups(frames()[name], col)


@shared_cache
def shared_summary(name):
    """One row per column: how many values are shared and how far the sharing groups spread."""
    rows = []
    for col in frames()[name].columns:
        g = shared_groups(name, col)
        if g.empty:
            continue
        top = g.iloc[0]
        rows.append({"column": col, "values_shared": len(g), "records_involved": int(g["records"].sum()),
                     "groups_over_1_city": int((g["cities"] > 1).sum()),
                     "groups_over_1_state": int((g["states"] > 1).sum()),
                     "groups_over_1_country": int((g["countries"] > 1).sum()),
                     "groups_over_1_type": int((g["brewery_types"] > 1).sum()),
                     "largest_group_value": str(top[col]), "largest_group_records": int(top["records"])})
    return pd.DataFrame(rows)


F = frames()
if any(v is None for v in F.values()):
    st.error("raw_breweries.db, analysis_all_records.db and analysis_unique_records.db must all be in data/.")
    st.stop()

raw = F["raw"].copy()
for c in ["brewery_type", "country", "state_province", "city"]:
    raw[c] = raw[c].astype("string").fillna("Unknown")
raw["missing"] = raw["latitude"].isna() | raw["longitude"].isna()

tabs = st.tabs(["Missing coordinates", "Missing & null values", "Unique values", "Shared values", "State labels",
                "Names vs websites"])

# ======================================================================= 1. missing coordinates
with tabs[0]:
    k = st.columns(4)
    k[0].metric("Raw records", f"{len(raw):,}")
    k[1].metric("Missing lat/lon", f"{int(raw['missing'].sum()):,}")
    k[2].metric("Share of raw data", f"{raw['missing'].mean()*100:.1f}%")
    k[3].metric("Only one coordinate missing", int((raw["latitude"].isna() ^ raw["longitude"].isna()).sum()))
    st.caption("Only the raw table has missing coordinates: all_records and unique_records keep just the records "
               "with both latitude and longitude, so their missing-coordinate rate is 0% everywhere.")

    st.header("Who is missing coordinates?")
    dims = {"Brewery type": "brewery_type", "Country": "country", "State / province": "state_province", "City": "city"}
    label = st.radio("Break down by", list(dims), horizontal=True)
    col = dims[label]
    g = raw.groupby(col).agg(total=("id", "size"), missing=("missing", "sum")).reset_index()
    g["missing_%"] = (g["missing"] / g["total"] * 100).round(1)
    g["share_of_all_missing_%"] = (g["missing"] / g["missing"].sum() * 100).round(1)
    a, b = st.columns(2)
    top = a.slider("Top N", 5, 40, 15)
    min_n = b.slider("Minimum records per category (rate chart)", 1, 100, 10)
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.bar(g.nlargest(top, "missing").sort_values("missing"), x="missing", y=col, orientation="h",
                           text="missing", hover_data=["total", "missing_%"],
                           title=f"Most missing records by {label.lower()}"), width="stretch")
    c2.plotly_chart(px.bar(g[g["total"] >= min_n].nlargest(top, "missing_%").sort_values("missing_%"),
                           x="missing_%", y=col, orientation="h", text="missing_%", hover_data=["total", "missing"],
                           title=f"Highest missing rate by {label.lower()} (≥{min_n} records)"),
                    width="stretch")
    st.dataframe(g.sort_values("missing", ascending=False), hide_index=True, width="stretch")

    st.subheader("Missing-coordinate rate within each category")
    st.caption("`coordinate_missing_rate(raw, column)` for brewery type, state/province, country and city: total "
               "records, records missing a coordinate, and the percentage, sorted by percentage and then size.")
    rate_tables = {c: qc.coordinate_missing_rate(F["raw"], c)
                   for c in ["brewery_type", "state_province", "country", "city"]}
    cfg = {"missing_coordinate_percentage": st.column_config.NumberColumn("missing_coordinate_%", format="%.2f")}
    r1, r2 = st.columns(2)
    for box, c in zip([r1, r2, r1, r2], rate_tables):
        box.markdown(f"**missing_rate_by_{'state' if c == 'state_province' else c.replace('brewery_', '')}**")
        box.dataframe(rate_tables[c], hide_index=True, width="stretch", height=280, column_config=cfg)

    st.subheader("Where the missing coordinates are, by state/province")
    located = F["raw"].dropna(subset=["latitude", "longitude"])
    centers = located.groupby(["country", "state_province"])[["latitude", "longitude"]].mean()
    by_state = (raw.groupby(["country", "state_province"])
                .agg(records=("id", "size"), missing=("missing", "sum")).join(centers).reset_index())
    by_state["missing_%"] = by_state["missing"] / by_state["records"] * 100
    shown = by_state[(by_state["missing"] > 0) & by_state["latitude"].notna()]
    unplaced = by_state[(by_state["missing"] > 0) & by_state["latitude"].isna()]
    m = point_map(shown, size="missing", color="missing_%", color_continuous_scale=SATURATED_SCALE,
                  range_color=(0, 100), hover_name="state_province",
                  hover_data={"country": True, "records": ":,", "missing": ":,", "missing_%": ":.1f"},
                  size_max=40, opacity=0.9, height=560, style=basemap,
                  title="Records without coordinates per state/province (size = count, colour = rate %)")
    m.update_layout(coloraxis_colorbar=dict(title="Missing %", thickness=14))
    st.plotly_chart(m, width="stretch", config={"scrollZoom": True})
    st.caption("Each bubble sits at the mean position of the state's records that do have coordinates. "
               + (f"{len(unplaced)} state(s) have no located record at all and cannot be placed: "
                  + ", ".join(f"{r['state_province']} ({r['country']}, {int(r['missing'])} missing)"
                              for _, r in unplaced.iterrows()) + "." if len(unplaced) else ""))

    st.subheader("Hierarchy of missing records: country → state/province → type")
    miss = raw[raw["missing"]]
    st.plotly_chart(px.treemap(miss, path=[px.Constant("All missing"), "country", "state_province", "brewery_type"]),
                    width="stretch")
    topc = miss["country"].value_counts().head(8).index
    ct = pd.crosstab(miss[miss["country"].isin(topc)]["brewery_type"], miss[miss["country"].isin(topc)]["country"])
    st.plotly_chart(px.imshow(ct, text_auto=True, aspect="auto", color_continuous_scale="Oranges",
                              title="Missing records: brewery type × country (top 8 countries)"),
                    width="stretch")

    st.divider()
    st.subheader("Questions answered in this tab")
    rt = rate_tables["brewery_type"].query("total_records >= 30")
    worst_t = rt.iloc[0]
    rc = rate_tables["country"].query("total_records >= 30")
    us_states = rate_tables["state_province"].merge(
        F["raw"][["state_province", "country"]].drop_duplicates(), on="state_province")
    us_states = us_states[(us_states["country"] == "United States") & (us_states["total_records"] >= 30)]
    addr_m = F["raw"]["address_1"].isna()
    zero_c = rc[rc["missing_coordinate_percentage"] == 0]["country"].tolist()
    zero_s = us_states[us_states["missing_coordinate_percentage"] == 0]["state_province"].tolist()
    plan_addr = F["raw"].loc[F["raw"]["brewery_type"] == "planning", "address_1"].isna().mean() * 100
    st.markdown(f"""
**Why can’t we use every source record for geographic analysis?**\\
The raw table holds **{len(raw):,} records**; **{int(raw['missing'].sum()):,}** have no usable coordinate (**{raw['missing'].mean()*100:.1f}%**). Distances and clusters need a position, so these records are left out of coordinate-based work rather than given guessed locations. They stay in the raw database.

**Is coordinate missingness random?**\\
No. **{worst_t['brewery_type']}** records lose **{worst_t['missing_coordinate_percentage']:.0f}%** of their coordinates, against {raw['missing'].mean()*100:.0f}% overall. Among countries with at least 30 records the rate runs from **0%** ({len(zero_c)} countries: {", ".join(zero_c)}) to **{rc.iloc[0]['missing_coordinate_percentage']:.1f}%** ({rc.iloc[0]['country']}). Inside the United States, states with at least 30 records range from **0%** ({", ".join(zero_s) or "none"}) to **{us_states.iloc[0]['missing_coordinate_percentage']:.1f}%** ({us_states.iloc[0]['state_province']}). The *Missing & null values* tab tests this formally.

**What explains the gaps?**\\
Two things. First, a missing street address: a record without `address_1` has no coordinates **{raw.loc[addr_m, 'missing'].mean()*100:.0f}%** of the time, and **{plan_addr:.0f}%** of planning breweries have no address, which is why that type loses most of its positions. Second, and larger, records that do have an address were simply never geocoded: only **{addr_m[raw['missing']].mean()*100:.1f}%** of the coordinate-less records lack an address. Those cluster in particular states, which points to how each region's records were collected rather than to the breweries themselves.

**Does a missing coordinate mean a brewery is not real?**\\
No. It means the source gives no position. It is a coverage limit, and geographic results describe the geocoded subset.
""")

# ======================================================================= 2. missing & null values
with tabs[1]:
    st.subheader("Missing values in every column of the three datasets")
    rows = [{"column": c, "dataset": n, "missing": int(d[c].isna().sum()), "pct": d[c].isna().mean() * 100}
            for n, d in F.items() for c in d.columns]
    mv = pd.DataFrame(rows)
    mv = mv[mv["column"].isin(mv.groupby("column")["missing"].max().loc[lambda s: s > 0].index)]
    heat = mv.pivot(index="column", columns="dataset", values="pct")[list(F)]
    text = mv.assign(t=[f"{m:,}<br>({p:.1f}%)" for m, p in zip(mv["missing"], mv["pct"])]) \
        .pivot(index="column", columns="dataset", values="t")[list(F)]
    heat = heat.loc[heat["raw"].sort_values(ascending=False).index]
    hf = px.imshow(heat, color_continuous_scale="Blues", zmin=0, zmax=100, aspect="auto",
                   labels={"color": "missing %", "x": "", "y": ""})
    hf.update_traces(text=text.loc[heat.index].values, texttemplate="%{text}")
    hf.update_layout(height=60 * len(heat) + 80, coloraxis_colorbar=dict(thickness=12))
    st.plotly_chart(hf, width="stretch")
    st.caption("Cell = missing records (share of that dataset). Columns never missing anywhere are left out.")

    ds = st.radio("Dataset to investigate", list(F), horizontal=True, key="null_ds")
    d = F[ds]
    targets = stats_store.missing_targets(d)

    st.subheader("Is missingness random? Chi-square test by brewery type, country, state and city")
    tests = stats_store.missingness(stats_table("missingness"), ds)  # chi-square tests precomputed (stats_store)
    if tests.empty:
        st.info("Nothing to test in this dataset.")
    else:
        vmat = tests.pivot(index="column", columns="by", values="cramers_v")[qc.CATEGORY_COLUMNS]
        vtxt = tests.assign(t=[f"V={v:.2f}<br>{'random' if 'random' in s else 'not random' if 'not' in s else 'weak'}"
                               for v, s in zip(tests["cramers_v"], tests["verdict"])]) \
            .pivot(index="column", columns="by", values="t")[qc.CATEGORY_COLUMNS]
        vf = px.imshow(vmat, color_continuous_scale="Purples", zmin=0, zmax=1, aspect="auto",
                       labels={"color": "Cramér's V", "x": "grouped by", "y": "missing column"})
        vf.update_traces(text=vtxt.loc[vmat.index].values, texttemplate="%{text}")
        vf.update_layout(height=70 * len(vmat) + 100, coloraxis_colorbar=dict(thickness=12))
        st.plotly_chart(vf, width="stretch")
        st.dataframe(tests.round(4), hide_index=True, width="stretch")
        st.caption("Each cell tests whether a column is missing equally often in every category. Categories with "
                   "fewer than 30 records are pooled into one 'other' group so the test stays valid. Cramér's V "
                   "measures the strength of the link: below 0.1 negligible, 0.1–0.3 weak, 0.3–0.5 moderate, above 0.5 "
                   "strong. 'not random' = p < 0.001 and V ≥ 0.1.")

    st.subheader("Drill down: missing rate of one column across one grouping")
    c1, c2, c3 = st.columns(3)
    tgt = c1.selectbox("Missing column", targets, key="drill_t") if targets else None
    by = c2.selectbox("Group by", qc.CATEGORY_COLUMNS, key="drill_by")
    mn = c3.slider("Minimum records per group", 1, 100, 20, key="drill_min")
    if tgt:
        mr = qc.missing_rate_by(d, tgt, by)
        top_rate = mr[mr["records"] >= mn].sort_values("missing_%", ascending=False).head(20)
        top_rate["label"] = [f"{m:,} of {r:,} ({p:.0f}%)" for m, r, p in
                             zip(top_rate["missing"], top_rate["records"], top_rate["missing_%"])]
        f = px.bar(top_rate.sort_values("missing_%"), x="missing_%", y=by, orientation="h", text="label",
                   color_discrete_sequence=[DATASET_COLORS[ds]],
                   title=f"{tgt}: highest missing rate by {by} (groups with ≥ {mn} records)")
        f.update_traces(textposition="outside", cliponaxis=False)
        f.update_layout(xaxis_range=[0, 125], xaxis_title="missing %", yaxis_title="", height=560)
        st.plotly_chart(f, width="stretch")
        st.dataframe(mr.round(2), hide_index=True, width="stretch", height=300)

    st.subheader("Do gaps travel together?")
    cm = qc.co_missingness(d)
    if cm.empty:
        st.info("Fewer than two columns have missing values in this dataset.")
    else:
        cm = cm[cm["records_missing_a"] >= 20]
        pm = cm.pivot(index="if_missing", columns="then_missing", values="p_b_given_a_%")
        cf = px.imshow(pm.round(1), text_auto=True, color_continuous_scale="Blues", zmin=0, zmax=100, aspect="auto",
                       labels={"color": "% also missing", "x": "…then this is missing", "y": "If this is missing…"})
        st.plotly_chart(cf, width="stretch")
        st.dataframe(cm.sort_values("ratio", ascending=False).round(2), hide_index=True, width="stretch",
                     height=280)
        st.caption("Read a row: among records missing the row column, the share that also miss the column above. "
                   "`ratio` compares that share with the column's overall missing rate; well above 1 means the two "
                   "gaps come together. Only columns missing in at least 20 records are used as conditions.")

    st.divider()
    st.subheader("Questions answered in this tab")
    t_raw = stats_store.missingness(stats_table("missingness"), "raw")
    t_raw = t_raw[t_raw["column"].isin(["coordinates", "phone", "website_url", "address_1"])]
    strongest = t_raw.sort_values("cramers_v", ascending=False).groupby("column").head(1).set_index("column")
    phone_full = qc.missing_rate_by(F["all_records"], "phone", "country").sort_values("missing_%", ascending=False)
    phone_full = phone_full[phone_full["records"] >= 30].iloc[0]
    addr_type = qc.missing_rate_by(F["all_records"], "address_1", "brewery_type").sort_values("missing_%",
                                                                                              ascending=False).iloc[0]
    co = qc.co_missingness(F["raw"]).set_index(["if_missing", "then_missing"])
    st.markdown(f"""
**Are the null values random accidents?**\\
No, not for the fields that matter. In the raw table every gap tested (coordinates, phone, website, street address) depends on where the brewery is or what kind of brewery it is (p < 0.001). The strongest links: coordinates by **{strongest.loc['coordinates', 'by']}** (V = {strongest.loc['coordinates', 'cramers_v']:.2f}), phone by **{strongest.loc['phone', 'by']}** (V = {strongest.loc['phone', 'cramers_v']:.2f}), website by **{strongest.loc['website_url', 'by']}** (V = {strongest.loc['website_url', 'cramers_v']:.2f}) and street address by **{strongest.loc['address_1', 'by']}** (V = {strongest.loc['address_1', 'cramers_v']:.2f}).

**Which information is systematically ignored, and where?**\\
Phone numbers are a country habit: in all_records **{phone_full['country']}** has no phone number for **{phone_full['missing_%']:.0f}%** of its {int(phone_full['records'])} records. The street address is a brewery-type habit: **{addr_type['brewery_type']}** records lack it **{addr_type['missing_%']:.0f}%** of the time.

**Do the gaps come together?**\\
Yes. In the raw table, a record without `address_1` also lacks coordinates **{co.loc[('address_1', 'coordinates'), 'p_b_given_a_%']:.0f}%** of the time, {co.loc[('address_1', 'coordinates'), 'ratio']:.1f}× the overall coordinate gap; records without coordinates lack a phone number **{co.loc[('coordinates', 'phone'), 'ratio']:.1f}×** and a website **{co.loc[('coordinates', 'website_url'), 'ratio']:.1f}×** more often than average. `street` and `address_1` are missing together every time because they are the same field (see *State labels*).

**What does this change for the analysis?**\\
The geocoded datasets under-represent planning breweries and the places with high gap rates, so every geographic count describes recorded, geocoded breweries rather than all breweries.
""")

# ======================================================================= 3. unique values
with tabs[2]:
    uv = qc.unique_values(F)
    st.subheader("Distinct values per column in raw, all_records and unique_records")
    uv["label"] = [f"{u:,} ({p:.1f}%)" for u, p in zip(uv["unique_values"], uv["unique_%"])]
    order = uv[uv["dataset"] == "raw"].sort_values("unique_%", ascending=False)["column"].tolist()
    f = px.bar(uv, x="unique_%", y="column", color="dataset", barmode="group", orientation="h", text="label",
               color_discrete_map=DATASET_COLORS, category_orders={"column": order, "dataset": list(DATASET_COLORS)},
               log_x=True, hover_data={"unique_values": ":,", "records": ":,", "records_sharing_a_value": ":,"})
    f.update_traces(textposition="outside", cliponaxis=False)
    f.update_layout(height=30 * len(uv) + 140, xaxis_title="Distinct values as % of the dataset's records (log scale)",
                    yaxis_title="", legend_title_text="Dataset", bargap=0.2)
    st.plotly_chart(f, width="stretch")
    wide = uv.pivot(index="column", columns="dataset",
                    values=["unique_values", "unique_%", "values_used_more_than_once", "records_sharing_a_value"])
    wide.columns = [f"{d}_{m}" for m, d in wide.columns]
    st.dataframe(wide.loc[order].round(2), width="stretch")
    st.caption("`unique_%` = distinct non-null values ÷ records. `values_used_more_than_once` counts values held by "
               "two or more records, and `records_sharing_a_value` counts those records. The *Shared values* tab "
               "opens up each of these groups.")

    st.divider()
    st.subheader("Questions answered in this tab")
    r_uv = uv[uv["dataset"] == "raw"].set_index("column")
    cg = F["raw"].groupby(["country", "state_province", "city"]).size()
    a_uv = uv[uv["dataset"] == "all_records"].set_index("column")
    st.markdown(f"""
**Which columns identify a brewery?**\\
Only `id` is unique on every record. `name` comes close (**{r_uv.loc['name', 'unique_values']:,}** distinct names for {len(F['raw']):,} raw records), which still leaves **{r_uv.loc['name', 'records_sharing_a_value']:,}** records sharing a name with another one: chains, and the near-duplicates on the Duplicates tab.

**Which columns are categories?**\\
`brewery_type` has {r_uv.loc['brewery_type', 'unique_values']} values, `country` {r_uv.loc['country', 'unique_values']}, `state_province` {r_uv.loc['state_province', 'unique_values']}. `city` has **{r_uv.loc['city', 'unique_values']:,}** distinct names in raw. Keyed by country + state + city there are **{len(cg):,}** cities, and **{(cg == 1).mean()*100:.0f}%** of them hold a single brewery ({(cg <= 2).mean()*100:.0f}% hold one or two): the city level is a long tail.

**What do the coordinates say?**\\
all_records holds {len(F['all_records']):,} records but only **{a_uv.loc['latitude', 'unique_values']:,}** distinct latitudes and **{a_uv.loc['longitude', 'unique_values']:,}** distinct longitudes, because hundreds of records share a point (see Duplicates & Identity).

**Does cleaning change the picture?**\\
Hardly. Going from raw to all_records removes the records without coordinates, which lowers every count but keeps the ranking of columns the same. all_records and unique_records differ by the single Paulaner record.
""")

# ======================================================================= 4. shared values
with tabs[3]:
    st.subheader("Records that share the same value")
    ds = st.radio("Dataset", list(F), horizontal=True, index=1, key="shared_ds")
    summ = shared_summary(ds)
    st.markdown("**Overview: every column with shared values**")
    st.dataframe(summ, hide_index=True, width="stretch")
    st.caption("A group = all records holding one value of the column. `groups_over_1_city` counts groups whose "
               "records sit in more than one city (same for state, country and brewery type).")

    cols = summ["column"].tolist()
    col = st.selectbox("Column to investigate", cols, index=cols.index("website_url") if "website_url" in cols else 0)
    g = shared_groups(ds, col)
    d = F[ds]
    in_groups = d[d[col].isin(g[col])]
    k = st.columns(5)
    k[0].metric("Values shared", f"{len(g):,}")
    k[1].metric("Records involved", f"{len(in_groups):,}", help=f"{len(in_groups) / len(d) * 100:.1f}% of {ds}")
    k[2].metric("Groups across >1 city", f"{int((g['cities'] > 1).sum()):,}")
    k[3].metric("Groups across >1 state", f"{int((g['states'] > 1).sum()):,}")
    k[4].metric("Groups across >1 country", f"{int((g['countries'] > 1).sum()):,}")
    st.dataframe(g.head(300), hide_index=True, width="stretch", height=320)

    c1, c2 = st.columns(2)
    for box, by, title in [(c1, "brewery_type", "Brewery types"), (c2, "country", "Countries")]:
        lf = qc.composition_lift(in_groups, d, by).head(10)
        long = lf.melt(id_vars=[by, "lift", "records"], value_vars=["share_%", "overall_share_%"], var_name="group",
                       value_name="share")
        long["group"] = long["group"].map({"share_%": f"Records sharing a {col}", "overall_share_%": f"All of {ds}"})
        f = px.bar(long, x="share", y=by, color="group", barmode="group", orientation="h",
                   hover_data={"lift": ":.2f", "records": True},
                   color_discrete_map={f"Records sharing a {col}": "#eb6834", f"All of {ds}": "#2a78d6"},
                   title=f"{title}: records sharing a {col} vs all records")
        f.update_layout(yaxis=dict(autorange="reversed"), yaxis_title="", xaxis_title="% of records",
                        legend_title_text="", height=420)
        box.plotly_chart(f, width="stretch")

    n_top = st.slider("Groups to map", 3, 20, 10, key="shared_top")
    top_vals = g[col].head(n_top).tolist()
    pts = d[d[col].isin(top_vals)].dropna(subset=["latitude", "longitude"]).copy()
    if pts.empty:
        st.info("None of these records has coordinates.")
    else:
        pts["group"] = pts[col].astype(str).str.slice(0, 45)
        m = point_map(pts, color="group", color_discrete_map=color_map(pts["group"].unique()), hover_name="name",
                      hover_data=["city", "state_province", "country", "brewery_type"], marker_size=9, height=560,
                      style=basemap, title=f"Records in the {n_top} largest {col} groups")
        m.update_layout(legend_title_text=col)
        st.plotly_chart(m, width="stretch", config={"scrollZoom": True})
    pick = st.selectbox("Show the records of one group", top_vals, key="shared_pick")
    st.dataframe(d[d[col] == pick], hide_index=True, width="stretch")

    st.divider()
    st.subheader("Questions answered in this tab")
    S = summ.set_index("column")
    web = shared_groups(ds, "website_url")
    web_chain = web[web["cities"] > 1].head(3)
    city_g = shared_groups(ds, "city")
    multi_state_city = city_g[city_g["states"] > 1].head(3)
    st.markdown(f"""
**What does a shared website reveal?**\\
Chains. **{S.loc['website_url', 'values_shared']}** websites are used by more than one record and **{S.loc['website_url', 'groups_over_1_city']}** of them span several cities, e.g. {", ".join(f"`{r['website_url']}` ({r['records']} records, {r['cities']} cities, {r['states']} {'state' if r['states'] == 1 else 'states'})" for _, r in web_chain.iterrows())}. These are multi-site companies, most often brewpub groups ({int((web[web['cities'] > 1]['dominant_type'] == 'brewpub').sum())} of the {S.loc['website_url', 'groups_over_1_city']}), so one brand appears as many records across a region.

**What does a shared address or phone reveal?**\\
Shared premises. The largest `address_1` group has **{S.loc['address_1', 'largest_group_records']} records at {S.loc['address_1', 'largest_group_value']}**: one site hosting many brands (the same stack seen on the Duplicates tab). Shared phones mark the same sites; the {S.loc['phone', 'groups_over_1_state']} phone groups that cross a state line are companies using one number for several sites, and none crosses a country.

**What does a shared city name reveal?**\\
That a city name alone is not a location. **{S.loc['city', 'groups_over_1_state']}** city names occur in more than one state and **{S.loc['city', 'groups_over_1_country']}** in more than one country, e.g. {", ".join(f"{r['city']} ({r['states']} states in {r['countries']} {'country' if r['countries'] == 1 else 'countries'})" for _, r in multi_state_city.iterrows())}. Every city-level count in this project is therefore keyed by country + state + city.

**What does a shared postal code reveal?**\\
Postal codes repeat across countries (**{S.loc['postal_code', 'groups_over_1_country']}** codes appear in more than one country), so a code is only meaningful together with its country.

**What does a shared name reveal?**\\
**{S.loc['name', 'values_shared']}** names are used more than once and **{S.loc['name', 'groups_over_1_city']}** of them in more than one city: chains with branches, plus a few generic names.
""")

# ======================================================================= 5. state labels
with tabs[4]:
    st.header("Are `state` and `state_province` the same?")
    st.caption("Both columns are compared after trimming surrounding spaces and ignoring letter case.")
    state_equal_n = None  # records where the two fields agree (None when `state` is no longer stored)
    if {"state", "state_province"} <= set(raw.columns):
        def norm(x): return x.astype("string").str.strip().str.casefold().replace("", pd.NA)
        a, b = norm(raw["state"]), norm(raw["state_province"])
        same = (a.notna() & b.notna() & a.eq(b)).fillna(False)
        cats = {"Both missing": a.isna() & b.isna(), "Only state missing": a.isna() & b.notna(),
                "Only state_province missing": a.notna() & b.isna(), "Both present and equal": same,
                "Both present but different": (a.notna() & b.notna() & ~same).fillna(False)}
        summ = pd.DataFrame({"comparison": list(cats), "record_count": [int(v.sum()) for v in cats.values()]})
        summ["percentage"] = (summ["record_count"] / len(raw) * 100).round(2)
        st.dataframe(summ, hide_index=True, width="stretch")
        bad = ~(cats["Both missing"] | same)
        state_equal_n = int((~bad).sum())
        if bad.any():
            st.error(f"The columns disagree on {int(bad.sum()):,} record(s). Review them before treating `state` "
                     "as redundant:")
            st.dataframe(raw.loc[bad, ["id", "name", "city", "country", "state", "state_province"]], hide_index=True,
                         width="stretch")
        else:
            st.success(f"Identical on all {len(raw):,} raw records. `state` carries no extra information, so the "
                       "project uses `state_province` as its only state/province field.")
    else:
        st.info("`state` is no longer in raw_breweries.db — it was already dropped after the equivalence check.")

    st.subheader("Other columns that repeat each other")
    twins = qc.identical_columns(F["raw"])
    st.dataframe(twins, hide_index=True, width="stretch")

    st.subheader("One region, several labels: `state_province_harmonized`")
    ht = qc.harmonization_table(F["raw"])
    st.dataframe(ht, hide_index=True, width="stretch")
    st.caption("The raw `state_province` column is never changed. `state_province_harmonized` maps each label above "
               "to one name per real region, and state-level analyses use it so a region is not split in two. One "
               "row is a single-record fix rather than a spelling: see the last question below.")

    st.divider()
    st.subheader("Questions answered in this tab")
    be = ht[ht["country"] == "Belgium"]
    st.markdown(f"""
**Should `state` and `state_province` be merged or one dropped?**\\
{(f"One of them can go. After trimming spaces and ignoring letter case the two columns hold the same value on all **{len(raw):,}** raw records: no record fills one and leaves the other empty, and no record disagrees. The analysis datasets are subsets of the raw table, so the same holds there. `state` is redundant, and every count, center, distance and cluster in this project is grouped by `state_province`." if state_equal_n == len(raw) else "The comparison above is the evidence: keep both columns until every disagreement listed there is resolved.")}

**Is any other column a copy?**\\
Yes: {", ".join(f"`{r['column']}` and `{r['identical_to']}`" for _, r in twins.iterrows())} match on all {len(F['raw']):,} raw records, missing values included. The source repeats them under two names, and the Open Brewery DB API documentation confirms it: `state` and `street` are listed there as deprecated copies of `state_province` and `address_1`.

**Are the state labels themselves consistent?**\\
Not everywhere. Belgium names its regions in three languages: Flanders appears as {", ".join(f"`{s}`" for s in be[be['state_province_harmonized'] == 'Flanders']['state_province'])} next to `Flanders`, and Wallonia as {", ".join(f"`{s}`" for s in be[be['state_province_harmonized'] == 'Wallonia']['state_province'])} next to `Wallonia`. Left as is, Belgium shows **{F['raw'].loc[F['raw']['country'] == 'Belgium', 'state_province'].nunique()}** region labels for **{qc.harmonize_states(F['raw'][F['raw']['country'] == 'Belgium'])['state_province_harmonized'].nunique()}** regions, and Flanders' and Wallonia's counts are split. One US record is filed under `MIssouri`. The harmonized column fixes the grouping without touching the source value.

**Is every record filed under the right region?**\\
All but one that a simple test can find. We looked for records that lie within 40 km of another region's capital in the same country and more than 300 km from their own region's capital. Only **Frontier Beer Company** is wrong: its city is Salt River, a suburb of Cape Town, and its point lies 4 km from Cape Town, the Western Cape capital, yet its label is `Gauteng`, whose capital is 1,259 km away. The point and the city agree, so the label is the error; the harmonized column files this one record (by its id) under Western Cape. Records near a border, such as the Wisconsin breweries in Hudson and River Falls next to Saint Paul, sit close to another region's capital but carry the right label.
""")

# ======================================================================= 6. names vs websites
with tabs[5]:
    st.subheader("Does a brewery's name show up in its website address?")
    ds = st.radio("Dataset", list(F), horizontal=True, key="nw_ds")
    d = F[ds].assign(link=lambda x: qc.name_website_links(x))
    w = d[d["link"] != "no website"]
    order = [c for c in qc.NAME_LINK_ORDER if c != "no website"]
    k = st.columns(4)
    k[0].metric("Records with a website", f"{len(w):,}", help=f"{len(w) / len(d) * 100:.1f}% of {ds}")
    k[1].metric("Full name in the domain", f"{(w['link'] == 'full name in domain').mean() * 100:.1f}%")
    k[2].metric("Social-media page instead of a site", f"{(w['link'] == 'social-media page').mean() * 100:.1f}%")
    k[3].metric("No visible link", f"{(w['link'] == 'no visible link').mean() * 100:.1f}%")
    sh = w["link"].value_counts().reindex(order).fillna(0).rename_axis("link").reset_index(name="records")
    sh["label"] = [f"{n:,} ({n / len(w) * 100:.1f}%)" for n in sh["records"]]
    f = px.bar(sh, x="records", y="link", orientation="h", text="label", color_discrete_sequence=["#2a78d6"],
               title="How the name appears in the website address (records with a website)")
    f.update_traces(textposition="outside", cliponaxis=False)
    f.update_layout(yaxis=dict(autorange="reversed"), yaxis_title="", height=320, margin=dict(r=120))
    st.plotly_chart(f, width="stretch")
    c1, c2 = st.columns(2)
    for box, col, mn in [(c1, "country", 30), (c2, "brewery_type", 10)]:
        big = w[col].value_counts()
        ct = pd.crosstab(w[col], w["link"], normalize="index").reindex(columns=order).fillna(0) * 100
        ct = ct.loc[big[big >= mn].index]
        hm = px.imshow(ct.round(0), text_auto=True, aspect="auto", color_continuous_scale="Blues", zmin=0, zmax=100,
                       labels={"color": "% of records", "x": "", "y": ""},
                       title=f"By {col.replace('_', ' ')} (≥ {mn} records with a website)")
        hm.update_layout(height=40 * len(ct) + 160, coloraxis_showscale=False)
        box.plotly_chart(hm, width="stretch")
    nolink = w[w["link"] == "no visible link"]
    hosts = nolink["website_url"].str.lower().str.replace(r"^[a-z]+://", "", regex=True).str.replace(r"^www\d?\.", "",
                                                                                                   regex=True) \
        .str.split("/").str[0]
    shared_hosts = hosts.value_counts()
    st.markdown("**Websites behind 'no visible link' that serve several breweries**")
    st.dataframe(shared_hosts[shared_hosts > 1].rename_axis("website").reset_index(name="records").head(15),
                 hide_index=True, width="stretch")
    pick = st.selectbox("Show examples of", order, index=order.index("no visible link"), key="nw_pick")
    st.dataframe(w[w["link"] == pick][["name", "website_url", "city", "country", "brewery_type"]].head(25),
                 hide_index=True, width="stretch")
    st.caption("The domain label is the part of the host before the top-level domain (ballastpoint in "
               "ballastpoint.com). Generic words (brewing, brewery, company, co, beer, brauerei, brouwerij, ...) are "
               "ignored when matching, and German umlauts are tried both as plain letters and written out (ü → ue).")

    st.divider()
    st.subheader("Questions answered in this tab")
    cw = pd.crosstab(w["country"], w["link"], normalize="index").reindex(columns=order).fillna(0) * 100
    cw = cw.loc[w["country"].value_counts()[lambda x: x >= 30].index]
    social = cw["social-media page"].idxmax()
    weak = cw["no visible link"].idxmax()
    st.markdown(f"""
**Do brewery names intersect with their website domains?**\\
Mostly, yes. Of the {len(w):,} {ds} records that have a website, **{(w['link'] == 'full name in domain').mean() * 100:.1f}%** carry the brewery's distinctive name in the domain and another {(w['link'] == 'part of the name in domain').mean() * 100:.1f}% carry part of it. A domain is the brewery's own address in most of the world.

**Where does the link break, and why?**\\
In **{weak}** ({cw.loc[weak, 'no visible link']:.0f}% with no visible link) records often point to a website shared by several breweries, such as a brewers' collective, or to a parent company. In **{social}** {cw.loc[social, 'social-media page']:.0f}% of the websites are social-media pages rather than an own site. The shared websites in the table above are the same shared premises and groups found in the *Shared values* tab.

**What is this useful for?**\\
Matching records to the right brewery. A record whose website carries another name is either part of a group or a data-entry mix-up, and the name–domain link is a cheap check for both when records are merged or deduplicated.
""")

