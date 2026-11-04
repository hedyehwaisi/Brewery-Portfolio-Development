import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import distribution_core as dc
import stats_store
from common import pick_any_dataset, pick_basemap, harmonized, shared_cache, stats_table
from geo_plots import point_map, type_style
from palette import color_map

st.set_page_config(page_title="Brewery Types", page_icon="🍻", layout="wide")
st.title("🍻 Brewery types — what kinds of breweries, where, and which patterns")
name = pick_any_dataset(default="raw")
basemap = pick_basemap()
min_n = st.sidebar.slider("Minimum records per place (rates, quotients, tests)", stats_store.TYPE_MIN_RECORDS.start,
                          stats_store.TYPE_MIN_RECORDS.stop - 1, 30)

df = harmonized(name)
geo = df.dropna(subset=["latitude", "longitude"])
type_cmap, type_order = type_style(harmonized("raw"))  # one colour per type, identical on every page/dataset
STATE = dc.STATE_COL
LEVEL_NAMES = {"country": "Country", "state": "State / province", "city": "City"}
st.caption(f"Dataset: **{name}** ({len(df):,} records). States use harmonized labels; cities are keyed by "
           "country + state + city.")


# The chi-square, Ward/silhouette and permutation tests below are precomputed for every setting of this page
# (stats_store, brewery_dashboard_statistics.ipynb), so the app does not need SciPy or scikit-learn.
@shared_cache
def colocation(ds, country, exclude_same_point):
    return stats_store.type_colocation(stats_table("type_colocation"), ds, country, exclude_same_point)


@shared_cache
def profiles(ds, country, mn):
    return stats_store.type_profiles(stats_table("type_profile_states"), stats_table("type_profile_mix"),
                            stats_table("type_profile_silhouette"), ds, country, mn)


def centers(level):
    """Mean position of each place's geocoded records (used to place bubbles on maps)."""
    g = geo.assign(place=dc.place_label(geo, level)).groupby("place")[["latitude", "longitude"]].mean()
    return g


tabs = st.tabs(["Counts & definitions", "Where each type is", "Type mix of top places", "Dominant type", "Patterns"])

# ======================================================================= 1. counts & definitions
with tabs[0]:
    tc = dc.type_counts(df)
    tc["label"] = [f"{r:,} ({p:.2f}%)" for r, p in zip(tc["records"], tc["share_%"])]
    f = px.bar(tc, x="records", y="brewery_type", orientation="h", text="label", color="brewery_type",
               color_discrete_map=type_cmap, title=f"Records per brewery type in {name}")
    f.update_traces(textposition="outside", cliponaxis=False)
    f.update_layout(showlegend=False, yaxis=dict(autorange="reversed"), yaxis_title="", xaxis_title="Records",
                    height=480, margin=dict(r=150))
    st.plotly_chart(f, width="stretch")
    st.dataframe(tc.drop(columns="label").round(2), hide_index=True, width="stretch",
                 column_config={"definition": st.column_config.TextColumn(width="large")})
    st.caption("Definitions are the official ones from openbrewerydb.org/documentation. `taproom`, `cidery` and "
               "`beergarden` occur in the data but are not defined in the API documentation; `large` and `bar` are "
               "marked deprecated there.")

    st.divider()
    st.subheader("Questions answered in this tab")
    top2 = tc.head(2)
    rare = tc[tc["share_%"] < 1]
    undocumented = tc[~tc["brewery_type"].isin(dc.TYPE_DEFINITIONS)]
    st.markdown(f"""
**Which types make up the dataset?**\\
Two types carry it: **{top2.iloc[0]['brewery_type']}** ({top2.iloc[0]['share_%']:.2f}%) and **{top2.iloc[1]['brewery_type']}** ({top2.iloc[1]['share_%']:.2f}%), together **{top2['share_%'].sum():.2f}%** of {name}. The other {len(tc) - 2} types share the remaining {100 - top2['share_%'].sum():.2f}%; {len(rare)} of them are below 1% each ({", ".join(rare['brewery_type'])}).

**Can every type be trusted the same way?**\\
No. {len(undocumented)} types ({", ".join(undocumented['brewery_type'])}) are not defined in the API documentation, and `large` and `bar` are deprecated. Their counts describe how contributors labelled records, so they are reported but not over-interpreted. `closed` and `planning` describe a status (closed, not yet open) rather than a business model.
""")

# ======================================================================= 2. where each type is
with tabs[1]:
    types = dc.type_counts(df)["brewery_type"].tolist()
    c1, c2, c3 = st.columns(3)
    t_pick = c1.selectbox("Brewery type", types, key="where_type")
    level = c2.radio("Level", list(LEVEL_NAMES), format_func=LEVEL_NAMES.get, horizontal=True, key="where_level")
    n = c3.slider("Places to show", 5, 30, 15, key="where_n")
    sub = df[df["brewery_type"] == t_pick]
    k = st.columns(4)
    k[0].metric("Records", f"{len(sub):,}", help=f"{len(sub) / len(df) * 100:.2f}% of {name}")
    k[1].metric("Countries", sub["country"].nunique())
    k[2].metric("States / provinces", sub[["country", STATE]].drop_duplicates().shape[0])
    k[3].metric("Cities", sub[["country", STATE, "city"]].drop_duplicates().shape[0])

    pc = dc.counts(sub, level).rename(columns={"share_%": "share_of_type_%"})
    allc = dc.counts(df, level).set_index("place")["records"]
    pc["place_records"] = pc["place"].map(allc)
    pc["share_of_place_%"] = pc["records"] / pc["place_records"] * 100
    pc["location_quotient"] = pc["share_of_place_%"] / (len(sub) / len(df) * 100)
    top = pc.head(n).copy()
    top["label"] = [f"{r:,} ({a:.1f}% of type · {b:.1f}% of place)"
                    for r, a, b in zip(top["records"], top["share_of_type_%"], top["share_of_place_%"])]
    f = px.bar(top, x="records", y="place", orientation="h", text="label",
               color_discrete_sequence=[type_cmap[t_pick]],
               hover_data={"place_records": ":,", "location_quotient": ":.2f"},
               title=f"Where `{t_pick}` records are: top {len(top)} {LEVEL_NAMES[level].lower()} places")
    f.update_traces(textposition="outside", cliponaxis=False)
    f.update_layout(yaxis=dict(autorange="reversed"), yaxis_title="", xaxis_title="Records",
                    height=32 * len(top) + 140, margin=dict(r=260))
    st.plotly_chart(f, width="stretch")
    st.dataframe(pc.drop(columns=["rank"]).round(2), hide_index=True, width="stretch", height=260)
    st.caption("`share_of_type_%`: the place's part of all records of this type. `share_of_place_%`: the type's part "
               "of the place's records. `location_quotient`: share_of_place_% divided by the type's share overall "
               "(1 = as common as everywhere, 2 = twice as common).")

    pts = geo[geo["brewery_type"] == t_pick]
    if len(pts):
        m = point_map(pts, hover_name="name", hover_data=["city", STATE, "country"], marker_size=6, height=520,
                      color_discrete_sequence=[type_cmap[t_pick]], style=basemap,
                      title=f"Every geocoded `{t_pick}` record ({len(pts):,} of {len(sub):,})")
        st.plotly_chart(m, width="stretch", config={"scrollZoom": True})

    st.subheader(f"Every type at a glance: where it is concentrated ({name})")
    rows = []
    for t in types:
        s = df[df["brewery_type"] == t]
        cc, ss, ci = dc.counts(s, "country").iloc[0], dc.counts(s, "state").iloc[0], dc.counts(s, "city").iloc[0]
        rows.append({"brewery_type": t, "records": len(s), "countries": s["country"].nunique(),
                     "top_country": cc["place"], "top_country_share_of_type_%": cc["share_%"],
                     "top_state": ss["place"], "top_state_share_of_type_%": ss["share_%"],
                     "top_city": ci["place"], "top_city_share_of_type_%": ci["share_%"]})
    glance = pd.DataFrame(rows)
    st.dataframe(glance.round(1), hide_index=True, width="stretch")

    st.divider()
    st.subheader("Questions answered in this tab")
    single = glance[glance["countries"] == 1]
    g = glance.set_index("brewery_type")
    st.markdown(f"""
**To which countries, states and cities does each type belong?**\\
The table above answers it for every type. Some types are effectively national: {", ".join(f"**{r['brewery_type']}** only in {r['top_country']}" for _, r in single.iterrows()) or "no type is confined to one country"}. {f"**{g.loc['brewpub', 'top_country_share_of_type_%']:.0f}%** of all brewpubs are in {g.loc['brewpub', 'top_country']}" if 'brewpub' in g.index else ""}{f", and **{g.loc['micro', 'top_country_share_of_type_%']:.0f}%** of all micro breweries in {g.loc['micro', 'top_country']}" if 'micro' in g.index else ""}.

**How should a type's top place be read?**\\
With both shares. The United States tops almost every type simply because it holds most records; the location quotient corrects for that size. A place with a high quotient for a type is where that type is unusually common, even if the absolute count is small.
""")

# ======================================================================= 3. type mix of top places
with tabs[2]:
    c1, c2 = st.columns(2)
    level = c1.radio("Level", list(LEVEL_NAMES), format_func=LEVEL_NAMES.get, horizontal=True, index=1,
                     key="mix_level")
    n = c2.slider("Top places by records", 5, 40, 20, key="mix_n")
    topp = dc.counts(df, level).head(n)
    mix = dc.type_mix(df, level, places=topp["place"].tolist())
    long = mix.rename_axis("place").reset_index().melt(id_vars="place", var_name="brewery_type", value_name="share_%")
    long = long[long["share_%"] > 0]
    f = px.bar(long, x="share_%", y="place", color="brewery_type", orientation="h", color_discrete_map=type_cmap,
               category_orders={"place": topp["place"].tolist(), "brewery_type": type_order["brewery_type"]},
               title=f"Type mix of the {len(topp)} largest {LEVEL_NAMES[level].lower()} places (each bar = 100%)",
               hover_data={"share_%": ":.1f"})
    f.update_layout(barmode="stack", xaxis_title="% of the place's records", yaxis_title="",
                    height=26 * len(topp) + 160, legend_title_text="Type", xaxis_range=[0, 100])
    st.plotly_chart(f, width="stretch")

    lq = dc.location_quotient(df, level, min_records=1).reindex(topp["place"]).dropna(how="all")
    lq = lq.loc[:, lq.max() > 0]
    z = np.log2(lq.clip(lower=1 / 8, upper=8))
    hf = px.imshow(z, aspect="auto", zmin=-3, zmax=3,
                   color_continuous_scale=[[0, "#2a78d6"], [0.5, "#f2f2f2"], [1, "#eb6834"]],
                   labels={"color": "LQ (log2)", "x": "", "y": ""},
                   title="Location quotient: how much more (orange) or less (blue) common each type is than overall")
    hf.update_traces(text=lq.round(2).astype(str).values, texttemplate="%{text}")
    hf.update_layout(height=26 * len(lq) + 160,
                     coloraxis_colorbar=dict(tickvals=[-3, -2, -1, 0, 1, 2, 3],
                                             ticktext=["≤⅛", "¼", "½", "1", "2", "4", "≥8"], title="LQ"))
    st.plotly_chart(hf, width="stretch")
    st.caption("Grey = the type is as common as in the whole dataset; colour scale is logarithmic and capped at ⅛× "
               "and 8×. A 0 means the type is absent from that place.")

    st.divider()
    st.subheader("Questions answered in this tab")
    st_counts = dc.counts(df, "state")
    lead = st_counts.iloc[0]
    lead_country = lead["country"]
    in_country = df[df["country"] == lead_country]
    lead_mix = dc.type_mix(df, "state", places=[lead["place"]]).iloc[0]
    nat_mix = in_country["brewery_type"].value_counts(normalize=True) * 100
    comp = pd.DataFrame({"state": lead_mix, "country": nat_mix}).fillna(0)
    comp["diff_pp"] = comp["state"] - comp["country"]
    above = comp[comp["diff_pp"] > 1].sort_values("diff_pp", ascending=False)
    raw = harmonized("raw")
    raw_lead = raw[(raw["country"] == lead_country) & (raw[STATE] == lead[STATE])]
    raw_nat = raw[raw["country"] == lead_country]
    why = ""
    if name != "raw" and "planning" in above.index:
        rp = (raw_lead["brewery_type"] == "planning").mean() * 100
        rn = (raw_nat["brewery_type"] == "planning").mean() * 100
        why = (f" The planning surplus is not a {lead[STATE]} speciality. In raw, planning is **{rp:.1f}%** of "
               f"{lead[STATE]} and **{rn:.1f}%** of {lead_country}: almost the same. {lead[STATE]} keeps "
               f"{len(in_country[in_country[STATE] == lead[STATE]]) / len(raw_lead) * 100:.0f}% of its records after "
               f"geocoding, while {lead_country} as a whole keeps {len(in_country) / len(raw_nat) * 100:.0f}%, and "
               "planning records are the ones that usually lack coordinates. The other states lost their planning "
               "breweries in cleaning; this one did not.")
    tops = dc.dominant_types(df, "state").head(10)
    dom_counts = tops["dominant_type"].value_counts()
    ct_top = dc.dominant_types(df, "city").head(10)
    city_keys = ct_top[["country", STATE, "city"]]
    in_cities = df.merge(city_keys, on=["country", STATE, "city"])
    in_states = df.merge(city_keys[["country", STATE]].drop_duplicates(), on=["country", STATE])
    rest = in_states.merge(city_keys, on=["country", STATE, "city"], how="left", indicator=True)
    rest = rest[rest["_merge"] == "left_only"]
    cmix = in_cities["brewery_type"].value_counts(normalize=True) * 100
    smix = rest["brewery_type"].value_counts(normalize=True) * 100
    cdiff = (cmix - smix.reindex(cmix.index).fillna(0)).sort_values()
    st.markdown(f"""
**What is the brewery-type focus of the state with the most breweries, and why?**\\
**{lead['place']}** ({lead['records']:,} records) is led by **{lead_mix.idxmax()}** ({lead_mix.max():.1f}%), then {lead_mix.drop(lead_mix.idxmax()).idxmax()} ({lead_mix.drop(lead_mix.idxmax()).max():.1f}%). Against {lead_country} as a whole it leans towards {", ".join(f"**{t}** (+{r['diff_pp']:.1f} pp)" for t, r in above.head(3).iterrows()) or "no type in particular"}.{why}

**Which types do the states with the most breweries have?**\\
Of the 10 largest states, {", ".join(f"{v} are led by **{t}**" for t, v in dom_counts.items())}. {", ".join(f"{r['place']} ({r['dominant_type']} {r['dominant_share_%']:.0f}%)" for _, r in tops[tops['dominant_type'] != 'micro'].iterrows())}{" break the micro pattern." if (tops['dominant_type'] != 'micro').any() else ""}

**And the cities with the most breweries?**\\
{", ".join(f"{r['place'].split(',')[0]}: {r['dominant_type']} {r['dominant_share_%']:.0f}%" for _, r in ct_top.iterrows())}. Taken together, these 10 cities hold **{cmix.get('brewpub', 0):.1f}%** brewpubs and **{cmix.get('micro', 0):.1f}%** micro breweries, against {smix.get('brewpub', 0):.1f}% and {smix.get('micro', 0):.1f}% in the rest of their own states. The biggest differences: {", ".join(f"{t} {v:+.1f} pp" for t, v in pd.concat([cdiff.head(2), cdiff.tail(2)]).items())}.{(" " + ", ".join(r['place'].split(',')[0] for _, r in ct_top[ct_top['dominant_type'] == 'closed'].iterrows()) + " is led by closed records, a trace of how its state's database is maintained (see Patterns).") if (ct_top['dominant_type'] == 'closed').any() else ""}
""")

# ======================================================================= 4. dominant type
with tabs[3]:
    c1, c2 = st.columns(2)
    level = c1.radio("Level", list(LEVEL_NAMES), format_func=LEVEL_NAMES.get, horizontal=True, index=1,
                     key="dom_level")
    mn = c2.slider("Minimum records per place", 1, 100, 10 if level != "country" else 1, key="dom_min")
    dt = dc.dominant_types(df, level, min_records=mn)
    k = st.columns(3)
    k[0].metric("Places", f"{len(dt):,}")
    k[1].metric("Most common leader", dt["dominant_type"].value_counts().index[0],
                help=f"leads {dt['dominant_type'].value_counts().iat[0]} places")
    k[2].metric("Median winning margin", f"{dt['margin_pp'].median():.1f} pp")
    c1, c2 = st.columns([1, 2])
    vc = dt["dominant_type"].value_counts().rename_axis("dominant_type").reset_index(name="places")
    f = px.bar(vc, x="places", y="dominant_type", orientation="h", text="places", color="dominant_type",
               color_discrete_map=type_cmap, title=f"Which type leads each {LEVEL_NAMES[level].lower()}?")
    f.update_layout(showlegend=False, yaxis=dict(autorange="reversed"), yaxis_title="")
    c1.plotly_chart(f, width="stretch")
    cen = centers(level)
    dm = dt.join(cen, on="place").dropna(subset=["latitude", "longitude"])
    m = point_map(dm, color="dominant_type", size="records", color_discrete_map=type_cmap,
                  category_orders={"dominant_type": type_order["brewery_type"]},
                  hover_name="place", hover_data={"records": ":,", "dominant_share_%": ":.1f", "runner_up": True,
                                                  "margin_pp": ":.1f"},
                  size_max=35, opacity=0.9, height=520, style=basemap,
                  title=f"Leading type per {LEVEL_NAMES[level].lower()} (bubble at the place's mean position)")
    m.update_layout(legend_title_text="Leading type")
    c2.plotly_chart(m, width="stretch", config={"scrollZoom": True})
    st.dataframe(dt.round(1), hide_index=True, width="stretch", height=320)
    st.caption("`margin_pp` = leader's share minus runner-up's share, in percentage points. A small margin means the "
               "place is mixed rather than dominated.")

    st.divider()
    st.subheader("Questions answered in this tab")
    ds_ = dc.dominant_types(df, "state", min_records=30)
    dco = dc.dominant_types(df, "country", min_records=30)
    close = ds_.nsmallest(3, "margin_pp")
    dcity = dc.dominant_types(df, "city", min_records=10)
    st.markdown(f"""
**What is the most frequent brewery type in each state?**\\
Among states with at least 30 records, {", ".join(f"**{t}** leads {v}" for t, v in ds_['dominant_type'].value_counts().items())}. The closest contests are {", ".join(f"{r['place']} ({r['dominant_type']} vs {r['runner_up']}, {r['margin_pp']:.1f} pp)" for _, r in close.iterrows())}.

**And in each country?**\\
{", ".join(f"{r['place']}: **{r['dominant_type']}** {r['dominant_share_%']:.0f}%" for _, r in dco.iterrows())}. {len(dco[dco['dominant_share_%'] >= 70])} of these {len(dco)} countries give at least 70% of their records one label; the mixed ones are {", ".join(f"{r['place']} ({r['dominant_share_%']:.0f}%)" for _, r in dco[dco['dominant_share_%'] < 70].iterrows())}.

**And in each city?**\\
Among cities with at least 10 records, {", ".join(f"**{t}** leads {v}" for t, v in dcity['dominant_type'].value_counts().items())}.
""")

# ======================================================================= 5. patterns
with tabs[4]:
    st.subheader("1. Does the type mix depend on the place?")
    assoc = stats_store.type_association(stats_table("type_association"), name, min_n)
    st.dataframe(assoc.round(4), hide_index=True, width="stretch")
    st.caption(f"Chi-square test of independence between place and brewery type, places with ≥ {min_n} records. "
               "Cramér's V: below 0.1 negligible, 0.1–0.3 weak, 0.3–0.5 moderate, above 0.5 strong.")

    st.subheader("2. Country labelling conventions")
    lqc = dc.location_quotient(df, "country", min_records=min_n)
    lqc = lqc.loc[dc.counts(df, "country").query("records >= @min_n")["place"], lqc.max() > 0]
    mixc = dc.type_mix(df, "country", places=lqc.index.tolist())
    hf = px.imshow(mixc.round(1), text_auto=True, aspect="auto", color_continuous_scale="Blues", zmin=0, zmax=100,
                   labels={"color": "% of country", "x": "", "y": ""},
                   title=f"Type mix of every country with ≥ {min_n} records (% of the country's records)")
    hf.update_layout(height=34 * len(mixc) + 160)
    st.plotly_chart(hf, width="stretch")

    st.subheader("3. Groups of US states with a similar type mix")
    countries = dc.counts(df, "country")["place"].tolist()
    pc_country = st.selectbox("Country", countries, index=countries.index("United States")
                              if "United States" in countries else 0, key="prof_country")
    assign, prof, sil = profiles(name, pc_country, min_n) or (None, None, None)
    if assign is None or len(assign) < 4:
        st.info(f"{pc_country} has too few states with ≥ {min_n} records to group.")
    else:
        assign["profile"] = "Profile " + assign["profile"].astype(str)
        c1, c2 = st.columns([1, 2])
        sf = px.line(sil, x="k", y="silhouette", markers=True, title="Silhouette score per number of groups")
        sf.update_layout(xaxis=dict(dtick=1), yaxis_title="silhouette (higher = better separated)")
        c1.plotly_chart(sf, width="stretch")
        prof_show = prof.copy()
        prof_show.index = ["Profile " + str(i) for i in prof_show.index]
        c2.dataframe(prof_show.round(1), width="stretch")
        c2.caption("Mean type share (%) of the states in each group. The number of groups is the one with the "
                   "highest silhouette score.")
        state_c = geo[geo["country"] == pc_country].groupby(STATE)[["latitude", "longitude"]].mean()
        am = assign.join(state_c, on=STATE).dropna(subset=["latitude", "longitude"])
        pm = point_map(am, color="profile", size="records",
                       color_discrete_map=color_map(sorted(am["profile"].unique())), hover_name=STATE,
                       hover_data={"records": ":,"}, size_max=35, opacity=0.9, height=480, style=basemap,
                       title="States coloured by type-mix group")
        st.plotly_chart(pm, width="stretch", config={"scrollZoom": True})
        st.dataframe(assign.sort_values(["profile", "records"], ascending=[True, False]), hide_index=True,
                     width="stretch", height=240)

    st.subheader("4. Do breweries of the same type sit next to each other?")
    c1, c2 = st.columns(2)
    co_country = c1.selectbox("Country", countries, index=countries.index("United States")
                              if "United States" in countries else 0, key="coloc_country")
    excl = c2.checkbox("Ignore neighbours on the very same coordinate", value=False,
                       help="Records stacked on one point (shared sites) count as neighbours unless this is ticked.")
    if geo[geo["country"] == co_country].shape[0] < stats_store.COLOCATION_MIN_GEOCODED:
        st.info("Too few geocoded records in this country.")
    else:
        col = colocation(name, co_country, excl)
        show = col[col["records"] >= 10]
        long = show.melt(id_vars=["brewery_type", "records", "p_value", "ratio"],
                         value_vars=["observed_%", "expected_%"], var_name="what", value_name="pct")
        long["what"] = long["what"].map({"observed_%": "Observed", "expected_%": "Expected by chance"})
        f = px.bar(long, x="pct", y="brewery_type", color="what", barmode="group", orientation="h",
                   color_discrete_map={"Observed": "#eb6834", "Expected by chance": "#2a78d6"},
                   hover_data={"records": True, "p_value": ":.3f", "ratio": ":.2f"},
                   title="Share of records whose nearest neighbour has the same type")
        f.update_layout(yaxis=dict(autorange="reversed"), yaxis_title="", xaxis_title="%", legend_title_text="",
                        height=40 * len(show) + 140)
        st.plotly_chart(f, width="stretch")
        st.dataframe(col.round(3), hide_index=True, width="stretch")
        st.caption("Expected = mean of 199 random reshuffles of the types within each state (every state keeps its "
                   "own type mix and every brewery keeps its position). p_value = share of reshuffles at least as "
                   "extreme as the observed value. Types with fewer than 10 records are left out of the chart.")

    st.divider()
    st.subheader("Questions answered in this tab")
    us = "United States" if "United States" in countries else countries[0]
    a_us, p_us, s_us = profiles(name, us, min_n)
    best_k = int(s_us.loc[s_us["silhouette"].idxmax(), "k"])
    cl_hi = p_us["closed"].idxmax() if "closed" in p_us else None
    col_all = colocation(name, us, False).set_index("brewery_type")
    col_ex = colocation(name, us, True).set_index("brewery_type")
    strong = col_ex[(col_ex["p_value"] < 0.05) & (col_ex["records"] >= 30) & (col_ex.index != "all types")] \
        .sort_values("ratio", ascending=False)
    lqx = dc.location_quotient(df, "country", min_records=min_n)
    mixx = dc.type_mix(df, "country", places=lqx.index.tolist())
    one_label = mixx[mixx.max(axis=1) >= 85]
    st.markdown(f"""
**Is there a pattern by country? Is there a common "culture"?**\\
Yes, and it is the strongest pattern in the type column (Cramér's V {assoc.set_index('level').loc['country', 'cramers_v']:.2f} by country). {len(one_label)} countries put at least 85% of their records under one label: {", ".join(f"{c} {mixx.loc[c].idxmax()} {mixx.loc[c].max():.0f}%" for c in one_label.index)}. Real brewing landscapes are not that uniform, so this reads as each country's records being labelled by one contributor or one convention. Compare types across countries with care; compare them within a country with more confidence.

**Do US states form groups by type mix?**\\
They split best into **{best_k}** groups (silhouette {s_us['silhouette'].max():.2f}). The split is driven by status labels, not business models: one group carries **{p_us.loc[cl_hi, 'closed']:.0f}%** closed records on average, the other {p_us.drop(cl_hi)['closed'].mean():.0f}%. That gap ({p_us.loc[cl_hi, 'closed'] - p_us.drop(cl_hi)['closed'].mean():.0f} pp) is the largest difference between the groups; micro differs by {abs(p_us.loc[cl_hi, 'micro'] - p_us.drop(cl_hi)['micro'].mean()):.0f} pp and brewpub by {abs(p_us.loc[cl_hi, 'brewpub'] - p_us.drop(cl_hi)['brewpub'].mean()):.0f} pp, partly because the closed records take their place. Some states keep closed breweries in the database and others remove them, so the grouping reveals maintenance practice first and business models second.

**Do breweries of a type cluster in space beyond what their state's mix explains?**\\
Yes. Across {us}, a brewery's nearest neighbour has the same type {col_all.loc['all types', 'observed_%']:.1f}% of the time against {col_all.loc['all types', 'expected_%']:.1f}% expected (p = {col_all.loc['all types', 'p_value']:.3f}). Ignoring records stacked on one point, the clearest cases are {", ".join(f"**{t}** ×{r['ratio']:.1f}" for t, r in strong.head(4).iterrows())}. Contract and proprietor brewers depend on someone else's brewhouse, so they gather around the breweries that host them. Planning records cluster only through shared addresses: with stacked points ignored their ratio falls to {col_ex.loc['planning', 'ratio']:.1f} (from {col_all.loc['planning', 'ratio']:.1f}).
""" if "planning" in col_ex.index and cl_hi is not None else "")
