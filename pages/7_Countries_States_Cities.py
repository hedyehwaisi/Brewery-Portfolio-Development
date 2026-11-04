import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import distribution_core as dc
from common import pick_any_dataset, pick_basemap, harmonized
from geo_plots import point_map, type_style
from palette import color_map, DATASET_COLORS

st.set_page_config(page_title="Countries, States & Cities", page_icon="🗺️", layout="wide")
st.title("🗺️ Countries, states and cities — how the hierarchy holds together")
name = pick_any_dataset(default="raw")
basemap = pick_basemap()
n_states = st.sidebar.slider("Top states", 5, 30, 10)
n_cities = st.sidebar.slider("Top cities", 5, 40, 10)
n_countries = st.sidebar.slider("Top countries", 3, 10, 5)

df = harmonized(name)
raw = harmonized("raw")
geo = df.dropna(subset=["latitude", "longitude"])
STATE = dc.STATE_COL
type_cmap, type_order = type_style(raw)
KIND_COLORS = {"A: top state": "#2a78d6", "B: outside top states": "#eb6834"}
st.caption(f"Dataset: **{name}** ({len(df):,} records) · top {n_states} states, top {n_cities} cities, top "
           f"{n_countries} countries (sidebar). States use harmonized labels; cities are keyed by country + state + city.")

city_centers = geo.assign(place=dc.place_label(geo, "city")).groupby("place")[["latitude", "longitude"]].mean()
state_counts, city_counts, country_counts = dc.counts(df, "state"), dc.counts(df, "city"), dc.counts(df, "country")

tabs = st.tabs(["Top states & their cities", "Top cities vs top states", "City share of state: type A / B",
                "City distribution", "Insights by level"])

# ======================================================================= 1. top states and their cities
with tabs[0]:
    coverage = st.slider("Coverage target: cities that together hold this % of a state's breweries", 20, 90, 50, 5)
    det, summ = dc.cities_of_top_states(df, n_states=n_states, coverage=coverage, n_top_cities=n_cities)
    summ["state"] = summ[STATE] + ", " + summ["country"]
    show = summ[["state_rank", "state", "records", "cities", "cities_for_coverage",
                 "cities_for_coverage_pct_of_cities", "top_city", "top_city_share_pct", "cities_in_overall_top"]]
    st.dataframe(show.round(1), hide_index=True, width="stretch",
                 column_config={"cities_for_coverage": st.column_config.NumberColumn(f"cities for {coverage}%"),
                                "cities_for_coverage_pct_of_cities": st.column_config.NumberColumn(
                                    f"those cities as % of all cities", format="%.1f"),
                                "cities_in_overall_top": st.column_config.NumberColumn(f"cities in top {n_cities}")})
    long = summ.melt(id_vars=["state"], value_vars=["cities_for_coverage", "cities"], var_name="what",
                     value_name="n")
    long["what"] = long["what"].map({"cities_for_coverage": f"Cities holding {coverage}% of the breweries",
                                     "cities": "All cities with a brewery"})
    f = px.bar(long, x="n", y="state", color="what", barmode="group", orientation="h", text="n",
               color_discrete_map={f"Cities holding {coverage}% of the breweries": "#eb6834",
                                   "All cities with a brewery": "#2a78d6"},
               title=f"How many cities carry each top state?")
    f.update_layout(yaxis=dict(autorange="reversed"), yaxis_title="", xaxis_title="Cities", legend_title_text="",
                    height=40 * len(summ) + 150)
    st.plotly_chart(f, width="stretch")

    pick = st.selectbox("State to open", summ["state"].tolist(), key="tsc_state")
    one = det[(det[STATE] + ", " + det["country"]) == pick]
    shown = one[one["needed_for_coverage"] | (one["rank_in_state"] <= 5)].copy()
    shown["group"] = np.where(shown["in_top_cities"], f"Also in the top {n_cities} cities overall",
                              "Not in the overall top")
    shown["label"] = [f"{r} ({s:.1f}% · cum. {c:.0f}%)" for r, s, c in
                      zip(shown["records"], shown["share_of_state_%"], shown["cumulative_share_%"])]
    f = px.bar(shown, x="share_of_state_%", y="city", orientation="h", text="label", color="group",
               color_discrete_map={f"Also in the top {n_cities} cities overall": "#eb6834",
                                   "Not in the overall top": "#2a78d6"},
               title=f"{pick}: the cities that hold {coverage}% of its breweries")
    f.update_traces(textposition="outside", cliponaxis=False)
    f.update_layout(yaxis=dict(autorange="reversed"), yaxis_title="", xaxis_title="% of the state's breweries",
                    legend_title_text="", height=max(320, 28 * len(shown) + 140), margin=dict(r=170))
    st.plotly_chart(f, width="stretch")

    cov = det[det["needed_for_coverage"]].join(city_centers, on="place").dropna(subset=["latitude", "longitude"])
    cov["state"] = cov[STATE] + ", " + cov["country"]
    m = point_map(cov, color="state", size="records", color_discrete_map=color_map(summ["state"]),
                  hover_name="place", hover_data={"records": True, "share_of_state_%": ":.1f"}, size_max=30,
                  opacity=0.9, height=540, style=basemap,
                  title=f"Cities that hold {coverage}% of each top state (size = breweries)")
    m.update_layout(legend_title_text="State")
    st.plotly_chart(m, width="stretch", config={"scrollZoom": True})

    st.divider()
    st.subheader("Questions answered in this tab")
    top3 = det.groupby("state_rank").head(3)
    lines = []
    for rk, g in top3.groupby("state_rank"):
        s = summ[summ["state_rank"] == rk].iloc[0]
        lines.append(f"- **{s['state']}** ({s['cities']} cities): "
                     + ", ".join(f"{r['city']} {r['share_of_state_%']:.1f}%" for _, r in g.iterrows())
                     + f" — {s['cities_for_coverage']} cities reach {coverage}%")
    state_lines = "\n".join(lines)
    concentrated = summ.nsmallest(3, "cities_for_coverage_pct_of_cities")
    spread = summ.nlargest(3, "cities_for_coverage_pct_of_cities")
    with_top = summ[summ["cities_in_overall_top"] > 0]
    st.markdown(f"""
**Which cities in the top states have the most breweries?**\\
The three largest cities of each top state and their share of it:

{state_lines}

**Which cities make up the largest share of each top state?**\\
It depends on how the state is built. The most city-centred top states need only a few cities for {coverage}%: {", ".join(f"{r['state']} ({r['cities_for_coverage']} of {r['cities']} cities)" for _, r in concentrated.iterrows())}. The most spread-out ones need many: {", ".join(f"{r['state']} ({r['cities_for_coverage']} of {r['cities']})" for _, r in spread.iterrows())}.

**Do the top states contain the top cities?**\\
Only {len(with_top)} of the top {n_states} states contain at least one of the top {n_cities} cities ({", ".join(f"{r['state']}: {r['cities_in_overall_top']}" for _, r in with_top.iterrows()) or "none"}). A state can be large because of one big hub or because of many small towns; the two routes produce very different cities.
""")

# ======================================================================= 2. top cities vs top states
with tabs[1]:
    kinds = dc.city_state_kinds(df, n_states=n_states, n_cities=n_cities, n_countries=n_countries,
                                reference=raw if name != "raw" else None)
    ref_col = "share_of_state_in_reference_%" if name != "raw" else "share_of_state_%"

    def reason(r):
        if r["city_type"].startswith("A"):
            return "inside a top state"
        if r[ref_col] >= 75:
            return "city-state or one-city region"
        if r["share_of_state_%"] >= 75 and r[ref_col] < 50:
            return "share inflated by records lost in geocoding"
        return f"hub of a state outside the top {n_states}"

    kinds["why"] = kinds.apply(reason, axis=1)
    k = st.columns(3)
    k[0].metric(f"Top {n_cities} cities in a top-{n_states} state", int(kinds["city_type"].str.startswith("A").sum()))
    k[1].metric("…outside the top states", int(kinds["city_type"].str.startswith("B").sum()))
    k[2].metric(f"…in a top-{n_countries} country", int(kinds["state_in_top_countries"].sum()))
    f = px.scatter(kinds, x="state_records", y="records", color="city_type", size="share_of_state_%",
                   text=kinds["city"], color_discrete_map=KIND_COLORS, log_x=True, size_max=40,
                   hover_data={"place": True, "state_rank": True, "share_of_state_%": ":.1f", "why": True},
                   title=f"Top {n_cities} cities: city size vs size of its state (bubble = city's share of its state)")
    f.update_traces(textposition="top center")
    f.update_layout(xaxis_title="Breweries in the city's state (log scale)", yaxis_title="Breweries in the city",
                    legend_title_text="", height=560)
    st.plotly_chart(f, width="stretch")
    cols = ["rank", "place", "records", "state_records", "state_rank", "share_of_state_%"] \
        + (["share_of_state_in_reference_%"] if name != "raw" else []) + ["country_rank", "city_type", "why"]
    st.dataframe(kinds[cols].round(1), hide_index=True, width="stretch")
    if name != "raw":
        st.caption("`share_of_state_in_reference_%` is the same share computed on raw, which keeps the records "
                   "without coordinates. A share far above its raw value means the state lost many records in "
                   "cleaning, not that the city dominates it.")

    st.subheader(f"Are the top {n_states} states inside the top {n_countries} countries?")
    ts = state_counts.head(n_states).merge(country_counts[["country", "rank"]].rename(columns={"rank": "country_rank"}),
                                           on="country")
    ts["in_top_countries"] = ts["country_rank"] <= n_countries
    st.dataframe(ts[["rank", "place", "records", "share_%", "country_rank", "in_top_countries"]].round(2),
                 hide_index=True, width="stretch")

    st.divider()
    st.subheader("Questions answered in this tab")
    A = kinds[kinds["city_type"].str.startswith("A")]
    B = kinds[kinds["city_type"].str.startswith("B")]
    dom_states = dc.dominant_types(df, "state").head(n_states)
    dom_cities = dc.dominant_types(df, "city").head(n_cities)
    expl = []
    for w, g in B.groupby("why"):
        names_ = ", ".join(g["city"])
        if w.startswith("city-state"):
            expl.append(f"**{names_}**: the state or region is drawn around one metro, so the city *is* most of the "
                        f"state ({', '.join(f'{v:.0f}%' for v in g[ref_col])} of it even in raw)")
        elif w.startswith("share inflated"):
            expl.append(f"**{names_}**: looks dominant only because the rest of its state lost coordinates "
                        f"({', '.join(f'{a:.0f}% here vs {b:.0f}% in raw' for a, b in zip(g['share_of_state_%'], g[ref_col]))})")
        else:
            expl.append(f"**{names_}**: a real hub that concentrates a state ranked "
                        f"{', '.join(f'#{int(r)}' for r in g['state_rank'])} "
                        f"({', '.join(f'{v:.0f}%' for v in g['share_of_state_%'])} of its breweries) while the rest "
                        "of the state stays thin")
    cities_mix = df.merge(kinds[["country", STATE, "city"]], on=["country", STATE, "city"])["brewery_type"] \
        .value_counts(normalize=True) * 100
    states_mix = df.merge(ts[["country", STATE]], on=["country", STATE])["brewery_type"].value_counts(normalize=True) * 100
    st.markdown(f"""
**Are the cities with the most breweries located in the states with the most breweries?**\\
Partly. **{len(A)} of the top {n_cities} cities** sit in a top-{n_states} state ({", ".join(A['city'])}) and **{len(B)}** do not ({", ".join(B['city']) or "none"}).

**What does the "yes" part tell us?**\\
Where a top city sits in a top state, the state's size is partly built on that hub: {", ".join(f"{r['city']} holds {r['share_of_state_%']:.0f}% of {r[STATE]}" for _, r in A.head(4).iterrows())}. These states have a deep base of breweries *and* at least one city large enough to anchor it.

**Why can a city rank high when its state does not?**\\
{"; ".join(expl) + "." if expl else "Every top city sits in a top state in this selection."}

**Which brewery types dominate the highest-frequency states and cities?**\\
The top {n_states} states together are **{states_mix.index[0]}** {states_mix.iat[0]:.0f}% and **{states_mix.index[1]}** {states_mix.iat[1]:.0f}%; the top {n_cities} cities are **{cities_mix.index[0]}** {cities_mix.iat[0]:.0f}% and **{cities_mix.index[1]}** {cities_mix.iat[1]:.0f}%. Individually, {", ".join(f"{r['place'].split(',')[0]} ({r['dominant_type']})" for _, r in dom_states[dom_states['dominant_type'] != 'micro'].iterrows()) or "no top state"} {"is" if (dom_states['dominant_type'] != 'micro').sum() == 1 else "are"} the exception among states, and {", ".join(f"{r['place'].split(',')[0]} ({r['dominant_type']})" for _, r in dom_cities[dom_cities['dominant_type'] != 'micro'].iterrows()) or "no top city"} among cities.

**Are the top states inside the top countries?**\\
{int(ts['in_top_countries'].sum())} of the top {n_states} states belong to a top-{n_countries} country; {int((ts['country_rank'] == 1).sum())} of them are in {country_counts.iloc[0]['place']} alone. Size at the state level is mostly inherited from the country: the United States is split into 50+ states and still fills the ranking, while smaller countries enter it only through their single dominant region.

**What can we conclude?**\\
Brewery counts in this dataset are organised around metropolitan hubs, but a state's rank depends on how many hubs and small towns it adds up, not on its largest city. Ranking cities and ranking states therefore answer different questions: cities show where breweries cluster, states show where the brewing base is broad. Any state- or city-level comparison must also be read against coverage, because the regions that lost coordinates fall in the geocoded ranking without having fewer breweries.
""")

# ======================================================================= 3. type A / type B
with tabs[2]:
    n_ab = st.slider("Cities to classify", 10, 100, max(20, n_cities), 5, key="ab_n")
    ab = dc.city_state_kinds(df, n_states=n_states, n_cities=n_ab, n_countries=n_countries)
    abm = ab.join(city_centers, on="place").dropna(subset=["latitude", "longitude"])
    c1, c2 = st.columns(2)
    f = px.scatter(ab, x="share_of_state_%", y="records", color="city_type", color_discrete_map=KIND_COLORS,
                   hover_name="place", hover_data={"state_records": True, "state_rank": True},
                   text=ab["city"].where(ab["rank"] <= 15, ""),
                   title=f"Top {n_ab} cities: size vs share of their state")
    f.update_traces(textposition="top center", marker=dict(size=11, line=dict(width=1, color="#151515")))
    f.update_layout(xaxis_title="City's share of its state (%)", yaxis_title="Breweries in the city",
                    legend_title_text="", height=520)
    c1.plotly_chart(f, width="stretch")
    m = point_map(abm, color="city_type", size="records", color_discrete_map=KIND_COLORS, hover_name="place",
                  hover_data={"share_of_state_%": ":.1f", "state_rank": True}, size_max=30, opacity=0.9, height=520,
                  style=basemap, title="Type A and type B cities")
    m.update_layout(legend_title_text="")
    c2.plotly_chart(m, width="stretch", config={"scrollZoom": True})
    st.dataframe(ab[["rank", "place", "records", "state_records", "state_rank", "share_of_state_%", "city_type"]].round(1),
                 hide_index=True, width="stretch", height=300)

    mixes = []
    for kind in KIND_COLORS:
        keys = ab[ab["city_type"] == kind][["country", STATE, "city"]]
        if len(keys):
            m_ = df.merge(keys, on=["country", STATE, "city"])["brewery_type"].value_counts(normalize=True) * 100
            mixes.append(m_.rename(kind))
    mixdf = pd.concat(mixes, axis=1).fillna(0)
    long = mixdf.rename_axis("brewery_type").reset_index().melt(id_vars="brewery_type", var_name="kind",
                                                                value_name="share")
    f = px.bar(long, x="share", y="kind", color="brewery_type", orientation="h", color_discrete_map=type_cmap,
               category_orders={"brewery_type": type_order["brewery_type"]},
               title="Brewery-type mix of type A vs type B cities (each bar = 100%)")
    f.update_layout(barmode="stack", xaxis_title="% of the group's records", yaxis_title="", height=280,
                    legend_title_text="Type")
    st.plotly_chart(f, width="stretch")

    st.divider()
    st.subheader("Questions answered in this tab")
    A2, B2 = ab[ab["city_type"].str.startswith("A")], ab[ab["city_type"].str.startswith("B")]
    dmix = (mixdf.iloc[:, 0] - mixdf.iloc[:, -1]).sort_values() if mixdf.shape[1] == 2 else pd.Series(dtype=float)
    st.markdown(f"""
**What are type A and type B cities?**\\
Both have many breweries. A **type A** city sits in one of the top {n_states} states: breweries are common in the city *and* across its state. A **type B** city sits in a state outside the top {n_states}: the city is a hub, but its state is not a large brewing state overall. Each city's share of its state tells how much of the state the city carries.

**How many of each?**\\
Of the top {n_ab} cities, **{len(A2)}** are type A and **{len(B2)}** type B. Type A cities hold a median **{A2['share_of_state_%'].median():.1f}%** of their state; type B cities **{B2['share_of_state_%'].median():.1f}%**. Type B cities carry their states; type A cities are one hub among several.

**Do the two kinds differ by brewery type?**\\
{("Yes. " + ", ".join(f"{t}: {v:+.1f} pp" for t, v in pd.concat([dmix.head(2), dmix.tail(2)]).items()) + " (type A minus type B). ") if len(dmix) else ""}Type B includes the city-states and one-city regions abroad, so it inherits their national labelling conventions (see Brewery Types → Patterns).
""")

# ======================================================================= 4. city distribution
with tabs[3]:
    stats = pd.DataFrame({lv: dc.concentration_stats(dc.counts(df, lv)["records"]) for lv in ["country", "state", "city"]}).T
    st.dataframe(stats.round(3), width="stretch")
    st.caption("groups = places at that level. Gini: 0 = every place has the same count, 1 = one place has all. "
               "HHI = sum of squared shares; effective_groups = 1/HHI, the number of equal-sized places that would "
               "give the same concentration. zipf_exponent = slope of log(count) against log(rank) for places with "
               "≥ 3 records (≈1 is the classic rank-size law, below 1 a flatter tail).")

    c1, c2 = st.columns(2)
    rs = pd.concat([dc.counts(df, lv)[["rank", "records"]].assign(level=lv) for lv in ["state", "city"]])
    f = px.scatter(rs, x="rank", y="records", color="level", log_x=True, log_y=True,
                   color_discrete_map={"state": "#2a78d6", "city": "#eb6834"},
                   title="Rank-size plot (log-log): states vs cities")
    f.update_traces(marker=dict(size=5))
    for lv, colr in [("state", "#2a78d6"), ("city", "#eb6834")]:
        z = stats.loc[lv, "zipf_exponent"]
        top = dc.counts(df, lv)
        top = top[top["records"] >= 3]
        if len(top) >= 5 and not np.isnan(z):
            b = np.polyfit(np.log(top["rank"]), np.log(top["records"]), 1)
            xs = np.array([1, top["rank"].max()])
            f.add_trace(go.Scatter(x=xs, y=np.exp(b[1]) * xs ** b[0], mode="lines", showlegend=True,
                                   name=f"{lv} fit, exponent {z:.2f}", line=dict(color=colr, dash="dash", width=2)))
    f.update_layout(xaxis_title="Rank (log)", yaxis_title="Breweries (log)", legend_title_text="", height=460)
    c1.plotly_chart(f, width="stretch")
    lz = pd.concat([dc.lorenz(dc.counts(df, lv)["records"]).assign(level=lv) for lv in ["country", "state", "city"]])
    f = px.line(lz, x="share_of_groups_%", y="share_of_records_%", color="level",
                color_discrete_map={"country": "#1baf7a", "state": "#2a78d6", "city": "#eb6834"},
                title="Lorenz curves: how evenly breweries spread over places")
    f.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", name="perfect equality",
                           line=dict(color="#8a8a8a", dash="dot")))
    f.update_layout(xaxis_title="% of places (smallest first)", yaxis_title="% of breweries", legend_title_text="",
                    height=460)
    c2.plotly_chart(f, width="stretch")

    sizes = city_counts["records"].value_counts().sort_index().rename_axis("breweries_in_city").reset_index(name="cities")
    f = px.bar(sizes, x="breweries_in_city", y="cities", log_y=True, color_discrete_sequence=["#eb6834"],
               title="How many cities have 1, 2, 3 … breweries? (log scale)")
    f.update_layout(xaxis_title="Breweries in the city", yaxis_title="Cities (log)", height=360)
    st.plotly_chart(f, width="stretch")

    rows = []
    for c, g in df.groupby("country"):
        if len(g) >= 50:
            s = dc.concentration_stats(dc.counts(g, "city")["records"])
            rows.append({"country": c, "records": len(g), "cities": s["groups"],
                         "single_brewery_cities_%": s["single_record_groups_%"], "top_city_share_%": s["top_1_share_%"],
                         "gini": s["gini"], "zipf_exponent": s["zipf_exponent"]})
    per_country = pd.DataFrame(rows).sort_values("records", ascending=False)
    st.markdown("**City distribution inside each country (countries with ≥ 50 records)**")
    st.dataframe(per_country.round(2), hide_index=True, width="stretch")

    st.divider()
    st.subheader("Questions answered in this tab")
    cs_, ss_ = stats.loc["city"], stats.loc["state"]
    st.markdown(f"""
**Why does the city column behave differently from the others?**\\
Because it is a long, flat tail. There are **{int(cs_['groups']):,}** cities for {len(df):,} records: the median city has **{cs_['median']:.0f}** brewery, the mean is {cs_['mean']:.2f}, and **{cs_['single_record_groups_%']:.0f}%** of cities have exactly one. The largest city holds only {cs_['top_1_share_%']:.1f}% of all records and the top 10 cities {cs_['top_10_share_%']:.1f}%, against {ss_['top_10_share_%']:.1f}% for the top 10 states.

**What do the statistics say about how breweries spread?**\\
At the state level the distribution is steep (Zipf exponent **{ss_['zipf_exponent']:.2f}**, Gini {ss_['gini']:.2f}): a few states dominate. At the city level it is flat (Zipf exponent **{cs_['zipf_exponent']:.2f}**, Gini {cs_['gini']:.2f}): even the big hubs are only a few dozen breweries, and most of the volume sits in thousands of small towns with one or two each. The city Gini is *lower* than the state Gini not because cities are balanced but because almost all of them are equally tiny. In other words, breweries are a small-town phenomenon that also has a handful of metropolitan hubs, not a big-city phenomenon.

**Does this hold in every country?**\\
The share of single-brewery cities ranges from {per_country['single_brewery_cities_%'].min():.0f}% ({per_country.loc[per_country['single_brewery_cities_%'].idxmin(), 'country']}) to {per_country['single_brewery_cities_%'].max():.0f}% ({per_country.loc[per_country['single_brewery_cities_%'].idxmax(), 'country']}). Countries with a steep city distribution (Zipf ≥ 1: {", ".join(per_country.loc[per_country['zipf_exponent'] >= 1, 'country']) or "none"}) have one or two dominant cities in the data; the others spread their records over many towns.
""")

# ======================================================================= 5. insights by level
with tabs[4]:
    if name != "raw":
        st.subheader("How geocoding changes the ranking (raw vs this dataset)")
        c1, c2 = st.columns(2)
        for box, lv in [(c1, "state"), (c2, "city")]:
            shift = dc.rank_shift(df, raw, lv, n=15)
            box.markdown(f"**{lv.title()}s: top 15 in raw**")
            box.dataframe(shift.round(1), hide_index=True, width="stretch")
    typ = dc.type_counts(df)
    co1, co2 = country_counts.iloc[0], country_counts.iloc[1]
    st1 = state_counts.iloc[0]
    n50 = int((state_counts["records"].cumsum() / len(df) * 100 < 50).sum()) + 1
    cstat = dc.concentration_stats(city_counts["records"])
    mixx = dc.type_mix(df, "country", places=country_counts.query("records >= 30")["place"].tolist())
    one_label = mixx[mixx.max(axis=1) >= 85]
    state_shift_text = city_shift_text = ""
    if name != "raw":
        shift_s = dc.rank_shift(df, raw, "state", n=15)
        drops = shift_s[shift_s["rank_change"] <= -5]
        if len(drops):
            state_shift_text = " Geocoding reshapes this level: " + ", ".join(
                f"{r['place'].split(',')[0]} keeps {r['records_kept_%']:.0f}% of its records and moves from "
                f"#{int(r['rank_reference'])} to #{int(r['rank_dataset'])}" for _, r in drops.iterrows()) + "."
        shift_c = dc.rank_shift(df, raw, "city", n=15)
        drops = shift_c[shift_c["rank_change"] <= -5]
        if len(drops):
            city_shift_text = " The cities that fall furthest after geocoding: " + ", ".join(
                f"{r['place'].rsplit(',', 1)[0]} #{int(r['rank_reference'])} → #{int(r['rank_dataset'])}"
                for _, r in drops.iterrows()) + "."
    changed = df[df["state_province"] != df[STATE]][["country", STATE]].drop_duplicates()
    merged = df.merge(changed, on=["country", STATE])
    gain = (merged.groupby(STATE).size()
            - merged.groupby([STATE, "state_province"]).size().groupby(level=0).max())
    harm_text = (f" Harmonizing merges {merged['state_province'].nunique()} raw labels into {len(gain)} regions; "
                 "each merged region gains only "
                 + ", ".join(f"{int(v)} record{'s' if v != 1 else ''} ({k})" for k, v in gain.items())
                 + " over its largest original label, so the ranking barely moves." if len(gain) else "")
    st.subheader("Insights by level")
    st.markdown(f"""
**Country.** {co1['place']} holds **{co1['share_%']:.1f}%** of {name}, {co1['records'] / co2['records']:.1f}× the next country ({co2['place']}, {co2['share_%']:.1f}%). {len(country_counts)} countries appear, but {int((country_counts['records'] < 50).sum())} of them have fewer than 50 records, so any "global" statement is really a statement about the United States plus a handful of countries.

**State / province.** **{n50}** states hold half of all records; {st1['place']} alone holds {st1["share_%"]:.1f}%.{harm_text}{state_shift_text}

**City.** {int(cstat['groups']):,} cities, {cstat['single_record_groups_%']:.0f}% with a single brewery, and a flat rank-size curve (Zipf exponent {cstat['zipf_exponent']:.2f}). Brewing is spread over small towns, punctuated by a few hubs such as {", ".join(city_counts.head(4)["city"])}.{city_shift_text}

**Brewery type.** {typ.iloc[0]['brewery_type']} ({typ.iloc[0]['share_%']:.1f}%) and {typ.iloc[1]['brewery_type']} ({typ.iloc[1]['share_%']:.1f}%) make up {typ.head(2)['share_%'].sum():.1f}% of records. Types follow national labelling habits ({", ".join(f"{c} {mixx.loc[c].idxmax()} {mixx.loc[c].max():.0f}%" for c in one_label.index)}), US states split by whether they keep closed records, and contract/proprietor brewers sit next to their host breweries (Brewery Types → Patterns).

**What this means for the rest of the analysis.** Spatial results describe the United States first. State comparisons need harmonized labels and must be read against coverage. City-level hubs are few and well defined, which is exactly what density-based clustering should pick up; the long tail of single-brewery towns is what it will call noise.
""")
