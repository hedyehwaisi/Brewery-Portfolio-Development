import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import prediction_core as pc
from common import harmonized, pick_basemap, table, shared_cache
from geo_plots import point_map, type_style
from geographic_utils import haversine_km

st.set_page_config(page_title="Prediction", page_icon="🔮", layout="wide")
st.title("🔮 Where would a new brewery open, and what type would it be?")
basemap = pick_basemap()
STATE = pc.STATE

params_t = table("prediction_results.db", "params")
if params_t is None:
    st.error("data/prediction_results.db not found. Run brewery_prediction.ipynb."); st.stop()
P = dict(zip(params_t["key"], params_t["value"].astype(float)))
ALPHA, BETA, BW = P["alpha"], P["beta"], P["bandwidth_km"]
st.caption(f"Trained on the {len(pc.operating(harmonized('raw'))):,} operating breweries of raw (planning and closed "
           f"records excluded), harmonized states, corrected coordinates. Settings chosen by 5-fold cross-validation in "
           f"brewery_prediction.ipynb: α = {ALPHA:g}, β = {BETA:g}, density bandwidth = {BW:g} km.")


@shared_cache
def model():
    return pc.PlaceTypeModel(ALPHA, BETA).fit(pc.operating(harmonized("raw")))


@shared_cache
def centers(level):
    g = harmonized("raw").dropna(subset=["latitude", "longitude"])
    return g.groupby(pc.LEVELS[level])[["latitude", "longitude"]].mean()


@shared_cache(show_spinner="Estimating the density surface…")
def density(country, state, brewery_type):
    g = pc.operating(harmonized("all_records"))
    g = g[g["country"] == country]
    if state != "All states":
        g = g[g[STATE] == state]
    if brewery_type != "Any type":
        g = g[g["brewery_type"] == brewery_type]
    if len(g) < 5:
        return None, g
    pad_la = max((g["latitude"].max() - g["latitude"].min()) * 0.08, 0.3)
    pad_lo = max((g["longitude"].max() - g["longitude"].min()) * 0.08, 0.3)
    la = (g["latitude"].quantile(0.005) - pad_la, g["latitude"].quantile(0.995) + pad_la)
    lo = (g["longitude"].quantile(0.005) - pad_lo, g["longitude"].quantile(0.995) + pad_lo)
    grid = pc.kde_grid(pc.kde_fit(g["latitude"], g["longitude"], BW), la, lo, n=110)
    return grid, g


m = model()
raw_op = pc.operating(harmonized("raw"))
type_cmap, type_order = type_style(harmonized("raw"))
LEVEL_NAMES = {"country": "Country", "state": "State / province", "city": "City"}
tabs = st.tabs(["Where would it open?", "What type would it be?", "Where exactly? (density)",
                "The pipeline: planning breweries", "How reliable is this?"])

# ======================================================================= 1. place
with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    level = c1.radio("Predict the", list(LEVEL_NAMES), format_func=LEVEL_NAMES.get, index=1, key="p_level")
    countries = raw_op["country"].value_counts().index.tolist()
    country = c2.selectbox("Country", countries, disabled=level == "country", key="p_country")
    states = raw_op[raw_op["country"] == country][STATE].value_counts().index.tolist()
    state = c3.selectbox("State / province", states, disabled=level != "city", key="p_state")
    types = ["Any type (predict it)"] + m.types
    btype = c4.selectbox("Brewery type", types, key="p_type")
    n_show = st.slider("Places to show", 5, 30, 12, key="p_n")
    parent = () if level == "country" else (country,) if level == "state" else (country, state)
    dist = (m.place_distribution(level, parent) if btype.startswith("Any")
            else m.place_given_type(level, btype, parent))
    show = dist.head(n_show).rename("probability").reset_index().rename(columns={"index": "place"})
    show["label"] = [f"{p * 100:.1f}%" for p in show["probability"]]
    show["kind"] = np.where(show["place"] == pc.NEW_PLACE, "a place with no brewery yet", "existing brewery place")
    where = "the world" if level == "country" else country if level == "state" else f"{state}, {country}"
    f = px.bar(show, x="probability", y="place", orientation="h", text="label", color="kind",
               color_discrete_map={"existing brewery place": "#2a78d6", "a place with no brewery yet": "#eb6834"},
               title=f"Most likely {LEVEL_NAMES[level].lower()} for a new "
                     f"{'brewery' if btype.startswith('Any') else btype + ' brewery'} in {where}")
    f.update_traces(textposition="outside", cliponaxis=False)
    f.update_layout(yaxis=dict(autorange="reversed"), yaxis_title="", xaxis_tickformat=".0%", legend_title_text="",
                    height=34 * len(show) + 160, margin=dict(r=80))
    st.plotly_chart(f, width="stretch")

    cen = centers(level)
    pts = show[show["place"] != pc.NEW_PLACE].copy()
    pk = [parent + (p,) if len(parent) else p for p in pts["place"]]
    pts["latitude"] = [cen["latitude"].get(k, np.nan) for k in pk]
    pts["longitude"] = [cen["longitude"].get(k, np.nan) for k in pk]
    pts = pts.dropna(subset=["latitude", "longitude"])
    if len(pts):
        mp = point_map(pts, size="probability", hover_name="place", hover_data={"label": True, "probability": False},
                       color_discrete_sequence=["#2a78d6"], size_max=40, opacity=0.9, height=480, style=basemap,
                       text="place", title="The same places on the map (bubble = probability)")
        st.plotly_chart(mp, width="stretch", config={"scrollZoom": True})

    if btype.startswith("Any"):
        top_place = next((p for p in dist.index if p != pc.NEW_PLACE), None)
        key = list(parent) + [top_place]
        td = m.type_distribution(*(key + [None] * (3 - len(key)))).head(5)
        rd = m.type_distribution(*(list(parent) + [None] * (3 - len(parent)))).head(5)
        c1, c2 = st.columns(2)
        for box, d_, ttl in [(c1, rd, f"Type of a new brewery anywhere in {where}"),
                             (c2, td, f"Type of a new brewery in {top_place}, the most likely place")]:
            t_ = d_.rename("probability").reset_index().rename(columns={"index": "brewery_type"})
            g_ = px.bar(t_, x="probability", y="brewery_type", orientation="h", color="brewery_type",
                        color_discrete_map=type_cmap, text=[f"{v * 100:.1f}%" for v in t_["probability"]], title=ttl)
            g_.update_layout(showlegend=False, yaxis=dict(autorange="reversed"), yaxis_title="",
                             xaxis_tickformat=".0%", height=300)
            box.plotly_chart(g_, width="stretch")

    st.divider()
    st.subheader("Questions answered in this tab")
    top = dist.drop(pc.NEW_PLACE, errors="ignore").head(3)
    p_new = dist.get(pc.NEW_PLACE, 0) * 100
    pnew_txt = "under 0.1%" if p_new < 0.1 else f"{p_new:.1f}%"
    st.markdown(f"""
**If a new {'brewery' if btype.startswith('Any') else btype + ' brewery'} opened in {where}, where would it most likely be?**\\
{", ".join(f"**{p}** ({v * 100:.1f}%)" for p, v in top.items())}. There is a **{pnew_txt}** chance it opens in a {LEVEL_NAMES[level].lower()} that has no brewery in the data yet.

**How is that number made?**\\
From the share of today's operating breweries in each place, lightly smoothed (α = {ALPHA:g}) so small places are not ruled out, and a Good–Turing estimate for brand-new places: the more places in the region hold a single brewery, the more likely the next one is somewhere new. Choosing a type re-weights every place by how common that type is there (Bayes' rule).

**And its type?**\\
When no type is chosen, the type is predicted for the region and for its most likely place: a place's own mix, pulled toward its region's (β = {BETA:g} pseudo-breweries), so a town with three breweries does not dictate the answer on its own.
""")

# ======================================================================= 2. type
with tabs[1]:
    c1, c2, c3 = st.columns(3)
    tc = c1.selectbox("Country", countries, key="t_country")
    ts = c2.selectbox("State / province", ["(whole country)"] + raw_op[raw_op["country"] == tc][STATE]
                      .value_counts().index.tolist(), key="t_state")
    cities = [] if ts == "(whole country)" else raw_op[(raw_op["country"] == tc) & (raw_op[STATE] == ts)]["city"] \
        .value_counts().index.tolist()
    tci = c3.selectbox("City", ["(whole state)"] + cities, disabled=ts == "(whole country)", key="t_city")
    s_ = None if ts == "(whole country)" else ts
    ci_ = None if (s_ is None or tci == "(whole state)") else tci
    rows = [("World (global mix)", m.t_global), (tc, m.type_distribution(tc))]
    if s_:
        rows.append((s_, m.type_distribution(tc, s_)))
    if ci_:
        rows.append((ci_ + " (shrunk toward " + s_ + ")", m.type_distribution(tc, s_, ci_)))
        raw_mix = m.t["city"].loc[(tc, s_, ci_)]
        rows.append((ci_ + " (its own records only)", raw_mix / raw_mix.sum()))
    long = pd.concat([d.rename("probability").reset_index().rename(columns={"index": "brewery_type"})
                      .assign(level=name) for name, d in rows])
    long = long[long["probability"] > 0.001]
    f = px.bar(long, x="probability", y="level", color="brewery_type", orientation="h", color_discrete_map=type_cmap,
               category_orders={"level": [r[0] for r in rows], "brewery_type": type_order["brewery_type"]},
               title="Predicted type of a new brewery at each level (each bar = 100%)")
    f.update_layout(barmode="stack", xaxis_tickformat=".0%", yaxis_title="", legend_title_text="Type",
                    height=110 + 60 * len(rows))
    st.plotly_chart(f, width="stretch")
    final = rows[-2][1] if ci_ else rows[-1][1]
    n_here = int(m.t["city"].loc[(tc, s_, ci_)].sum()) if ci_ else int(m.n["state"].get((tc, s_), 0)) if s_ \
        else int(m.n["country"].get(tc, 0))
    st.divider()
    st.subheader("Questions answered in this tab")
    st.markdown(f"""
**What type would a new brewery in {ci_ or s_ or tc} be?**\\
Most likely **{final.index[0]}** ({final.iat[0] * 100:.1f}%), then {final.index[1]} ({final.iat[1] * 100:.1f}%). That rests on {n_here:,} operating breweries {'in this city' if ci_ else 'in this state' if s_ else 'in this country'}, pulled toward the {'state' if ci_ else 'national' if s_ else 'global'} mix.

**Why shrink a place's own mix?**\\
A city with two breweries, both brewpubs, would otherwise predict "brewpub" with certainty. Shrinkage lets small places speak only as loudly as their evidence allows; large places keep almost their own mix. Cross-validation chose β = {BETA:g}: the predictions that score best on breweries the model has not seen.

**Does the city matter?**\\
Hardly. On held-out breweries, the state alone already gives 68% top-1 accuracy (global mix: 55%); adding the city changes the log-loss by less than 1%. Brewery types follow the labelling habits of each country and state far more than the town.
""")

# ======================================================================= 3. density
with tabs[2]:
    g_all = pc.operating(harmonized("all_records"))
    c1, c2, c3 = st.columns(3)
    dc_ = c1.selectbox("Country", g_all["country"].value_counts().index.tolist(), key="d_country")
    ds_ = c2.selectbox("State / province", ["All states"] + g_all[g_all["country"] == dc_][STATE]
                       .value_counts().index.tolist(), key="d_state")
    dt_ = c3.selectbox("Brewery type", ["Any type"] + sorted(g_all["brewery_type"].unique()), key="d_type")
    grid, pts = density(dc_, ds_, dt_)
    if grid is None:
        st.info("Fewer than 5 geocoded breweries match this choice; the density would mean nothing.")
    else:
        dm = px.density_map(grid[grid["share"] > grid["share"].max() * 0.01], lat="latitude", lon="longitude",
                            z="share", radius=12, map_style=basemap, color_continuous_scale="Viridis", height=600,
                            center={"lat": float(pts["latitude"].mean()), "lon": float(pts["longitude"].mean())},
                            zoom=3 if ds_ == "All states" else 5,
                            title=f"Where the next {'' if dt_ == 'Any type' else dt_ + ' '}brewery is most likely "
                                  f"({ds_ if ds_ != 'All states' else dc_})")
        dm.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(dm, width="stretch", config={"scrollZoom": True})
        # top spots: greedy picks at least 2 bandwidths apart, named after the nearest brewery city
        cc = pts.groupby(pc.LEVELS["city"])[["latitude", "longitude"]].mean().reset_index()
        picks = []
        for _, r in grid.sort_values("share", ascending=False).iterrows():
            if all(haversine_km(r["latitude"], r["longitude"], q["latitude"], q["longitude"]) > 2 * BW for q in picks):
                picks.append(r)
            if len(picks) == 10:
                break
        spots = pd.DataFrame(picks)
        d_city = [np.argmin((cc["latitude"] - r["latitude"]) ** 2 + ((cc["longitude"] - r["longitude"])
                                                                     * np.cos(np.radians(r["latitude"]))) ** 2)
                  for _, r in spots.iterrows()]
        spots["nearest_brewery_city"] = cc["city"].iloc[d_city].to_numpy() + ", " + cc[STATE].iloc[d_city].to_numpy()
        spots["relative_density"] = spots["share"] / grid["share"].mean()
        st.markdown("**The ten densest spots** (at least two bandwidths apart)")
        st.dataframe(spots[["nearest_brewery_city", "latitude", "longitude", "relative_density"]].round(3),
                     hide_index=True, width="stretch")
        st.caption(f"Gaussian kernel density of {len(pts):,} geocoded operating breweries on the sphere, bandwidth "
                   f"{BW:g} km (chosen by held-out likelihood). relative_density = how many times denser than the "
                   "average cell of the map.")

    st.divider()
    st.subheader("Questions answered in this tab")
    st.markdown(f"""
**Where exactly would a new brewery most likely be?**\\
{(f"In {ds_ if ds_ != 'All states' else dc_}, around **{spots.iloc[0]['nearest_brewery_city']}**, then " + ", ".join(spots['nearest_brewery_city'].iloc[1:4]) + ".") if grid is not None else "Choose a region with enough breweries."} The map spreads every existing brewery over a {BW:g}-km kernel and adds them up, so the bright areas are where breweries already crowd together.

**Why {BW:g} km?**\\
It is the bandwidth that best predicts held-out breweries: narrower kernels overfit single towns, wider ones blur separate hubs into one (the same trade-off DBSCAN's eps faces).

**Does this fit the openings in preparation?**\\
The geocoded planning breweries sit in denser places than a typical brewery (mean log-density {P['planning_kde_log_density']:.2f} against {table('prediction_results.db', 'kde_cv')['heldout_log_likelihood'].max():.2f}): new breweries open where breweries already are. Caveat: only 108 planning records have coordinates and most are in California.
""")

# ======================================================================= 4. pipeline
with tabs[3]:
    pipe = table("prediction_results.db", "pipeline_by_state")
    min_pl = st.slider("Minimum planning breweries per state", 1, 30, 10, key="pipe_min")
    pp = pipe[pipe["planning"] >= min_pl].sort_values("lift")
    pp["label"] = [f"{l:.2f}× ({p} planning, {o} operating)" for l, p, o in zip(pp["lift"], pp["planning"], pp["operating"])]
    f = px.bar(pp, x="lift", y=STATE, orientation="h", text="label", color="lift",
               color_continuous_scale=[[0, "#2a78d6"], [0.5, "#e6e6e6"], [1, "#eb6834"]],
               range_color=(0, 2), title="Pipeline lift: share of planning breweries ÷ share of operating breweries")
    f.add_vline(x=1, line_dash="dot")
    f.update_traces(textposition="outside", cliponaxis=False)
    f.update_layout(yaxis_title="", coloraxis_showscale=False, height=30 * len(pp) + 140, margin=dict(r=220))
    st.plotly_chart(f, width="stretch")
    sc = centers("state").reset_index()
    pm = pipe.merge(sc[sc["country"] == "United States"], on=STATE, how="left").dropna(subset=["latitude"])
    pm = pm[pm["planning"] >= min_pl]
    mm = point_map(pm, size="planning", color="lift", color_continuous_scale=[[0, "#2a78d6"], [0.5, "#e6e6e6"],
                                                                                [1, "#eb6834"]],
                   range_color=(0, 2), hover_name=STATE, hover_data={"planning": True, "operating": True,
                                                                     "lift": ":.2f"},
                   size_max=35, opacity=0.95, height=480, style=basemap,
                   title="Planning breweries per state (size) and lift (orange > 1 > blue)")
    mm.update_layout(coloraxis_colorbar=dict(title="lift", thickness=12))
    st.plotly_chart(mm, width="stretch", config={"scrollZoom": True})
    st.dataframe(pipe.round(2), hide_index=True, width="stretch", height=300)

    st.divider()
    st.subheader("Questions answered in this tab")
    up = pipe[pipe["planning"] >= 10].nlargest(5, "lift")
    down = pipe[pipe["planning"] >= 10].nsmallest(3, "lift")
    st.markdown(f"""
**Where are openings being prepared?**\\
The {int(pipe['planning'].sum())} planning breweries (all in the United States) concentrate in {", ".join(pipe.head(4)[STATE])}. Size alone explains part of that, so the lift compares each state's share of planning breweries with its share of operating ones.

**Which states are growing faster than their size suggests?**\\
{", ".join(f"**{r[STATE]}** ({r['lift']:.1f}×)" for _, r in up.iterrows())}: the South and the East Coast. Below 1: {", ".join(f"{r[STATE]} ({r['lift']:.2f}×)" for _, r in down.iterrows())}, states with a large operating base and a comparatively thin pipeline.

**Is the difference real?**\\
Yes, though modest: a chi-square test of planning vs operating state mix gives p = {P['planning_p_value']:.1e} and Cramér's V = {P['planning_cramers_v']:.2f}. The next wave of breweries is not a copy of the current map.

**Caveat.** Planning records are as complete as their contributors make them; a state with no planning record may simply not record them.
""")

# ======================================================================= 5. reliability
with tabs[4]:
    summ = table("prediction_results.db", "summary")
    st.markdown("**Cross-validated accuracy (5 folds, operating breweries)**")
    st.dataframe(summ.round(3), hide_index=True, width="stretch")
    c1, c2, c3 = st.columns(3)
    cvp = table("prediction_results.db", "cv_places")
    f = px.line(cvp, x="alpha", y="log_loss", color="level", markers=True, log_x=True,
                title="Place model: held-out log-loss by α")
    c1.plotly_chart(f, width="stretch")
    cvt = table("prediction_results.db", "cv_types")
    f = px.line(cvt.melt(id_vars="beta", value_vars=["log_loss", "log_loss_state_only"]), x="beta", y="value",
                color="variable", markers=True, log_x=True, title="Type model: held-out log-loss by β")
    f.add_hline(y=float(cvt["log_loss_global"].iloc[0]), line_dash="dot", annotation_text="global mix")
    c2.plotly_chart(f, width="stretch")
    kcv = table("prediction_results.db", "kde_cv")
    f = px.line(kcv, x="bandwidth_km", y="heldout_log_likelihood", markers=True,
                title="Density: held-out log-likelihood by bandwidth")
    c3.plotly_chart(f, width="stretch")
    pt = table("prediction_results.db", "planning_test")
    st.markdown("**Out-of-sample test on the planning breweries (future openings the model never saw)**")
    st.dataframe(pt.round(3), hide_index=True, width="stretch")

    st.divider()
    st.subheader("Questions answered in this tab")
    s_ = summ.set_index("task")
    ptt = pt.set_index("test")
    st.markdown(f"""
**How much better than guessing are these predictions?**\\
On breweries held out of training, the state of a new brewery is predicted with log-loss {s_.loc['state, given the country', 'log_loss']:.2f} against {s_.loc['state, given the country', 'baseline_log_loss']:.2f} for a uniform guess, and its city with {s_.loc['city, given the state', 'log_loss']:.2f} against {s_.loc['city, given the state', 'baseline_log_loss']:.2f}. Its type is right on the first guess {s_.loc['type, given the city', 'top1_%']:.0f}% of the time, against {float(cvt.loc[cvt['beta'] == BETA, 'top1_global_%'].iloc[0]):.0f}% for always guessing the most common type.

**Does it hold for real future openings?**\\
On the planning breweries the place model still beats a uniform guess (state {ptt.loc['state', 'log_loss']:.2f} vs {ptt.loc['state', 'log_loss_uniform']:.2f}; city {ptt.loc['city', 'log_loss']:.2f} vs {ptt.loc['city', 'log_loss_uniform']:.2f}), but less well than in cross-validation, because the geography of openings is shifting (see *The pipeline*). Good–Turing expected **{ptt.loc['new_city', 'predicted_%']:.1f}%** of them in a city with no brewery yet; **{ptt.loc['new_city', 'observed_%']:.1f}%** actually are.

**What does the model assume?**\\
That a new brewery resembles today's operating ones, place by place. The data has no opening dates, so it cannot say *how many* will open or *when*; it says where the next one is most likely and what it would most likely be, and the planning test measures how far that holds.
""")
