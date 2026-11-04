"""Spatial Clustering page.

Every DBSCAN, HDBSCAN and OPTICS run of the parameter grids is read from the allruns_* tables of
data/clustering_results.db, written by brewery_clustering_all_runs.ipynb. Nothing is fitted at page load: the page
decodes the stored labels of the selected run and describes its clusters with cluster_analysis.cluster_table.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import cluster_analysis as ca
import clustering_store as store
import origins_core as oc
from common import pick_basemap, harmonized, shared_cache
from geo_plots import add_halo, point_map, SATURATED_SCALE
from palette import color_map, NOISE_COLOR, OTHER_COLOR

st.set_page_config(page_title="Spatial Clustering | Brewery Analysis", page_icon="🌍", layout="wide")
st.title("Spatial Clustering")
st.markdown(
    "Every setting of three density-based algorithms is stored: **264 DBSCAN** (`eps` × `min_samples`), **336 "
    "HDBSCAN** (`min_cluster_size` × `min_samples`) and **33 OPTICS** (`min_samples` × `xi`) runs per dataset, on "
    "corrected coordinates. Pick any run in the sidebar; compare all of them below. No configuration is declared "
    "best in advance: the comparison section ranks runs on criteria you can weigh yourself.")

ALG_COLORS = {"DBSCAN": "#2a78d6", "HDBSCAN": "#eb6834", "OPTICS": "#1baf7a"}
MERGE_COLORS = {"joins distinct hubs": "#eb6834", "single hub or region": "#2a78d6"}


# ============================================================ cached readers
# Per-run results are bounded (RUN_CACHE entries each) so browsing many runs does not grow memory without limit.
RUN_CACHE = 64


@shared_cache(show_spinner=False)
def all_runs(sig):
    return store.allruns_runs()


@shared_cache(show_spinner=False)
def records(dataset, sig):
    """Corrected, harmonized records in the order of the stored label arrays."""
    order = store.allruns_records(dataset)
    return harmonized(dataset).set_index("id").loc[order["id"]].reset_index()


@shared_cache(show_spinner=False, max_entries=RUN_CACHE)
def labels(run_id, sig):
    return store.allruns_labels(run_id)


@shared_cache(show_spinner=False)
def reference():
    return oc.load_reference()


@shared_cache(show_spinner="Describing every cluster of this run…", max_entries=RUN_CACHE)
def clusters(run_id, hub_records, hub_inside, hub_sep, sig):
    dataset = run_id.split("|")[0]
    return ca.cluster_table(records(dataset, sig), labels(run_id, sig), hub_records, hub_inside, hub_sep,
                            state_capitals=reference()["state"])


@shared_cache(show_spinner=False)
def run_neighbours(dataset, algorithm, sig):
    runs = all_runs(sig)
    sub = runs[(runs["dataset"] == dataset) & (runs["algorithm"] == algorithm)].copy()
    if algorithm == "DBSCAN":
        return ca.neighbours(sub, [("eps_km", ca.EPS_KM), ("min_samples", ca.MIN_SAMPLES)])
    if algorithm == "OPTICS":
        return ca.neighbours(sub, [("min_samples", ca.MIN_SAMPLES), ("xi", ca.OPTICS_XI)])
    num, dflt = sub[sub["min_samples"].notna()], sub[sub["min_samples"].isna()].assign(min_samples=-1)
    axes = [("min_cluster_size", ca.MIN_CLUSTER_SIZE), ("min_samples", ca.MIN_SAMPLES)]
    return {**ca.neighbours(num, axes), **ca.neighbours(dflt, [axes[0], ("min_samples", [-1])])}


@shared_cache(show_spinner=False, max_entries=RUN_CACHE)
def recurrence(run_id, sig):
    dataset, algorithm = run_id.split("|")[:2]
    nbs = run_neighbours(dataset, algorithm, sig).get(run_id, [])
    if not nbs:
        return pd.Series(dtype=float), nbs
    lab = labels(run_id, sig)
    rec = pd.concat([ca.cluster_recurrence(lab, labels(n, sig)) for n in nbs], axis=1).mean(axis=1)
    return rec, nbs


@shared_cache(show_spinner=False, max_entries=RUN_CACHE)
def hdbscan_extras(run_id, sig):
    return store.allruns_hdbscan_extras(run_id)


@shared_cache(show_spinner=False)
def optics(dataset, ms, sig):
    return store.allruns_optics(dataset, ms)


@shared_cache(show_spinner=False)
def dbscan_star(dataset, sig):
    return store.allruns_dbscan_star(dataset)


# ============================================================ sidebar
st.sidebar.header("Clustering settings")
dataset_name = st.sidebar.radio("Dataset", list(store.DATASETS.values()))
dataset = next(k for k, v in store.DATASETS.items() if v == dataset_name)
algorithm = st.sidebar.radio("Algorithm", ["DBSCAN", "HDBSCAN", "OPTICS"], horizontal=True)
if algorithm == "DBSCAN":
    eps = st.sidebar.select_slider("eps (km)", ca.EPS_KM, value=45)
    ms = st.sidebar.select_slider("min_samples", ca.MIN_SAMPLES, value=15)
    run = ca.run_id(dataset, "DBSCAN", eps_km=eps, min_samples=ms)
elif algorithm == "HDBSCAN":
    mcs = st.sidebar.select_slider("min_cluster_size", ca.MIN_CLUSTER_SIZE, value=15,
                                   help="The minimum number of breweries needed to form a cluster.")
    ms = st.sidebar.select_slider("min_samples", ["default"] + ca.MIN_SAMPLES, value=4,
                                  help="'default' = min_cluster_size, as in scikit-learn.")
    run = ca.run_id(dataset, "HDBSCAN", min_cluster_size=mcs, min_samples=None if ms == "default" else ms)
else:
    ms = st.sidebar.select_slider("min_samples", ca.MIN_SAMPLES, value=10)
    xi = st.sidebar.select_slider("xi", ca.OPTICS_XI, value=0.05)
    run = ca.run_id(dataset, "OPTICS", min_samples=ms, xi=xi)
top_n = st.sidebar.slider("Largest clusters to display", 5, 30, 15, step=5)
show_noise = st.sidebar.checkbox("Include noise on map", value=True)
show_all = st.sidebar.checkbox("Show all cluster points on map", value=True)
basemap = pick_basemap()
with st.sidebar.expander("Hub-merge settings"):
    hub_records = st.slider("A hub is a city with at least … breweries", 5, 30, 10)
    hub_inside = st.slider("… of which at least … sit in the cluster", 1, 10, 3)
    hub_sep = st.slider("Hubs farther apart than … km count as distinct", 25, 500, 100, 25)
st.sidebar.divider()
if st.sidebar.button("Reload stored results", help="Clear cached results and re-read the database."):
    st.cache_resource.clear()
    st.rerun()

SIG = store.signature()
missing = store.allruns_missing()
if missing:
    st.error("data/clustering_results.db lacks " + ", ".join(missing) + ". Run brewery_clustering_all_runs.ipynb.")
    st.stop()
runs = all_runs(SIG)
dr = runs[runs["dataset"] == dataset].copy()
sel = dr.set_index("run_id").loc[run]
df = records(dataset, SIG)
lab = labels(run, SIG)
ct = clusters(run, hub_records, hub_inside, hub_sep, SIG)
st.sidebar.caption(f"Run: `{run}`  \nSource: `data/clustering_results.db`")

# ============================================================ key findings across all runs
st.subheader(f"What the {len(runs):,} runs show")
star_all = dbscan_star(dataset, SIG)
db = dr[dr["algorithm"] == "DBSCAN"]
hd = dr[dr["algorithm"] == "HDBSCAN"]
op = dr[dr["algorithm"] == "OPTICS"]
zero = dr[dr["clusters_merging_hubs"] == 0]
chain = db[db["min_samples"] == 15].sort_values("eps_km").set_index("eps_km")
k1 = st.columns(4)
k1[0].metric("Runs stored for this dataset", f"{len(dr):,}")
k1[1].metric("DBSCAN runs that merge no hubs", f"{int((db['clusters_merging_hubs'] == 0).sum())} of {len(db)}")
k1[2].metric("HDBSCAN runs that merge no hubs", f"{int((hd['clusters_merging_hubs'] == 0).sum())} of {len(hd)}")
k1[3].metric("OPTICS runs that merge no hubs", f"{int((op['clusters_merging_hubs'] == 0).sum())} of {len(op)}")
st.markdown(f"""
- **A single radius chains hubs together.** With `min_samples` 15, the share of breweries sitting in clusters that
  join two hubs more than 100 km apart grows from **{chain.iloc[0]['records_in_hub_merging_clusters_%']:.0f}%** at
  {chain.index[0]:.0f} km to **{chain.loc[45, 'records_in_hub_merging_clusters_%']:.0f}%** at 45 km and
  **{chain.iloc[-1]['records_in_hub_merging_clusters_%']:.0f}%** at {chain.index[-1]:.0f} km; at 45 km the largest cluster
  reaches {chain.loc[45, 'largest_cluster_max_km']:.0f} km from its center. The only DBSCAN settings that avoid every
  merge leave {zero[zero['algorithm'] == 'DBSCAN']['noise_share_pct'].min():.0f}% or more of breweries as noise.
- **HDBSCAN separates hubs at a lower price.** Its runs that merge no hubs leave
  {zero[zero['algorithm'] == 'HDBSCAN']['noise_share_pct'].min():.0f}–{zero[zero['algorithm'] == 'HDBSCAN']['noise_share_pct'].max():.0f}%
  of breweries as noise.
- **HDBSCAN contains DBSCAN.** A flat cut of the HDBSCAN tree at any `eps` reproduces DBSCAN exactly on its core
  points ({int((star_all['ari_core_points'] > 0.9999).sum())} of {len(star_all)} settings); the two differ only on
  border points.
- **OPTICS finds the tightest groups** (median within-cluster distance {op['median_cluster_median_km'].median():.0f} km
  against {hd['median_cluster_median_km'].median():.0f} km for HDBSCAN) but is the least stable when its settings move
  one step (median ARI {op['mean_neighbour_ari'].median():.2f}).
""")

# ============================================================ dataset overview
st.divider()
st.subheader("Dataset overview")
status = df["coordinate_status"].value_counts()
k = st.columns(4)
k[0].metric("Records clustered", f"{len(df):,}")
k[1].metric("Coordinates corrected", int(status[status.index.str.startswith("corrected")].sum()),
            help="Latitude/longitude swapped back (8), one coordinate sign reversed (Mescan Brewery) and one "
                 "misplaced duplicate (Liefmans) fixed.")
k[2].metric("Flagged far from their city", int(status.get("flagged: far from its city", 0)),
            help="Kept at their recorded position; see Geographic Analysis → Outliers & coordinate checks.")
k[3].metric("Runs stored", f"{len(dr):,}")
with st.expander("Stored results and integrity"):
    st.dataframe(store.allruns_verification(), hide_index=True, width="stretch")
    st.dataframe(store.allruns_hdbscan_check().round(3), hide_index=True, width="stretch")
    st.caption("All label arrays come from brewery_clustering_all_runs.ipynb. The second table checks the `hdbscan` "
               "library against scikit-learn's HDBSCAN (adjusted Rand index; differences come from records stacked on "
               "identical coordinates).")

# ============================================================ compare all runs
st.divider()
st.subheader("Compare all runs and pick the best of each algorithm")
st.markdown("Every run gets a score from five criteria, each turned into a percentile rank among all runs of this "
            "dataset (0 = best, 1 = worst) and averaged with the weights below. Hub merges use the stored defaults "
            "(hub ≥ 10 breweries, ≥ 3 inside, > 100 km apart).")
w = st.columns(5)
weights = {"noise_share_pct": w[0].slider("Noise", 0.0, 3.0, 1.0, 0.5),
           "records_in_hub_merging_clusters_%": w[1].slider("Records in hub-merging clusters", 0.0, 3.0, 2.0, 0.5),
           "instability": w[2].slider("Instability (1 − neighbour ARI)", 0.0, 3.0, 1.0, 0.5),
           "non_recurrence": w[3].slider("Clusters not recurring", 0.0, 3.0, 1.0, 0.5),
           "median_cluster_p90_km": w[4].slider("Cluster spread (P90 km)", 0.0, 3.0, 1.0, 0.5)}
f1, f2 = st.columns(2)
min_clusters = f1.slider("Ignore runs with fewer clusters than", 1, 50, 5)
max_noise = f2.slider("Ignore runs with more noise than (%)", 10, 100, 60, 5)
cmp_ = dr.assign(instability=1 - dr["mean_neighbour_ari"], non_recurrence=1 - dr["median_cluster_recurrence"])
cmp_ = cmp_[(cmp_["cluster_count"] >= min_clusters) & (cmp_["noise_share_pct"] <= max_noise)].copy()
ranks = pd.DataFrame({c: cmp_[c].rank(pct=True).fillna(1.0) for c in weights})
total = sum(weights.values()) or 1.0
cmp_["score"] = sum(ranks[c] * wt for c, wt in weights.items()) / total
cmp_ = cmp_.sort_values("score")
best = cmp_.groupby("algorithm").head(1)
show_cols = ["run_id", "algorithm", "score", "cluster_count", "noise_share_pct", "clusters_merging_hubs",
             "records_in_hub_merging_clusters_%", "mean_neighbour_ari", "median_cluster_recurrence",
             "median_cluster_median_km", "median_cluster_p90_km", "largest_cluster_share_pct", "diagnostic"]
st.markdown("**Best run of each algorithm under these weights**")
st.dataframe(best[show_cols].round(3), hide_index=True, width="stretch")
fig = px.scatter(cmp_, x="noise_share_pct", y="records_in_hub_merging_clusters_%", color="algorithm",
                 size="cluster_count", size_max=22, color_discrete_map=ALG_COLORS, hover_name="run_id",
                 hover_data={"score": ":.3f", "mean_neighbour_ari": ":.2f", "cluster_count": True},
                 title="Every run: noise vs breweries in hub-merging clusters (lower-left is better on both)")
fig.add_trace(go.Scatter(x=best["noise_share_pct"], y=best["records_in_hub_merging_clusters_%"], mode="markers+text",
                         text=["best " + a for a in best["algorithm"]], textposition="top center",
                         name="best per algorithm", marker=dict(symbol="star", size=18, color="#151515")))
if run in cmp_["run_id"].values:
    r_ = cmp_.set_index("run_id").loc[run]
    fig.add_trace(go.Scatter(x=[r_["noise_share_pct"]], y=[r_["records_in_hub_merging_clusters_%"]], mode="markers",
                             name="selected run", marker=dict(size=22, color="rgba(0,0,0,0)",
                                                              line=dict(width=3, color="#151515"))))
fig.update_layout(height=520, xaxis_title="Noise (% of breweries)", yaxis_title="Breweries in hub-merging clusters (%)")
st.plotly_chart(fig, width="stretch")
with st.expander(f"All {len(cmp_):,} runs that pass the filters, ranked"):
    st.dataframe(cmp_[show_cols].round(3), hide_index=True, width="stretch")
st.download_button("Download every run (CSV)", data=dr.to_csv(index=False).encode("utf-8-sig"),
                   file_name=f"clustering_all_runs_{dataset}.csv", mime="text/csv")

# ============================================================ metric comparison
st.divider()
st.subheader("Metric comparison")
metric_options = {"Number of clusters": "cluster_count", "Clustered records (%)": "clustered_share_pct",
                  "Noise records (%)": "noise_share_pct", "Largest cluster share (%)": "largest_cluster_share_pct",
                  "Median cluster size": "median_cluster_size", "Clusters joining distinct hubs": "clusters_merging_hubs",
                  "Breweries in hub-merging clusters (%)": "records_in_hub_merging_clusters_%",
                  "Clusters crossing a state line": "clusters_over_1_state",
                  "Typical cluster spread (median km)": "median_cluster_median_km",
                  "Mean ARI with neighbouring settings": "mean_neighbour_ari",
                  "Median cluster recurrence": "median_cluster_recurrence"}
metric_label = st.selectbox("Choose a metric", list(metric_options))
mcol = metric_options[metric_label]
fig = px.strip(dr, x="algorithm", y=mcol, color="algorithm", color_discrete_map=ALG_COLORS, hover_name="run_id",
               title=f"{metric_label}: every run (the selected run is ringed)")
fig.add_trace(go.Scatter(x=[sel["algorithm"]], y=[sel[mcol]], mode="markers", name="selected run",
                         marker=dict(size=18, color="rgba(0,0,0,0)", line=dict(width=3, color="#151515"))))
fig.update_layout(yaxis_title=metric_label, showlegend=False, height=420)
st.plotly_chart(fig, width="stretch")

# ============================================================ parameter sensitivity
st.divider()
st.subheader("Parameter sensitivity")
st.markdown("Each cell is one stored run of the selected algorithm. A setting is more trustworthy when neighbouring "
            "cells give similar results; the selected run is outlined.")
sens_label = st.selectbox("Heatmap metric", list(metric_options), index=6, key="sens_metric")
scol = metric_options[sens_label]
g = dr[dr["algorithm"] == algorithm].copy()
if algorithm == "DBSCAN":
    rk, ck, rt, ctt = "eps_km", "min_samples", "eps (km)", "min_samples"
elif algorithm == "HDBSCAN":
    rk, ck, rt, ctt = "min_cluster_size", "min_samples", "min_cluster_size", "min_samples"
    g["min_samples"] = g["min_samples"].fillna(-1)
else:
    rk, ck, rt, ctt = "min_samples", "xi", "min_samples", "xi"
pv = g.pivot_table(index=rk, columns=ck, values=scol, aggfunc="first").sort_index()
pv.columns = ["default" if c == -1 else f"{c:g}" for c in pv.columns]
pv.index = [f"{v:g}" for v in pv.index]
hm = px.imshow(pv, aspect="auto", color_continuous_scale="Viridis",
               text_auto=".0f" if scol in {"cluster_count", "clusters_merging_hubs", "clusters_over_1_state"} else ".2f",
               labels={"x": ctt, "y": rt, "color": sens_label}, title=f"{algorithm}: {sens_label}")
hm.update_xaxes(type="category", side="top")
hm.update_yaxes(type="category")
sel_r = f"{sel[rk]:g}"
sel_c = "default" if pd.isna(sel[ck]) else f"{sel[ck]:g}"
if sel_r in pv.index and sel_c in pv.columns:
    xi_, yi_ = list(pv.columns).index(sel_c), list(pv.index).index(sel_r)
    hm.add_shape(type="rect", x0=xi_ - 0.5, x1=xi_ + 0.5, y0=yi_ - 0.5, y1=yi_ + 0.5,
                 line=dict(color="#ff3b00", width=3))
hm.update_layout(height=max(380, 26 * len(pv) + 140))
st.plotly_chart(hm, width="stretch")

with st.expander("k-distance curve (choosing DBSCAN's eps)"):
    kc1, kc2 = st.columns(2)
    k_ms = kc1.selectbox("min_samples (k)", ca.MIN_SAMPLES, index=ca.MIN_SAMPLES.index(15))
    k_cap = kc2.slider("Max distance shown (km)", 50, 2000, 200, step=50)
    kd = np.sort(store.allruns_core_distance(dataset, k_ms))
    kfig = px.line(pd.DataFrame({"rank": np.arange(len(kd)), "k_distance_km": kd}), x="rank", y="k_distance_km",
                   title=f"Sorted distance to the {k_ms}th nearest brewery (itself included)")
    if algorithm == "DBSCAN":
        kfig.add_hline(y=sel["eps_km"], line_dash="dot", annotation_text=f"selected eps = {sel['eps_km']:g} km")
    kfig.update_yaxes(range=[0, k_cap], title="k-distance (km)")
    kfig.update_xaxes(title="Breweries sorted by k-distance")
    st.plotly_chart(kfig, width="stretch")
    st.caption("The same distance is HDBSCAN's core distance. A good eps sits near the knee, where the curve starts "
               "rising steeply; computed on corrected coordinates.")

# ============================================================ one tree, every radius
st.divider()
st.subheader("DBSCAN and HDBSCAN: one tree, every radius")
star = dbscan_star(dataset, SIG)
c1, c2 = st.columns(2)
for box, col, title in [(c1, "ari_all_points", "all breweries"), (c2, "ari_core_points", "DBSCAN core points only")]:
    pvs = star.pivot_table(index="eps_km", columns="min_samples", values=col).sort_index()
    pvs.index, pvs.columns = [f"{v:g}" for v in pvs.index], [f"{v:g}" for v in pvs.columns]
    f = px.imshow(pvs, aspect="auto", zmin=0.5, zmax=1, color_continuous_scale="Blues", text_auto=".2f",
                  labels={"x": "min_samples", "y": "eps (km)", "color": "ARI"},
                  title=f"DBSCAN vs a cut of the HDBSCAN tree — {title}")
    f.update_xaxes(type="category", side="top")
    f.update_yaxes(type="category")
    f.update_layout(height=560)
    box.plotly_chart(f, width="stretch")
st.caption("HDBSCAN builds one tree of mutual reachability distances per `min_samples`; cutting it at a distance "
           "gives DBSCAN*, which is DBSCAN without border points. On core points the two agree in every setting "
           "(ARI ≥ 0.99997): the only gap is a DBSCAN cluster built around a single core point, which the tree cut, "
           "needing two points for a cluster, leaves as noise. On all points the agreement drops where DBSCAN "
           "attaches many border points to clusters. This is the HDBSCAN talk's argument: one HDBSCAN fit holds "
           "every DBSCAN radius at once.")

# ============================================================ explore the selected run
st.divider()
st.subheader(f"Explore the selected run: {run.split('|', 1)[1]}")
k = st.columns(5)
k[0].metric("Clusters", f"{int(sel['cluster_count']):,}")
k[1].metric("Clustered share", f"{sel['clustered_share_pct']:.1f}%")
k[2].metric("Noise share", f"{sel['noise_share_pct']:.1f}%")
k[3].metric("Clusters joining distinct hubs", int(ct["merges_distinct_hubs"].sum()) if len(ct) else 0,
            help="With the hub-merge settings in the sidebar.")
k[4].metric("Mean ARI with neighbouring runs",
            f"{sel['mean_neighbour_ari']:.2f}" if pd.notna(sel["mean_neighbour_ari"]) else "–")
if ct.empty:
    st.info("This run has no clusters: every brewery is noise.")
else:
    sort_by = st.radio("Order clusters by", ["size", "density (densest first)", "density (sparsest first)",
                                             "spread (P90 km)", "states spanned"], horizontal=True)
    order = {"size": ("size", False), "density (densest first)": ("density_per_1000km2", False),
             "density (sparsest first)": ("density_per_1000km2", True), "spread (P90 km)": ("p90_km", False),
             "states spanned": ("states", False)}[sort_by]
    view = ct.sort_values(order[0], ascending=order[1], na_position="last")
    cols = ["label", "size", "share_of_all_%", "center_latitude", "center_longitude", "min_latitude", "max_latitude",
            "min_longitude", "max_longitude", "median_km", "p90_km", "max_km", "hull_area_km2", "density_per_1000km2",
            "countries", "states", "cities", "dominant_country", "dominant_state", "dominant_state_share_%",
            "dominant_type", "dominant_type_share_%", "type_mix", "nearest_city", "nearest_city_km",
            "km_from_state_center", "direction_from_state_center", "state_capital", "km_from_state_capital",
            "direction_from_state_capital", "hubs_inside", "hub_names", "merges_distinct_hubs"]
    st.dataframe(view[[c for c in cols if c in view]].head(top_n).round(2), hide_index=True, width="stretch")
    st.caption("Spread = distance of the cluster's breweries from its center (median, 90th percentile, maximum). "
               "Density = breweries per 1,000 km² of the convex hull (blank when the points lie on a line or one "
               "spot). Direction and distance are measured from the dominant state's brewery center and capital to "
               "the cluster center.")
    st.download_button("Download this run's cluster table (CSV)", data=ct.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{run.replace('|', '_')}_clusters.csv", mime="text/csv")

    # ---------------- geographic cluster map
    st.markdown("#### Geographic cluster map")
    m_df = df.assign(cluster=lab)
    top_ids = ct.head(top_n)["cluster"].tolist()
    name_of = ct.set_index("cluster")["label"]
    m_df["map_group"] = np.where(m_df["cluster"] == -1, "Noise",
                                 np.where(m_df["cluster"].isin(top_ids), m_df["cluster"].map(name_of), "Other clusters"))
    if not show_noise:
        m_df = m_df[m_df["cluster"] != -1]
    if not show_all:
        m_df = m_df[(m_df["cluster"] == -1) | m_df["cluster"].isin(top_ids)]
    group_order = [name_of[c] for c in top_ids] + ["Other clusters", "Noise"]
    gm = px.scatter_map(m_df, lat="latitude", lon="longitude", color="map_group",
                        color_discrete_map=color_map(group_order, fixed={"Noise": NOISE_COLOR,
                                                                         "Other clusters": OTHER_COLOR}),
                        category_orders={"map_group": group_order}, hover_name="name",
                        hover_data={"city": True, "state_province_harmonized": True, "country": True,
                                    "brewery_type": True, "cluster": True, "latitude": ":.4f", "longitude": ":.4f"},
                        zoom=1, height=720, opacity=0.8, map_style=basemap,
                        center={"lat": float(m_df["latitude"].mean()), "lon": float(m_df["longitude"].mean())})
    gm.update_traces(marker={"size": 7})
    add_halo(gm)
    gm.update_layout(legend_title_text="Cluster", margin={"l": 0, "r": 0, "t": 10, "b": 0})
    st.plotly_chart(gm, width="stretch", config={"scrollZoom": True})
    st.caption("Borders and place names come from the basemap chosen in the sidebar; their detail grows as you zoom in.")

    # ---------------- cluster centers map
    st.markdown("#### Cluster centers: which regions carry the clustering")
    cc_ = ct.assign(kind=np.where(ct["merges_distinct_hubs"], "joins distinct hubs", "single hub or region"),
                    place=ct["label"].str.split(" · ").str[1])
    cm = point_map(cc_, lat="center_latitude", lon="center_longitude", size="size", color="kind",
                   color_discrete_map=MERGE_COLORS, hover_name="label",
                   hover_data={"size": True, "p90_km": ":.0f", "states": True, "nearest_city": True,
                               "dominant_type": True},
                   text=cc_["place"].where(cc_.index < 12, ""), size_max=45, opacity=0.9, height=600, style=basemap,
                   title="Center of every cluster (size = breweries; orange = the cluster joins distinct hubs)")
    cm.update_layout(legend_title_text="")
    st.plotly_chart(cm, width="stretch", config={"scrollZoom": True})

    # ---------------- clusters vs places
    st.markdown("#### Where clusters sit within their states, and who clusters")
    c1, c2 = st.columns(2)
    dirs = ct.groupby("direction_from_state_center")["size"].count().reindex(ca.COMPASS).fillna(0)
    pf = go.Figure(go.Barpolar(r=dirs.values, theta=dirs.index, marker_color="#2a78d6",
                               hovertemplate="%{theta}: %{r} clusters<extra></extra>"))
    pf.update_layout(title="Direction of each cluster center from its state's brewery center", height=420,
                     polar=dict(angularaxis=dict(direction="clockwise", rotation=90)))
    c1.plotly_chart(pf, width="stretch")
    near_cap = int((ct["km_from_state_capital"] <= 50).sum()) if "km_from_state_capital" in ct else 0
    near_cen = int((ct["km_from_state_center"] <= 50).sum())
    c1.caption(f"{near_cen} of {len(ct)} clusters have their center within 50 km of their state's brewery center, "
               f"{near_cap} within 50 km of the state capital.")
    lv = c2.radio("Clustered share by", ["country", "state", "brewery type"], horizontal=True, key="share_lv")
    col_ = {"country": "country", "state": "state_province_harmonized", "brewery type": "brewery_type"}[lv]
    shr = (df.assign(clustered=lab != -1).groupby(col_).agg(records=("id", "size"), clustered=("clustered", "mean"))
           .reset_index())
    shr["clustered_%"] = shr["clustered"] * 100
    shr = shr[shr["records"] >= (10 if lv != "brewery type" else 1)].nlargest(20, "records").sort_values("clustered_%")
    f = px.bar(shr, x="clustered_%", y=col_, orientation="h", text="records", color_discrete_sequence=["#2a78d6"],
               title=f"Share of breweries in a cluster, by {lv} (label = breweries)")
    f.add_vline(x=sel["clustered_share_pct"], line_dash="dot", annotation_text="overall")
    f.update_layout(yaxis_title="", xaxis_title="% clustered", height=420)
    c2.plotly_chart(f, width="stretch")

    # ---------------- hub merges
    st.markdown("#### Are the merges geographically logical?")
    mg = ct[ct["merges_distinct_hubs"]]
    if mg.empty:
        st.success(f"No cluster of this run joins two hubs more than {hub_sep} km apart (hub = city with ≥ "
                   f"{hub_records} breweries, ≥ {hub_inside} of them in the cluster).")
    else:
        st.dataframe(mg[["label", "size", "states", "countries", "max_km", "hubs_inside", "max_hub_separation_km",
                         "hub_names"]].round(0), hide_index=True, width="stretch")
        pts = df.assign(cluster=lab)
        pts = pts[pts["cluster"].isin(mg["cluster"])].assign(cluster_name=lambda x: x["cluster"].map(name_of))
        mm = point_map(pts, color="cluster_name", color_discrete_map=color_map(mg["label"]), hover_name="name",
                       hover_data=["city", "state_province_harmonized"], marker_size=6, height=560, style=basemap,
                       title="Clusters that join distinct hubs (black = hub cities)")
        h = ca.hubs(df, hub_records)
        line_cls = getattr(go, "Scattermap", None) or go.Scattermapbox
        mm.add_trace(line_cls(lat=h["hub_latitude"], lon=h["hub_longitude"], mode="markers", name="hub city",
                              text=h["hub"], hoverinfo="text", marker=dict(size=11, color="#151515", opacity=0.9)))
        mm.update_layout(legend_title_text="")
        st.plotly_chart(mm, width="stretch", config={"scrollZoom": True})

    # ---------------- stability of this run
    st.markdown("#### Does this grouping survive a one-step change of its parameters?")
    rec, nbs = recurrence(run, SIG)
    if not nbs:
        st.info("This run has no neighbouring run in the grid.")
    else:
        rix = runs.set_index("run_id")
        nb_tab = pd.DataFrame({"neighbour run": [n.split("|", 1)[1] for n in nbs],
                               "ARI with this run": [ca.ari(lab, labels(n, SIG)) for n in nbs],
                               "clusters": [int(rix.loc[n, "cluster_count"]) for n in nbs]})
        c1, c2 = st.columns([2, 3])
        c1.dataframe(nb_tab.round(3), hide_index=True, width="stretch")
        ctr = ct.assign(recurrence=ct["cluster"].map(rec))
        c1.metric("Clusters that recur (Jaccard ≥ 0.5 on average)", f"{(ctr['recurrence'] >= 0.5).mean() * 100:.0f}%")
        rm = point_map(ctr, lat="center_latitude", lon="center_longitude", size="size", color="recurrence",
                       color_continuous_scale="Viridis", range_color=(0, 1), hover_name="label",
                       hover_data={"recurrence": ":.2f"}, size_max=35, height=420, style=basemap,
                       title="Cluster centers coloured by recurrence in neighbouring runs (1 = reappears exactly)")
        c2.plotly_chart(rm, width="stretch", config={"scrollZoom": True})

    # ---------------- algorithm-specific views
    if algorithm == "HDBSCAN":
        st.markdown("#### HDBSCAN: condensed tree, persistence and outlier scores")
        ex = hdbscan_extras(run, SIG)
        tree = ex["condensed"].copy()
        finite = tree.loc[np.isfinite(tree["birth_km"]), "birth_km"]
        top_km = float(finite.max()) * 1.5 if len(finite) else 1e4
        tree["birth_km"] = tree["birth_km"].replace(np.inf, top_km)
        tree["death_km"] = tree["death_km"].clip(lower=0.01)
        kids = tree.groupby("parent")["node"].apply(list).to_dict()
        size_of = tree.set_index("node")["size_at_birth"]
        pos = {}
        stack = [(int(tree.loc[tree["parent"] == -1, "node"].iloc[0]), 0.0, 1.0)]
        while stack:  # lay the tree out left to right; each child gets a share of its parent's width
            node, x0, x1 = stack.pop()
            pos[node] = (x0, x1)
            ch = kids.get(node, [])
            tot = float(size_of.loc[ch].sum()) if ch else 0.0
            start = x0
            for c in ch:
                width = (x1 - x0) * float(size_of.loc[c]) / max(tot, 1.0)
                stack.append((int(c), start, start + width))
                start += width
        tree = tree[tree["node"].isin(pos)]
        tree["x0"] = tree["node"].map(lambda n: pos[n][0])
        tree["x1"] = tree["node"].map(lambda n: pos[n][1])
        chosen = tree["selected_label"] >= 0
        tf = go.Figure(go.Bar(x=(tree["x0"] + tree["x1"]) / 2, width=(tree["x1"] - tree["x0"]) * 0.92,
                              base=tree["death_km"], y=tree["birth_km"] - tree["death_km"],
                              marker_color=np.where(chosen, "#eb6834", "#b8c4d6"),
                              customdata=np.column_stack([tree["size_at_birth"], tree["selected_label"],
                                                          tree["birth_km"], tree["death_km"]]),
                              hovertemplate="breweries when it appears: %{customdata[0]:.0f}<br>selected as cluster: "
                                            "%{customdata[1]:.0f} (−1 = not selected)<br>appears below "
                                            "%{customdata[2]:,.0f} km, splits or dissolves at %{customdata[3]:,.1f} km"
                                            "<extra></extra>"))
        tf.update_layout(height=520, yaxis_type="log", yaxis_title="mutual reachability distance (km, log)",
                         xaxis=dict(showticklabels=False, title="clusters (width = breweries when the cluster appears)"),
                         title="Condensed tree: each bar is a cluster living between two distances; orange = selected")
        tf.update_yaxes(autorange="reversed")
        st.plotly_chart(tf, width="stretch")
        c1, c2 = st.columns(2)
        per = pd.DataFrame({"cluster": np.arange(len(ex["persistence"])), "persistence": ex["persistence"]})
        per["label"] = per["cluster"].map(name_of)
        f = px.bar(per.nlargest(20, "persistence").sort_values("persistence"), x="persistence", y="label",
                   orientation="h", color_discrete_sequence=["#eb6834"],
                   title="Most persistent clusters (how long and how large they live in the tree)")
        f.update_layout(yaxis_title="", height=520)
        c1.plotly_chart(f, width="stretch")
        gl = df.assign(glosh=ex["glosh"], cluster=lab).sort_values("glosh")
        gmap = point_map(gl, color="glosh", color_continuous_scale=SATURATED_SCALE, range_color=(0, 1),
                         hover_name="name", hover_data={"glosh": ":.2f", "cluster": True}, marker_size=5, height=520,
                         style=basemap, halo=False, title="GLOSH outlier score (1 = most unusual within its part of the tree)")
        c2.plotly_chart(gmap, width="stretch", config={"scrollZoom": True})
    elif algorithm == "OPTICS":
        st.markdown("#### OPTICS reachability plot")
        o = optics(dataset, int(sel["min_samples"]), SIG).copy()
        o["cluster"] = lab[o["position"].to_numpy()]
        o["reachability_km"] = o["reachability_km"].replace(np.inf, ca.OPTICS_MAX_EPS_KM)
        o["group"] = np.where(o["cluster"] == -1, "Noise", "Cluster")
        f = px.bar(o, x="order", y="reachability_km", color="group",
                   color_discrete_map={"Noise": NOISE_COLOR, "Cluster": "#1baf7a"},
                   title="Reachability distance in OPTICS order: valleys are clusters, peaks separate them")
        f.update_traces(marker_line_width=0)
        f.update_layout(bargap=0, height=420, yaxis_type="log", yaxis_title="reachability (km, log; capped at 520)",
                        xaxis_title="breweries in OPTICS order", legend_title_text="")
        st.plotly_chart(f, width="stretch")
    else:
        srow = star[(star["eps_km"] == sel["eps_km"]) & (star["min_samples"] == sel["min_samples"])].iloc[0]
        st.info(f"This DBSCAN run has **{int(srow['border_points']):,} border points**. The HDBSCAN tree cut at "
                f"{sel['eps_km']:g} km agrees with it at ARI {srow['ari_all_points']:.3f} over all breweries and "
                f"{srow['ari_core_points']:.3f} over core points: the only disagreement is where the border points go.")

    # ---------------- density contours
    st.markdown("#### Density: where breweries are thick on the ground")
    c1, c2, c3 = st.columns(3)
    radius = c1.slider("Heat-map smoothing radius (px)", 3, 30, 10)
    countries = df["country"].value_counts().index.tolist()
    region = c2.selectbox("Contour region", countries, key="contour_region")
    states_in = ["All states"] + df[df["country"] == region]["state_province_harmonized"].value_counts().index.tolist()
    reg_state = c3.selectbox("State / province", states_in, key="contour_state")
    dm = px.density_map(df, lat="latitude", lon="longitude", radius=radius, map_style=basemap, zoom=1, height=520,
                        center={"lat": float(df["latitude"].mean()), "lon": float(df["longitude"].mean())},
                        color_continuous_scale="Viridis", title="Brewery density with this run's cluster centers (white)")
    pt_cls = getattr(go, "Scattermap", None) or go.Scattermapbox
    dm.add_trace(pt_cls(lat=ct["center_latitude"], lon=ct["center_longitude"], mode="markers", name="cluster center",
                        text=ct["label"], hoverinfo="text", marker=dict(size=8, color="white")))
    dm.update_layout(margin=dict(l=0, r=0, t=40, b=0), coloraxis_showscale=False)
    st.plotly_chart(dm, width="stretch", config={"scrollZoom": True})
    sub = df[df["country"] == region]
    subc = ct[ct["dominant_country"] == region]
    if reg_state != "All states":
        sub = sub[sub["state_province_harmonized"] == reg_state]
        subc = subc[subc["dominant_state"] == reg_state]
    cf = go.Figure(go.Histogram2dContour(x=sub["longitude"], y=sub["latitude"], colorscale="Blues", ncontours=14,
                                         contours=dict(coloring="fill", showlines=True), showscale=False,
                                         hoverinfo="skip"))
    cf.add_trace(go.Scatter(x=sub["longitude"], y=sub["latitude"], mode="markers", name="breweries",
                            marker=dict(size=3, color="#555", opacity=0.5)))
    cf.add_trace(go.Scatter(x=subc["center_longitude"], y=subc["center_latitude"], mode="markers+text",
                            text=subc["label"].str.split(" · ").str[1].str.split(",").str[0], textposition="top center",
                            name="cluster centers",
                            marker=dict(size=11, color="#eb6834", line=dict(width=1.5, color="#151515"))))
    lat0 = float(sub["latitude"].mean()) if len(sub) else 0.0
    cf.update_layout(height=560, title=f"Contours of brewery density — {region if reg_state == 'All states' else reg_state}",
                     xaxis_title="longitude", yaxis_title="latitude", legend=dict(orientation="h", y=-0.12),
                     yaxis=dict(scaleanchor="x", scaleratio=1 / max(np.cos(np.radians(lat0)), 0.2)))
    st.plotly_chart(cf, width="stretch")
    st.caption("The contour plot counts breweries per grid cell and draws lines of equal density; this run's cluster "
               "centers in the region are marked. The axes keep the local east-west/north-south scale.")

# ============================================================ noise analysis
st.divider()
st.subheader("Noise analysis")
noise = df.assign(cluster=lab)[lab == -1]
if noise.empty:
    st.success("This run classified no brewery as noise.")
else:
    c1, c2 = st.columns(2)
    nc = (df.assign(noise=lab == -1).groupby("country").agg(total=("noise", "size"), noise=("noise", "sum"))
          .reset_index())
    nc["noise_rate_%"] = nc["noise"] / nc["total"] * 100
    nc = nc.sort_values("total", ascending=False)
    c1.metric("Noise records", f"{len(noise):,}")
    f = px.bar(nc.head(15), x="country", y="noise_rate_%", hover_data=["total", "noise"], title="Noise rate by country",
               color_discrete_sequence=[NOISE_COLOR])
    f.update_layout(xaxis_tickangle=-45, yaxis_title="Noise rate (%)")
    c1.plotly_chart(f, width="stretch")
    by_type = (df.assign(noise=lab == -1).groupby("brewery_type")["noise"].agg(["size", "mean"])
               .rename(columns={"size": "records", "mean": "noise_rate"}).reset_index())
    by_type["noise_rate_%"] = by_type["noise_rate"] * 100
    c2.markdown("##### Noise records by region")
    c2.dataframe(noise.groupby(["country", "state_province_harmonized"]).size().rename("noise records").reset_index()
                 .sort_values("noise records", ascending=False).head(30), hide_index=True, width="stretch")
    c2.markdown("##### Noise rate by brewery type")
    c2.dataframe(by_type.drop(columns="noise_rate").sort_values("noise_rate_%", ascending=False).round(1),
                 hide_index=True, width="stretch")
    st.download_button("Download all noise records (CSV)", data=noise.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{run.replace('|', '_')}_noise.csv", mime="text/csv")

# ============================================================ methodology
st.divider()
with st.expander("Methodology and limitations"):
    st.markdown(f"""
**Where the results come from.** brewery_clustering_all_runs.ipynb fits every setting once and stores the labels
(`allruns_*` tables). The page decodes the labels of the selected run; the cluster table, maps and stability views
are computed from them on the fly. Coordinates are the corrected ones (8 swaps undone, one sign error reversed, one misplaced duplicate fixed);
18 records far from their city are kept and flagged.

**Distance.** Haversine on radians; every radius is in kilometres divided by the Earth's mean radius
({ca.R:,.4f} km). No feature scaling, so `eps` keeps its geographic meaning.

**One radius for the whole map.** A per-state `eps` would redefine "dense" from state to state, like calling people
tall above 170 cm in one city and above 190 cm in another: both could report "50% tall" while measuring different
things. Every run uses one radius for every brewery.

**Parameters.** DBSCAN: `eps` (neighbourhood radius) and `min_samples` (breweries within `eps`, itself included, for a
core point). HDBSCAN: `min_cluster_size`, the minimum number of breweries needed to form a cluster, and `min_samples`
(scikit-learn convention; the `hdbscan` library receives `min_samples − 1` and builds the exact minimum spanning
tree). OPTICS: `min_samples` and `xi`, the minimum relative drop in reachability that marks a cluster boundary;
reachability search capped at {ca.OPTICS_MAX_EPS_KM} km.

**Hub merges.** A hub is a city with at least {hub_records} breweries (sidebar). A cluster joins distinct hubs when
at least two hubs each put {hub_inside} or more breweries in it and their centers are more than {hub_sep} km apart. The
ranking above uses the stored defaults (10, 3, 100 km).

**Stability.** For each run, the runs one grid step away along each parameter; ARI compares whole labellings and,
for each cluster, the best Jaccard overlap with a neighbour's clusters measures recurrence.

**Noise** (label −1) means a brewery did not meet the density rule of that run. It is not an error flag.
""")

# ============================================================ questions
st.divider()
st.header("Questions answered on this page")
best_txt = "; ".join(f"**{r['algorithm']}** `{r['run_id'].split('|', 2)[2]}` (noise {r['noise_share_pct']:.0f}%, "
                     f"{r['records_in_hub_merging_clusters_%']:.0f}% of breweries in hub-merging clusters, neighbour ARI "
                     f"{r['mean_neighbour_ari']:.2f})" for _, r in best.iterrows())
top_c = ct.head(5) if len(ct) else ct
largest_txt = ("Its largest clusters are " + ", ".join(
    f"{r['label'].split(' · ')[1]} ({int(r['size'])} breweries, {r['states']} state{'s' if r['states'] > 1 else ''}, "
    f"P90 {r['p90_km']:.0f} km)" for _, r in top_c.iterrows()) + ".") if len(top_c) else ""
st.markdown(f"""
**Why DBSCAN, HDBSCAN and OPTICS?**\\
All three are density-based: they need no preset number of clusters and can leave sparse breweries as noise. DBSCAN
uses one fixed radius; HDBSCAN builds the whole hierarchy of radii and keeps the clusters that persist longest; OPTICS
orders breweries by reachability and adapts the threshold locally. Running all of them over full grids shows which
conclusions depend on the method and which do not.

**Which run is best?**\\
Under the weights chosen above: {best_txt}. No run wins every criterion. Less noise is bought with merged hubs, and
fewer merges with more noise, so the weights state the trade-off openly instead of hiding it in one metric.

**Are the merges geographically logical?**\\
Up to roughly 25 km, DBSCAN merges stay inside metropolitan regions (San Diego with Los Angeles, Seattle with Tacoma,
the Bay Area). From about 30–45 km it chains whole corridors: Brooklyn to hubs over 300 km away, Brussels to München,
Baltimore to Boston. A cluster spanning Belgium, the Netherlands and Bavaria is not a brewery hub but a chain of
them, which matters more here than how large the largest cluster becomes. HDBSCAN keeps hubs apart in many more
settings because it does not use one radius everywhere.

**What does the selected run say about places?**\\
`{run.split('|', 1)[1]}` finds {int(sel['cluster_count'])} clusters and leaves {sel['noise_share_pct']:.1f}% as noise.
{largest_txt} The centers map shows which regions carry the clustering; the direction chart shows where clusters sit
relative to their state's brewery center, and the clustered-share chart which countries, states and brewery types
cluster most.

**What do these results not prove?**\\
They describe spatial patterns among the geocoded Open Brewery DB records. Shared coordinates, uneven geocoding and
the US-heavy sample shape every run; clusters are not market boundaries.
""")
