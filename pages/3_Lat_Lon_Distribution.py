import plotly.express as px
import streamlit as st
from common import pick_dataset, pick_basemap, analysis
from geo_plots import band_map, dist_plots, band_table, point_map, type_style

st.set_page_config(page_title="Latitude & Longitude Distribution", page_icon="🧭", layout="wide")
st.title("🧭 Latitude & longitude distribution")
df = analysis(pick_dataset())
df = df.assign(brewery_type=df["brewery_type"].fillna("Unknown").astype(str))
# one distinct colour per brewery type, fixed from the full dataset so it does not shift with the range selection
type_cmap, type_order = type_style(df)
basemap = pick_basemap()
width = st.sidebar.slider("Band width (degrees)", 1, 15, 5)

# ---------------- 1. range selector -> records on a labelled map ----------------
st.header("Select a latitude / longitude range")
la0, la1 = float(df.latitude.min() // 1), float(-(-df.latitude.max() // 1))
lo0, lo1 = float(df.longitude.min() // 1), float(-(-df.longitude.max() // 1))
c1, c2 = st.columns(2)
lat_rng = c1.slider("Latitude range (°)", la0, la1, (la0, la1), step=0.5)
lon_rng = c2.slider("Longitude range (°)", lo0, lo1, (lo0, lo1), step=0.5)
sel = df[df.latitude.between(*lat_rng) & df.longitude.between(*lon_rng)]

k = st.columns(5)
k[0].metric("Records in range", f"{len(sel):,}")
k[1].metric("Share of dataset", f"{len(sel)/len(df)*100:.1f}%")
k[2].metric("Countries", sel["country"].nunique())
k[3].metric("States / provinces", sel[["country", "state_province"]].drop_duplicates().shape[0])
k[4].metric("Cities", sel[["country", "state_province", "city"]].drop_duplicates().shape[0])

if sel.empty:
    st.warning("No breweries inside this range — widen the sliders.")
else:
    m, side = st.columns([3, 2])
    m.plotly_chart(point_map(sel, color="brewery_type", hover_name="name",
                             color_discrete_map=type_cmap, category_orders=type_order,
                             hover_data=["city", "state_province", "country", "latitude", "longitude"], opacity=0.8, style=basemap,
                             title=f"{len(sel):,} breweries in the selected range (zoom in for place names)"),
                   width="stretch")
    with side:
        for col, label in [("country", "Countries"), ("state_province", "States / provinces"), ("city", "Cities")]:
            top = sel[col].value_counts().head(10).rename_axis(col).reset_index(name="records")
            fig = px.bar(top.sort_values("records"), x="records", y=col, orientation="h", text="records",
                         title=f"Top {label.lower()} in range")
            fig.update_layout(height=230, margin=dict(l=0, r=0, t=35, b=0), yaxis_title="")
            st.plotly_chart(fig, width="stretch")
    with st.expander("Records in range"):
        st.dataframe(sel[["name", "brewery_type", "city", "state_province", "country", "latitude", "longitude"]],
                     hide_index=True, width="stretch")

# ---------------- 2. distribution of the values themselves ----------------
st.header("Distribution of coordinate values")
st.caption("Each band is coloured by how many breweries fall inside it (red outline = your selected range). "
           "This shows the distribution of the values, not individual brewery locations.")
left, right = st.columns(2)
for box, col, rng, scale in [(left, "latitude", lat_rng, "Viridis"), (right, "longitude", lon_rng, "Plasma")]:
    with box:
        st.subheader(col.title())
        st.plotly_chart(band_map(df, col, width, scale, highlight=rng), width="stretch")
        hist, ecdf = dist_plots(df, col, width)
        st.plotly_chart(hist, width="stretch")
        st.plotly_chart(ecdf, width="stretch")
        s = df[col].describe(percentiles=[.1, .25, .5, .75, .9]).to_frame().T
        s["skew"] = df[col].skew()
        st.dataframe(s.round(2), width="stretch")
        st.dataframe(band_table(df, col, width).query("records > 0").round(2), hide_index=True, width="stretch")
st.info("Latitude concentrates in the northern mid-latitudes (US/Europe); longitude is bimodal (Americas vs Europe/Oceania).")


st.divider()
st.header("Questions answered on this page")
lat_med, lon_med = df["latitude"].median(), df["longitude"].median()
st.markdown(
    f"""
**What does the selected range contain?**\\
**{len(sel):,} of {len(df):,} records ({len(sel)/len(df)*100:.1f}%)** in **{sel['country'].nunique():,} countries**. The sliders only filter the view; the data itself is unchanged (coordinates are the corrected ones).

**How are latitude and longitude distributed?**\\
Latitude is packed into the northern mid-latitudes (median {lat_med:.1f}°), where the United States and Europe lie; the southern records are Australia, New Zealand and South Africa. Longitude has two peaks, the Americas and Europe/Oceania, with a median of {lon_med:.1f}° because most records are American. The bands and histograms count records per degree; they are not a map of density per square kilometre, and a degree of longitude shrinks toward the poles.

**Do two longitude peaks mean breweries have two natural centers?**\\
No. They mirror where the records come from. A dataset with more Asian or South American records would draw a different picture.

**Can these plots find brewery hubs?**\\
Only the broad belts. Hubs need distances in kilometres, which the Geographic Analysis and Spatial Clustering pages provide.

**What do the summary statistics add?**\\
The median and quantiles locate the typical record and the spread; skewness shows which side the tail is on. They are statistics of angles, so they describe where records sit, not how far apart they are.
"""
)
