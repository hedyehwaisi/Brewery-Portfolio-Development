import plotly.express as px
import streamlit as st
from common import pick_dataset, pick_basemap, analysis, shared_cache
from geo_plots import build_coordinate_details, point_map, SATURATED_SCALE

st.set_page_config(page_title="Coordinate Groups", page_icon="📍", layout="wide")
st.title("📍 Records sharing the same coordinates")
key = pick_dataset()
basemap = pick_basemap()
df = analysis(key)


@shared_cache
def details(k):
    return build_coordinate_details(analysis(k))


det = details(key)
if det.empty:
    st.info("No shared coordinates."); st.stop()
min_size = st.sidebar.slider("Min records per location", 2, int(det["record_count"].max()), 2)
det = det[det["record_count"] >= min_size]
same, diff = det[det["same_identity"]], det[~det["same_identity"]]

k = st.columns(4)
k[0].metric("Shared locations", f"{len(det):,}")
k[1].metric("Records involved", f"{int(det['record_count'].sum()):,}")
k[2].metric("Identical identity", f"{len(same):,}")
k[3].metric("Differing identity", f"{len(diff):,}")
st.caption("Identity = name, city, state/province, country, address_1, postal_code, phone, website_url. "
           "A missing value vs a present value counts as a difference.")


def show(col, d, title, colorbar=None, **kw):
    if d.empty:
        col.info(f"No locations for: {title}"); return
    fig = point_map(d, title=f"{title} ({len(d):,})", size="record_count", hover_name="names",
                    hover_data=["city", "state_province", "country", "record_count", "differing_columns"],
                    opacity=0.9, style=basemap, **kw)
    if colorbar:
        fig.update_layout(coloraxis_colorbar=colorbar)
    col.plotly_chart(fig, width="stretch", config={"scrollZoom": True})


l, r = st.columns(2)
show(l, same, "Same coordinates, identical identity", color_discrete_sequence=["#2a9d8f"])
# n_differences is a small integer (1-6): saturated violet -> vermilion scale with no pale stop, so even the
# lowest value stands out from the basemap; one colour-bar tick per possible value
n_max = int(diff["n_differences"].max()) if not diff.empty else 1
show(r, diff, "Same coordinates, differing identity", color="n_differences",
     color_continuous_scale=SATURATED_SCALE, range_color=(1, max(n_max, 2)),
     colorbar=dict(title="Identity fields<br>that differ", tickvals=list(range(1, n_max + 1)), thickness=14))
st.caption("Marker size = records stacked on the location; colour = how many of the eight identity fields differ "
           "between them (violet = 1, vermilion = the most). The scale has no pale colour, and every marker has a "
           "dark ring, so it reads on both basemaps.")

a, b = st.columns(2)
dist = det.groupby(["record_count", "same_identity"]).size().reset_index(name="locations")
a.plotly_chart(px.bar(dist, x="record_count", y="locations", color="same_identity", barmode="stack",
                      title="Locations by number of stacked records"), width="stretch")
if not diff.empty:
    cnt = diff["differing_columns"].str.split(", ").explode().value_counts().rename_axis("column").reset_index(name="locations")
    b.plotly_chart(px.bar(cnt.sort_values("locations"), x="locations", y="column", orientation="h",
                          title="Which columns differ at shared coordinates?"), width="stretch")
st.subheader("Largest stacks")
st.dataframe(det.sort_values("record_count", ascending=False).drop(columns=["ids"]).head(100),
             hide_index=True, width="stretch")
with st.expander("All individual records at shared coordinates"):
    m = df.merge(det[["latitude", "longitude", "same_identity"]], on=["latitude", "longitude"])
    st.dataframe(m.sort_values(["latitude", "longitude"]), hide_index=True, width="stretch")


st.divider()
st.header("Questions answered on this page")
st.markdown(
    f"""
**Are records that share a coordinate duplicates?**\\
Rarely. The current selection has **{len(det):,} shared points** holding **{int(det['record_count'].sum()):,} records**. **{int(det['same_identity'].sum()):,}** of those points hold records that agree on every identity field, and **{int((~det['same_identity']).sum()):,}** hold records that differ in at least one. Most shared points are several businesses at one street address (Duplicates & Identity tab); coordinate equality alone would merge them wrongly.

**Why does it matter for clustering?**\\
A dozen records on one point look like a dense neighbourhood to a density-based algorithm. When those records are a brewing collective, that density is real; when they are a town-center fallback of the geocoder, it is not. The map and the identity columns tell the two apart before a dense spot is read as a hub.

**What does "identical identity" mean here?**\\
The same name, city, state/province, country, address, postal code, phone and website. A value present in one record and missing in the other counts as a difference. It is a transparent rule for this audit, not a legal definition of a business.

**Should these records be removed?**\\
No. Only records that meet the project's duplicate rule (identical in every column but `id`) were removed. This page is for checking, not deleting.
"""
)
