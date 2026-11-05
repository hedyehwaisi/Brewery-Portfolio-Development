# LinkedIn post (draft)

Written in the first person, since it is posted from a personal profile. Check both links before posting: the
repository must be public, and the Render service may need a minute to wake up.

---

**Where do breweries cluster, and where would the next one open?**

I've published my first end-to-end data science project: an interactive dashboard built on the 11,959 brewery records
of Open Brewery DB.

A few things the data taught me:

- **Missing data has a pattern.** 2,468 records have no coordinates, including 83% of the breweries still in planning.
  Every map in the project therefore describes geocoded breweries only.
- **Valid coordinates can still be wrong.** Checking each record against the center of its own city turned up 8
  swapped latitude/longitude pairs and an Irish brewery stored in northern Germany. A range check catches none of
  them.
- **One radius does not fit a whole map.** DBSCAN starts chaining brewery hubs together at about 30 km; at 45 km,
  Brussels and München end up in one cluster. Keeping hubs apart costs roughly a quarter of the breweries as noise.
  So I stored all 1,266 DBSCAN, HDBSCAN and OPTICS runs, and the dashboard can show any of them on the map.
- **Place predicts type, but mostly through labelling habits.** Knowing a brewery's state lifts the top-1 accuracy of
  guessing its type from 55% to 68%. German breweries are mostly labelled brewpub, Australian ones micro.
- **The pipeline moves.** Breweries in planning lean towards Georgia, Tennessee and Texas, at more than twice those
  states' share of today's breweries.

The lesson I'll keep: a larger cluster or a higher share of clustered points does not make a better model. The map
has to agree.

Live dashboard: https://brewery-dashboard.onrender.com/
Code and notebooks: https://github.com/hedyehwaisi/Brewery-Portfolio-Development

Feedback is very welcome.

#DataScience #GeospatialAnalysis #Python #Streamlit #Clustering
