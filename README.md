# Open Brewery DB: where breweries are, how they cluster, and where the next one would open

We took the 11,959 brewery records of an Open Brewery DB snapshot and asked three questions: how good is the
geographic information, how are breweries spread across countries, states and cities, and what do density-based
clustering and a simple probability model say about where breweries gather and where the next one is likely to
appear. Everything is reproducible from the SQLite files in `data/`, and the results live in a multi-page Streamlit
dashboard.

- **Live dashboard:** https://brewery-dashboard.onrender.com/
- **Repository:** https://github.com/hedyehwaisi/Brewery-Portfolio-Development
- **Data source:** [Open Brewery DB](https://www.openbrewerydb.org/), plus [GeoNames](https://www.geonames.org)
  (CC BY 4.0) and [OpenStreetMap](https://www.openstreetmap.org/copyright) Nominatim (ODbL) for capitals and city centers
- **Stack:** Python, pandas, NumPy, SciPy, scikit-learn, hdbscan, SQLite, Streamlit, Plotly

## What we found

**The data needed more cleaning than it looked.**
- 2,468 records (20.6%) have no coordinates, and the gaps are not random. Planning breweries lose 83%, Sweden 83%,
  Vermont 56% and Texas 52%, while Australia, Belgium and New Zealand lose none.
- `state` repeats `state_province` and `street` repeats `address_1` on every record.
- Belgium writes its three regions with seven labels in three languages.
- Checking each record against the center of the city it names found **8 records with latitude and longitude
  swapped**. Our first rule (latitude below −60°) caught only 2 of them. It also found one longitude with the wrong
  sign (an Irish brewery stored in northern Germany), one misplaced duplicate and 18 records far from their own city.
- One record carries the wrong province: a Cape Town brewery labelled Gauteng.

**Shared coordinates are mostly shared buildings.**
- 800 records sit on 336 shared points. At 299 of those points every record gives the same street address: brewing
  collectives and shared sites, such as 19 brands at one address in Christchurch, not geocoding errors.

**The geography is a US story with a long small-town tail.**
- The United States holds 64% of the geocoded records.
- 70% of cities have exactly one brewery. The city rank-size curve is flat (Zipf exponent 0.64) where the state
  curve is steep (1.41).
- Half of the ten largest brewery cities sit outside the ten largest states. Berlin, Auckland and Wellington are
  one-city regions, Chicago is a hub in a mid-ranked state, and Toronto looks dominant only because the rest of
  Ontario lost its coordinates.
- Dropping records without coordinates moves Texas from #7 to #15 and Austin from #7 to #25.

**Brewery types follow labelling habits more than business models.**
- Germany calls 87% of its breweries brewpubs, while Australia calls 96% of its breweries micro.
- US states split into two groups by whether they keep closed breweries in the data.
- Contract and incubator brewers sit next to the breweries that host them (4–6× more often than chance).

**Clustering shows why one radius is not enough.**
- We stored every setting: 264 DBSCAN, 336 HDBSCAN and 33 OPTICS runs per dataset.
- From about 30 km, DBSCAN chains separate hubs. At 45 km one cluster runs from Brussels to München (25 states).
  Only 2 DBSCAN settings keep hubs apart, and they leave 63–66% of breweries as noise; 14 HDBSCAN settings do so
  at 25–38% noise.
- A flat cut of the HDBSCAN tree reproduces DBSCAN on its core points in all 528 settings (ARI ≥ 0.99997; the only
  gap is DBSCAN clusters built around a single core point), the point of the HDBSCAN talk by Healy and McInnes.

**Where the next brewery would open.**
- A probability model beats a uniform guess on held-out breweries (city log-loss 2.82 vs 4.20). Knowing the state
  lifts the type prediction from 55% to 68% right on the first guess; knowing the city adds almost nothing.
- The 634 breweries in planning lean south and east: Georgia 2.7×, Tennessee 2.2× and Texas 2.2× their share of
  operating breweries.
- Good–Turing expected 29.5% of new breweries in towns that have none yet; 35.0% of the planning breweries are.

## The dashboard

| Page | What it answers |
|---|---|
| **Main app** – Overview, Pipeline & Quality, Duplicates & Identity, Missing Coordinates, Geography, Concentration, All vs Unique, Methodology | What the data holds, how records were cleaned, which records are duplicates or share a point, and the swapped coordinates |
| **Data Quality** | Missing values and whether they are random, unique and shared values, state labels, names vs websites |
| **Coordinate Groups** | Every point shared by several records and how their identities differ |
| **Latitude & Longitude** | How coordinate values are distributed, by range and band |
| **Geographic Analysis** | Centers, distances and dispersion around two origins (brewery center and capital or city center), coordinate checks and outliers |
| **Spatial Clustering** | Any of the 1,266 runs on the map, a ranking that picks the best run of each algorithm, hub merges, stability, condensed tree, reachability, density contours |
| **Brewery Types** | Counts and definitions, where each type is, type mix of top places, dominant type, patterns |
| **Countries, States & Cities** | Top states and their cities, top cities vs top states, type A/B cities, city distribution |
| **Prediction** | Most likely country, state, city and type of a new brewery, a density map, the planning pipeline, accuracy |

Every tab, and every single-view page, ends with a "Questions answered" section written from the numbers it shows. Maps can switch between
OpenStreetMap and CARTO Positron in the sidebar.

## Data and pipeline

| File | Contents |
|---|---|
| `data/raw_breweries.db` | The snapshot: 11,959 records, including those without coordinates |
| `data/analysis_all_records.db` | 9,488 geocoded records after removing 3 exact duplicates |
| `data/analysis_unique_records.db` | 9,487 records: one Paulaner record that differs only by website removed |
| `data/coordinate_corrections.csv` | 28 records found by the city check: 8 swapped back, 1 sign reversed, 1 copied from its twin, 18 flagged. Applied when data is loaded; the databases keep the original values |
| `data/reference_places.db` | National capitals (23/23), state capitals (207/209) and city centers (95% of raw records), each with its match method |
| `data/clustering_results.db` | Every clustering run (labels, metrics, stability, condensed trees, OPTICS reachability) in `allruns_*` tables, plus the summary tables of the first clustering pass |
| `data/prediction_results.db` | Tuned prediction settings, cross-validation and planning-test results |
| `data/dashboard_stats.db` | The tests the dashboard shows that need SciPy or scikit-learn, computed for every page setting so the app runs without either library |
| `data/results.db` | Geographic tables of the first pass (before corrections) |

The notebooks run in this order:

1. `merged_brewery_cleaned_for_eda.ipynb` – snapshot, duplicates, the two analysis datasets, exploratory analysis.
2. `brewery_geographic_analysis.ipynb` – centers, distances, dispersion and outliers (first pass, before corrections).
3. `tools/build_reference_places.py` – capitals and city centers from GeoNames (see the script's docstring for the
   files to download).
4. `brewery_clustering_all_runs.ipynb` – coordinate corrections and every DBSCAN, HDBSCAN and OPTICS run.
5. `brewery_prediction.ipynb` – tuning and validation of the prediction models.
6. `brewery_dashboard_statistics.ipynb` – the chi-square, Spearman, Ward and permutation tests the pages show,
   stored for every page setting.

`brewery_clustering_revised.ipynb` is the first clustering pass and is kept for the record; the dashboard no longer
needs it.

## Data preprocessing

The project performs data-quality checks, handles duplicate records, and uses case and whitespace normalization when comparing the `state` and `state_province` fields.

For geographic analysis, records with incomplete coordinates are excluded rather than having missing locations imputed. Spatial distances are calculated using the Haversine metric, with DBSCAN's neighbourhood radius converted from kilometres to angular distance as required.

Conventional numerical feature scaling is not described as part of the current geographic clustering workflow. This preserves the interpretation of the clustering parameters in geographic terms.

Duplicate handling, text normalization, and geographic distance conversion are treated as distinct preprocessing operations.

## Methods in brief

- **Geography.** Every brewery is compared with two origins at country, state and city level. Origin 1 is the mean
  position of the place's breweries; origin 2 is its capital, or at city level its GeoNames center. Dispersion is
  summarised by medians, quantiles and the mean-to-median ratio, which the dashboard reads out automatically.
- **Clustering.** DBSCAN, HDBSCAN and OPTICS, all with the Haversine metric and one radius for the whole map; a
  per-state radius would redefine "dense" from state to state.
  - Each run is judged on noise, on clusters that join distinct hubs (cities with 10+ breweries, more than 100 km
    apart), on spread, and on whether its clusters reappear when a parameter moves one step.
  - No setting was declared best in advance; the dashboard ranks them with weights you can change.
- **Prediction.** Place shares with Good–Turing smoothing for new places, type mixes shrunk toward the parent region,
  and a kernel density map. All settings were chosen by 5-fold cross-validation and tested on the planning breweries.

## What the project can and cannot predict

It can say **where the next brewery is most likely** at country, state and city level, **what type it would most
likely be**, and **how likely it is to open in a town that has no brewery yet**. Choose a type, and it says where that
type is most likely. These are probabilities learned from today's operating breweries, checked on breweries the
model never saw and on the 634 that are being planned. On those future openings the model still beats a uniform
guess, but less well than in cross-validation, because the geography of openings is moving south and east.

It cannot say **how many** breweries will open or **when**. The data has no opening dates, so there is nothing to
learn a rate from. A real forecast would need dated openings and closings and outside drivers such as population,
income or regulation. Predicting exact names or website addresses was dropped as an objective: it has no use. What
is useful is the name–website check now on the Data Quality page, which links 71% of websites to the brewery's own
name and exposes shared group sites.

## Challenges along the way

The full list, with what each attempt gave us and what is still open, is in
[`docs/CHALLENGES.md`](docs/CHALLENGES.md). In short:
- **Resolved:**
  - coordinate quality (swaps, a wrong sign, misplaced and stacked records);
  - multilingual and wrong region labels;
  - matching thousands of place names to an outside gazetteer;
  - keeping all 1,266 clustering runs inside a 30 MB database;
  - finding out why the `hdbscan` library disagreed with DBSCAN (its default approximate tree);
  - fitting the dashboard into a 512 MB host (ten visitors at once);
  - an inexact kernel density in scikit-learn far from the data, replaced by an exact one.
- **Open:**
  - population-adjusted density;
  - region areas for true coverage;
  - dated openings for real forecasting;
  - geocoding the 1,862 records that have an address but no coordinates;
  - a capital for Flanders, whose seat lies outside the region.

## Run it locally

Python 3.12 is the documented version (the code is also tested on 3.13).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-deploy.txt
python -m streamlit run app.py
```

On Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1` and run the same `python -m ...` commands.

Tests and notebooks need the full environment:

```bash
python -m pip install -r requirements.txt
python -m pytest tests -q
```

The tests pin the facts the dashboard text relies on and check the stored results: every clustering run's labels
against its stored metrics, the DBSCAN/HDBSCAN agreement on core points, the corrections log, the reference places and the
prediction results. To rebuild everything, run the notebooks in the order above with
`python -m jupyter nbconvert --to notebook --execute --inplace <notebook>`. The clustering notebook takes about
6 minutes and the prediction notebook about 2 on an 8-core laptop; no GPU is needed.

## Deployment

The dashboard runs on Render as a native Python web service. Build with `pip install -r requirements-deploy.txt` and
start with `streamlit run app.py --server.address 0.0.0.0 --server.port $PORT --server.headless true`, as recorded in
`render.yaml`.

We measured the app on a real Streamlit server with simulated visitors who open every page. Four changes brought it
well inside Render's 512 MB: one shared copy of each cached table (`common.shared_cache`), no SciPy or scikit-learn at
run time (their statistics are precomputed in `data/dashboard_stats.db`), pyarrow's jemalloc memory pool and capped
malloc arenas (both set in `render.yaml`). One visitor now needs about 270 MB, six at once about 420 MB and ten at
once under 490 MB, against 440 MB for one visitor before. CPU is now the limit on the free plan (0.1 CPU), so Starter
is the plan to use while the link is shared widely. [`DEPLOYMENT.md`](DEPLOYMENT.md) has the measurements, the plan
comparison and how Docker fits.

## Limits worth keeping in mind

- Spatial results describe geocoded breweries, and geocoding is uneven across types and places.
- The data is US-heavy, and countries label brewery types in their own way.
- Counts are records in a community database, not a census, a market share or a measure of size.
- Shared points, flagged records and approximate geocodes still shape local density; the dashboard shows where.
- Patterns are associations. Nothing here explains why a brewery opened where it did.

## Project structure

```text
.
├── app.py                          # main dashboard (8 tabs)
├── pages/                          # Data Quality, Coordinate Groups, Lat/Lon, Geographic Analysis,
│                                   # Spatial Clustering, Brewery Types, Countries/States/Cities, Prediction
├── common.py                       # cached loaders (with coordinate corrections), dataset and basemap pickers
├── quality_core.py                 # data quality, identity, corrections, harmonized labels, names vs websites
├── distribution_core.py            # frequencies, hierarchy, brewery types, concentration
├── origins_core.py                 # distances to brewery centers, capitals and city centers; coordinate checks
├── cluster_analysis.py             # clustering grids, cluster geography, hub merges, stability
├── clustering_store.py             # read-only access to the stored clustering runs
├── prediction_core.py              # place/type probabilities, kernel density, planning pipeline
├── stats_store.py                  # builds and reads data/dashboard_stats.db (precomputed statistical tests)
├── analysis_core.py                # coverage and extent helpers from the first geographic pass
├── geo_plots.py, palette.py, geographic_utils.py
├── tools/build_reference_places.py # builds data/reference_places.db from GeoNames
├── tools/memory_probe.py           # memory and CPU of a real server under simulated visitors
├── *.ipynb                         # the notebooks listed above
├── data/                           # SQLite databases and the corrections log
├── tests/                          # regression, integrity and runtime-import tests
├── docs/                           # analytical questions, data dictionary, challenges, LinkedIn draft
├── requirements.txt / requirements-deploy.txt / render.yaml / DEPLOYMENT.md
```

## License

No license is asserted yet. Before adding one, check the terms of the bundled data: Open Brewery DB's own terms,
GeoNames (CC BY 4.0, attribution required) and OpenStreetMap (ODbL, attribution and share-alike for derived
databases).
