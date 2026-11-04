# Challenges and open questions

The problems we ran into while building the project: how we solved the ones we could, what came out of them, and
why the rest are still open. Every number refers to the bundled Open Brewery DB snapshot.

## Solved

### 1. Missing coordinates are not random
**Problem.** 2,468 of 11,959 records (20.6%) have no coordinates. Dropping them silently would bias every map.
**What we did.** We measured where the gaps fall before deciding anything: by type, country, state and city, with a
chi-square test and Cramér's V for each field.
**Result.** The gaps are structured. 83% of planning breweries have no coordinates, as do 30 of 36 Swedish records and
29 of 52 in Vermont, while Australia, Belgium and New Zealand have none missing. We keep the raw snapshot for counts
and state on every spatial result that it describes geocoded breweries only.

### 2. Columns that repeat each other
**Problem.** `state` and `state_province`, and `street` and `address_1`, looked like separate fields.
**What we did.** We compared them on every raw record after trimming spaces and ignoring case.
**Result.** Both pairs are identical on all 11,959 records. The Open Brewery DB documentation lists `state` and
`street` as deprecated copies. We use `state_province` and `address_1`.

### 3. One region, several labels
**Problem.** Belgium writes its three regions with seven labels in Dutch, French and English, one US record says
`MIssouri`, and one Cape Town brewery is filed under Gauteng.
**What we did.** We added `state_province_harmonized`, which merges the language variants and fixes the typo. For
the Cape Town record we checked every record against its own region's capital and the capitals of its country's other
regions. Only that record was wrong, so it is fixed by its id. The source column stays untouched.
**Result.** State-level counts, centers, type mixes and predictions no longer split a region in two.

### 4. Duplicates that are not exact
**Problem.** Three pairs are identical except for `id`. Six Belgian pairs agree on everything but their coordinates.
Two Paulaner records differ only by their website.
**What we did.** We removed the exact duplicates and measured the distance within each Belgian pair. We also built two
analysis datasets, all records and unique records, so every result can be checked with and without the Paulaner
twin.
**Result.** Five Belgian pairs are one site geocoded twice, less than 100 m apart. The two Liefmans records are 51 km
apart, so one of them is misplaced. The two datasets lead to the same conclusions throughout.

### 5. Wrong coordinates that pass every range check
**Problem.** Our first rule (latitude below −60°) found two records with swapped latitude and longitude. A swap or a
wrong sign that stays inside the valid range is invisible to range checks.
**What we did.** We compared every record with the GeoNames center of the city it names. Where the city has no
center, we compared it with its state capital instead. We then tested whether exchanging the two values, or reversing
one sign, brings the record home.
**Result.** 8 swapped records (4 in Bolzano, 2 in Portugal, 2 in the US), one longitude with the wrong sign (Mescan
Brewery in Mayo, stored in northern Germany), one misplaced Liefmans twin and 18 records more than 50 km from their
city. The fixes go into `data/coordinate_corrections.csv`, which every loader applies. The databases keep the
original values, and the 18 far-off records are flagged, not moved, because their true position is unknown.

### 6. Matching places to a gazetteer
**Problem.** To use capitals and real city centers we had to match the dataset's labels to GeoNames. Region names
differ by language and level (Irish counties, Dutch provinces, Singapore and Åland as "states"), and the same city
name exists in several states.
**What we did.** We used alias tables for the labels that differ, a seat priority (national capital, then
first-level seat, then second-level seat), and, for same-name places in the right state, the one nearest to the
city's breweries (the most populous one when the city has no geocoded brewery).
Names found only in another state are left unmatched rather than guessed. Cities with at least two breweries that
GeoNames could not place were looked up once in OpenStreetMap Nominatim.
**Result.** All 23 national capitals, 207 of 209 state capitals and 5,213 of 5,756 city centers (about 95% of raw
records); 68 of those came from Nominatim. Every row records how it was matched.

### 7. Stacked records
**Problem.** 800 records sit on just 336 coordinate points, which looked like geocoding failures.
**What we did.** We compared the addresses of the records at each shared point.
**Result.** At 299 of the 336 points every record gives the same street address: brewing collectives and shared
sites (19 brands at one address in Christchurch), not errors. We keep them.

### 8. Keeping every clustering run
**Problem.** The first clustering notebook kept the labels of only five hand-picked settings, so no other setting
could be inspected or compared.
**What we did.** We fit the whole grid: 264 DBSCAN, 336 HDBSCAN and 33 OPTICS runs per dataset, 1,266 in all. Labels
are stored as compressed int16 arrays, probabilities and outlier scores as uint8, condensed trees as float32.
**Result.** Every run fits in a 30 MB database, under GitHub's 50 MB warning, and the dashboard can show any of
them. The old per-candidate tables were dropped once nothing used them.

### 9. Two HDBSCAN conventions and an approximate tree
**Problem.** We wanted to show that a flat cut of the HDBSCAN tree is DBSCAN without border points. The `hdbscan`
library counts `min_samples` without the point itself, unlike scikit-learn. Its default approximate spanning tree
also broke the equivalence in 156 of 528 settings.
**What we did.** We pass `min_samples − 1` and build the exact tree (`approx_min_span_tree=False`).
**Result.** The cut matches DBSCAN on core points in all 528 settings. The only gap is a DBSCAN cluster built around
a single core point, which the cut, needing two points, leaves as noise; we verified that this explains every
difference.

### 10. Choosing clustering settings without ground truth
**Problem.** There is no correct clustering of breweries, and single scores reward the wrong things. Coverage
rewards one big cluster, and stability rewards clusters so merged that a small change cannot alter them.
**What we did.** We used one radius for the whole map (a per-state radius would change the meaning of "dense" from
state to state) and added a geographic test: does a cluster join two distinct brewery hubs more than 100 km apart?
We then ranked runs on hub merges, noise, compactness and stability together.
**Result.** DBSCAN chains hubs from about 30 km (Brussels to München at 45 km). Only 2 of 264 DBSCAN settings keep
hubs apart, and they leave 63–66% of breweries as noise. HDBSCAN manages it in 14 settings at 25–38% noise. The
dashboard shows the trade-off rather than declaring a winner.

### 11. Slow parallel OPTICS
**Problem.** OPTICS with `n_jobs=-1` took 155 s per fit.
**What we did.** We timed it with the default single process.
**Result.** 5 s per fit. The parallel neighbour search costs more than it saves at this size.

### 12. The GPU
**Problem.** We wanted to run the heavy computation on the RTX 4060.
**What we did.** We checked the GPU library (cuML) for the distance metric these algorithms need.
**Result.** It does not offer Haversine for them, and projecting to a flat metric would distort distances across
continents. The CPU run takes about 6 minutes, so we stayed on the CPU.

### 13. Zero probabilities in the prediction
**Problem.** A model that gives a new town zero probability fails completely (infinite log-loss) on the first
brewery that opens somewhere new.
**What we did.** We reserve probability for "a place with no brewery yet" with Good–Turing in its add-one form,
(N1 + 1) / (N + 1), smooth known places with a cross-validated pseudo-count, and pull each city's type mix towards its
state's.
**Result.** Every held-out record gets a finite score. Good–Turing expects 29.5% of new breweries in a town without
one; 35.0% of planning breweries are.

### 14. No dates to validate a forecast
**Problem.** The snapshot has no opening dates, so a forecast cannot be checked against what happened next.
**What we did.** We use planning breweries as a stand-in for future openings. The model is trained on operating
breweries and scored on planning ones.
**Result.** It beats a uniform guess on those future openings, but scores worse than in its own cross-validation.
The pipeline also leans to different states: Georgia carries 2.7 times its share of today's breweries, Tennessee
and Texas 2.2 times.

### 15. Readable maps on any basemap
**Problem.** Category colours clashed with each other and with map tiles, and points beyond ±85° latitude fall off
the Web Mercator tiles.
**What we did.** We switched to a validated palette, added a dark ring under every marker, put a basemap switch
(OpenStreetMap or CARTO Positron) in the sidebar, and drew such points on a globe projection.
**Result.** Markers stay distinct on both basemaps, and swapped records near the South Pole stay visible before
their correction.

### 16. Fitting a 512 MB host
**Problem.** One visitor touring every page used 440–453 MB on a real Streamlit server, and three at once 539 MB,
above the 512 MB of Render's free and Starter plans.
**What we did.** We measured before each change, with simulated visitors on a real server (`tools/memory_probe.py`):
- one shared copy per cached table (`common.shared_cache`) instead of `st.cache_data`'s pickled copy plus a fresh
  copy per call; pandas copy-on-write keeps callers from changing it;
- the statistics that need SciPy or scikit-learn precomputed for every page setting
  (`brewery_dashboard_statistics.ipynb`), and the convex hull, adjusted Rand index and kernel density rewritten in
  NumPy and tested against the libraries, so the app no longer imports either (about 80 MB);
- pyarrow's jemalloc memory pool, which hands back the memory used to send tables to the browser far better than
  its default pool;
- bounded per-run caches, capped malloc arenas and a chunked density computation.
**Result.** One visitor needs about 270 MB, six at once 413–422 MB and ten at once 449–486 MB. Drawing maps without
their marker rings would have saved only 7–8 MB, so the rings stay.

### 17. An inexact kernel density
**Problem.** Rewriting the kernel density in NumPy, we compared it with scikit-learn's `KernelDensity` and found them
far apart at small bandwidths: at 5 km, points 90–130 km from the nearest brewery were off by tens to hundreds of
log units, in both directions.
**What we did.** We checked both against an independent sum in plain Python with exact summation (`math.fsum`).
The direct NumPy sum agrees with it; scikit-learn's ball-tree shortcut does not.
**Result.** The exact version (`prediction_core.HaversineKDE`) now serves the notebook and the dashboard. The
cross-validated bandwidth is still 30 km, but the old scores had overrated 10 and 20 km, and the held-out
log-likelihood at 30 km is 2.29, not 2.33.

### 18. Notebooks written over several sessions
**Problem.** The early notebooks carry mixed execution orders, scratch cells and a few wrong checks, and the first
clustering notebook's save cell drops every table in the results database.
**What we did.** We documented them in place with markdown notes and comments and changed no code or output, so the
stored outputs remain a record of what was run.
**Result.** Each oddity is explained where it appears, and the dangerous save cell carries a warning.

## Open

### Population-adjusted density
Breweries per resident would separate brewing culture from plain population size. It needs consistent population
figures for states and cities in 23 countries. GeoNames has city populations, but not for every matched place or at
the region level.

### Region areas
Density per km² of a state, and coverage maps, need region boundaries. We use convex hulls of the breweries
themselves, which describe where breweries are, not the region they belong to. Natural Earth or national boundary
files would fill this gap.

### A real forecast
Predicting where and when the next brewery opens needs dated openings and closures from repeated snapshots or
another source. Until then the prediction page gives probabilities learned from one snapshot.

### The 2,468 records without coordinates
1,862 of them have a street address and could be geocoded, which would also fill the gaps in Sweden, Vermont and the
planning breweries. Geocoding them would change the analysis datasets and needs a geocoder with acceptable terms of
use for about 1,900 lookups. Its output would also need the same checks as the existing coordinates.

### The 18 flagged records
Their true positions are unknown: a geocoder may have matched the brewery's name to another town (Armstrong Brewing
in South San Francisco sits at Armstrong, British Columbia). Each needs a manual lookup.

### Places without a reference point
Two state capitals are unmatched: Flanders, whose seat Brussels lies outside the region, and the Isle of Man's
"Middle". 381 geocoded records sit in cities with no GeoNames or Nominatim center, so they are only checked for sign
errors, against their state capital.

### Type labels across countries
Germany labels 87% of its breweries brewpub and Australia 96% micro. The labels follow national habits, so type
comparisons hold within a country but not across countries. A common definition would need data the source does
not have, such as production volume.

### CPU on the free plan
Memory now fits, but Render's free instance has a tenth of a CPU core: after a restart the opening page needs about
2.8 CPU-seconds, roughly half a minute there. Making the first page lighter (fewer tables computed before anything is
shown) would help; a paid instance removes the problem.
