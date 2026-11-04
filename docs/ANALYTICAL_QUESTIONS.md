# From raw questions to analytical answers

The project started from a brainstorm of questions about brewery distribution, concentration, brewery types,
dispersion, clustering and prediction. This page turns each one into a question the data can answer, says where the
answer lives, and what it is. Overlapping questions are grouped instead of each getting its own model.

## Questions and where they are answered

| Raw idea | Analytical question | Where | Answer | Status |
|---|---|---|---|---|
| Which country, state, city and type has the most records? | How is the snapshot distributed across levels and types? | Overview; Countries, States & Cities; Brewery Types | The United States holds 63.6% of geocoded records; California (915) and San Diego (90) lead their levels; micro and brewpub dominate | Answered for the snapshot, not for all real breweries |
| What is the shape of state and city counts? | How skewed are group sizes at each level? | Countries, States & Cities; Concentration tab | Among geocoded records, states follow a steep rank-size curve (Zipf exponent 1.41) while cities form a flat long tail (0.64), and 70% of cities have a single brewery | Answered |
| Do top states rely on a few cities? | How much of a state comes from its leading cities, and are top cities in top states? | Countries, States & Cities | Half of the ten largest cities are outside the ten largest states (Type B cities) | Answered |
| Do patterns persist from country to state to city? | Do ranks hold across levels and datasets? | Countries, States & Cities (rank shift) | Mostly, but geocoding gaps move ranks: Texas is 7th in the raw data and 15th among geocoded records | Answered |
| Which geographic level is "better"? | Which level suits which question? | Overview, Countries/States/Cities, Methodology | States show broad concentration, cities show local dominance that state totals hide; no single winner | Answered conceptually |
| How many records have coordinates, and is missingness random? | Does coverage vary by type and place? | Data Quality | 2,468 of 11,959 lack coordinates; planning breweries 83%, Sweden 30 of 36, Vermont 29 of 52; chi-square and Cramér's V per field | Answered; the cause behind it is unknown |
| Are `state` and `state_province` interchangeable? | Are the two fields equal after normalization? | Data Quality: State labels | Equal on all 11,959 raw records; `state` is a deprecated copy | Answered |
| Are region labels consistent? | Does one region appear under several labels, or a record under the wrong one? | Data Quality: State labels | Belgium uses seven labels for three regions, one US record says `MIssouri`, one Cape Town record is labelled Gauteng; all handled in `state_province_harmonized` | Answered |
| Are the coordinates right? | Does each record lie in the city it names? | Geographic Analysis: Outliers & coordinate checks | 8 swaps, 1 sign error, 1 misplaced twin corrected; 18 records far from their city flagged | Answered; the 18 need manual lookup |
| How dispersed are breweries within each state? | How far are breweries from the state's center, and how does that vary? | Geographic Analysis | Median, IQR, P90, shares within 10–100 km, automatic mean-to-median reading, on corrected coordinates | Answered |
| Are fixed 25/50/100 km thresholds meaningful? | Can spread be compared without one universal radius? | Geographic Analysis: Spatial dispersion | Robust summaries carry the comparison; the radius shares are shown as context | Answered |
| Are brewery centers the same as capitals or major cities? | Is the capital an important brewery city? | Geographic Analysis: Capitals & central cities | Among the 74 states with at least 30 records, the capital is the main brewery city in 35; a city's distance from the center rarely predicts its share | Answered (capitals from GeoNames) |
| What do DBSCAN, HDBSCAN and OPTICS reveal? | Which settings give geographically sensible groups? | Spatial Clustering; `brewery_clustering_all_runs.ipynb` | All 1,266 runs stored and comparable; DBSCAN chains hubs from about 30 km, HDBSCAN keeps them apart in 14 settings at 25–38% noise, OPTICS gives the tightest groups but the least stable | Answered as a trade-off, no single correct clustering |
| Can `eps` merge separate hubs? | Does a cluster join distinct brewery hubs? | Spatial Clustering: hub merges | Yes: Brussels to München at 45 km, Baltimore to Boston (772 km apart) | Answered |
| Which clustering metric is best? | How to weigh coverage, noise, compactness, hub merges and stability? | Spatial Clustering: ranking | Weighted ranking across measures; stability alone rewards over-merging | Answered methodologically |
| Can silhouette select the clustering? | Is a generic internal score suitable here? | Spatial Clustering: methodology | Not used to select: noise, irregular shapes and Haversine distances break its assumptions | Deliberately not used |
| Is HDBSCAN really DBSCAN at every radius? | Does a flat cut of the HDBSCAN tree reproduce DBSCAN? | Spatial Clustering: DBSCAN*; notebook | Yes on core points in all 528 settings; the only gap is DBSCAN clusters around a single core point | Answered |
| Do brewery types differ by place? | How does the type mix vary after accounting for size? | Brewery Types | Location quotients, chi-square and Cramér's V, state type profiles (Hellinger + Ward); labels follow national habits (Germany 87% brewpub, Australia 96% micro) | Answered |
| Do breweries of one type sit together? | Is a same-type nearest neighbour more common than chance? | Brewery Types: co-location | Permutation test within states: overall a same-type nearest neighbour is more common than chance, most clearly for contract and proprietor breweries | Answered |
| Do names match website domains? | How often does a brewery's name appear in its website address? | Data Quality: Names vs websites | 71.3% of raw websites contain the full name, 16.2% part of it; shared collective sites and social-media pages explain most of the rest | Answered |
| Can we predict exact names or URLs? | Is free-text prediction meaningful here? | — | No: high-cardinality strings with no target | Not pursued |
| Can we predict where a new brewery will open? | Given a country or state, which place is most likely, and of which type? | Prediction | As probabilities: state log-loss 2.95 against 3.41 for a uniform guess, city 2.82 against 4.20, type top-1 68% against 55%; planning breweries lean to Georgia (2.7×), Tennessee and Texas (2.2×) | Answered as probabilities, not a dated forecast |
| Does frequency mean market dominance? | What population do the results describe? | README, Methodology, every count | Records in Open Brewery DB, geocoded unevenly; not sales, production or market share | Must stay qualified |

## Why density-based clustering

The questions are whether brewery locations form dense groups, whether isolated breweries can be told apart from
them, and whether apparent hubs survive a change in the definition of "nearby". Density-based methods suit that: they
need no preset number of clusters and they can label noise.

- **DBSCAN** has readable parameters: a radius (`eps`) and the number of breweries needed within it
  (`min_samples`). One fixed radius struggles where density varies, and it chains hubs together.
- **HDBSCAN** builds the whole hierarchy of radii and keeps the clusters that persist, so it handles varying density.
  It still depends on `min_cluster_size` and `min_samples` and pays for clean clusters with noise.
- **OPTICS** orders breweries by reachability and cuts clusters where the reachability plot is steep (`xi`).
- **Haversine distance** throughout, because latitude and longitude are angles on a sphere. Degrees are not scaled:
  scaling would turn "45 km" into a number with no geographic meaning.

## What we compare runs on

1. Cluster count: structure, not quality.
2. Clustered and noise share: coverage, which can come from over-merging; noise is not error.
3. Largest cluster share: a warning sign for merged hubs.
4. Spread and hull density: how compact a cluster is.
5. Hub merges: whether a cluster joins distinct brewery hubs more than 100 km apart.
6. Stability: ARI and cluster recurrence against the neighbouring settings.
7. All records against unique records: sensitivity to the one-record difference.

A good run is readable on the map and holds up when its parameters move one step. No run is chosen for maximising
coverage or minimising cluster count.

## Findings that can be stated safely

- The raw snapshot has 11,959 records; 2,468 lack coordinates, and the gaps are not random.
- The geocoded datasets hold 9,488 (all records) and 9,487 (unique records) breweries.
- The snapshot is US-heavy (63.6% of geocoded records); it is not a balanced global sample.
- 28 records have wrong or doubtful coordinates; 10 are corrected and 18 flagged.
- At the radii where DBSCAN keeps noise low it merges distinct hubs; keeping hubs apart costs roughly a quarter or
  more of the breweries as noise with any algorithm.
- Where a brewery is says more about its type than about anything else in the data, and that is a labelling habit of
  each country as much as a business pattern.

These describe the snapshot. They are not causal claims or estimates for all breweries in the world.

## Next questions worth answering

1. Breweries per resident, with consistent population data for states and cities.
2. True coverage and density per region, with region boundaries.
3. A dated forecast, once openings and closures can be observed over time.
4. The 1,862 records with an address but no coordinates, geocoded and checked like the rest.
5. Type comparisons across countries, if a common type definition becomes available.
